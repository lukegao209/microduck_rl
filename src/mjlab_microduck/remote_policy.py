"""Non-blocking client for Microduck's experimental remote-policy protocol.

The simulation control loop must never wait for UDP. A worker thread owns the
socket while :class:`RemotePolicyClient` only exchanges bounded queue items and
accepts a response when it is fresh and matches the current policy generation.
"""

from __future__ import annotations

import json
import math
import os
import queue
import secrets
import select
import socket
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Self

PROTOCOL_VERSION = 1
OBSERVATION_LEN = 61
ACTION_LEN = 14
MAX_DATAGRAM_LEN = 4096


@dataclass
class RemotePolicyStats:
    submitted: int = 0
    request_queue_replaced: int = 0
    received: int = 0
    accepted: int = 0
    invalid: int = 0
    stale: int = 0
    send_errors: int = 0
    reconnects: int = 0


@dataclass(frozen=True)
class _Pending:
    sent_at: float
    sent_at_us: int
    network: str
    generation: int


class RemotePolicyClient:
    """Submit observations without blocking and take the newest fresh action."""

    def __init__(self, endpoint: str, max_action_age_ms: float = 40.0):
        if max_action_age_ms <= 0:
            raise ValueError("max_action_age_ms must be greater than zero")

        self.endpoint = endpoint
        self.max_action_age = max_action_age_ms / 1000.0
        self._socket = _connect_udp(endpoint)
        self._requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._responses: queue.Queue[tuple[dict[str, Any], float]] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._stats = RemotePolicyStats()
        self._stats_lock = threading.Lock()
        self._started = time.monotonic()
        self._session_id = secrets.randbits(64) ^ (os.getpid() << 32)
        self._generation = 0
        self._network: str | None = None
        self._next_sequence = 1
        self._last_applied_sequence = 0
        self._pending: dict[int, _Pending] = {}
        self._using_remote = False
        self._model_id: str | None = None
        self._last_rtt_ms: float | None = None
        self._last_inference_us: int | None = None
        self._next_reconnect_at = 0.0
        self._worker = threading.Thread(
            target=self._network_worker,
            name="remote-policy",
            daemon=True,
        )
        self._worker.start()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def using_remote(self) -> bool:
        return self._using_remote

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def stats(self) -> RemotePolicyStats:
        with self._stats_lock:
            return RemotePolicyStats(**asdict(self._stats))

    def status(self) -> str:
        stats = self.stats()
        source = "remote" if self._using_remote else "local"
        timing = ""
        if self._last_rtt_ms is not None and self._last_inference_us is not None:
            timing = (
                f" rtt={self._last_rtt_ms:.2f}ms"
                f" infer={self._last_inference_us}us"
            )
        return (
            f"{source} accepted={stats.accepted}/{stats.submitted}"
            f" stale={stats.stale} invalid={stats.invalid}"
            f" neterr={stats.send_errors} reconnect={stats.reconnects}{timing}"
        )

    def deactivate(self, reason: str = "policy transition") -> None:
        """Invalidate in-flight actions when remote walking is not selected."""
        if self._network is None:
            return
        self._transition_to(None)
        if self._using_remote:
            print(f"[remote-policy] {reason}; local ONNX resumed immediately")
        self._using_remote = False

    def submit_and_take(
        self,
        observation: Any,
        network: str = "walk",
    ) -> tuple[float, ...] | None:
        """Submit this tick and return a fresh action already received, if any.

        ``None`` is the ordinary local-fallback signal. It covers startup,
        network delay/loss, malformed output, a stopped server and transitions.
        """
        values = _observation_values(observation)
        if network != self._network:
            self._transition_to(network)

        now = time.monotonic()
        self._pending = {
            sequence: pending
            for sequence, pending in self._pending.items()
            if now - pending.sent_at <= self.max_action_age
        }

        sequence = self._next_sequence
        self._next_sequence = 1 if sequence == (2**64 - 1) else sequence + 1
        sent_at_us = int((now - self._started) * 1_000_000)
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self._session_id,
            "generation": self._generation,
            "sequence": sequence,
            "sent_at_us": sent_at_us,
            "network": network,
            "observation": values,
        }
        self._enqueue_request(request)
        self._pending[sequence] = _Pending(
            sent_at=now,
            sent_at_us=sent_at_us,
            network=network,
            generation=self._generation,
        )
        self._increment("submitted")

        newest: tuple[int, tuple[float, ...], dict[str, Any], _Pending, float] | None = None
        while True:
            try:
                response, received_at = self._responses.get_nowait()
            except queue.Empty:
                break
            self._increment("received")
            accepted = self._validate_response(response, received_at, network)
            if accepted is None:
                continue
            response_sequence, action, pending = accepted
            if newest is None or response_sequence > newest[0]:
                newest = (response_sequence, action, response, pending, received_at)

        if newest is None:
            self._fallback()
            return None

        response_sequence, action, response, pending, received_at = newest
        self._last_applied_sequence = response_sequence
        self._model_id = response["model_id"]
        self._last_inference_us = response["inference_us"]
        self._last_rtt_ms = (received_at - pending.sent_at) * 1000.0
        self._increment("accepted")
        if not self._using_remote:
            print(
                "[remote-policy] fresh walking actions active "
                f"model={self._model_id} rtt={self._last_rtt_ms:.2f}ms"
            )
        self._using_remote = True
        return action

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=1.0)
        self._socket.close()

    def _transition_to(self, network: str | None) -> None:
        self._generation = 1 if self._generation == (2**64 - 1) else self._generation + 1
        self._network = network
        self._last_applied_sequence = 0
        self._pending.clear()
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                break

    def _enqueue_request(self, request: dict[str, Any]) -> None:
        try:
            self._requests.put_nowait(request)
            return
        except queue.Full:
            pass

        try:
            replaced = self._requests.get_nowait()
            self._pending.pop(replaced["sequence"], None)
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            return
        self._increment("request_queue_replaced")

    def _validate_response(
        self,
        response: dict[str, Any],
        received_at: float,
        network: str,
    ) -> tuple[int, tuple[float, ...], _Pending] | None:
        sequence = response.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            self._increment("invalid")
            return None
        pending = self._pending.pop(sequence, None)
        if pending is None:
            self._increment("invalid")
            return None

        identity_matches = (
            response.get("protocol_version") == PROTOCOL_VERSION
            and response.get("session_id") == self._session_id
            and response.get("generation") == pending.generation == self._generation
            and response.get("sent_at_us") == pending.sent_at_us
            and response.get("network") == pending.network == network == self._network
            and sequence > self._last_applied_sequence
        )
        if not identity_matches:
            self._increment("invalid")
            return None
        if received_at - pending.sent_at > self.max_action_age:
            self._increment("stale")
            return None

        action = response.get("action")
        if not isinstance(action, list) or len(action) != ACTION_LEN:
            self._increment("invalid")
            return None
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in action
        ):
            self._increment("invalid")
            return None
        model_id = response.get("model_id")
        inference_us = response.get("inference_us")
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(inference_us, int)
            or isinstance(inference_us, bool)
            or inference_us < 0
        ):
            self._increment("invalid")
            return None
        return sequence, tuple(float(value) for value in action), pending

    def _fallback(self) -> None:
        if self._using_remote:
            print("[remote-policy] action unavailable; local ONNX resumed immediately")
        self._using_remote = False

    def _network_worker(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                request = None

            if request is not None:
                try:
                    payload = json.dumps(
                        request,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode()
                    if len(payload) > MAX_DATAGRAM_LEN:
                        raise ValueError(
                            f"request is {len(payload)} bytes, limit is {MAX_DATAGRAM_LEN}"
                        )
                    self._socket.send(payload)
                except (OSError, TypeError, ValueError):
                    self._increment("send_errors")
                    self._reconnect_socket()

            try:
                readable, _, _ = select.select([self._socket], [], [], 0.001)
            except (OSError, ValueError):
                if self._stop.is_set():
                    return
                self._increment("send_errors")
                self._reconnect_socket()
                continue
            if not readable:
                continue

            while not self._stop.is_set():
                try:
                    payload = self._socket.recv(MAX_DATAGRAM_LEN + 1)
                except BlockingIOError:
                    break
                except OSError:
                    # A connected UDP socket can surface an ICMP port-unreachable
                    # response here after the server stops. It is transient: keep
                    # the worker alive so a server restarted at the same endpoint
                    # is discovered by subsequent requests.
                    self._increment("send_errors")
                    self._reconnect_socket()
                    break
                if len(payload) > MAX_DATAGRAM_LEN:
                    self._increment("invalid")
                    continue
                try:
                    response = json.loads(payload)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._increment("invalid")
                    continue
                if not isinstance(response, dict):
                    self._increment("invalid")
                    continue
                self._enqueue_response((response, time.monotonic()))

    def _enqueue_response(self, response: tuple[dict[str, Any], float]) -> None:
        try:
            self._responses.put_nowait(response)
            return
        except queue.Full:
            pass
        try:
            self._responses.get_nowait()
        except queue.Empty:
            pass
        try:
            self._responses.put_nowait(response)
        except queue.Full:
            pass

    def _reconnect_socket(self) -> None:
        now = time.monotonic()
        if now < self._next_reconnect_at:
            return
        # Keep retrying fast enough for control testing without opening and
        # closing a socket on every 50 Hz tick while a server is offline.
        self._next_reconnect_at = now + 0.05
        try:
            self._socket.close()
        except OSError:
            pass
        try:
            self._socket = _connect_udp(self.endpoint)
        except OSError:
            self._increment("send_errors")
            return
        self._increment("reconnects")

    def _increment(self, field: str) -> None:
        with self._stats_lock:
            setattr(self._stats, field, getattr(self._stats, field) + 1)


def _connect_udp(endpoint: str) -> socket.socket:
    host, port = _split_endpoint(endpoint)
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)
    if not addresses:
        raise ValueError(f"could not resolve remote-policy endpoint {endpoint!r}")
    family, socktype, proto, _, address = addresses[0]
    udp = socket.socket(family, socktype, proto)
    udp.connect(address)
    udp.setblocking(False)
    return udp


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    if endpoint.startswith("["):
        closing = endpoint.find("]")
        if closing < 0 or closing + 1 >= len(endpoint) or endpoint[closing + 1] != ":":
            raise ValueError(f"invalid remote-policy endpoint {endpoint!r}")
        host = endpoint[1:closing]
        port_text = endpoint[closing + 2 :]
    else:
        try:
            host, port_text = endpoint.rsplit(":", 1)
        except ValueError as error:
            raise ValueError(f"invalid remote-policy endpoint {endpoint!r}") from error
    if not host:
        raise ValueError(f"invalid remote-policy endpoint {endpoint!r}")
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError(f"invalid remote-policy endpoint {endpoint!r}") from error
    if not 0 < port <= 65535:
        raise ValueError(f"invalid remote-policy port {port}")
    return host, port


def _observation_values(observation: Any) -> list[float]:
    values = observation.tolist() if hasattr(observation, "tolist") else list(observation)
    if len(values) != OBSERVATION_LEN:
        raise ValueError(
            f"remote observation has {len(values)} values, expected {OBSERVATION_LEN}"
        )
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("remote observation contains NaN or infinity")
    return converted
