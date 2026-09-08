import json
import socket
import threading
import time

import pytest

from mjlab_microduck.remote_policy import (
    ACTION_LEN,
    OBSERVATION_LEN,
    RemotePolicyClient,
)


class _Responder:
    def __init__(self, *, delay=0.0, mutate=None, port=0):
        self.delay = delay
        self.mutate = mutate
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(0.05)
        self.endpoint = f"127.0.0.1:{self.socket.getsockname()[1]}"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join(timeout=1.0)
        self.socket.close()

    def _run(self):
        while not self.stop.is_set():
            try:
                payload, peer = self.socket.recvfrom(4096)
            except TimeoutError:
                continue
            request = json.loads(payload)
            if self.delay:
                time.sleep(self.delay)
            response = {
                "protocol_version": request["protocol_version"],
                "session_id": request["session_id"],
                "generation": request["generation"],
                "sequence": request["sequence"],
                "sent_at_us": request["sent_at_us"],
                "network": request["network"],
                "model_id": "test-model",
                "inference_us": 12,
                "action": [0.25] * ACTION_LEN,
            }
            if self.mutate:
                self.mutate(response)
            self.socket.sendto(json.dumps(response).encode(), peer)


def _wait_for_action(client, timeout=0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        action = client.submit_and_take([0.0] * OBSERVATION_LEN)
        if action is not None:
            return action
        time.sleep(0.002)
    return None


def test_fresh_matching_action_is_accepted_without_blocking():
    with _Responder() as server, RemotePolicyClient(server.endpoint, 100.0) as client:
        started = time.monotonic()
        assert client.submit_and_take([0.0] * OBSERVATION_LEN) is None
        assert time.monotonic() - started < 0.01

        action = _wait_for_action(client)
        assert action == pytest.approx([0.25] * ACTION_LEN)
        assert client.using_remote
        assert client.model_id == "test-model"
        assert client.stats().accepted >= 1


def test_stale_response_falls_back_to_local():
    with _Responder(delay=0.03) as server, RemotePolicyClient(server.endpoint, 5.0) as client:
        deadline = time.monotonic() + 0.12
        actions = []
        while time.monotonic() < deadline:
            actions.append(client.submit_and_take([0.0] * OBSERVATION_LEN))
            time.sleep(0.002)

        assert all(action is None for action in actions)
        assert not client.using_remote
        assert client.stats().stale + client.stats().invalid > 0


def test_wrong_identity_is_rejected():
    def corrupt_session(response):
        response["session_id"] += 1

    with (
        _Responder(mutate=corrupt_session) as server,
        RemotePolicyClient(server.endpoint, 100.0) as client,
    ):
        assert _wait_for_action(client, timeout=0.08) is None
        assert client.stats().invalid > 0


def test_client_recovers_when_server_restarts_on_the_same_endpoint():
    first_server = _Responder()
    endpoint = first_server.endpoint
    port = first_server.socket.getsockname()[1]
    with RemotePolicyClient(endpoint, 100.0) as client:
        with first_server:
            assert _wait_for_action(client) is not None

        deadline = time.monotonic() + 0.08
        while time.monotonic() < deadline:
            client.submit_and_take([0.0] * OBSERVATION_LEN)
            time.sleep(0.002)
        assert not client.using_remote
        assert client.stats().reconnects <= 3

        with _Responder(port=port):
            assert _wait_for_action(client, timeout=0.12) is not None
            assert client.using_remote


@pytest.mark.parametrize("endpoint", ["localhost", "localhost:nope", "localhost:70000"])
def test_invalid_endpoint_is_rejected(endpoint):
    with pytest.raises(ValueError):
        RemotePolicyClient(endpoint)
