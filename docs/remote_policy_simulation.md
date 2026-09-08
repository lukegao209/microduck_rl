# Remote walking-policy simulation rehearsal

This rehearsal keeps the 50 Hz MuJoCo control loop local while sending each
61-float walking observation to `microduck-remote-policy-server`. The server
returns 14 action offsets from the walking ONNX. Network work happens on a
background thread: the simulation loop never waits for UDP and runs the local
walking ONNX whenever no matching, fresh remote action is available.

Only the walking policy is remote in this slice. Standing, sit/stand and other
behaviours remain local. Changing away from walking increments the policy
generation and invalidates queued or in-flight walking actions.

## Start the server

The ONNX Runtime shared library must match the Rust server architecture. The
development setup below uses the arm64 runtime installed in `microduck_rl`:

```bash
cd /Users/luke/Develop/pollen-robotics/microduck-remote-policy-server

rustup target add aarch64-apple-darwin --toolchain 1.89.0

ORT_DYLIB_PATH=/Users/luke/Develop/pollen-robotics/microduck_rl/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.24.4.dylib \
  cargo +1.89.0 run --release --target aarch64-apple-darwin \
  --bin microduck-remote-policy-server -- \
  --bind 127.0.0.1:7547 \
  --walking ../microduck/policies/alpha_walking.onnx
```

The ready message includes the SHA-256 identity of the loaded model. Port
`7547` is used here so an existing transport-only server on `7447` can remain
untouched.

## Start MuJoCo

In a second terminal:

```bash
cd /Users/luke/Develop/pollen-robotics/microduck_rl

DYLD_LIBRARY_PATH=/Users/luke/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib \
  .venv/bin/mjpython scripts/infer_policy.py \
  --walking ../microduck/policies/alpha_walking.onnx \
  --standing ../microduck/policies/alpha_stand.onnx \
  --new-cmd-obs \
  --lin-vel-x 0.3 \
  --remote-policy 127.0.0.1:7547
```

The first control tick uses local ONNX while the first UDP request is in
flight. Once fresh responses arrive, the terminal reports
`[remote-policy] fresh walking actions active` and the periodic status contains
`remote[...]`. Keyboard controls are read from the terminal, not the MuJoCo
window.

## Verify failover and recovery

1. With the robot walking, stop the server with `Ctrl+C`. The next tick without
   a fresh action prints `local ONNX resumed immediately`; MuJoCo keeps walking
   from its already-loaded local model.
2. Restart the same server at the same endpoint. The client reconnects in the
   background and automatically resumes fresh remote actions.
3. Press `Space` to clear the velocity command. If the standing policy becomes
   active, queued and in-flight walking actions are invalidated. Press the up
   arrow to select walking again.

The default action-age limit is 40 ms. Change it only for experiments:

```bash
--remote-max-action-age-ms 40
```

This is a simulation/bench prototype, not powered-robot acceptance. Raw UDP is
unauthenticated, model identity is reported but not yet pinned to a robot
manifest, and two-host Wi-Fi jitter/loss still needs closed-loop testing.
