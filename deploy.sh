#!/bin/bash
# Cross-build for the trashcan and push over HTTP (the box has wget, no scp).
# Usage: ./deploy.sh [cargo args...]   then: ssh akuma /tmp/akuma-wgpu selftest ...
# Needs: x86_64-linux-musl-gcc (musl-cross) and the rust target; box reachable as `ssh akuma`.
set -euo pipefail
cd "$(dirname "$0")"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=x86_64-linux-musl-gcc
cargo build --release --target x86_64-unknown-linux-musl "$@"
OUT=target/x86_64-unknown-linux-musl/release
HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
PORT=${PORT:-8099}
python3 -m http.server "$PORT" --bind 0.0.0.0 --directory "$OUT" >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1
ssh akuma "wget -q -O /tmp/akuma-wgpu http://$HOST_IP:$PORT/akuma-wgpu && chmod +x /tmp/akuma-wgpu && ls -l /tmp/akuma-wgpu"
