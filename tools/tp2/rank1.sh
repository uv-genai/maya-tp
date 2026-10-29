#!/usr/bin/env bash
# Maya TP2 rank 1 (worker) — runs on strix-halo-2.
# Peer is rank 0 on strix-halo-1 (192.168.1.22).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MAYA_CONFIG="${MAYA_CONFIG:-$ROOT/maya-maya-s-v2-iq2_xxs-hip.json}"
ENGINE="${ENGINE:-$ROOT/build-hip/maya}"
BOOTSTRAP_HOST="${BOOTSTRAP_HOST:-192.168.1.22}"

RDEV=$(ibv_devices | awk '/usb4_rdma/{print $2, $1}' | sort | head -n1 | awk '{print $2}')
[ -n "$RDEV" ] || { echo "no usb4_rdma device found" >&2; exit 1; }

PACK=$(jq -r '(.args // []) as $a | [range(0; $a|length) | select($a[.] == "--glm-pack" or $a[.] == "--pack") | $a[.+1]][0] // ""' "$MAYA_CONFIG")

echo "rank1: pack=$PACK rdma=$RDEV bootstrap=$BOOTSTRAP_HOST"

# Wait for rank 0's bootstrap port before starting the engine.
# The engine uses a non-blocking connect and dies on EINPROGRESS if
# rank 0 is not yet listening.
echo "rank1: waiting for $BOOTSTRAP_HOST:18515 ..."
for _ in $(seq 1 60); do
  if (echo >/dev/tcp/"$BOOTSTRAP_HOST"/18515) 2>/dev/null; then
    echo "rank1: bootstrap port is open"
    break
  fi
  sleep 2
done

exec "$ENGINE" --serve --glm-pack "$PACK" \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host "$BOOTSTRAP_HOST" \
  --tp-bootstrap-port 18515 \
  --tp-rdma-device "$RDEV" \
  --max-context "${CONTEXT:-4096}"
