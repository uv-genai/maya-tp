#!/usr/bin/env bash
# Maya TP2 rank 1 (worker) — runs on strix-halo-2.
# Peer is rank 0 on strix-halo-1 (192.168.1.22).
# TCP bootstrap over LAN; RDMA data over usb4_rdma* (thunderbolt-ibverbs).
# Start rank 0 first, then this within 30 s.
#
# Usage: rank1.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MAYA_CONFIG="${MAYA_CONFIG:-$ROOT/maya-maya-s-v2-iq2_xxs-hip.json}"
ENGINE="${ENGINE:-$ROOT/build-hip/maya}"
BOOTSTRAP_HOST="${BOOTSTRAP_HOST:-192.168.1.22}"

# Device names shift after re-plugs; striping covers every rail anyway.
RDEV=$(ibv_devices | awk '/usb4_rdma/{print $2, $1}' | sort | head -n1 | awk '{print $2}')
[ -n "$RDEV" ] || { echo "no usb4_rdma device found" >&2; exit 1; }

# Read the pack path from the config.
PACK=$(python3 -c "
import json
cfg = json.load(open('$MAYA_CONFIG'))
args = cfg.get('args', [])
for flag in ('--glm-pack', '--pack'):
    if flag in args:
        print(args[args.index(flag) + 1])
        break
")

echo "rank1: pack=$PACK rdma=$RDEV bootstrap=$BOOTSTRAP_HOST"

exec "$ENGINE" --glm-pack "$PACK" \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host "$BOOTSTRAP_HOST" \
  --tp-bootstrap-port 18515 \
  --tp-rdma-device "$RDEV" \
  --max-context "${CONTEXT:-4096}" \
  "${EXTRA_ARGS:-}"
