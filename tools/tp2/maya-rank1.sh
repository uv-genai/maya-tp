#!/bin/bash
# tools/tp2/maya-rank1.sh — start TP2 rank 1 (worker) on the second host.
#
# Usage: maya-rank1.sh [options]
#   --pack DIR          model pack (default from maya-<model>.json)
#   --host HOST         rank 0's hostname (default strix-halo-1.local)
#   --port N            bootstrap port (default 18515)
#   --rdma-device D     RDMA device (default usb4_rdma5)
#   --config FILE       maya-*.json config (default: first found)
#   --world N           TP world size (default 2)
#
# Prefer starting this BEFORE rank 0, or within rank 0's 120 s accept window.
# Do not probe the bootstrap port with /dev/tcp.
set -e
cd "$(dirname "$0")/../.."

PORT=18515
RDMA_DEV=usb4_rdma5
WORLD=2
CONFIG=""
PACK=""
HOST="strix-halo-1.local"

while [ $# -gt 0 ]; do
  case "$1" in
    --pack) PACK="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --rdma-device) RDMA_DEV="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --world) WORLD="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# //; s/^#//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# find config if not given
if [ -z "$CONFIG" ]; then
  CONFIG=$(ls maya-*.json 2>/dev/null | head -1 || true)
fi

# find pack from config if not given
if [ -z "$PACK" ] && [ -n "$CONFIG" ]; then
  PACK=$(python3 -c "
import json, sys
cfg = json.load(open('$CONFIG'))
args = cfg.get('args', [])
for flag in ('--glm-pack', '--pack'):
    if flag in args:
        print(args[args.index(flag) + 1])
        break
" 2>/dev/null || true)
fi

if [ -z "$PACK" ]; then
  echo "error: no pack found (give --pack or a maya-*.json with --glm-pack)" >&2
  exit 1
fi

# find engine binary
ENGINE="build-hip/maya"
[ -x "$ENGINE" ] || ENGINE="build/maya"
if [ ! -x "$ENGINE" ]; then
  echo "error: engine binary not found (build with cmake first)" >&2
  exit 1
fi

echo "=== TP2 rank 1 (worker) ==="
echo "  pack:         $PACK"
echo "  rank 0 host:  $HOST"
echo "  rdma device:  $RDMA_DEV"
echo "  bootstrap:    port $PORT"
echo "  world size:   $WORLD"
echo ""

# start the engine (rank 1, worker — no HTTP server)
exec "$ENGINE" --glm-pack "$PACK" \
  --tp-world-size "$WORLD" --tp-rank 1 \
  --tp-bootstrap-host "$HOST" --tp-bootstrap-port "$PORT" \
  --tp-rdma-device "$RDMA_DEV" \
  --max-context 4096 \
  "$@"
