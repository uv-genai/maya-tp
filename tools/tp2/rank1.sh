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
CONTEXT="${CONTEXT:-$(jq -r '(.args // []) as $a | [$a | to_entries[] | select(.value == "--max-context") | $a[.key+1]][-1] // 4096' "$MAYA_CONFIG")}"

echo "rank1: pack=$PACK rdma=$RDEV bootstrap=$BOOTSTRAP_HOST seqs=${SEQS:-1} context=$CONTEXT"

# Do NOT probe the bootstrap port with /dev/tcp: that completes a TCP
# connection and rank 0's single accept() takes it as the peer handshake,
# then fails (EAGAIN / EISDIR) when the probe closes.  The engine's own
# connect retries until rank 0 is listening (MAYA_TP_BOOTSTRAP_TIMEOUT).
# Use one sequence while multi-sequence TP2 remains gated (#11).
exec "$ENGINE" --glm-pack "$PACK" \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host "$BOOTSTRAP_HOST" \
  --tp-bootstrap-port 18515 \
  --tp-rdma-device "$RDEV" \
  --parallel "${SEQS:-1}" \
  --max-context "$CONTEXT"
