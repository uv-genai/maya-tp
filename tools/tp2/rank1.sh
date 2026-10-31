#!/usr/bin/env bash
# Maya TP2 rank 1 (worker) — runs on strix-halo-2.
# Peer is rank 0 on strix-halo-1 (192.168.1.22).
#
# Usage: rank1.sh [--config PATH] [--model NAME | --pack PATH]
#   --config    base maya-*.json (else MAYA_CONFIG / model-derived / Maya-S default)
#   --model     short model id under MAYA_MODELS_DIR (else MAYA_MODEL)
#   --pack      explicit pack directory (else MAYA_PACK); overrides config --glm-pack
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
# shellcheck source=model_env.sh
source "$HERE/model_env.sh"

while (( $# )); do
  case "$1" in
    --config) MAYA_CONFIG="${2:?--config needs PATH}"; shift 2 ;;
    --model) MAYA_MODEL="${2:?--model needs NAME}"; MAYA_PACK=""; shift 2 ;;
    --pack) MAYA_PACK="${2:?--pack needs PATH}"; shift 2 ;;
    --help|-h)
      sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unexpected argument: $1" >&2; exit 2 ;;
  esac
done

MAYA_CONFIG=$(maya_resolve_config "$ROOT")
[ -f "$MAYA_CONFIG" ] || { echo "config not found: $MAYA_CONFIG" >&2; exit 1; }

ENGINE="${ENGINE:-$ROOT/build-hip/maya}"
BOOTSTRAP_HOST="${BOOTSTRAP_HOST:-192.168.1.22}"

RDEV=$(ibv_devices | awk '/usb4_rdma/{print $2, $1}' | sort | head -n1 | awk '{print $2}')
[ -n "$RDEV" ] || { echo "no usb4_rdma device found" >&2; exit 1; }

if [ -n "${MAYA_PACK:-}" ] || [ -n "${MAYA_MODEL:-}" ]; then
  PACK=$(maya_resolve_pack)
  BASE_PACK=$(jq -r '(.args // []) as $a | [range(0; $a|length) | select($a[.] == "--glm-pack" or $a[.] == "--pack") | $a[.+1]][0] // ""' "$MAYA_CONFIG")
  if [ "$PACK" != "$BASE_PACK" ]; then
    echo "rank1: overriding pack $BASE_PACK -> $PACK" >&2
  fi
else
  PACK=$(jq -r '(.args // []) as $a | [range(0; $a|length) | select($a[.] == "--glm-pack" or $a[.] == "--pack") | $a[.+1]][0] // ""' "$MAYA_CONFIG")
fi
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
