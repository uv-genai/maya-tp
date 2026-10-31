#!/usr/bin/env bash
# Maya TP2 rank 0 (OpenAI server) — runs on strix-halo-1.
# Start rank1.sh on strix-halo-2 first (or within this engine's 120 s
# bootstrap accept window). Do NOT probe port 18515 with /dev/tcp.
#
# Usage: rank0.sh [PORT] [--config PATH] [--model NAME | --pack PATH]
#   PORT        HTTP port (default 8000)
#   --config    base maya-*.json (else MAYA_CONFIG / model-derived / Maya-S default)
#   --model     short model id under MAYA_MODELS_DIR (else MAYA_MODEL)
#   --pack      explicit pack directory (else MAYA_PACK); overrides config --glm-pack
#
# HTTP API: http://192.168.1.22:PORT/v1 (OpenAI chat/completions)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
# shellcheck source=model_env.sh
source "$HERE/model_env.sh"

PORT=8000
PORT_SET=0
while (( $# )); do
  case "$1" in
    --config) MAYA_CONFIG="${2:?--config needs PATH}"; shift 2 ;;
    --model) MAYA_MODEL="${2:?--model needs NAME}"; MAYA_PACK=""; shift 2 ;;
    --pack) MAYA_PACK="${2:?--pack needs PATH}"; shift 2 ;;
    --help|-h)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      if (( PORT_SET )); then echo "unexpected argument: $1" >&2; exit 2; fi
      PORT="$1"; PORT_SET=1; shift ;;
  esac
done

BASE_CONFIG=$(maya_resolve_config "$ROOT")
[ -f "$BASE_CONFIG" ] || { echo "config not found: $BASE_CONFIG" >&2; exit 1; }

WORK=$(mktemp -d /tmp/maya-tp2-rank0-XXXXXX)
trap 'rm -rf "$WORK"' EXIT

if [ -n "${MAYA_PACK:-}" ] || [ -n "${MAYA_MODEL:-}" ]; then
  PACK=$(maya_resolve_pack)
  BASE_PACK=$(jq -r '(.args // []) as $a | [range(0; $a|length) | select($a[.] == "--glm-pack" or $a[.] == "--pack") | $a[.+1]][0] // ""' "$BASE_CONFIG")
  if [ "$PACK" != "$BASE_PACK" ]; then
    echo "rank0: overriding pack $BASE_PACK -> $PACK" >&2
    maya_config_with_pack "$BASE_CONFIG" "$PACK" > "$WORK/base.json"
    BASE_CONFIG="$WORK/base.json"
  fi
fi

# Device names shift after re-plugs; pick the lowest-GUID rail so both ranks agree.
RDEV=$(ibv_devices | awk '/usb4_rdma/{print $2, $1}' | sort | head -n1 | awk '{print $2}')
[ -n "$RDEV" ] || { echo "no usb4_rdma device found" >&2; exit 1; }

# Generate a TP2 config from the base config: add tp block and --tp-* flags.
# Multi-sequence TP2 remains gated (#11); llama-swap uses one sequence.
SEQS="${SEQS:-1}"
CONTEXT="${CONTEXT:-$(jq -r '(.args // []) as $a | [$a | to_entries[] | select(.value == "--max-context") | $a[.key+1]][-1] // 4096' "$BASE_CONFIG")}"
TP_CONFIG="$WORK/rank0.json"

jq --arg rdev "$RDEV" --argjson seqs "$SEQS" --arg context "$CONTEXT" \
   '.tp = {world_size: 2, rank: 0, bootstrap_port: 18515, rdma_device: $rdev}
    | .args += ["--tp-world-size","2","--tp-rank","0","--tp-bootstrap-port","18515","--tp-rdma-device",$rdev,
                "--parallel",($seqs|tostring),"--max-context",$context]' \
   "$BASE_CONFIG" > "$TP_CONFIG"

TOKENIZER=$(jq -r '.tokenizer // ""' "$BASE_CONFIG")
PACK=$(jq -r '(.args // []) as $a | [range(0; $a|length) | select($a[.] == "--glm-pack" or $a[.] == "--pack") | $a[.+1]][0] // ""' "$BASE_CONFIG")

echo "rank0: port=$PORT rdma=$RDEV config=$TP_CONFIG pack=$PACK seqs=$SEQS context=$CONTEXT"

# Keep env for child rank scripts / tests that re-enter this launcher.
export MAYA_CONFIG="$BASE_CONFIG" MAYA_PACK="${MAYA_PACK:-}" MAYA_MODEL="${MAYA_MODEL:-}"

exec "$ROOT/.venv/bin/python" "$ROOT/serve/server.py" --engine strata --config "$TP_CONFIG" \
  --host 0.0.0.0 --port "$PORT" \
  --tokenizer "$TOKENIZER"
