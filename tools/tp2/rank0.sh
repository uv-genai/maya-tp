#!/usr/bin/env bash
# Maya TP2 rank 0 (OpenAI server) — runs on strix-halo-1.
# Start rank1.sh on strix-halo-2 first (or within this engine's 120 s
# bootstrap accept window). Do NOT probe port 18515 with /dev/tcp.
#
# Usage: rank0.sh [PORT]   (default 8000)
#
# HTTP API: http://192.168.1.22:PORT/v1 (OpenAI chat/completions)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PORT="${1:-8000}"
MAYA_CONFIG="${MAYA_CONFIG:-$ROOT/maya-maya-s-v2-iq2_xxs-hip.json}"

# Device names shift after re-plugs; pick the lowest-GUID rail so both ranks agree.
RDEV=$(ibv_devices | awk '/usb4_rdma/{print $2, $1}' | sort | head -n1 | awk '{print $2}')
[ -n "$RDEV" ] || { echo "no usb4_rdma device found" >&2; exit 1; }

# Generate a TP2 config from the base config: add tp block and --tp-* flags.
# Multi-sequence TP2 remains gated (#11); llama-swap uses one sequence.
SEQS="${SEQS:-1}"
CONTEXT="${CONTEXT:-$(jq -r '(.args // []) as $a | [$a | to_entries[] | select(.value == "--max-context") | $a[.key+1]][-1] // 4096' "$MAYA_CONFIG")}"
TP_CONFIG=$(mktemp /tmp/maya-tp2-rank0-XXXXXX.json)
trap 'rm -f "$TP_CONFIG"' EXIT

jq --arg rdev "$RDEV" --argjson seqs "$SEQS" --arg context "$CONTEXT" \
   '.tp = {world_size: 2, rank: 0, bootstrap_port: 18515, rdma_device: $rdev}
    | .args += ["--tp-world-size","2","--tp-rank","0","--tp-bootstrap-port","18515","--tp-rdma-device",$rdev,
                "--parallel",($seqs|tostring),"--max-context",$context]' \
   "$MAYA_CONFIG" > "$TP_CONFIG"

TOKENIZER=$(jq -r '.tokenizer // ""' "$MAYA_CONFIG")

echo "rank0: port=$PORT rdma=$RDEV config=$TP_CONFIG seqs=$SEQS context=$CONTEXT"

exec "$ROOT/.venv/bin/python" "$ROOT/serve/server.py" --engine strata --config "$TP_CONFIG" \
  --host 0.0.0.0 --port "$PORT" \
  --tokenizer "$TOKENIZER"
