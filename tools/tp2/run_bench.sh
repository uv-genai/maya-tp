#!/bin/bash
# TP2 controlled benchmark: rank 0 on strix-halo-1, rank 1 here.
#
# Usage: run_bench.sh [--model NAME | --pack PATH | -]
#   MAYA_PACK / MAYA_MODEL also work. Default: Maya-S-v2-IQ2_XXS.
set -e
cd "$(dirname "$0")/../.."
# shellcheck source=model_env.sh
source tools/tp2/model_env.sh

while (( $# )); do
  case "$1" in
    --model) MAYA_MODEL="${2:?--model needs NAME}"; MAYA_PACK=""; shift 2 ;;
    --pack) MAYA_PACK="${2:?--pack needs PATH}"; shift 2 ;;
    -) shift ;;
    *) echo "unexpected argument: $1" >&2; exit 2 ;;
  esac
done

P=$(maya_resolve_pack)
[ -d "$P" ] || { echo "pack not found: $P" >&2; exit 1; }
TOKS="${TOKS:-154822,25062,287,29905,371,25,7487,785,6722,315,9621,374}"
export MAYA_GLM_NO_MTP=1
export STRATA_GLM_TIMING=1

echo "=== pack: $P (model=${MAYA_MODEL:-}) ==="
echo "=== rank 0 on strix-halo-1 ==="
ssh strix-halo-1.local "MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 /tmp/maya-tp2 --glm-pack '$P' --tokens '$TOKS' --max-new 32 --greedy --spec-follow-save /tmp/r0.txt --max-context 4096 --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5" > /tmp/r0_log.txt 2>&1 &
R0PID=$!

sleep 2
echo "=== rank 1 here ==="
./build-hip/maya --glm-pack "$P" --tokens "$TOKS" --max-new 32 --greedy \
  --spec-follow-save /tmp/r1.txt --max-context 4096 \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host strix-halo-1.local --tp-bootstrap-port 18515 \
  --tp-rdma-device usb4_rdma5 2>&1 | tee /tmp/r1_log.txt | grep -E "timing|ms/token|error|Error|fail|pair|INFO|pack"

echo "=== rank 0 output ==="
wait $R0PID || true
grep -E "timing|ms/token|error|Error|fail|pair|INFO|pack" /tmp/r0_log.txt | head -15

echo "=== pair parity ==="
python tools/tp2/greedy_parity.py /tmp/r0.txt /tmp/r1.txt --min-agree 1.0 2>&1 || true
