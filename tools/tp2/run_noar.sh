#!/bin/bash
# Quick no-AR measurement: TP2 mixer slice without all-reduce.
set -e
cd "$(dirname "$0")/../.."
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374

# clean up any stale rank 0
ssh strix-halo-1.local "pkill -x maya-tp2 2>/dev/null" || true
sleep 1

# start rank 0
ssh strix-halo-1.local "MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 MAYA_TP_NOAR=1 /tmp/maya-tp2 --glm-pack '$P' --tokens '$TOKS' --max-new 32 --greedy --max-context 4096 --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5" > /tmp/r0_noar.txt 2>&1 &
sleep 6

# start rank 1
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 MAYA_TP_NOAR=1 ./build-hip/maya --glm-pack "$P" --tokens "$TOKS" --max-new 32 --greedy \
  --max-context 4096 --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host strix-halo-1.local --tp-bootstrap-port 18515 \
  --tp-rdma-device usb4_rdma5 2>&1 | tee /tmp/r1_noar.txt | grep -E "timing|ms/token|error"

echo "=== rank 0 ==="
wait || true
grep -E "timing|ms/token" /tmp/r0_noar.txt || true
