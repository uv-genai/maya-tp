#!/bin/bash
# Profile TP2 AR stall breakdown.
set -e
cd "$(dirname "$0")/../.."
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374

ssh strix-halo-1.local "pkill -x maya-tp2 2>/dev/null" || true
sleep 1

ssh strix-halo-1.local "MAYA_GLM_NO_MTP=1 STRATA_GLM_PROF=8 /tmp/maya-tp2 --glm-pack '$P' --tokens '$TOKS' --max-new 32 --greedy --max-context 4096 --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5" > /tmp/r0_prof2.txt 2>&1 &
sleep 8

MAYA_GLM_NO_MTP=1 STRATA_GLM_PROF=8 ./build-hip/maya --glm-pack "$P" --tokens "$TOKS" --max-new 32 --greedy \
  --max-context 4096 --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host strix-halo-1.local --tp-bootstrap-port 18515 \
  --tp-rdma-device usb4_rdma5 > /tmp/r1_prof2.txt 2>&1

echo "=== rank 1 profile ==="
grep -E "prof|ms/token" /tmp/r1_prof2.txt | head -35
echo "=== rank 0 profile ==="
wait || true
grep -E "prof|ms/token" /tmp/r0_prof2.txt | head -35
