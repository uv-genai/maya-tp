#!/bin/sh
# tools/tp2/run_comm_test.sh - run tp_comm_rdma_test on both hosts and compare.
#
#   tools/tp2/run_comm_test.sh                 # rank0 on strix-halo-1, rank1 here
#   tools/tp2/run_comm_test.sh --device usb4_rdma5
#
# Expects the test binary in build-tp2/tp_comm_rdma_test (or $BIN) and SSH to $RANK0_HOST.
set -e
cd "$(dirname "$0")/../.." || exit 1
BIN="${BIN:-build-tp2/tp_comm_rdma_test}"
RANK0_HOST="${RANK0_HOST:-strix-halo-1.local}"
PORT="${PORT:-18515}"
DEV=""
[ -x "$BIN" ] || { echo "build $BIN first (cmake -DSTRATA_ENABLE_TP2_RDMA=ON -DSTRATA_ENABLE_HIP=ON)"; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEV="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
DEVSRC=""
[ -n "$DEV" ] && DEVSRC="--rdma-device $DEV"
if [ ! -x /usr/bin/ssh ]; then echo "ssh not found"; exit 1; fi

# rank 1 here, rank 0 on $RANK0_HOST (start rank 0 first)
echo "rank 0 on $RANK0_HOST ..."
ssh "$RANK0_HOST" "cd $PWD && $BIN --rank 0 --bootstrap-port $PORT $DEVSRC" &
R0=$!
sleep 1
echo "rank 1 here ..."
"$BIN" --rank 1 --bootstrap-host "$RANK0_HOST" --bootstrap-port $PORT $DEVSRC
R1=$?
wait $R0
R0s=$?
echo "rank0 exit $R0s  rank1 exit $R1"
[ "$R0s" = 0 ] && [ "$R1" = 0 ]
