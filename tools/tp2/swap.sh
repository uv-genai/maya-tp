#!/usr/bin/env bash
# llama-swap launcher for the two-host Maya TP2 pair.
#
#   run PORT   : start rank 1 on strix-halo-2 (ssh), then exec rank 0 on PORT
#   stop       : stop both ranks (used as cmdStop by llama-swap)
#
# Only one TP2 pair can run at a time: both ranks share bootstrap port 18515.
# Rank 0 logs to stdout (llama-swap logs) and llama-swap polls
# http://127.0.0.1:$PORT/health until the pair is ready.
set -euo pipefail

exec > >(tee -a "$HOME/maya-tp2-swap.log") 2>&1
echo "=== maya-tp2-swap $(date '+%F %T') args: $*"

RANK1_HOST="${RANK1_HOST:-strix-halo-2.local}"
HERE=$(cd "$(dirname "$0")" && pwd)

stop_all() {
  echo "maya-tp2-swap: stopping rank 0 (local) and rank 1 ($RANK1_HOST)"
  pkill -x maya 2>/dev/null || true
  pkill -f "serve/server.py" 2>/dev/null || true
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pkill -x maya' 2>/dev/null || true
  # Wait for both ranks to exit so the next pair can bind the bootstrap port.
  for _ in $(seq 1 30); do
    if ! pgrep -x maya >/dev/null 2>&1 \
       && ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pgrep -x maya >/dev/null' 2>/dev/null; then
      break
    fi
    sleep 1
  done
  sleep 1
}

case "${1:?usage: maya-tp2-swap.sh run PORT | maya-tp2-swap.sh stop}" in
  stop)
    stop_all
    ;;
  run)
    PORT="${2:?maya-tp2-swap.sh run PORT: PORT missing}"
    echo "maya-tp2-swap: starting rank 1 on $RANK1_HOST"
    # Don't start while a previous pair still holds the bootstrap port.
    for _ in $(seq 1 15); do
      ss -tln | grep -qE ':(18515) ' || break
      sleep 1
    done
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
      "nohup $HERE/rank1.sh >/home/ugo/maya-tp2-rank1-swap.log 2>&1 &"
    sleep 2   # both ranks pair within 30 s of each other
    echo "maya-tp2-swap: starting rank 0 on port $PORT"
    exec "$HERE/rank0.sh" "$PORT"
    ;;
  *)
    echo "usage: maya-tp2-swap.sh run PORT | maya-tp2-swap.sh stop" >&2
    exit 2
    ;;
esac
