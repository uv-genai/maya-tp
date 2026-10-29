#!/usr/bin/env bash
# llama-swap: two-host Maya TP2 (RDMA) GLM-5.3-Flash.
#
#   run PORT : rank 1 on $RANK1_HOST, then exec rank 0 on PORT (cmdStart)
#   stop     : stop both ranks (cmdStop)
#
# Only one TP2 pair at a time (shared bootstrap port 18515).
# Example llama-swap config:
#   "maya-glm5-tp2":
#     cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT}
#     cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
#     proxy: http://127.0.0.1:${PORT}
set -euo pipefail

exec > >(tee -a "$HOME/maya-tp2-swap.log") 2>&1
echo "=== maya-tp2-swap $(date '+%F %T') args: $*"

RANK1_HOST="${RANK1_HOST:-strix-halo-2.local}"
HERE=$(cd "$(dirname "$0")" && pwd)
TP2="$HERE/../tp2"

stop_all() {
  echo "maya-tp2-swap: stopping rank 0 (local) and rank 1 ($RANK1_HOST)"
  pkill -x maya 2>/dev/null || true
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pkill -x maya' 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! pgrep -x maya >/dev/null 2>&1 \
       && ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
            'pgrep -x maya >/dev/null' 2>/dev/null; then
      break
    fi
    sleep 1
  done
  sleep 1
}

case "${1:?usage: tp2.sh run PORT | tp2.sh stop}" in
  stop)
    stop_all
    ;;
  run)
    PORT="${2:?tp2.sh run PORT: PORT missing}"
    stop_all
    echo "maya-tp2-swap: starting rank 1 on $RANK1_HOST"
    for _ in $(seq 1 15); do
      ss -tln | grep -qE ':(18515) ' || break
      sleep 1
    done
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
      "nohup $TP2/rank1.sh >/home/ugo/maya-tp2-rank1-swap.log 2>&1 </dev/null &" \
      2>/dev/null || true
    sleep 2
    echo "maya-tp2-swap: starting rank 0 on port $PORT"
    exec "$TP2/rank0.sh" "$PORT"
    ;;
  *)
    echo "usage: tp2.sh run PORT | tp2.sh stop" >&2
    exit 2
    ;;
esac
