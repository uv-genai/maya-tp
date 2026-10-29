#!/bin/bash
# llama-swap launcher for Maya TP2 — based on run_bench.sh.
#
#   run PORT : start rank 1 on strix-halo-2 (ssh), then exec rank 0 on PORT
#   stop     : stop both ranks (cmdStop)
#
# Example llama-swap config:
#   "maya-glm5-tp2":
#     cmd: /home/ugo/software/project-maya/tools/tp2/tp2-swap.sh run ${PORT}
#     cmdStop: /home/ugo/software/project-maya/tools/tp2/tp2-swap.sh stop
#     proxy: http://127.0.0.1:${PORT}
set -euo pipefail

cd /home/ugo/software/project-maya
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
ENGINE=/home/ugo/software/project-maya/build-hip/maya
RDEV=usb4_rdma5
RANK1_HOST=strix-halo-2.local

stop_all() {
  echo "tp2-swap: stopping rank 0 (local) and rank 1 ($RANK1_HOST)"
  kill -9 $(pgrep -x maya) 2>/dev/null || true
  ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'kill -9 $(pgrep -x maya)' 2>/dev/null || true
  for _ in $(seq 1 15); do
    if ! pgrep -x maya >/dev/null 2>&1 \
       && ! ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pgrep -x maya >/dev/null' 2>/dev/null; then
      break
    fi
    sleep 1
  done
}

case "${1:?usage: tp2-swap.sh run PORT | tp2-swap.sh stop}" in
  stop)
    stop_all
    ;;
  run)
    PORT="${2:?tp2-swap.sh run PORT: PORT missing}"
    stop_all

    # rank 1 on strix-halo-2 — same pattern as run_bench.sh:
    # the & backgrounds the SSH session, keeping the remote engine alive
    echo "tp2-swap: starting rank 1 on $RANK1_HOST"
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
      "$ENGINE --glm-pack '$P' --tp-world-size 2 --tp-rank 1 \
       --tp-bootstrap-host 192.168.1.22 --tp-bootstrap-port 18515 \
       --tp-rdma-device $RDEV --max-context 4096 --serve" \
      > /home/ugo/rank1.log 2>&1 &
    R1PID=$!

    sleep 2

    # rank 0 locally — the OpenAI server (foreground for llama-swap)
    echo "tp2-swap: starting rank 0 on port $PORT"
    TP_CONFIG=$(mktemp /tmp/maya-tp2-XXXXXX.json)
    trap 'rm -f "$TP_CONFIG"' EXIT
    jq --arg rdev "$RDEV" \
       '.tp = {world_size: 2, rank: 0, bootstrap_port: 18515, rdma_device: $rdev}
        | .args += ["--tp-world-size","2","--tp-rank","0","--tp-bootstrap-port","18515","--tp-rdma-device",$rdev]' \
       maya-maya-s-v2-iq2_xxs-hip.json > "$TP_CONFIG"
    TOKENIZER=$(jq -r '.tokenizer // ""' maya-maya-s-v2-iq2_xxs-hip.json)
    exec .venv/bin/python serve/server.py --engine strata --config "$TP_CONFIG" \
      --host 0.0.0.0 --port "$PORT" --tokenizer "$TOKENIZER"
    ;;
  *)
    echo "usage: tp2-swap.sh run PORT | tp2-swap.sh stop" >&2
    exit 2
    ;;
esac
