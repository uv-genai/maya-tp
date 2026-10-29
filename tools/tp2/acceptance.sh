#!/bin/bash
# tools/tp2/acceptance.sh — TP2 serve acceptance: pair up, chat, peer health, kill/recover.
#
# Usage: tools/tp2/acceptance.sh [--serve-host HOST] [--port PORT]
#
# Run from rank 1 (strix-halo-2). Rank 0 is started via SSH.
# Tests:
#   1. Start rank1 worker + rank0 server; wait for /health peer=up
#   2. curl chat/completions
#   3. Kill rank1; next chat fails; /health degraded peer=down
#   4. Restart both; health peer=up; chat works again
set -euo pipefail
cd "$(dirname "$0")/../.."

HOST="${HOST:-strix-halo-1.local}"
PORT="${PORT:-18080}"
PASS=0
FAIL=0

check() {
  local desc="$1" expected="$2" actual="$3"
  if echo "$actual" | grep -qF "$expected"; then
    echo "  PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $desc (expected '$expected', got: ${actual:0:200})"
    FAIL=$((FAIL + 1))
  fi
}

health() { curl -s -m 5 "http://$HOST:$PORT/health" 2>/dev/null || echo '{}'; }
chat() {
  curl -s -m 120 "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"maya","messages":[{"role":"user","content":"Say hello."}],"max_tokens":8}' \
    2>/dev/null || echo '{}'
}

start_pair() {
  pkill -9 -x maya 2>/dev/null || true
  ssh -o BatchMode=yes "$HOST" 'pkill -9 -x maya 2>/dev/null; fuser -k '"$PORT"'/tcp 2>/dev/null; pkill -9 -f "[s]erve/server" 2>/dev/null; true'
  sleep 1
  setsid nohup bash tools/tp2/rank1.sh >/tmp/tp2-acc-r1.log 2>&1 </dev/null &
  disown
  sleep 1
  ssh -o BatchMode=yes "$HOST" "cd $PWD && setsid nohup bash tools/tp2/rank0.sh $PORT >/tmp/tp2-acc-r0.log 2>&1 </dev/null & echo started"
  for i in $(seq 1 36); do
    if health | grep -q '"peer"'; then return 0; fi
    sleep 5
  done
  echo "pair did not become ready" >&2
  return 1
}

echo "=== TP2 acceptance ($HOST:$PORT) ==="
echo "--- 1. start pair ---"
start_pair
H=$(health)
check "health peer up" '"peer": "up"' "$H"
check "status ok" '"status": "ok"' "$H"

echo "--- 2. chat ---"
R=$(chat)
check "chat content" '"content"' "$R"

echo "--- 3. kill rank1 ---"
pkill -9 -x maya 2>/dev/null || true
sleep 1
R=$(chat)
check "chat fails cleanly" 'TP2 pair down' "$R"
H=$(health)
check "health degraded" '"status": "degraded"' "$H"
check "peer down" '"peer": "down"' "$H"

echo "--- 4. restart both ---"
start_pair
H=$(health)
check "recover peer up" '"peer": "up"' "$H"
R=$(chat)
check "chat after recover" '"content"' "$R"

echo ""
echo "=== results: $PASS passed, $FAIL failed ==="
[ "$FAIL" = 0 ]
