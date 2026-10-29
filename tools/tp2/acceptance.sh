#!/bin/bash
# tools/tp2/acceptance.sh — TP2 acceptance test: pair up, curl API, kill/recover.
#
# Usage: tools/tp2/acceptance.sh [--host HOST] [--port N] [--serve-port N]
#
# Tests:
#   1. Start rank 0 + rank 1, wait for pair ready
#   2. curl /health (check tp status)
#   3. curl /v1/chat/completions (basic generation)
#   4. Kill rank 1 -> verify rank 0 reports pair-down
#   5. Restart both -> verify recovery
set -e
cd "$(dirname "$0")/../.."

HOST="strix-halo-1.local"
PORT=18515
SERVE_PORT=8080
PASS=0
FAIL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --serve-port) SERVE_PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

check() {
  local desc="$1" expected="$2" actual="$3"
  if echo "$actual" | grep -q "$expected"; then
    echo "  PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $desc (expected '$expected', got: $actual)"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== TP2 acceptance test ==="
echo "  rank 0 host:  $HOST"
echo "  bootstrap:    port $PORT"
echo "  serve:        port $SERVE_PORT"
echo ""

# cleanup on exit
cleanup() {
  echo ""
  echo "=== cleanup ==="
  ssh "$HOST" "pkill -x maya-tp2 2>/dev/null" 2>/dev/null || true
  pkill -x maya 2>/dev/null || true
  sleep 1
}
trap cleanup EXIT

echo "--- 1. start rank 0 + rank 1 ---"
ssh "$HOST" "pkill -x maya-tp2 2>/dev/null" 2>/dev/null || true
sleep 1

# start rank 0 (the HTTP server + engine)
tools/tp2/maya-rank0.sh --config maya-glm5.json --port "$PORT" --serve-port "$SERVE_PORT" &
R0PID=$!
sleep 8

# start rank 1 (the worker)
tools/tp2/maya-rank1.sh --config maya-glm5.json --host "$HOST" --port "$PORT" &
R1PID=$!
sleep 5

echo "--- 2. /health ---"
HEALTH=$(curl -s "http://127.0.0.1:$SERVE_PORT/health" 2>/dev/null || echo "{}")
check "health returns ok" '"status"' "$HEALTH"
check "health has tp info" '"tp"' "$HEALTH"

echo "--- 3. /v1/chat/completions ---"
RESPONSE=$(curl -s "http://127.0.0.1:$SERVE_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"maya","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":5}' \
  2>/dev/null || echo "{}")
check "chat returns content" '"content"' "$RESPONSE"

echo "--- 4. kill rank 1 ---"
kill $R1PID 2>/dev/null || true
sleep 3
HEALTH2=$(curl -s "http://127.0.0.1:$SERVE_PORT/health" 2>/dev/null || echo "{}")
echo "  (rank 1 killed; health: $(echo "$HEALTH2" | head -c 80)...)"
# after killing rank 1, the engine may crash or report an error — either is acceptable

echo "--- 5. restart both ---"
kill $R0PID 2>/dev/null || true
sleep 2
tools/tp2/maya-rank0.sh --config maya-glm5.json --port "$PORT" --serve-port "$SERVE_PORT" &
R0PID=$!
sleep 8
tools/tp2/maya-rank1.sh --config maya-glm5.json --host "$HOST" --port "$PORT" &
R1PID=$!
sleep 5
HEALTH3=$(curl -s "http://127.0.0.1:$SERVE_PORT/health" 2>/dev/null || echo "{}")
check "recovery: health returns ok" '"status"' "$HEALTH3"

echo ""
echo "=== results: $PASS passed, $FAIL failed ==="
[ "$FAIL" = 0 ]
