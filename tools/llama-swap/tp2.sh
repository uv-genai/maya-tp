#!/usr/bin/env bash
# Run on strix-halo-1: tp2.sh run PORT [options] | tp2.sh stop
# llama-swap cmdStop: tp2.sh stop. Default: simple decode; MTP drafts default to 2.
#
# Options for run:
#   --mtp | --no-mtp     opt into / force simple decode
#   --drafts N           fixed MTP draft width (1..8, default 2)
#   --context N          max context on both ranks
#   --config PATH        base maya-*.json (else MAYA_CONFIG / model-derived / Maya-S)
#   --model NAME         model id under MAYA_MODELS_DIR (else MAYA_MODEL)
#   --pack PATH          pack directory (else MAYA_PACK); overrides config --glm-pack
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TP2="$ROOT/tools/tp2"
# shellcheck source=../tp2/model_env.sh
source "$TP2/model_env.sh"
RANK1_HOST="${RANK1_HOST:-strix-halo-2.local}"
CONFIG=""
ACTION="${1:-help}"
shift || true
usage() {
  echo 'usage: tp2.sh run PORT [--mtp|--no-mtp] [--drafts N] [--context N] [--config PATH] [--model NAME | --pack PATH] | tp2.sh stop'
}
if [[ "$ACTION" == help || "$ACTION" == --help || "$ACTION" == -h ]]; then usage; exit 0; fi
if [[ "$ACTION" != run && "$ACTION" != stop ]]; then usage >&2; exit 2; fi

MTP="${MAYA_TP_MTP:-${STRATA_TP_MTP:-0}}"
DRAFTS="${MAYA_GLM_MTP_FIXED:-${STRATA_GLM_MTP_FIXED:-2}}"
if [[ "$ACTION" == run ]]; then
  PORT="${1:?tp2.sh run: PORT missing}"
  shift
  while (( $# )); do
    case "$1" in
      --mtp) MTP=1; shift ;;
      --no-mtp) MTP=0; shift ;;
      --drafts) DRAFTS="${2:?--drafts needs N}"; shift 2 ;;
      --context) CONTEXT="${2:?--context needs N}"; shift 2 ;;
      --config) CONFIG="${2:?--config needs PATH}"; shift 2 ;;
      --model) MAYA_MODEL="${2:?--model needs NAME}"; MAYA_PACK=""; shift 2 ;;
      --pack) MAYA_PACK="${2:?--pack needs PATH}"; shift 2 ;;
      *) usage >&2; exit 2 ;;
    esac
  done
  [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT <= 65535 )) || { echo 'invalid HTTP port' >&2; exit 2; }
  [[ "$MTP" == 0 || "$MTP" == 1 ]] || { echo 'MAYA_TP_MTP must be 0 or 1' >&2; exit 2; }
  [[ "$DRAFTS" =~ ^[1-8]$ ]] || { echo 'drafts must be 1..8' >&2; exit 2; }
  if [[ -z "$CONFIG" ]]; then
    CONFIG=$(maya_resolve_config "$ROOT")
  fi
  [[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }
  # Match rank 0's configured context instead of silently giving the worker 4096.
  CONTEXT="${CONTEXT:-$(jq -r '(.args // []) as $a | [$a | to_entries[] | select(.value == "--max-context") | $a[.key+1]][-1] // 4096' "$CONFIG")}"
  [[ "$CONTEXT" =~ ^[0-9]+$ ]] && (( CONTEXT > 0 )) || { echo 'invalid context' >&2; exit 2; }
  export MAYA_CONFIG="$CONFIG" CONTEXT SEQS=1 MAYA_TP_MTP="$MTP"
  export MAYA_GLM_MTP_FIXED="$DRAFTS" MAYA_GLM_MTP_DRAFT="$DRAFTS"
  export MAYA_PACK="${MAYA_PACK:-}" MAYA_MODEL="${MAYA_MODEL:-}"
  if [[ "$MTP" == 1 ]]; then
    unset MAYA_GLM_NO_MTP STRATA_GLM_NO_MTP MAYA_GLM_NO_SPEC STRATA_GLM_NO_SPEC
  fi
fi

exec > >(tee -a "$HOME/maya-tp2-swap.log") 2>&1
echo "=== maya-tp2-swap $(date '+%F %T') action=$ACTION model=${MAYA_MODEL:-} pack=${MAYA_PACK:-} config=${MAYA_CONFIG:-}"
stop_all() {
  echo "maya-tp2-swap: stopping both ranks"
  # Kill the HTTP supervisor first; otherwise it can respawn its engine.
  pkill -9 -f '[s]erver.py --engine strata --config /tmp/maya-tp2' 2>/dev/null || true
  pkill -9 -x maya 2>/dev/null || true
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pkill -9 -x maya' 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! pgrep -x maya >/dev/null 2>&1 &&
       ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pgrep -x maya >/dev/null' 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo 'maya-tp2-swap: old engines did not stop' >&2
  return 1
}
stop_all
[[ "$ACTION" == stop ]] && exit 0

for _ in $(seq 1 15); do
  ss -tln | grep -qE ':18515 ' || break
  sleep 1
done
if ss -tln | grep -qE ':18515 '; then echo 'bootstrap port 18515 still busy' >&2; exit 1; fi

echo "maya-tp2-swap: MTP=$MTP drafts=$DRAFTS context=$CONTEXT config=$CONFIG model=${MAYA_MODEL:-} pack=${MAYA_PACK:-}"
PEER_ENV=(env -u MAYA_GLM_NO_MTP -u STRATA_GLM_NO_MTP -u MAYA_GLM_NO_SPEC -u STRATA_GLM_NO_SPEC
          "MAYA_CONFIG=$CONFIG" "MAYA_TP_MTP=$MTP" "MAYA_GLM_MTP_FIXED=$DRAFTS"
          "MAYA_GLM_MTP_DRAFT=$DRAFTS" "CONTEXT=$CONTEXT" SEQS=1
          "MAYA_PACK=${MAYA_PACK:-}" "MAYA_MODEL=${MAYA_MODEL:-}"
          "MAYA_MODELS_DIR=${MAYA_MODELS_DIR:-}")
printf -v PEER_CMD '%q ' "${PEER_ENV[@]}" "$TP2/rank1.sh"
# Background just the fully redirected command; no `cd && command &` wrapper holding SSH pipes.
ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
  "setsid nohup $PEER_CMD > /home/ugo/maya-tp2-rank1-swap.log 2>&1 </dev/null &"
exec "$TP2/rank0.sh" "$PORT"
