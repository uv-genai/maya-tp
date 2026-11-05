#!/usr/bin/env bash
# Owned GLM engine/API route acceptance. Deliberately loses its recorded peer,
# verifies all-route failure and later-GEN rejection; never clears existing jobs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/tools/tp2/model_env.sh"
BASE=$(maya_resolve_config "$ROOT")
WORK=$(mktemp -d /tmp/maya-tp2-acceptance-XXXXXX)
trap 'rm -rf "$WORK"' EXIT
PACK=""
if [[ -n "${MAYA_MODEL:-}${MAYA_PACK:-}" ]]; then PACK=$(maya_resolve_pack); fi
maya_config_with_pack "$BASE" "$PACK" > "$WORK/base.json"
"$ROOT/.venv/bin/python" "$ROOT/tools/tp2/validate_batch.py" \
  --name "acceptance-$(date +%s)-$$" --config "$WORK/base.json" \
  --engine "${ENGINE:-$ROOT/build-hip/maya}" --peer "${RANK1_HOST:-strix-halo-2.local}" \
  --lead-address "${BOOTSTRAP_HOST:-192.168.1.22}" --port "${TP_PORT:-19151}" \
  --context "${CONTEXT:-32768}" --seqs "${SEQS:-4}" --clients 4 \
  --mtp "${MAYA_TP_MTP:-0}" --scenario peer-loss "$@"
