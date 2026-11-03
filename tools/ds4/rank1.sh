#!/usr/bin/env bash
# DeepSeek TP2 worker. Usage: rank1.sh --model Maya-XS | --config FILE
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds4/tp2_launch.py" rank1 "$@"
