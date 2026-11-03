#!/usr/bin/env bash
# DeepSeek TP2 API rank. Usage: rank0.sh [PORT] --model Maya-XS | --config FILE
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds4/tp2_launch.py" rank0 "$@"
