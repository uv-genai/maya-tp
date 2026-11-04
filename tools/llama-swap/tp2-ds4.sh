#!/usr/bin/env bash
# DeepSeek pair. Usage: tp2-ds4.sh run PORT --model Maya-XS | tp2-ds4.sh stop PORT
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ds4/tp2_launch.py" "$@"
