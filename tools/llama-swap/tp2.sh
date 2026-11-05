#!/usr/bin/env bash
# Owned GLM TP2 pair: run PORT [--parallel 1..4] [--mtp] [model/config options].
# cmdStop: stop PORT, or stop-pid ${PID}; only this launch's processes are stopped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec "$ROOT/.venv/bin/python" "$ROOT/tools/tp2/launch.py" "$@"
