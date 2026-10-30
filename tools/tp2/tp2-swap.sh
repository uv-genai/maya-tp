#!/usr/bin/env bash
# Compatibility entry point: use the same llama-swap launcher and MTP options.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/../llama-swap/tp2.sh" "$@"
