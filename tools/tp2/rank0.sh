#!/usr/bin/env bash
# Maya TP2 rank 0 (OpenAI server) — runs on strix-halo-1.
# Start this first, then rank1.sh on strix-halo-2 within 30 s.
#
# Usage: rank0.sh [PORT]   (default 8000)
#
# HTTP API: http://192.168.1.22:PORT/v1 (OpenAI chat/completions)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PORT="${1:-8000}"
MAYA_CONFIG="${MAYA_CONFIG:-$ROOT/maya-maya-s-v2-iq2_xxs-hip.json}"
ENGINE="${ENGINE:-$ROOT/build-hip/maya}"

# Device names shift after re-plugs; pick the lowest-GUID rail so both ranks agree.
RDEV=$(ibv_devices | awk '/usb4_rdma/{print $2, $1}' | sort | head -n1 | awk '{print $2}')
[ -n "$RDEV" ] || { echo "no usb4_rdma device found" >&2; exit 1; }

# Generate a TP2 config from the base config: add tp block and --tp-* flags.
TP_CONFIG=$(mktemp /tmp/maya-tp2-rank0-XXXXXX.json)
trap 'rm -f "$TP_CONFIG"' EXIT
python3 - "$MAYA_CONFIG" "$TP_CONFIG" "$RDEV" << 'PYEOF'
import json, sys
src, dst, rdev = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(src))
cfg["tp"] = {"world_size": 2, "rank": 0, "bootstrap_port": 18515, "rdma_device": rdev}
args = cfg.setdefault("args", [])
for k, v in [("--tp-world-size", "2"), ("--tp-rank", "0"),
             ("--tp-bootstrap-port", "18515"), ("--tp-rdma-device", rdev)]:
    if k in args[:-1]:
        args[args.index(k) + 1] = v
    else:
        args += [k, v]
json.dump(cfg, open(dst, "w"), indent=1)
print(f"rank0: tp config -> {dst} (rdma={rdev} port={sys.argv[1]})")
PYEOF

TOKENIZER="${TOKENIZER:-$(python3 -c "import json; print(json.load(open('$MAYA_CONFIG')).get('tokenizer',''))")}"

exec python3 "$ROOT/serve/server.py" --engine strata --config "$TP_CONFIG" \
  --host 0.0.0.0 --port "$PORT" \
  --tokenizer "$TOKENIZER"
