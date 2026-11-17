#!/usr/bin/env bash
# tools/local_gates.sh - the source gates every commit must pass on the host before it is pushed. There is no GitHub
# CI for this fork (the user's rule, 2026-11-17): code is validated here, at the exact commit, and only then pushed.
# Engine/GPU changes additionally need the gfx1151 HIP builds and paired TP2 checks (CONTRIBUTING.md, AGENTS.md).
#
#   tools/local_gates.sh                 run every gate in order; stops at the first failure
#
# Python: MAYA_PYTHON, else <checkout>/.venv/bin/python, else python3 (needs numpy, jinja2, regex, pyyaml, tqdm,
# psutil and jsonschema, as ./maya.sh --setup installs them). llama.cpp's gguf-py at setup.py's LLAMA_CPP_COMMIT:
# STRATA_GGUF_PY, else <checkout>/third_party/llama.cpp/gguf-py (tools/_paths.py).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${MAYA_PYTHON:-}
if [ -z "$PY" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY=python3; fi
fi
if [ -z "${STRATA_GGUF_PY:-}" ]; then
    if [ -d "$ROOT/third_party/llama.cpp/gguf-py" ]; then export STRATA_GGUF_PY="$ROOT/third_party/llama.cpp/gguf-py"
    else echo "local gates: set STRATA_GGUF_PY to llama.cpp's gguf-py (setup.py LLAMA_CPP_COMMIT)" >&2; exit 2; fi
fi
CXX=${CXX:-c++}
OUT=$(mktemp -d "${TMPDIR:-/tmp}/maya-local-gates.XXXXXX")
trap 'rm -rf "$OUT"' EXIT
step() { printf '\n== %s\n' "$*"; }

step "HIP mapping: every CUDA name the HIP build compiles is mapped (include/strata/hip_compat)"
"$PY" tools/check_hip_compat.py
step "Server tests"
"$PY" -m unittest discover -s serve -p "test_*.py" -t .
step "Installer and tools tests"
"$PY" -m unittest discover -s tools -p "test_*.py"

step "GLM memory policy, TP2 batch framing, env switches, Flash state layout, tuning-table headers (CPU only)"
"$CXX" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror tests/glm_memory_test.cpp -o "$OUT/glm_memory_test"
"$OUT/glm_memory_test"
"$CXX" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -Iinclude tests/glm_tp_batch_test.cpp -o "$OUT/glm_tp_batch_test"
"$OUT/glm_tp_batch_test"
"$CXX" -std=c++20 -O2 -UNDEBUG -Wall -Wextra -Werror -Iinclude tests/env_switch_test.cpp -o "$OUT/env_switch_test"
"$OUT/env_switch_test"
"$CXX" -std=c++20 -O2 -UNDEBUG -Wall -Wextra -Werror -Itests/host -Iinclude tests/glm_dsa/glm_state_layout_test.cpp \
    src/core/glm_state_layout.cpp -o "$OUT/glm_state_layout_test"
"$OUT/glm_state_layout_test" --flash-only
"$CXX" -std=c++20 -O2 -UNDEBUG -Wall -Wextra -Werror tests/hipblaslt_tuning_header_test.cpp \
    -o "$OUT/hipblaslt_tuning_header_test"
"$OUT/hipblaslt_tuning_header_test"

step "DeepSeek CPU row preparation (CPU only)"
"$CXX" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror tests/ds4_cpu_prepare_rows_test.cpp -o "$OUT/ds4_cpu_prepare_rows_test"
"$OUT/ds4_cpu_prepare_rows_test"
"$CXX" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -DDS4_CPU_PREPARE_STUB_TEST=1 tests/ds4_cpu_prepare_rows_test.cpp \
    src/ds4/ds4_iq_cpu.cpp -o "$OUT/ds4_cpu_prepare_stub_test"
"$OUT/ds4_cpu_prepare_stub_test"

step "DeepSeek TP2 batch control (CPU only)"
for suite in ds4_tp_control_test ds4_tp_identity_test ds4_tp_worker_state_test ds4_tp_model_worker_test \
             ds4_tp_channel_test ds4_tp_batch_test ds4_tp_batch_channel_test; do
    "$CXX" -std=c++20 -O2 -UNDEBUG -Wall -Wextra -Werror -Itests/host -Iinclude "tests/$suite.cpp" -o "$OUT/$suite"
    "$OUT/$suite"
done

head=$(git rev-parse HEAD 2>/dev/null || echo unknown)
dirty=$(git status --porcelain 2>/dev/null | grep -c . || true)
printf '\nlocal gates: all passed at %s (%s uncommitted paths)\n' "$head" "$dirty"
