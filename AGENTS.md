# AGENTS.md

Project Maya: local inference for very large MoE models (GLM-5.3, DeepSeek-V4.1, MiniMax-M3.1). C++20 CUDA/HIP engine + stdlib Python server/installer. No GPU is required for CI checks.

## Platform (hard rule)

**This fork targets HIP/ROCm on gfx1151 (Strix Halo) Linux only.** Optimize, build, and measure for that one platform. Do not add CUDA/NVIDIA support or multi-backend abstractions. The `tools/check_hip_compat.py` gate exists so the HIP build's CUDA-name shims (`include/strata/hip_compat/`) keep compiling — it is not NVIDIA work.

**Interconnect is RDMA on the APU only** (Thunderbolt `usb4_rdma*`, as in gufo): target **~37 Gb/s, ~6 µs latency**. No TCP data path; sockets are for bootstrap/control only.

## GitHub workflow (mandatory)

**All development and progress is tracked in GitHub issues on `uv-genai/maya-tp`.** Every agent session must follow this:

1. **One issue per workstream** (a plan phase, feature, bug, or experiment) — **not** one umbrella issue with comments. If a plan lists P1…P7, open one issue per phase and work in that issue. The umbrella epic (if any) only indexes them.
2. **Comment on the issue while making progress** — status, decisions, blockers, measurements, commit SHAs. Do not wait until the end; comment at meaningful milestones (after analysis, after a design decision, after each phase/PR, when a gate passes or fails).
3. **Link commits and PRs to the issue** (`refs #N` / `closes #N` in commit or PR body).
4. **Close or leave open explicitly** when the workstream ends — with a final comment summarizing outcome.

Use `gh` against `uv-genai/maya-tp` (this repo's `origin`). Do not track work only in chat or local notes.

## Git remotes / branches

| | |
| --- | --- |
| `origin` | `https://github.com/uv-genai/maya-tp.git` (our fork — push here) |
| `upstream` | `https://github.com/mw00/project-maya.git` (official Maya — read/merge only) |
| `main` | tracks `origin/main` (uv-genai/maya-tp) |
| `maya-upstream` | tracks `upstream/main` (official) |

Push only to `origin` (`uv-genai`). Never push to `upstream`.

**Release tags:** new Maya-TP releases use **`tp-vMAJOR.MINOR.PATCH`** (for example `tp-v1.1.43`),
so they cannot be confused with upstream Maya's `v*` tags. Fetch upstream tags as `upstream/v*`.
Keep `VERSION` numeric. Existing published legacy `v*` tags are immutable; do not rename/delete them
or create a release merely because an upstream tag was merged.

## Layout

| Path | Role |
| --- | --- |
| `src/`, `include/strata/` | Engine C++ (three binaries: `maya`=GLM, `maya-ds4`=DeepSeek, `maya-mm3`=MiniMax) |
| `serve/` | OpenAI/Anthropic-compatible API, Responses API, web UI, vision glue |
| `tools/` | Installer, calibration, packing, quant helpers |
| `tests/` | C++ host tests (CMake-registered when present; omitted from some published trees) |
| `maya.py` / `maya.sh` / `setup.sh` | Installer and runner (creates/reuses `.venv`) |
| `CMakeLists.txt`, `cmake/` | Engine build; HIP backend in `cmake/hip_backend.cmake` |
| `docs/` | SETTINGS, SETUP, MODELS, AMD_* — trust code/CI over prose |

Configs are `maya-<quant>[-hip].json` (gitignored). Use `build-hip/` for this fork;
inherited upstream references to `build/` describe its other build configurations.

## Process and deployment ownership

The user launches their own processes. Agents may stop only validation processes they started,
identified by recorded PIDs/process groups and launch logs. Stop an owned server before its owned
engine so it cannot respawn. Do not use name-wide `pkill`, port-wide killing, or service-manager
commands to clear the hosts. Do not inspect or manage unrelated services, including llama-swap,
unless the current task explicitly requests it. Earlier validation permission is not standing
authorization to stop user processes.

Read the current state in `Handoff.md` before deploying: the remote canonical checkout contains
protected user edits, and production and review binaries may differ. Use isolated review trees and
explicit configs for future validation; do not pull/reset, rebuild or overwrite the protected
production checkout as routine test setup.

## Commands (CI / local, no GPU)

Run these in order after installing Python deps into a venv (see `.github/workflows/tests.yml`):

```sh
# 1) HIP mapping check — every CUDA/cuBLAS name used in sources must exist in include/strata/hip_compat/
python tools/check_hip_compat.py

# 2) Server tests (imports as package `serve.*`)
python -m unittest discover -s serve -p "test_*.py" -t .

# 3) Installer / tools tests
python -m unittest discover -s tools -p "test_*.py"
```

The workflow also runs CPU-only GLM memory-policy and DeepSeek row-preparation/
native-experts-OFF stub checks after these three gates. Their exact compiler
commands are in `.github/workflows/tests.yml`.

Single test:

```sh
python -m unittest serve.test_ds41_conformance -v
python -m unittest tools.test_maya_hip -v
```

`tools/_paths.py` needs llama.cpp's `gguf-py`: either run setup once (`third_party/llama.cpp/`) or set `STRATA_GGUF_PY`. CI pins the same commit as `setup.py` (`LLAMA_CPP_COMMIT`).

## User-facing installer (GPU machines)

```sh
./maya.sh --setup              # or ./setup.sh (alias)
./maya.sh --check              # hardware only
./maya.sh --rebuild            # recompile engine
./maya.sh --backend hip --gpu 0 --check   # AMD
./maya.sh --bench              # with Maya stopped
./maya.sh --report             # maya-report.txt for issues
```

Installer builds set `STRATA_BUILD_TESTS=OFF`. Prefer `./maya.sh` over raw cmake unless you need a custom tree.

## Engine build (CMake)

- **Build HIP only:** `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF`, targeting `gfx1151`. Options accept `STRATA_*` or `MAYA_*` (`MAYA_X` sets `STRATA_X` in CMake; runtime env aliases are `MAYA_*` → `STRATA_*`).
- **HIP requires MMQ:** `-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON` (`./maya.sh --backend hip` sets this).
- Enable tests only when you need C++ suites: `-DSTRATA_BUILD_TESTS=ON`.
- The inherited HIP backend contains other architecture paths; this fork's build and measurement target is `gfx1151` — see `cmake/hip_backend.cmake`.
- Many kernel tests are **parity binaries**; run them as `build*/glm_*_parity --selftest` (and `ctest` where registered).

The gfx1151 RDMA configure example is in `docs/TP2-RUN.md`. `docs/AMD_RDNA4.md` is an inherited
gfx1201 reference, not this fork's target build recipe.

## Conventions that are easy to miss

1. **New CUDA / cuBLAS names need a HIP mapping.** The AMD build compiles the same sources with `include/strata/hip_compat/` forced in. Missing `#define`s break every HIP build (v1.0.21). Always run `python tools/check_hip_compat.py` after touching CUDA runtime calls.
2. **Do not change the NVIDIA path for AMD.** Gate AMD behavior with `#if defined(STRATA_USE_HIP)` / related flags; leave CUDA behavior identical unless you have NVIDIA measurements.
3. **New settings are `MAYA_*` environment variables.** Code still reads `STRATA_*`; both work, `MAYA_*` wins when both set. Default off or preserve current behavior unless measured better. Document every new knob in `docs/SETTINGS.md`.
4. **Python is stdlib unittest, not pytest.** Serve tests must stay importable as `serve.<module>` (`discover -t .`). Fixtures: `serve/fixtures/`, synthetic GGUFs from `tools/`.
5. **llama.cpp is pinned.** Do not casually change `LLAMA_CPP_COMMIT` in `setup.py` (also used by CI). ggml / MMQ live under `third_party/llama.cpp`; a GLM MMQ patch is applied from the pinned checkout.
6. **Per-model engines, not one generic backend.** `maya` (GLM: `src/core`, `src/prefill`, …), `maya-ds4` (`src/ds4`), and `maya-mm3` (`src/mm3`) share kernels but not control flow. A change in one engine usually does not apply to the others. Server routing and templates differ per model family (DeepSeek: vendored `serve/ds41_encoding.py` + fixtures; others: Jinja).
7. **Pack integrity.** The pack index next to model shards is hashed; engine refuses mismatched shards. Use `./maya.sh --repack` after intentional pack changes.
8. **GLM TP2 launchers take the model as input.** Do not hardcode Maya-S paths in `tools/tp2/*` or `tools/llama-swap/tp2.sh`. Use `tools/tp2/model_env.sh` with `--model NAME`/`MAYA_MODEL`, `--pack PATH`/`MAYA_PACK`, `--config PATH`/`MAYA_CONFIG` and `MAYA_MODELS_DIR`. Pass the **same** selection to both ranks. Document new flags in `docs/TP2-RUN.md` “Choosing a model”. Example pack root is `/home/ugo/software/Maya-data/models/<Name>/pack`. DeepSeek uses the launcher JSON described below.
9. **A fatal TP transport error is sticky on its first occurrence.** A queued collective may fail before the mirror-send guard. Preserve `StrataEngine._request_error` and the pre-write `generate` guard: health must report the pair down immediately, with the original cause retained, and later GEN must not reach the engine. Never clear communicator poison or retry/restart one rank. Regression: `python -m unittest serve.test_tp2_peer_loss` (#88).

DeepSeek TP2 and DSpark have paired XS/S validation and user acceptance, including
hours of GLM/DeepSeek use and DeepSWE benchmarks. PR #122 merged the work into
`main` at `75e18e8`, with feature-head and merged-main CI passing.
`tools/ds4/tp2_rank.py` requires an explicit
pack, local checked manifest and engine; `tools/ds4/tp2_config.py` prints a dedicated rank-0
server config. `tools/ds4/rank0.sh` and `rank1.sh` read per-model launcher JSON;
`tools/llama-swap/tp2-ds4.sh run PORT --model NAME` manages a pair over SSH. Its
`stop PORT` and `stop-pid PID` actions use owned control sockets and recorded
identities. Keep model contents, executable, context and feature choices matched on both
ranks. DeepSeek uses its own launchers and engine. See `docs/DEEPSEEK-TP2-RUN.md`
and `Handoff.md` for the accepted runtime paths and numerical/capacity follow-ups.

## PRs (from CONTRIBUTING.md)

- Base on `main`; one topic per PR.
- Perf / output-affecting changes: measure before/after on the same harness; note if answers change (expert placement can flip bits).
- Engine/performance changes require gfx1151 HIP measurements. Preserve inherited CUDA behavior when editing shared code; NVIDIA builds and measurements are outside this fork's validation scope.
- CI must pass: HIP map check + `serve` + `tools` tests. GPU parity suites run on maintainers' machines before merge.

## Session handoff

`Handoff.md` is the **live** restart document (state, issue map, performance, gotchas, how to run). `Handoff-Prompt.md` is the prompt to give a fresh session. **Keep `Handoff.md` current** — update it whenever state changes, and comment on the relevant issue.

`README.md`, this file and the fork's operational docs are fork-owned. Merge applicable upstream
documentation semantically; do not replace them wholesale. Follow the user's requested upstream
batch size: for sequential work, finish affected validation plus exact-head and post-merge CI before
the next tag; for a cumulative merge, review the entire delta and validate it as one checkpoint. Historical design
documents and measurement reports describe their dated checkpoints; [docs/README.md](docs/README.md)
identifies the current guides. v1.1.37–v1.1.46 was integrated sequentially; #85 integrates v1.1.47–v1.1.49
directly from the latest tag, as explicitly requested by the user. Upstream
v1.1.50–v1.1.53 was integrated through PRs #126/#129/#131/#134, ending at
`de184d9`. Each checkpoint passed paired GPU validation, exact-head CI and
post-merge main CI. Final merged-main CI is 41834772034. See Handoff.md §47 and
`docs/upstream/v1.1.53.md`. New v53 engine experiments are deferred in #133.
Preserve the paired handling of the DeepSeek `END` notice. It marks the pair down
and retains the first fatal cause, with both ranks required to restart. The
optional prompt memo remains off by default and has real GLM/DeepSeek tokenizer
identity checks.

## Key references

- `CONTRIBUTING.md` — PR / bug / bench expectations  
- `.github/workflows/tests.yml` — exact CI commands and gguf-py pin  
- `docs/SETTINGS.md` — full `MAYA_*` / `STRATA_*` knob list  
- `docs/TP2-RUN.md` — TP2 ops: model selection (`--model`/`--pack`), serve/bench examples  
- `docs/DEEPSEEK-TP2-RUN.md` — DeepSeek TP2/DSpark launch, user acceptance and validation limits
- `docs/SETUP.md` — install options, build failure fixes  
- `docs/MODELS.md` — per-model files and engine options  
- `docs/AMD_MAYA.md`, `docs/AMD_RDNA4.md` — HIP install and validation  
