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

Configs are `maya-<quant>[-hip].json` (gitignored). Build dirs: `build/` (CUDA), `build-hip/` (HIP).

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

- **CUDA or HIP, never both.** Options accept `STRATA_*` or `MAYA_*` (`MAYA_X` sets `STRATA_X` in CMake; runtime env aliases are `MAYA_*` → `STRATA_*`).
- **HIP requires MMQ:** `-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON` (`./maya.sh --backend hip` sets this).
- Enable tests only when you need C++ suites: `-DSTRATA_BUILD_TESTS=ON`.
- CUDA: default arch is `120`; min compute capability **6.0** (Pascal experimental; Volta+ supported). HIP: wave32 RDNA3/4 (`gfx1100`, `gfx1201`, `gfx1151`, …) with optional wave64 CDNA path — see `cmake/hip_backend.cmake`.
- Many kernel tests are **parity binaries**; run them as `build*/glm_*_parity --selftest` (and `ctest` where registered).

Manual HIP configure example is in `docs/AMD_RDNA4.md`.

## Conventions that are easy to miss

1. **New CUDA / cuBLAS names need a HIP mapping.** The AMD build compiles the same sources with `include/strata/hip_compat/` forced in. Missing `#define`s break every HIP build (v1.0.21). Always run `python tools/check_hip_compat.py` after touching CUDA runtime calls.
2. **Do not change the NVIDIA path for AMD.** Gate AMD behavior with `#if defined(STRATA_USE_HIP)` / related flags; leave CUDA behavior identical unless you have NVIDIA measurements.
3. **New settings are `MAYA_*` environment variables.** Code still reads `STRATA_*`; both work, `MAYA_*` wins when both set. Default off or preserve current behavior unless measured better. Document every new knob in `docs/SETTINGS.md`.
4. **Python is stdlib unittest, not pytest.** Serve tests must stay importable as `serve.<module>` (`discover -t .`). Fixtures: `serve/fixtures/`, synthetic GGUFs from `tools/`.
5. **llama.cpp is pinned.** Do not casually change `LLAMA_CPP_COMMIT` in `setup.py` (also used by CI). ggml / MMQ live under `third_party/llama.cpp`; a GLM MMQ patch is applied from the pinned checkout.
6. **Per-model engines, not one generic backend.** `maya` (GLM: `src/core`, `src/prefill`, …), `maya-ds4` (`src/ds4`), and `maya-mm3` (`src/mm3`) share kernels but not control flow. A change in one engine usually does not apply to the others. Server routing and templates differ per model family (DeepSeek: vendored `serve/ds41_encoding.py` + fixtures; others: Jinja).
7. **Pack integrity.** The pack index next to model shards is hashed; engine refuses mismatched shards. Use `./maya.sh --repack` after intentional pack changes.

## PRs (from CONTRIBUTING.md)

- Base on `main`; one topic per PR.
- Perf / output-affecting changes: measure before/after on the same harness; note if answers change (expert placement can flip bits).
- AMD / HIP: accepted only with your measurements once NVIDIA still builds and behaves the same.
- CI must pass: HIP map check + `serve` + `tools` tests. GPU parity suites run on maintainers' machines before merge.

## Key references

- `CONTRIBUTING.md` — PR / bug / bench expectations  
- `.github/workflows/tests.yml` — exact CI commands and gguf-py pin  
- `docs/SETTINGS.md` — full `MAYA_*` / `STRATA_*` knob list  
- `docs/SETUP.md` — install options, build failure fixes  
- `docs/MODELS.md` — per-model files and engine options  
- `docs/AMD_MAYA.md`, `docs/AMD_RDNA4.md` — HIP install and validation  
