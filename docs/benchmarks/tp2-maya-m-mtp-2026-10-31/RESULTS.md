# Maya-M TP2 + MTP validation — 2026-10-31

Same suite used for Maya-S (#16/#22), run on **Maya-M** after packing both hosts (#29).

## Environment

| | |
| --- | --- |
| Hosts | strix-halo-1 (rank0) / strix-halo-2 (rank1), HIP/ROCm gfx1151, `usb4_rdma5` |
| Pack | `/home/ugo/software/Maya-data/models/Maya-M/pack` (identical MD5s on both hosts) |
| Binary | `5dbc93560e0eb024dcbdc0d9eb60ea57` (local `build-hip/maya`, rank0 `build-hip/maya`, `/tmp/maya-tp2`) |
| Context | 32768 (API), 4096 (CLI pair) |
| MTP | `MAYA_TP_MTP=1`, fixed/cap **7** (serve/API) or **2** (`bench_mtp.py` default) |
| Launch | `rank0.sh`/`rank1.sh` with `--model Maya-M` / `MAYA_MODEL=Maya-M` |

Engines stopped after the run. Binary hashes rechecked.

## Results

| Test | Result | Artifact |
| --- | --- | --- |
| MTP serve smoke (`draft_n > 0`) | PASS | — |
| `mtp_acceptance.py` (3 turns + reuse, cancel/follow-up, limits 1/2/3, peer up) | **PASS** | `maya-m-mtp-acceptance.log` |
| `regression_collectives.py --mode mtp` (990 / 7290 / 14550 prompts, cache reuse ≥7000, stream DONE, peer up) | **PASS** | `mtp-matrix.json`, `maya-m-mtp-matrix.log` |
| `regression_collectives.py --mode simple` (same prompts, `draft_n==0`) | **PASS** | `simple-matrix.json`, `maya-m-simple-matrix.log` |
| CLI pair parity simple (32 tokens) | **32/32** | `maya-m-cli-parity.log` |
| CLI pair parity 7-draft MTP (32 tokens, `mtp=1` in INFO) | **32/32** | `maya-m-cli-parity.log` |
| `acceptance.sh` MTP kill/recover (7 drafts) | **8 passed, 0 failed** | `maya-m-acceptance-mtp.log` |
| `bench_mtp.py --model Maya-M --tokens 256 --runs 4 --drafts 2` | **PASS** (raw pick diffs **0**) | `bench-mtp-results.json`, `maya-m-bench-mtp.log` |

MTP matrix observed `draft_n` / accepted on the large cases: e.g. 990-prompt 144/41, 7290-prompt 68/20,
14550-prompt 76/19. Reuse case cached **7289** tokens.

### Controlled MTP vs simple (bench_mtp, 256-token common reference, ABBA, warmups excluded)

| Mode | median ms/tok | median tok/s |
| --- | ---: | ---: |
| simple | **41.12** | **24.32** |
| MTP (2 drafts) | **38.03** | **26.29** |

MTP throughput **+8.11%**, latency **−7.50%** on this reference. Drafts 135/238 accepted.
Not a universal gain claim; one reference prompt, Maya-M only.

## Notes

- `--model Maya-M` rewrites the base Maya-S JSON’s `--glm-pack`/tokenizer; `/health` `model_label`
  still says Maya-S (config `model_name` field not swapped) — cosmetic.
- Launchpad gotcha: `env` requires options (`-u …`) **before** assignments.
- Not re-run here: C++ parity binaries, HIP map / serve / tools CI (unchanged engine).
