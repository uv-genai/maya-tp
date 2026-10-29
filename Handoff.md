# Handoff — Maya TP2 (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-29 · HEAD `e43c95d` · branch `main` on `uv-genai/maya-tp`

## 1. What we are building

Two-way **tensor parallel (TP2)** for **GLM-5.3-Flash Maya-S** (IQ2_XXS pack) across two twin hosts:

| Host | Role | IP |
| --- | --- | --- |
| `strix-halo-1.local` | rank 0 (API server) | 192.168.1.22 |
| `strix-halo-2.local` | rank 1 (worker, **this machine**) | 192.168.1.92 |

- **Platform (hard rule):** HIP/ROCm **gfx1151 (Strix Halo) Linux only**. No CUDA/NVIDIA work.
- **Interconnect (hard rule):** **RDMA only** over Thunderbolt `usb4_rdma*` (gufo's path), target **~37 Gb/s / ~6 µs**. No TCP data path (sockets for bootstrap/control only).
- **Reference implementation:** gufo at **`/home/ugo/projects/gufo`** (not `software/gufo`). See `TP2-SETUP.md`, `docs/models/qwen3.8-flash-next/TP2.md`, `src/models/qwen38_flash_next/kernels/rocm/{communicator,verbs}.*`.

## 2. Rules (must follow)

1. **Track all work through GitHub issues on `uv-genai/maya-tp`.** One issue per plan phase / workstream — **not** one umbrella with comments. Comment while making progress. Link commits (`refs #N` / `closes #N`). Close with a summary.
2. **Push only to `origin` (`uv-genai/maya-tp`). Never push to `upstream` (`mw00/project-maya`).**
3. **GPU is the inference path.** CPU is only for unit tests of slice math. Never run the model on CPU.
4. Perf claims need **controlled** measurements (fixed prompt, no draft-acceptance noise). Use `--spec-follow` for near-tie counts.
5. **NEVER use nested SSH** (ssh from strix-halo-1 to strix-halo-2). You are ON strix-halo-2. Run things locally. Use SSH only to reach strix-halo-1.
6. **Python must use `.venv/bin/python`** (not bare `python3`). Shell scripts use `jq` for JSON.
7. **Always kill old processes before starting new ones.** `pkill -9 maya; pkill -9 -f server.py`.

## 3. Git / GitHub

| | |
| --- | --- |
| `origin` | `https://github.com/uv-genai/maya-tp.git` (push here) |
| `upstream` | `https://github.com/mw00/project-maya.git` (read/merge only) |
| `main` | tracks `origin/main` |

## 4. Issue map

| # | Workstream | Status |
| --- | --- | --- |
| 1 | Umbrella (index only) | open |
| 2 | P1 partition + pack audit | **closed** |
| 3 | GPU-staged QueuePartial (6 µs / 37 Gb/s) | open |
| 4 | P4 greedy parity gate | **closed — PASS** |
| 5 | P5 fast path + prefill (perf) | **closed — +32% measured** |
| 6 | P6 serve integration | **open — BLOCKED on TP2 token mirroring** |
| 7 | P7 calibration / release | open |
| 8 | P2 RDMA transport | **closed** |
| 9 | MTP / spec decode TP | **closed — fixed** |
| 10 | Gather-based bit-exact variant | open (design alternative) |

## 5. What works (verified on the real Maya-S pack)

- **RDMA pair** live: `usb4_rdma5`, `roce_v2`, `gid_index=1`, `mtu=4096`, striping on.
- **Full TP2 slicing** (#5): mixer (KDA heads 64→32, DSA head-slice), experts (2048→1024), shared expert, dense FFN (12288→6144), attn_output column-slice. Two ARs per layer (mixer + FFN).
- **Benchmark mode works perfectly** (`run_bench.sh`): +32% vs single-host (56.81 → 38.50 ms/tok), pair parity 100% (32/32), spec-follow 31/32.
- **Engine stays alive as TP2 worker** (gufo serve behavior): `--tp-rank 1` without `--tokens` enters worker loop.
- **Transport:** `SendControl`/`RecvControl` on bootstrap TCP socket (with EAGAIN fix).

## 6. Performance status (measured 2026-10-29)

Controlled (no-MTP, 32 tokens, same prompt):

| Config | ms/tok | vs single |
| --- | --- | --- |
| single-host | 56.81 | — |
| TP2 slice, no AR | 36.57 | +35.6% |
| **TP2 full slice + AR** | **38.50** | **+32.0%** |

Pair parity 100% (32/32). Spec-follow 31/32 (1 near-tie).

**AR cost:** 90 exchanges × ~55 µs = 5 ms (13%). Link is 6 µs; overhead is driver-bound.

## 7. Architecture (what was built)

```
include/strata/core/tp_partition.hpp   TpPartition + tp_slice_expert
include/strata/core/tp_context.hpp     TpContext: rank/world/comm + all_reduce/start_reduce/finish_reduce
include/strata/net/{communicator,verbs,rdma_gid,tp_wire}.hpp   gufo Ibrverbs port
src/net/{verbs,rdma_gid}.cpp, src/net/tp_kernels.cu   RDMA + Stage/Wait/Add kernels
src/core/{tp_context,glm_model,glm_fast_path,glm_prefill,glm_window}.cu   slice + AR hooks
tools/tp2/{run_bench.sh,run_noar.sh,run_prof.sh,tp2-swap.sh,rank0.sh,rank1.sh,acceptance.sh}
tools/llama-swap/tp2.sh               llama-swap launcher (run PORT / stop)
tools/test_tp_config.py                10 CI tests
tools/calibrate_glm_tp.py              AR cost measurement
docs/TP2-RUN.md                        ops runbook
```

**Sharding:** All weights sliced under TP: mixer (KDA q/k/v row-slice, attn_output col-slice, per-head SSM state), experts (2048→1024), shared expert (2048→1024), dense FFN (12288→6144). DSA head-slice (MLA latent replicated). Two ARs per layer (mixer + FFN). Shared expert summed across ranks.

## 8. Critical gotchas

0. **Transport latency:** ~6 µs is the RDMA write transit; ~35 µs is the end-to-end blocking exchange. Both Maya and gufo have identical transport code. gufo's "6 µs" is the overlapped residual.
1. **Shard stride rule:** `Ly.gu_bytes` under TP is the **sliced** width; the GGUF shard's expert stride is always the **full** expert (`gu_row * n_ff_exp`).
2. **Per-layer width:** `ff_compute_for(il)` — the MTP draft layer stays full.
3. **Shared expert is summed across ranks** (sliced = partial). `MAYA_TP_SH_BOTH=1` forces both ranks.
4. **Drain contract:** `hipStreamSynchronize` before `CheckQueued`/`EndOperation`.
5. **Bootstrap before load:** `CreateIbrverbsCommunicator` → `TpContext::create` → `set_tp` → `load_pack_env`.
6. **Near-ties:** `dot(half0)+dot(half1) != dot(full)` in float32. Use `--spec-follow`.
7. **GCC 16 + HIP `<format>`:** compile with `-include format`.
8. **`ggml_quantize_chunk` is `(type, src, dst, start, nrows, n_per_row, imatrix)`**.
9. **`ibv_create_cq_ex` and `IBV_SEND_INLINE` are unsupported** by the Thunderbolt driver. Graceful fallback.
10. **Bootstrap socket is non-blocking** after connect. `RecvControl` must loop on EAGAIN.

## 9. How to run

### Benchmark (works perfectly)
```sh
# single-host
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P --tokens $TOKS --max-new 32 --greedy --max-context 4096

# TP2 pair (rank 0 first, rank 1 within 30 s)
bash tools/tp2/run_bench.sh
```

### Serve mode (BROKEN — see §10)
```sh
# llama-swap config:
#   "maya-glm5-tp2":
#     cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT}
#     cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
#     proxy: http://127.0.0.1:${PORT}
```

### Isolation flags
| Flag | Effect |
|---|---|
| `MAYA_TP_NOSLICE=1` | Full weights on both ranks |
| `MAYA_TP_NOAR=1` | Skip all-reduce |
| `MAYA_TP_SH_BOTH=1` | Shared expert on both ranks |
| `MAYA_GLM_NO_MTP=1` | Disable MTP |
| `GLM_CB_DIR=<dir>` | Dump layer outputs |

## 10. THE OPEN PROBLEM: serve-mode TP2 token mirroring

**Status: NOT WORKING. Only rank 0's GPU is used. Tokens never reach rank 1.**

**What gufo does** (`src/cli/serve/tp_executor.cpp`, `tp_control.cpp`):
- `TpMirroredRunner` sends a `TpInstruction` (kPrefill / kAdvance / kDecode) to rank 1 over a TCP control channel **before every model call**.
- Rank 1's `TpExecutor` executes the same call. Both ranks make identical calls in identical order.
- The mirroring is at the **caller level** (the runner), not inside the model's `forward()`.

**What Maya has tried (all failed):**
1. Mirror inside `forward()` — `prep_prompt` blocks before `forward()` is called, so the mirror never fires.
2. Send at start of `run_request` — rank 1 calls `forward()` immediately while rank 0 is stuck in `prep_prompt`, ARs mismatch.
3. Rank 1 worker loop calls `forward()` with dummy tokens — RecvControl gets EAGAIN (fixed) but rank 0 never sends.

**Root cause:** `prep_prompt` (generate.cpp:1820) does snapshot/checkpoint restore BEFORE calling `forward()` (line 2039). The "reading the prompt: N tokens" message comes from `serve/server.py`'s progress monitor, not the engine. `forward()` is never reached on rank 0.

**The fix (gufo's pattern):** Put `SendControl`/`RecvControl` at EVERY `model.forward()` call site in the request path:
- `prep_prompt` line 2039: `model.forward(part, lg, err)` — prompt processing
- `run_request` line 2318: `model.forward({tok}, lg, err)` — decode step
- Any other `forward()` call in `run_request`

Before each call: rank 0 sends the token IDs, rank 1 receives and calls `forward()` with the same IDs. The ARs synchronize them.

**Key files:**
- `src/program/generate.cpp` — `run_request` (line 2114), `prep_prompt` (line 1820), `forward()` calls at lines 2039, 2318
- `src/core/glm_model.cu` — `Glm5Model::forward()` (line 1001), `tp_` member
- `src/net/verbs.cpp` — `SendControl`/`RecvControl` (lines 700-730)
- `include/strata/net/communicator.hpp` — `SendControl`/`RecvControl` interface
- `tools/llama-swap/tp2.sh` — llama-swap launcher
- `tools/tp2/tp2-swap.sh` — swap launcher (rank 0 = serve/server.py, rank 1 = engine worker)

**Gufo reference files:**
- `/home/ugo/projects/gufo/src/cli/serve/tp_executor.cpp` — `TpMirroredRunner`, instruction mirroring
- `/home/ugo/projects/gufo/src/cli/serve/tp_control.hpp` — `TpInstruction`, `TpControlChannel`
- `/home/ugo/projects/gufo/src/cli/serve/tp_control.cpp` — control channel implementation

## 11. Key commits

`e43c95d` RecvControl EAGAIN fix · `5d32a39` token forwarding · `1e7737c` persistent worker · `c22a79a` launch scripts + config · `1e3a4a5` P6/P7 deliverables · `abb2f97` Handoff · `b92359d` Handoff · `0526f4e` llama-swap · `00d9781` swap scripts · `6c50d17` raw CQ polling · `9d9cbd1` transport experiments · `c8871be` TP2 full slice (+32%)
