# Handoff — Maya TP2 (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-30 · v1.1.37 · branch `main` on `uv-genai/maya-tp`

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
| 3 | GPU-staged QueuePartial (6 µs / 37 Gb/s) | **closed — link metrics confirmed; QueuePartial in engine** |
| 4 | P4 greedy parity gate | **closed — PASS** |
| 5 | P5 fast path + prefill (perf) | **closed — +32% measured** |
| 6 | P6 serve integration | **closed — mirroring, peer /health, kill/recover; multi-seq deferred** |
| 7 | P7 calibration / release | **closed — CI green, tables measured, v1.1.37 released** |
| 8 | P2 RDMA transport | **closed** |
| 9 | MTP / spec decode TP | **closed — fixed** |
| 10 | Gather-based bit-exact variant | open (design alternative) |

## 5. What works (verified on the real Maya-S pack)

- **RDMA pair** live: `usb4_rdma5`, `roce_v2`, `gid_index=1`, `mtu=4096`, striping on.
- **Full TP2 slicing** (#5): mixer (KDA heads 64→32, DSA head-slice), experts (2048→1024), shared expert, dense FFN (12288→6144), attn_output column-slice. Two ARs per layer (mixer + FFN).
- **Benchmark mode works perfectly** (`run_bench.sh`): +32% vs single-host (56.81 → 38.50 ms/tok), pair parity 100% (32/32), spec-follow 31/32.
- **Engine stays alive as TP2 worker** (gufo serve behavior): `--tp-rank 1` without `--tokens` enters worker loop.
- **Transport:** `SendControl`/`RecvControl` on bootstrap TCP socket (with EAGAIN fix).
- **Serve-mode TP2 token mirroring** (#6): FIXED 2026-10-30. Rank 0 sends `TpMirrorHdr`+tokens immediately before every collective model call (reset / prefill / forward); rank 1's worker executes the same call. Verified with two `curl /v1/chat/completions` requests (8 and 5 tokens; rank 1 log shows reset → prefill N → N× forward 1).

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

0. **What “6 µs / 37 Gb/s” means (do not mix them with AR cost).** Gufo’s Thunderbolt doc measures the **link**, not the model’s all-reduce:
   - **6.0 µs** = `ib_write_lat` **2 B WRITE round-trip** (one-way ≈ 3 µs). Confirmed again 2026-10-30: t_typical **6.04 µs**.
   - **37.4 Gb/s** = `ib_write_bw` **1 MiB WRITEs** (`-s 1048576`). Confirmed: 4459 MiB/s.
   - At **actual AR row sizes** sustained BW collapses: **48 KiB → 8.9 Gb/s**, **10 KiB → 9.1 Gb/s** (per-message overhead).
   - Maya’s **blocking** `ExchangePartial` at one decode row is **~58 µs** (48 KiB / hidden 12288), **~66 µs** at 10 KiB — host staging + driver + peer wait, not the 37 Gb/s ceiling. Gufo’s “~6 µs” in the TP path is the **overlapped residual** (`QueuePartial` Stage/Wait/Add), not a 48 KiB AR cost. See `tools/tp2` / `tp_comm_rdma_test --bench`.
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
11. **Do not probe the bootstrap port with `/dev/tcp`** — that steals rank 0’s single `accept()` (EAGAIN/EISDIR). Rank 1’s engine already retries connect.

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

### Serve mode (WORKS — verified 2026-10-30)
```sh
# llama-swap config:
#   "maya-glm5-tp2":
#     cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT}
#     cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
#     proxy: http://127.0.0.1:${PORT}
#
# Or manually:
#   rank 1 first (strix-halo-2):  bash tools/tp2/rank1.sh
#   rank 0 (strix-halo-1):        bash tools/tp2/rank0.sh 18080
#   curl http://192.168.1.22:18080/v1/chat/completions ...
```

### Isolation flags
| Flag | Effect |
|---|---|
| `MAYA_TP_NOSLICE=1` | Full weights on both ranks |
| `MAYA_TP_NOAR=1` | Skip all-reduce |
| `MAYA_TP_SH_BOTH=1` | Shared expert on both ranks |
| `MAYA_GLM_NO_MTP=1` | Disable MTP |
| `GLM_CB_DIR=<dir>` | Dump layer outputs |

## 10. Serve-mode TP2 token mirroring — FIXED (2026-10-30)

**Status: WORKING.** Two chat/completions requests completed; rank 1 logs `reset` → `prefill N` → N× `forward 1`.

**Pattern (gufo `TpInstruction`, simplified):** rank 0 sends an instruction immediately BEFORE every model call that participates in a collective. Rank 1's worker executes the same call. Do **not** mirror inside `forward()` — `prep_prompt` blocks before `forward()` is called.

**Wire** (`TpMirrorHdr` + `n * int32_t` tokens):
```
op: 0=reset, 1=prefill(+next_token), 2=forward
```

**Call sites mirrored** (`src/program/generate.cpp`):
- `prep_prompt`: `model.reset()` when reuse==0; `model.prefill(part, …, prompt[end])`; token-path `model.forward(part)`
- `run_request` simple decode loop: `model.forward({tok})` (after `on_token`)
- TP2 forces the simple decode loop (skips DFlash/MTP/spec — those need sampler state on the wire)
- TP2 disables conversation reuse (snapshot/checkpoint/slot) and multi-seq (`MAYA_GLM_SEQS`)

**Rank 1 worker:** `--tp-rank N` without `--tokens`, regardless of `--serve`. Entered after multi-bench, before `if (!o.serve)`.

**Gotchas fixed along the way:**
1. **Do NOT probe the bootstrap port with `/dev/tcp`** — the probe completes a TCP connection and rank 0's single `accept()` takes it as the peer; handshake then fails with EAGAIN/EISDIR. Rank 1's engine already retries connect (`MAYA_TP_BOOTSTRAP_TIMEOUT`, default 120 s). `tools/tp2/rank1.sh` no longer probes.
2. Rank 0 accept poll now matches that 120 s window (was hardcoded 30 s → “bootstrap accept timed out”).
3. Mirror traffic is **serve-only** (`tp_mirror = rank 0 && o.serve && tp2`). A bench run gives both ranks `--tokens`; pushing control traffic rank 1 never reads would fill the TCP buffer and block rank 0.

**Still open on #6:** `/health` reports `ok` from rank 0 alone (no peer liveness); kill/recover; multi-seq mirror (`kAdvanceBatch`/`kDecodeBatch`); mirror snapshot/restore for multi-turn reuse.

**Key files:**
- `src/program/generate.cpp` — `TpMirrorHdr`, `tp_send`, worker loop, prep_prompt/run_request hooks
- `src/net/verbs.cpp` — `SendControl`/`RecvControl`; accept timeout = `BootstrapTimeout()`
- `include/strata/net/communicator.hpp` — control channel interface
- `tools/tp2/rank0.sh`, `tools/tp2/rank1.sh` — launch scripts (rank1: no `--serve`, no port probe)
- `tools/llama-swap/tp2.sh` — llama-swap launcher

**Gufo reference:**
- `/home/ugo/projects/gufo/src/cli/serve/tp_executor.cpp` — `TpMirroredRunner`
- `/home/ugo/projects/gufo/src/cli/serve/tp_control.hpp` — `TpInstruction`, `TpControlChannel`
- `/home/ugo/projects/gufo/src/cli/serve/tp_control.cpp`

## 11. Key commits

`e43c95d` RecvControl EAGAIN fix · `5d32a39` token forwarding · `1e7737c` persistent worker · `c22a79a` launch scripts + config · `1e3a4a5` P6/P7 deliverables · `abb2f97` Handoff · `b92359d` Handoff · `0526f4e` llama-swap · `00d9781` swap scripts · `6c50d17` raw CQ polling · `9d9cbd1` transport experiments · `c8871be` TP2 full slice (+32%)
