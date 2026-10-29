# Handoff — Maya TP2 (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-29 · HEAD `e599e6c`+ · branch `main` on `uv-genai/maya-tp`

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

## 3. Git / GitHub

| | |
| --- | --- |
| `origin` | `https://github.com/uv-genai/maya-tp.git` (push here) |
| `upstream` | `https://github.com/mw00/project-maya.git` (read/merge only) |
| `main` | tracks `origin/main` |
| `maya-upstream` | tracks `upstream/main` (official) |

## 4. Issue map

| # | Workstream | Status |
| --- | --- | --- |
| 1 | Umbrella (index only) | open |
| 2 | P1 partition + pack audit | **closed** |
| 3 | GPU-staged QueuePartial (6 µs / 37 Gb/s) | open |
| 4 | P4 greedy parity gate | **closed — PASS** |
| 5 | P5 fast path + prefill (perf) | open — **active, mixer slice done** |
| 6 | P6 serve integration | open |
| 7 | P7 calibration / release | open |
| 8 | P2 RDMA transport | **closed** |
| 9 | MTP / spec decode TP | **closed — fixed** |
| 10 | Gather-based bit-exact variant | open (design alternative) |

## 5. What works (verified on the real Maya-S pack)

- **RDMA pair** live: `usb4_rdma5`, `roce_v2`, `gid_index=1`, `mtu=4096`, striping on.
- **Sliced expert pool:** 44.3 GB/rank (was 88.7 GB). Experts served from the 3 GGUF shards (pack has **no `experts.bin`**).
- **Pair parity 100%** (rank0 == rank1) on greedy 32-token runs, with and without MTP.
- **Correctness vs single-host:** `--spec-follow` reports **31/32 tokens are the model's own greedy picks** (1 near-tie) with MTP; **28/32** without MTP (4 near-ties). Layer dumps prove the trunk slice is mathematically correct (`ffn_out-3` maxdiff **7e-07**).
- **MTP verify window fixed** (#9): per-layer blob width, draft layer kept full, shared expert rank-0-only.
- **Mixer slice implemented + measured** (#5): KDA heads 64→32 + DSA head-slice, attn_output column-slice, per-head state halved, 2 ARs/layer. **+22.3% vs single-host** (44.15 ms/tok), pair parity 100%, spec-follow 31/32. `all_reduce` (QueuePartial) is correct for decode; `split_reduce` is worse (host blocks).
- **T2 split_reduce** (#5): `start_reduce`/`finish_reduce` implemented but NOT used in decode (hurts perf). Use only for prefill two-sub-batch (pending).
- Transport bench (blocking/queued): 27 Gb/s / 35 µs tight-loop; gufo's 6 µs / 37 Gb/s is the **overlapped** cost.

## 6. Performance status (measured 2026-10-29)

Controlled (no-MTP, 32 tokens, same prompt). Run-to-run variance ~5%.

| Config | ms/tok | vs single | AR cost |
| --- | --- | --- | --- |
| single-host | 56.81 | — | — |
| TP2 + AR (experts only, P4) | 50.61 | +9.4% | 3.2 ms (44 exch) |
| TP2, no AR (`MAYA_TP_NOAR=1`) | 36.57 | +35.6% | 0 |
| TP2 slice (mixer only) + AR | 44.15 | +22.3% | 7.6 ms (90 exch) |
| **TP2 full slice + AR** | **38.52** | **+32.0%** | **~5 ms (90 exch)** |

**Pair parity 100%** (32/32). **Spec-follow 31/32** (1 near-tie).

**What is sliced** (all under `head_compute_for` / `ff_compute` / `ff_compute_dense`):
- Mixer: KDA q/k/v row-slice, attn_output col-slice, per-head state/params; DSA q_b/k_b/v_b head-slice
- Routed experts: intra-expert 2048→1024 (P4)
- Shared expert: gate/up row-slice + down col-slice (this round)
- Dense FFN: gate/up row-slice + down col-slice (this round)

**AR cost**: 90 exchanges × ~55 µs = 5 ms (13% of 38.5 ms). Link is 6 µs; the rest is software stack.

## 7. Architecture (what was built)

```
include/strata/core/tp_partition.hpp   TpPartition + tp_slice_expert (expert band → compact blob)
include/strata/core/tp_context.hpp     TpContext: rank/world/comm + all_reduce()
include/strata/net/{communicator,verbs,rdma_gid,tp_wire}.hpp   gufo Ibrverbs port
src/net/{verbs,rdma_gid}.cpp, src/net/tp_kernels.cu   RDMA + Stage/Wait/Add kernels
src/core/{tp_context,glm_model,glm_fast_path,glm_prefill,glm_window}.cu   slice + AR hooks
tools/tp2/{audit_split,greedy_parity}.py, run_comm_test.sh
docs/{TP2-ANALYSIS,TP2-DESIGN,TP2-PLAN,TP2-RUN}.md
```

**Sharding:** intra-expert — each rank holds 1024 of 2048 intermediate units of every routed/shared expert (gate/up rows + down columns). Dense `n_ff` 12288→6144. **Mixers sliced (KDA heads 64→32, DSA head-slice); attn_output col-slice; per-head state.** Indexer/MLA-latent/hc replicated. Two all-reduces per layer (mixer + FFN). Shared expert summed across ranks (sliced).

## 8. Critical gotchas

0. **Transport latency — what the numbers mean.** ~6 µs is the RDMA write transit (link latency, `ib_write_lat`). ~35 µs is the end-to-end blocking exchange (GPU stage → thread sees ready → RDMA → peer writes back → GPU sees arrived → add → sync). Both Maya and gufo pay ~35 µs per blocking exchange — the code is identical. gufo "gets 6" because its profiler reports the **overlapped** residual (the AR hidden behind compute). The fix is T2 (StartPartial/FinishPartial + two sub-batches), not faster transport.

1. **Shard stride rule:** `Ly.gu_bytes` under TP is the **sliced** width; the GGUF shard's expert stride is always the **full** expert (`gu_row * n_ff_exp`). Using the sliced size as the shard stride reads the wrong expert's bytes. Slice from the full stride (see `pack_moe_tail`, warm-up, `fast_read_part`).
2. **Per-layer width:** `ff_compute_for(il)` — the NextN/MTP draft layer (`mtp_il_`) is **replicated** (full `n_ff`). Using the global width for it corrupts the verify window.
3. **Shared expert is rank-0-only** on every path (decode, prefill, window multi, `moe_win_experts`, per-token fallback). The all-reduce must not double it.
4. **Drain contract:** `hipStreamSynchronize` **before** `CheckQueued`/`EndOperation`. Ending a scope with exchanges in flight poisons the communicator.
5. **Bootstrap before load:** `CreateIbrverbsCommunicator` → `TpContext::create` → `set_tp` → `load_pack_env`. The warmup takes 20–40 s; connecting after it times out.
6. **Near-ties:** `dot(half0)+dot(half1) != dot(full)` in float32. Bit-exact needs issue **#10** (gather design). Use `--spec-follow` to count near-ties, not raw token agreement (one flip cascades).
7. **GCC 16 + HIP `<format>`:** compile with `-include format` (or `hipcc`). CMake: `$<$<COMPILE_LANGUAGE:CXX>:-include;format>` (not `-include=format`).
8. **`ggml_quantize_chunk` is `(type, src, dst, start, nrows, n_per_row, imatrix)`** — not a flat count.

## 9. How to run

```sh
# build (HIP, TP2)
cmake -S . -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_ENABLE_TP2_RDMA=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-hip --target maya -j $(nproc)

# pack (both hosts must have it)
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374

# single-host reference
./build-hip/maya --glm-pack $P --tokens $TOKS --max-new 32 --greedy \
  --spec-follow-save /tmp/single.txt --max-context 4096

# rank 0 on strix-halo-1, then rank 1 here (within 30 s)
ssh strix-halo-1 "/tmp/maya-tp2 --glm-pack $P --tokens $TOKS --max-new 32 --greedy \
  --spec-follow-save /tmp/r0.txt --max-context 4096 --tp-world-size 2 --tp-rank 0 \
  --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5"
./build-hip/maya --glm-pack $P --tokens $TOKS --max-new 32 --greedy \
  --spec-follow-save /tmp/r1.txt --max-context 4096 --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host strix-halo-1.local --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5

# correctness
python tools/tp2/greedy_parity.py /tmp/r0.txt /tmp/r1.txt --min-agree 1.0   # pair parity
./build-hip/maya ... --spec-follow /tmp/r0.txt   # log line: N differ from the model's own pick
```

**Isolation flags:** `MAYA_TP_NOSLICE=1` (full weights), `MAYA_TP_NOAR=1` (skip AR), `MAYA_TP_SH_BOTH=1` (shared on both ranks). `GLM_CB_DIR=<dir>` dumps `ffn_out-<il>.f32` / `l_out-<il>.f32` for A/B. `MAYA_GLM_NO_MTP=1` disables MTP.

**CI (no GPU):** `python tools/check_hip_compat.py` · `python -m unittest discover -s serve -p "test_*.py" -t .` · `python -m unittest discover -s tools -p "test_*.py"`

## 10. Next steps (priority)

1. **#5 — measure the mixer slice + T2**: controlled bench (no-MTP, 32 tokens, same prompt). Mixer slice + split_reduce AR both implemented. Target 30%+ vs single-host.
2. **#5 — prefill two-sub-batch overlap**: interleave two trunk batches a layer apart (gufo's `split_reduce` pair path). This is the remaining T2 win for prefill.
3. **#5 — remaining paths**: glm_prefill.cu / glm_rows.cu / glm_window.cu / glm_mtp_rows.cu still use `g.n_head` in compute (85 sites) — must use `head_compute_for(il)` before MTP or prefill with TP2.
4. **#3** — transport: AR cost ~5 ms (90 exch × 55 µs). Link is 6 µs. **Thunderbolt driver limitations**: `IBV_SEND_INLINE` unsupported, `ibv_create_cq_ex` unsupported (falls back to `ibv_poll_cq`). Tried and failed: spin-before-sleep (mutex contention), fused WaitAdd (parity race). Remaining options: pipelined worker (post send + defer CQE wait), or accept the current cost. Prefill two-sub-batch needs two independent token streams (concurrent requests).
5. **#6** — serve integration + launch scripts.
6. **#7** — calibration + controlled benches + release.
7. **#10** — only if bit-exact output is required (gather design).

## 11. Key commits

`0244b75` MTP window fix · `3136a69` shard-stride fix · `48911e1` native_down_cols · `94d88ef` GPU path · `a0b6e2a` QueuePartial · `110e9d1` RDMA transport · `05515a3` P1 partition
