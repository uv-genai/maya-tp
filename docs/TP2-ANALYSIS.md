# TP2 Analysis — Two-Host Tensor Parallelism for Maya GLM-5.3 (Maya-S)

**Goal:** run GLM-5.3-Flash **Maya-S** as a **2-way tensor-parallel (TP2)** job on two twin systems:

| Host | Role | Platform |
| --- | --- | --- |
| `strix-halo-1.local` (192.168.1.22) | rank 0 (server) | AMD Strix Halo, gfx1151, CachyOS, 128 GiB unified memory |
| `strix-halo-2.local` (192.168.1.92) | rank 1 (worker) | identical twin, this machine |

**Constraints for this document:** analysis only. No code, no processes (a benchmark is running on both hosts).

**Reference implementation:** gufo’s two-host TP2 + RDMA stack (Qwen3.8-Flash-Next). Tree is **`/home/ugo/projects/gufo`** (the prompt’s `/home/ugo/software/gufo` does not exist; that is the on-host deploy path used in gufo’s `TP2-SETUP.md`).

---

## 1. What exists in Maya today

### 1.1 Engine shape

- Three binaries; this work is **only** `maya` (GLM). Entry `src/program/generate.cpp`; model `Glm5Model` (`include/strata/core/glm_model.hpp:96`).
- **Single-token decode:** `step_layers` (`src/core/glm_model.cu:2211`) is the correctness reference. The production path is `fast_layers` (`src/core/glm_fast_path.cu:5611`), ~11 fused launches per layer, device router + host-mapped expert service ring.
- **Prefill:** `src/core/glm_prefill.cu` + hipBLASLt GEMMs (`src/prefill/gemm.cu`) and MMQ / RDNA4-fused expert GEMMs (`src/prefill/moe_mmq.cu`, `moe_fused_rdna4.cu`).
- **Serve:** `serve/server.py` spawns `maya --serve` and speaks a line protocol (`READY`/`GEN`/`T`/`STAT`/`DONE`). `--parallel N` = `STRATA_GLM_SEQS` concurrent conversations on **one** engine — not tensor parallel.

### 1.2 Multi-GPU on one node = pipeline, not TP

`--layer-split` builds a chain of `Glm5Model` parts (`split_next_`), each owning layers `[l0_, l1_)` on one device. The residual crosses boundaries as a **64 KiB pinned host hop** (`glm_model.cu:2109-2128`). Peer access is deliberately unused (`glm_model.hpp:116`). Not supported across a split: KV streaming, images, control vectors, helper caches, slot lending (`generate.cpp:4610-4612`).

GLM already has **one** column-split: the head split (`STRATA_GLM_HEAD_SPLIT`, `glm_model.hpp:343-370`) — `output.weight` columns joined over pinned host buffers. That is the in-tree precedent for vocab-parallel logits.

### 1.3 In-tree tensor parallel precedents (single host)

| Engine | Split | Reduce |
| --- | --- | --- |
| **maya-ds4** (`src/ds4/ds4_model.cu`) | `split128` column slices: FFN `n_ff`, head vocab; groups of `tp` equal-VRAM GPUs | `ArGroup` all-reduce (`ds4_kernels.cuh:414`): mapped rows + flags; 2-member `ar2` FP16-on-wire; `allreduce_host` bit-exact CPU sum |
| **maya-mm3** (`src/mm3/mm3_tp.cu`) | `upload_slice` per member: q/k/v columns, `o_proj` rows, 1/tp of every FFN width | Residual stays on host; members write partials; **host sums in member order** (`mm3_tp.hpp:6-14`) |

**Pattern:** host-orchestrated barrier, 128-block column/row weight slices, deterministic member-order reduction. All of it assumes **one machine** (host-mapped / pinned memory). There is **no** NCCL/RCCL/MPI/socket collective anywhere in the tree.

### 1.4 Maya-S (IQ2_XXS) geometry that must shard

From `include/strata/core/glm_model.hpp:66-93` and gufo’s dumped GGUF inventory (1412 tensors, 46 blocks):

| Quantity | Value | TP2 (world=2) |
| --- | --- | --- |
| `n_embd` | 4096 | replicated residual |
| blocks | 0–45 (blk.45 = NextN/MTP) | full layers on both ranks |
| KDA layers | 34 (`il % 4 != 3`), heads 64, `d_inner` 8192 | 32 heads/rank |
| DSA/MLA layers | 11 (`il % 4 == 3`), `kv_lora` 512 head-shared | 32 q-heads/rank; **latent replicated** |
| dense lead | layers 0–2, `n_ff_dense` 12288 | 6144/rank |
| MoE | 288 experts, top-8, `n_ff_exp` 2048, shared 1×2048 | **1024 intermediate units/rank** |
| head | `output.weight` 4096→154880 | column-split (head-split precedent) |
| pack size | 96.5 GB (experts ≈ 95%) | ~48 GB expert share + ~5 GB dense/emb per node |

Gufo’s Maya-S v2 tensor map (`docs/models/glm5-flash/WEIGHTS.md`, issue #3) already **validated TP2 column-split alignment for world=2**: intermediate 2048→1024, dense 12288→6144, all quant blocks 256-elem, every `ne1` split block-safe.

**Maya-S v2 quants that bound the slice:** IQ2_XXS gate/up, IQ2_S down (IQ3_XXS first/last four MoE layers, IQ4_XS blk.11/12), Q6_K attention/shared/dense/embed/head, Q8_0 small KDA + indexer, F32 routers. Super-block alignment (256 values for IQ2_XXS/IQ2_S class) must be respected at every split.

### 1.5 Expert tiers under a single node

VRAM pool (LFU slots per layer) → pinned RAM → SSD mmap. On one Strix Halo, Maya-S + KV + workspace does not leave the whole expert set resident; decode is dominated by expert traffic (~336 expert reads/token). **TP2 halves the expert bytes per node**, which is the main reason to do this: both halves can stay in the 128 GiB unified-memory pool and the RAM/SSD tiers mostly disappear.

---

## 2. What gufo already proved (the reference)

Gufo TP2 is **world-size 2 only**, one process per host, same binary (sha256), on the same twin Strix Halo pair with **RDMA over two Thunderbolt/USB4 cables** (striped, ~20–44 Gb/s).

### 2.1 Partition (proven shape)

From `src/models/qwen38_flash_next/distributed/tp_partition.hpp` and `docs/models/qwen3.8-flash-next/TP2.md`:

- **Intra-expert split, not expert-parallel.** Each rank holds `[ff_begin, ff_begin+ff_count)` of **every** routed and shared expert: rows of gate/up, matching columns of down. Both ranks compute **all** selected experts on their half; the MoE exchange sums the shares. Gufo measured this as **AR +5% (Q4) / +6% (Q8)** vs whole-expert ownership, with **no load imbalance** (`EXPERIMENTS.md`).
- Down-projection shares must land on **quantization-block boundaries** (32 for Q8_0/Q5_1, 256 for Q4_K/Q5_K); IQ2_XXS-class 256-super-block types are the same class of constraint (gufo refuses a split that would re-encode).
- Attention: KV heads split with their query groups; GDN/key-head splits where the model has them.
- **Replicated:** indexer, draft/MTP, HC mixer, embedding, PLE, vision; LM head either replicated or split with a 1-row argmax exchange (gufo splits the Q8_0 head for greedy drafts).

### 2.2 Collectives (the single primitive)

One primitive: **two-rank all-reduce = one exchange of partial sums**.

> Each rank adds local+peer; float add of two operands is commutative, so both ranks hold bit-identical sums.  
> (`communicator.hpp:12-15`)

Placement: end of every mixer block and every MoE block (after routed-down + shared), plus the split-argmax row. ~10 KiB per decode row per exchange on gufo’s 2560-wide hidden; Maya’s `n_embd=4096` → **16 KiB F32 per row per exchange** (or 8 KiB FP16), 2–3 exchanges per layer block.

Overlapped form (`StartPartial`/`FinishPartial`) runs prefill as two trunk batches a layer apart so one batch’s exchange crosses the link while the other computes.

### 2.3 Transport

`verbs.{hpp,cpp}` = libibverbs RC QP, **RDMA write with immediate**, 3×48 MiB rotating receive windows at fixed virtual addresses, event-driven completion channel, TCP bootstrap kept open only to detect a dead peer. Wire magic `GUFT` v7; collective header `(scope_id, operation_id, bytes)`; any mismatch **poisons** the communicator.

Thunderbolt path (already installed on these hosts): kernel module `thunderbolt-ibverbs` (`neuhaus/donnerkeule` `feat/write-striping`) exposes `usb4_rdma*` devices; `native_write_striping=1` spreads one QP over both cables. Userspace provider must match rdma-core 65. RoCE GID from dummy netdev `tbv0` (10.77.0.1/24 vs .2). Full bring-up: gufo `TP2-SETUP.md`.

### 2.4 Control plane (rank lockstep)

Rank 0 owns HTTP + scheduler + cache; just before each state-changing model call it sends a framed TCP instruction (`tp_control.*`, port 18516, shared token) to rank 1’s `TpExecutor`, which makes the **same** call on the same state. Both meet in the call’s collectives. Rank 0 mirrors **sampler RNG and draft counts** so both ranks run cycles of identical width.

**Hard invariant (the thing that breaks pairs):** both ranks must issue the **same** collectives — order, count, bytes. Size mismatches appear as `identity or size mismatch` (e.g. 10240 vs 122880 bytes = 1 vs 12 rows). Single-host-only features are gated with `TpWorldSize()==1` (draft-subset, prompt-lookup skip, `defer_residual_`). HIP graphs off under TP2. Component-cache restore is rank-0-only → TP2 uses mirrored snapshots.

### 2.5 Operational facts (from live use on these hosts)

- Start **rank 0 first**, rank 1 within 30 s; identical binaries and identical flags (handshake names any differing setting).
- Verbs **status 5 = `IBV_WC_MW_BIND_ERR`** — RDMA/link state after a bad restart, not a decode bug; reboot clears it. Size-mismatch ≠ status 5.
- Kill either rank → peer exits status 1; run under a supervisor.
- Measured with Q8+MTP: **1260–1280 prefill / ~43 decode tok/s** on striped USB4 (vs 860–930 / ~22 on soft-RoCE one cable).
- Both hosts must hold **identical model files**.

---

## 3. Gap analysis: Maya GLM vs gufo TP2

| Concern | Maya GLM today | Gufo has | Work for Maya |
| --- | --- | --- | --- |
| Network collective | none (mapped memory, host hops) | `Communicator` + verbs | **New** — port/reuse gufo transport |
| Weight sharding | layer split + head split only | intra-expert FF + head splits | **New** — slice pack blobs at load |
| All-reduce sites | none | after every mixer + MoE | **New** — insert in `step_layers` / `fast_layers` / prefill |
| Rank lockstep | single process | `TpMirroredRunner` + control channel | **New** — or a leaner 2-engine variant |
| Expert tiers | VRAM/RAM/SSD one node | not applicable (all resident) | **Simplify** under TP2 (halved bytes) |
| Parallel sequences | `STRATA_GLM_SEQS` local | sessions mirrored, same counts | Must **mirror** row groups / draft counts |
| Speculative (MTP/DFlash) | one-GPU chain, head split | mirrored draft counts | Restrict or mirror; draft-subset is single-host only in gufo |
| gfx1151 | validated HIP, unified memory | same hardware, GPU-visible host windows | Reuse; windows rely on APU coherence (already gufo’s model) |
| Pack format | `index.txt` + `dense.bin` + `experts.bin` blobs | GGUF shards | Slicer works on blob spans / 128-blocks |

**Not portable as-is:** gufo’s model adapter (qwen38_flash_next kernels, GDN splits). What *is* portable: transport (`verbs`/`Communicator`), control protocol shape, partition *rules*, prefill overlap trick, diagnostics, and the failure taxonomy.

---

## 4. Options considered

| Option | Idea | Verdict |
| --- | --- | --- |
| **A. Intra-expert FF TP2 (gufo-style)** | Each rank holds half of every expert’s `n_ff`; all-reduce after MoE and attention out | **Chosen.** Proven +5–6% vs expert-parallel in gufo; no route imbalance; matches Maya-S alignment (2048→1024) |
| B. Expert-parallel (144 experts/rank) | All-to-all activations to owners | Rejected for v1: top-8 almost always touches both nodes; all-to-all + imbalance; more new kernels |
| C. Data-parallel two engines | Each host runs full Maya-S | Doesn’t fit one 128 GiB node cleanly for Maya-S + KV; not the ask |
| D. Pipeline (layer-split) across hosts | Layers 0–22 on host A, 23–45 on host B | Rejected: residual hop becomes a per-token network hop; poor overlap; not tensor parallel |
| E. TCP-only first | Skip RDMA | **Keep as Phase-2 stepping stone** (protocol bring-up), but not the performance target — gufo measured ~2× gap |

**MoE under option A:** both ranks run the same top-8 from a replicated router (same F32 weights + deterministic stable top-8), each computes its half of gate/up→SwiGLU→down, one all-reduce of the weighted sum (can fuse with shared-expert partial). No expert movement over the network in the hot path.

**Head:** v1 replicates `output.weight` (simplest; head is <2% of bytes) and uses a 1-row split-argmax only if we later split it (gufo precedent). Prefer replicate first.

**KV / state:** each sequence’s KDA `S`/conv state splits by head naturally; MLA latent (`kv_lora=512`) is **replicated** (it is head-shared — splitting it forces extra all-reduces per layer); indexer cell selection must be identical (replicated indexer weights + same input ⇒ same selection).

---

## 5. Memory and performance sketch (Maya-S on 2× gfx1151)

| | One host | TP2 (each host) |
| --- | --- | --- |
| Expert share | ~92 GB (must tier) | ~46 GB (resident) |
| Dense / emb / head / HC | ~4.5 GB | ~4.5 GB (replicated) |
| Per-seq KV (32K ctx, rough) | tens of GB if many seqs | same per seq, but room left |
| Expert tier pressure | high (RAM/SSD) | low (all in unified pool) |
| Link traffic (decode) | 0 | 2–3 × 16 KiB × rows × layers/token… ≈ order 1–3 MB/token (FP32) |

Decode should be **faster than single-host** if expert residency eliminates SSD/RAM waits, even before counting bandwidth. Prefill should benefit from gufo’s batched overlap (two chunks a layer apart) and from reading only half the expert bytes per node. Exact numbers need measurement (benchmarks are running now; measure later).

---

## 6. Risks

1. **Collective desync** — any rank-divergent branch (MTP acceptance, prompt-lookup, speculative subsets, cache restore) hangs the pair. Mitigation: gufo’s `TpWorldSize()==1` gates + mirrored call surface + size check in the header.
2. **Near-tie token flips** — reduction order differs from one host; expected (gufo documents this). Acceptance is “same tokens up to documented near-ties,” not bit-identical to single-host.
3. **IQ2_XXS / IQ2_S super-block boundaries** — 256-value blocks; 2048/2=1024 is safe; any future uneven split must refuse at load (gufo rule).
4. **Thunderbolt verbs flakiness** — status 5 after bad restarts; document reboot/reload, never restart-loop.
5. **Different binaries** — always `sha256sum` both `maya` binaries; mismatched pairs look like “RDMA bugs.”
6. **gfx1151 unified memory** — host-registered RDMA windows + GPU staging already work on these hosts in gufo; keep the same fixed-window + `hipHostRegister` pattern.
7. **Scope creep** — DFlash2 / multi-MTP / images / KV-streaming across TP should be **out of v1** (same as gufo’s single-host gates).

---

## 7. Sources (highest signal)

- gufo TP2 design: `/home/ugo/projects/gufo/docs/models/qwen3.8-flash-next/TP2.md`  
- gufo bring-up: `/home/ugo/projects/gufo/TP2-SETUP.md`, `docs/thunderbolt-rdma/strix-halo-thunderbolt-rdma.md`  
- gufo layer map: `docs/RDMA-PORT-PLAN.md`  
- Partition: `src/models/qwen38_flash_next/distributed/tp_partition.hpp`  
- Transport: `src/models/qwen38_flash_next/kernels/rocm/{communicator.hpp,verbs.cpp}`  
- Control: `src/cli/serve/{tp_control,tp_executor}.*`  
- Maya GLM: `include/strata/core/glm_model.hpp`, `src/core/glm_model.cu`, `src/core/glm_fast_path.cu`, `src/core/glm_prefill.cu`  
- Maya TP precedents: `src/ds4/ds4_model.cu`, `src/mm3/mm3_tp.cu`  
- Maya-S quality/layout: `bench/results/MAYA-S.md`, `docs/GLM5-FLASH.md`, `tools/iq_pack.py`  
- Host/RDMA ops memory: gufo project MEMORY (TP2 diagnostics, serving-ops: Thunderbolt dual-link, rank topology)
