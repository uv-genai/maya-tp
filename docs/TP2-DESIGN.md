# TP2 Design — Maya GLM-5.3 Tensor Parallel over Two Strix Halo Hosts

Companion to [TP2-ANALYSIS.md](TP2-ANALYSIS.md). Design only — no code in this document.

**Product shape:** one logical Maya-S model, two ranks on `strix-halo-1` (rank 0) and `strix-halo-2` (rank 1), connected by the existing striped Thunderbolt RDMA (`usb4_rdma*`). Rank 0 exposes the usual OpenAI/Anthropic API via `serve/`; rank 1 is a lockstep worker.

---

## 1. Goals and non-goals

### Goals
- TP=2 **intra-expert tensor parallel** for GLM Maya-S (IQ2_XXS pack), decode and prefill.
- Both ranks bit-identical to each other on every collective result (gufo’s two-operand sum rule).
- Greedy outputs match single-host Maya-S except documented near-tie flips from reduction order.
- Reuse gufo’s proven transport + control discipline; stay idiomatic to Maya (pack loader, fast path, expert tiers).
- Work on gfx1151 HIP only for v1 (`STRATA_ENABLE_HIP`, wave32). No NVIDIA path changes.

### Non-goals (v1)
- world_size > 2, expert-parallel all-to-all, cross-host KV streaming, images on TP, DFlash2/MTP multi-draft across TP, component cache restore across ranks, bit-identical match to single-host logits.

---

## 2. Process and rank model

```
strix-halo-1 (rank 0)                    strix-halo-2 (rank 1)
┌─────────────────────────┐              ┌─────────────────────────┐
│ serve/server.py         │              │ maya --tp-worker        │
│  └─ maya --serve        │   control    │  └─ same Glm5Model      │
│      scheduler, cache   │◄──TCP 18516─►│      mirrored calls     │
│      token selection    │              │      no HTTP            │
│         │               │   RDMA       │         │               │
│    Communicator verbs   │◄═════════════►│    Communicator verbs   │
│    half FFN + heads     │  usb4_rdma*  │    half FFN + heads     │
└─────────────────────────┘              └─────────────────────────┘
        bootstrap TCP 18515 (QP metadata), token auth
```

- **One engine process per host**, same binary (sha256 must match), `--tp-world-size 2 --tp-rank {0,1}`.
- Rank 0 keeps Maya’s existing serve line protocol and `serve/server.py` surface. Rank 1 runs `--tp-worker`: no public port; answers only the control channel + collectives.
- Token selection, sampling, detokenize, HTTP stay on rank 0. Rank 1 only runs model state transitions that change activations/weights caches.
- Start order: rank 0 first, rank 1 within 30 s (gufo ops rule). Supervisor restarts both.

CMake: new option `STRATA_ENABLE_TP2_RDMA` (like gufo’s `GUFO_ENABLE_TP2_RDMA`), default **OFF**; links `libibverbs` / `rdma-core` only when ON. HIP builds otherwise unchanged (`python tools/check_hip_compat.py` still mandatory if any CUDA spelling is added — expect **none** in the transport).

---

## 3. Weight partition

### 3.1 Principle
Every rank loads the **same pack** from disk but uploads **only its slice** of split tensors. Slices are taken as **byte spans of existing quant blobs** (no re-encode), always on quant-block boundaries.

`TpPartition { rank, world=2, expert_ff, ff_begin, ff_count }` with `ff_begin = (expert_ff/2)*rank` — same struct idea as gufo `tp_partition.hpp`, validated at load.

### 3.2 Per-tensor rules (Maya-S)

| Tensor | Split axis | Rank r owns | After |
| --- | --- | --- | --- |
| `ffn_gate_exps`, `ffn_up_exps` (288×) | rows = `n_ff_exp` | `[1024·r, 1024·(r+1))` | local SwiGLU |
| `ffn_down_exps` | cols = `n_ff_exp` | same 1024-wide column band | **all-reduce** weighted sum |
| `ffn_{gate,up}_shexp`, `ffn_down_shexp` | same as routed | 1024 of 2048 | **all-reduce** with routed |
| `ffn_gate`, `ffn_up` (dense lead) | cols `n_ff_dense` | 6144 of 12288 | — |
| `ffn_down` (dense) | rows | 6144 | **all-reduce** |
| KDA `attn_q/k/v`, `ssm_*` per-head | heads 64→32 | heads `[32r, 32(r+1))` | none (state per head) |
| DSA `attn_q_b` / v absorb | heads 64→32 | 32 heads | none |
| `attn_output` (KDA and DSA) | rows (input heads) | 32-head band | **all-reduce** |
| `attn_kv_a_mqa`, `ssm_a`, norms, `hc_*` | **replicate** | full | — |
| Router `ffn_gate_inp`, `exp_probs_b` | **replicate** | full | identical top-8 |
| Indexer block | **replicate** | full | identical cell select |
| `token_embd`, `output.weight`, `output_norm` | **replicate** (v1) | full | greedy argmax local |
| NextN / blk.45 | **replicate** (v1) | full | optional later split |
| mHC mix | **replicate** (v1) | full | avoids an extra AR |

**Why replicate head/mHC/indexer (v1):** they are a small fraction of bytes; splitting them adds collectives and near-tie surface. Gufo’s split-head + 1-row argmax is a later optimization.

### 3.3 Quant block alignment
- IQ2_XXS / IQ2_S / IQ3_XXS / Q6_K / Q8_0 super-blocks are 256 values (ggml). `n_ff_exp=2048`, `n_ff_dense=12288` both divide cleanly for world=2.
- Load **refuses** any split that is not block-aligned (gufo rule). No `DS4_TP_GGUF_UNEVEN` analogue in v1.

### 3.4 Expert tiers under TP2
Each node’s `ExpertCache` / RAM / SSD tiers hold **only that node’s 1024-wide slice** of each expert blob. Routing still selects the same 8 experts; each node computes its slice for all 8. Residency bookkeeping is unchanged but sized for ~46 GB of expert bytes — on 128 GiB unified memory the whole slice should stay in the VRAM tier (the main win).

CPU lane and helper-GPU cache (`remote_experts.cpp`) remain **single-node** tools on each rank’s own slice.

---

## 4. Math: where the all-reduce sits

Residual stream `x` is replicated on both ranks (same values) at every block boundary.

Per transformer block (KDA or DSA + MoE):

1. Norms / mHC prep (replicated) → `x0`.
2. **Mixer** (KDA recurrence or MLA attention) on local heads → `attn_out` partial (`n_embd` wide).  
   → **AR₁** (`attn_out`), then `hc_post` + residual add (identical on both ranks).
3. Router (replicated) → same top-8 ids and weights.
4. Routed experts: local `gate/up` slice → SwiGLU → local `down` slice → weighted sum partial. Shared expert likewise (local slice).  
   → **AR₂** (`moe_out`), residual add.

Decode: 2 ARs × `n_embd` F32 = 32 KiB/layer/token (or 16 KiB if FP16 on the wire — decide in Phase 2 parity tests; gufo stays F32 partials in windows).  
Prefill: same ARs over `T × n_embd` rows per chunk.

**Invariants**
- Both ranks issue AR₁, AR₂ in the same order for every block, every row, every chunk.
- Router and indexer run on replicated weights + identical input ⇒ identical selection. Any future rank-local randomness is forbidden.
- Fused variant: AR₂ can include shared+weighted routed in one buffer (recommended).

**Head:** local full `output.weight` (replicated) ⇒ full logits on rank 0; rank 1 may skip the head entirely after the last AR (optimization).

---

## 5. Transport layer

### 5.1 Interface (port gufo’s `Communicator`)
Reuse the abstract API (`communicator.hpp`):

- `BeginOperation(scope_id)` / `EndOperation(scope_id)`
- `ExchangePartial(data, bytes, stream) -> peer_floats` (blocking)
- `StartPartial` / `FinishPartial` (overlap)
- `QueuePartial` + `CheckQueued` (GPU-stage path)

Two-rank sum: `local += peer` on the GPU after exchange. Both ranks get the same bits.

### 5.2 Verbs backend (port `verbs.cpp` skeleton)
- RC QP, `IBV_WR_RDMA_WRITE_WITH_IMM`, 3 rotating receive windows, fixed VAs, `ibv_reg_mr` LOCAL_WRITE|REMOTE_READ|REMOTE_WRITE, event CQ channel, TCP bootstrap held for peer-loss.
- Wire header `(scope_id, operation_id, bytes)` + magic; mismatch poisons.
- GID selection via IPv4-mapped RoCE v2 (port `rdma_gid.cpp`).
- Device pick: `--tp-rdma-device` default lowest-GUID `usb4_rdma*` (names shift on re-plug — gufo rule).

### 5.3 Sizing for Maya
Window must hold one prefill batch partial: `T × n_embd × 4` bytes. Gufo uses 48 MiB (4224 rows × 2560 F32). For Maya `n_embd=4096`: 48 MiB holds **3072 rows**. Keep 48 MiB × 3 windows and **cap prefill sub-batch rows ≤ 3072** under TP2 (or size windows to `2×` the cap). Decode 1-row exchanges are tiny.

### 5.4 Fallback
Phase 2 may implement the same `Communicator` over TCP (correctness only). Phase 3 replaces the backend with verbs. No dual-path at runtime except `--tp-transport=tcp|verbs` for bring-up.

---

## 6. Control plane (rank lockstep)

Mirror gufo’s **TpMirroredRunner / TpExecutor** shape, but against Maya’s serve protocol:

| Rank 0 | Instruction | Rank 1 |
| --- | --- | --- |
| `GEN` starts request | `kSingle` (prompt tokens, ctx, sampling flags) | open mirrored seq slot |
| prefill chunk | `kPrefill` (token span, chunk id) | same prefill |
| decode step / window / multi-row group | `kAdvance` / `kAdvanceBatch` (row list, **draft counts**, RNG if sampling) | same |
| reset / cancel | `kReset` / `kCancelPrepare` | same |
| end | `kEnd` (digest) | verify + drop |

Rules (from gufo’s failure history):
- Rank 0 mirrors **row group composition and draft counts**, not every micro-decision; RNG state travels with sampled steps.
- Every state-changing call goes through the wrapper. Cancellation is checked **between** model calls, never inside an exchange.
- Same collective sequence is mandatory; digest/ordinal mismatch fails the request, not the process, unless the communicator is poisoned.
- `--parallel` / `STRATA_GLM_SEQS` slots are **mirrored 1:1** (same slot ids on both ranks). v1: no rank-local-only slots.

Bootstrap handshake (TCP 18515): exchange QP/GID/window info **and** a settings hash (pack id, geometry, context, seqs, MTP on/off, quant recipe). Differing settings refuse to pair (HTTP 500-style error on rank 0).

---

## 7. Decode path integration

### 7.1 Reference path first (`step_layers`)
Prototype the AR insertion in `glm_model.cu:2211` with a `TpContext*` on `Glm5Model` (null = today’s single-node). Acceptance gate lives here (token parity).

### 7.2 Fast path (`fast_layers` / `fast_moe`)
- Insert AR₁ after mixer kernels, AR₂ after `moe_down` weighted sum (and shared), **before** `hc_post`.
- Prefer `QueuePartial` so the service-thread expert prefetch overlaps the exchange (gufo’s queued form).
- **Disable CUDA/HIP layer graphs under TP2** (gufo rule) — plan/capture divergence risk.
- Multi-seq window pipeline (`win_pipe_*`) must issue the same row order on both ranks (control channel carries the group).

### 7.3 Speculative decoding
v1: **single-token decode only** (`MTP off`, `DFlash off`) on TP2. Rationale: gufo needed mirrored draft counts and still gates draft-subset to single-host; Maya’s one-GPU MTP chain is a bigger lift. Document as a hard gate `tp_world==1` like gufo.

---

## 8. Prefill path integration

- Chunking (`prefill_chunks.hpp`) unchanged; both ranks run the same chunks.
- Expert MMQ / RDNA4-fused products run on the local `n_ff` slice only (half the expert bytes read — good).
- Overlap (gufo): run two sub-batches a layer apart so one’s AR crosses the link while the other computes. Optional in v1.1; correctness first (sequential chunks).
- Row cap 3072/chunk under TP2 until windows are larger.

---

## 9. State, KV, sessions

| State | Policy |
| --- | --- |
| KDA `S` + conv | split by head (local heads only) |
| MLA latent `kv_lora` (512) | **replicated** both ranks |
| Indexer ik/ig + pooled keys | replicated (selection identical) |
| Residual / norm | replicated |
| Session slots | mirrored ids; rank 0 owns HTTP session semantics |
| KV streaming / `--kv-resident` | **single-host only** in v1 (`tp_world==1` gate) |

---

## 10. Configuration and launch

### Engine flags (maya)
```
--tp-world-size 2 --tp-rank {0,1}
--tp-bootstrap-host HOST --tp-bootstrap-port 18515
--tp-control-port 18516 --tp-control-token TOKEN
--tp-rdma-device usb4_rdmaN --tp-rdma-port 1
--tp-gid-index N            # if RoCE GID ambiguous
--tp-worker                 # rank 1: no --serve HTTP
```

### `maya-*.json`
```json
{
  "tp": {
    "world_size": 2,
    "rank": 0,
    "bootstrap_host": "strix-halo-1.local",
    "bootstrap_port": 18515,
    "control_port": 18516,
    "control_token": "...",
    "rdma_device": "auto"
  }
}
```
`maya.py` / `./maya.sh` learn `--tp-rank` / `--tp-peer` and write these fields. Env aliases `MAYA_TP_*` map to `STRATA_TP_*` like other settings; document in `docs/SETTINGS.md`.

### Host networking (already provisioned)
Reuse gufo `TP2-SETUP.md` Stage 2: `thunderbolt_ibverbs` loaded, `native_write_striping=1`, `tbv0` 10.77.0.1/.2, provider matches rdma-core 65. Do **not** reconfigure while a benchmark runs.

Launch scripts (new): `tools/tp2/maya-rank0.sh`, `maya-rank1.sh` — copied to both hosts, byte-identical binaries.

---

## 11. Failure handling

| Event | Behavior |
| --- | --- |
| Peer process dies / link lost | both ranks exit; rank 0 marks `/ready` red first; supervisor restarts |
| Collective header mismatch | poison communicator; fail request; log size fingerprint |
| Verbs WC status ≠ SUCCESS (e.g. **5 = MW_BIND**) | poison + log; ops note: reboot / reload module; never restart-loop |
| Setting mismatch at handshake | refuse to pair, name the setting |
| Desynced draft/row counts | prevented by mirroring; if detected (ordinal/size), fail request |

Logging: `event=rdma_ready`, `event=load_completed`, `event=tp_pair_lost` (gufo-compatible names for ops muscle memory).

---

## 12. Parity and measurement

1. **Pair parity:** same prompt, greedy, both ranks — rank 0’s tokens must equal rank 1’s view of collectives (digest).
2. **Single-host vs TP2:** token dumps (`MAYA_GLM_MULTI_DUMP`-style) on a fixed prompt set; allow near-tie flips; count agreement rate (expect high 90s like gufo).
3. **Speed:** `./maya.sh --bench` methodology (3×256 greedy, prefill 2K/8K) on the pair vs one host; plus gufo-style pp2048/tg128 controlled runs — **no single-prompt anecdotes** (user rule from gufo work).
4. **Load check:** expert tier stats (`STAT vram_hit=…`) should show ~all-resident on each node.

---

## 13. File-level touch map (design intent)

| Area | New / change |
| --- | --- |
| Transport | `src/net/communicator.hpp`, `src/net/verbs.cpp`, `src/net/tcp_comm.cpp`, `src/net/rdma_gid.cpp` (port from gufo) |
| Partition | `src/core/tp_partition.hpp/.cpp`, slice helpers next to `weights.cpp` / `iq_pack` spans |
| Model | `Glm5Model` `TpContext`, AR hooks in `glm_model.cu`, `glm_fast_path.cu`, `glm_prefill.cu` |
| Serve | `generate.cpp` flags; `--tp-worker` mode; control protocol `src/serve/tp_control.cpp` (or extend `serve/` glue) |
| Build | `CMakeLists.txt` `STRATA_ENABLE_TP2_RDMA`, link ibverbs |
| Tools | `tools/tp2/maya-rank*.sh`, `tools/calibrate_glm_tp.py`, bench harness |
| Docs | `docs/SETTINGS.md` (`MAYA_TP_*`), `docs/TP2-RUN.md` (ops), this pair of design docs |

---

## 14. Open design choices (resolve during implementation)

1. **FP16 vs F32 partials on the wire** — F32 first (bit-stable, gufo); measure FP16 later (ds4 `ar2` precedent).
2. **Split head** — deferred; measure if logits AR becomes hot.
3. **Window size vs prefill row cap** — 48 MiB / 3072 rows vs bigger windows.
4. **Whether rank 1 also runs `serve/server.py`** — no; `--tp-worker` only (simpler, matches gufo).
5. **MTP on TP2** — later phase if decode headroom exists; requires gufo-grade draft-count mirroring.
