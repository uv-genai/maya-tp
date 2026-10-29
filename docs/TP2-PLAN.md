# TP2 Implementation Plan — Maya GLM Maya-S on Two Strix Halo Hosts

Implements [TP2-DESIGN.md](TP2-DESIGN.md); background in [TP2-ANALYSIS.md](TP2-ANALYSIS.md).  
**Rule for all phases in this session:** no code and no processes until the user says the benchmark is done. This file is the plan of record.

**Target commit discipline:** branch `tp2` (this tree is already on `tp2`); one PR per phase after Phase 3 when hardware runs are allowed.

---

## Phase 0 — Analysis & design (this deliverable)

| # | Task | Output | Status |
| --- | --- | --- | --- |
| 0.1 | Analyze Maya GLM + ds4/mm3 TP + gufo TP2/RDMA | `docs/TP2-ANALYSIS.md` | **done** |
| 0.2 | Design solution | `docs/TP2-DESIGN.md` | **done** |
| 0.3 | Implementation plan | `docs/TP2-PLAN.md` (this file) | **done** |

**Acceptance:** documents exist; partition math matches Maya-S geometry (2048→1024, 12288→6144, 256-block safe); reference paths point at real gufo files under `/home/ugo/projects/gufo`.

---

## Phase 1 — Partition + load-time slicing (CPU-only)

**Goal:** each rank can describe and (later) upload only its slice; refuse bad splits.

### Tasks
1.1 `include/strata/core/tp_partition.hpp` + `src/core/tp_partition.cpp`  
   - `TpPartition::Create(expert_ff, rank, world)` (equal split; reject remainder)  
   - `PlanRoutedBytes` / per-tensor span planner: given pack index + blob spans, return `(offset, bytes)` for gate/up rows and down columns.  
   - Unit tests (no GPU): `tests/tp_partition_test.cpp` — 2048/2, 12288/2, IQ2_XXS 256-block boundaries, refuse 129-value nonsense, dense-lead layers 0–2.

1.2 Pack span audit tool (no GPU): `tools/tp2/audit_split.py`  
   - Walk `index.txt` + `native_experts.txt` of a Maya-S pack; print every tensor’s split rule and whether block-safe.  
   - Must agree with gufo’s published “TP2 column-split alignment all GREEN” result.

1.3 Docs: add `MAYA_TP_*` stubs to `docs/SETTINGS.md` (off by default).

### Acceptance
- `python -m unittest tools.test_tp_partition` (or ctest host test) green.  
- `python tools/tp2/audit_split.py --pack <Maya-S>` prints all GREEN, exit 0.  
- No HIP / verbs code yet. `python tools/check_hip_compat.py` still green.

---

## Phase 2 — Communicator over TCP + control protocol (CPU-only)

**Goal:** two processes exchange partials and stay in lockstep **without** GPU or RDMA.

### Tasks
2.1 Port abstract `Communicator` (from gufo `communicator.hpp`) into `src/net/communicator.hpp`.  
2.2 `src/net/tcp_comm.cpp`: same `ExchangePartial` / `StartPartial` / `FinishPartial` / `QueuePartial` semantics over a socket; header `(scope, op_id, bytes)`; poison on mismatch.  
2.3 Control channel: `src/net/tp_control.*` — framed commands `kSingle/kPrefill/kAdvance/kAdvanceBatch/kReset/kEnd`, shared token, handshake settings hash.  
2.4 Host tests (no GPU):  
   - `tests/tp_comm_test.cpp` — two threads: N exchanges, overlap form, size-mismatch poison, peer-drop.  
   - `tests/tp_control_test.cpp` — mirror of gufo `tp_control_test` (ordering, auth, digest).  
2.5 Worker stub: `maya --tp-worker` that accepts instructions and runs a **fake** forward (zeros) so the pair can be smoke-tested with `nc`/scripts.

### Acceptance
- CPU test suite green in CI (`ctest` / unittest).  
- Two stub processes on localhost complete 10k exchanges with identical digests.  
- Handshake refuses mismatched settings.

---

## Phase 3 — Verbs RDMA backend (port from gufo)

**Goal:** drop-in `Communicator` on `usb4_rdma*` with the same tests.

### Tasks
3.1 CMake `STRATA_ENABLE_TP2_RDMA` (default OFF) + find `libibverbs`.  
3.2 Port `verbs.cpp` / `rdma_gid.cpp` (paths cited in analysis) → `src/net/verbs.cpp`, `src/net/rdma_gid.cpp`. Keep gufo wire magics or Maya-namespaced equivalents — prefer **reuse byte layout** so tooling/diagnosis match ops muscle memory.  
3.3 Window sizing: 3×48 MiB fixed VA + `hipHostRegister` when HIP present; `MaxPartialBytes` documented.  
3.4 Same unit tests as 2.4 against verbs backend (needs RDMA device; mark manual/`ctest -L rdma`).  
3.5 Ops doc `docs/TP2-RUN.md`: device pick, GID, status-5 / size-mismatch triage, start order, sha256 check.

### Acceptance
- On the twin hosts (when allowed): `ibv_devices` shows `usb4_rdma*`; pair exchange bench (`maya --tp-selftest-comm`) matches TCP results bit-for-bit on the sum.  
- CI still green without RDMA (option OFF).  
- **Do not touch** `thunderbolt_ibverbs` load state while user benchmarks run.

---

## Phase 4 — Reference forward with TP (correctness gate)

**Goal:** `step_layers` runs TP=2 and matches single-host greedy except near-ties.

### Tasks
4.1 `Glm5Model::TpContext` (null ⇒ unchanged single-node path — **NVIDIA/CUDA path untouched**).  
4.2 Load: `load_pack` slice upload via Phase-1 planner (dense arena + expert blobs).  
4.3 AR insertion in `step_layers` only: after mixer out, after MoE/shared sum. F32 partials.  
4.4 Head replicated; argmax on rank 0.  
4.5 Disable speculative + graphs when `tp_world==2` (assert/gates).  
4.6 Parity harness: `tools/tp2/greedy_parity.py`  
   - fixed prompt set (reuse gufo golden-style + Maya bench prompts)  
   - dump tokens from single-host and TP2 (`MAYA_GLM_MULTI_DUMP`-like)  
   - report agreement % and list flipped positions.

### Acceptance (hard gate — do not proceed until green)
- Both ranks produce **identical** token streams to each other (digest).  
- vs single-host: document flips; agreement high (target ≥ 95% on greedy factual prompts; investigate any structural mismatch).  
- `STRATA_USE_HIP` builds; CUDA-only users unchanged (option OFF ⇒ zero behavior change).

---

## Phase 5 — Fast path + prefill

**Goal:** production decode/prefill speed on the pair.

### Tasks
5.1 `fast_layers` / `fast_moe` AR hooks (`QueuePartial` + service-thread overlap).  
5.2 Prefill: same ARs on `T×n_embd`; row cap 3072/chunk (or raise windows).  
5.3 Expert slice residency: verify all-resident path (`STAT`); size `ExpertCache` for half-blob.  
5.4 Optional: two-sub-batch prefill overlap (gufo `split_reduce`).  
5.5 Metrics: extend `STAT` with `tp_ar_us`, `tp_wait_us`; `MAYA_GLM_TIMING` phases.

### Acceptance
- Fast-path tokens match Phase-4 reference on the same prompts.  
- Controlled bench (user’s methodology: pp2048 / tg128, **not** single-prompt anecdotes): TP2 vs single-host Maya-S on identical hardware class.  
- No `tp_pair_lost` / size mismatch in long soak (≥ 1 h multi-request).

---

## Phase 6 — Serve integration

**Goal:** `serve/server.py` + `maya-*.json` run the pair like a normal Maya.

### Tasks
6.1 Engine flags + config `tp` block (design §10); `maya.py`/`maya.sh` `--tp-rank` / `--tp-peer`.  
6.2 Rank 1 `--tp-worker` supervised by scripts `tools/tp2/maya-rank0.sh` / `maya-rank1.sh`.  
6.3 Mirror `STRATA_GLM_SEQS` slots; `--parallel` allowed only as mirrored counts.  
6.4 `/ready` semantics: red until pair up and `load_completed` on both.  
6.5 CI: unittest for config parsing + handshake hash; no GPU.

### Acceptance
- `curl` chat/completions against rank 0 works; kill rank 1 → clean pair-down; restart both → recover.  
- Multi-session smoke (2–4 chats) with stable streaming.

---

## Phase 7 — Calibration, docs, release hygiene

### Tasks
7.1 `tools/calibrate_glm_tp.py` (shape of `calibrate_ds4_tp.py`): AR overlap vs sequential, prefill chunk rows, CPU lane on each node.  
7.2 Settings: `MAYA_TP_AR_FP16`, `MAYA_TP_PREFILL_ROWS`, `MAYA_TP_OVERLAP` — default preserve Phase-5 behavior.  
7.3 Update `docs/SETUP.md` / `AMD_MAYA.md` pointer to `TP2-RUN.md`; CHANGELOG entry.  
7.4 Quality: one perplexity/next-token agreement run vs single-host (Maya-S bench methodology).  
7.5 PR to `main` per CONTRIBUTING (measurements in description; HIP-only feature; NVIDIA path untouched).

### Acceptance
- CI: hip_compat + serve + tools tests.  
- Documented before/after tables on the twin Strix Halos.  
- Ops runbook complete (start/stop/status, status-5, sha256).

---

## Dependency graph

```
P1 partition ──► P4 reference TP ──► P5 fast/prefill ──► P6 serve ──► P7 release
       │                ▲                    ▲
       └─► P2 TCP comm ─┴──► P3 verbs ───────┘
              (P2/P3 can proceed in parallel with P1 after interfaces freeze)
```

## Effort sketch (indicative)

| Phase | Nature | Risk |
| --- | --- | --- |
| P1 | small, pure CPU | low |
| P2 | medium protocol | medium (desync class bugs) |
| P3 | port + ops | medium (Thunderbolt verbs quirks) |
| P4 | core engine | **high** (parity) |
| P5 | performance | medium |
| P6 | plumbing | low |
| P7 | polish | low |

## Definition of done (product)

1. Two hosts run Maya-S TP2 behind one HTTP API.  
2. Greedy parity vs single-host documented (flips only near-ties).  
3. Controlled decode/prefill numbers better than single-host **or** clearly justified if not (expert-residency win should dominate).  
4. Failure modes match gufo ops (poison, pair-down, status-5 playbook).  
5. Single-host and NVIDIA builds bit-unchanged when `STRATA_ENABLE_TP2_RDMA=OFF`.
