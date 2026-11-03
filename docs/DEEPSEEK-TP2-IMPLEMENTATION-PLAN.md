# DeepSeek two-host tensor parallelism and DSpark: implementation plan

Status: proposed, 2026-11-03. Planning [#109](https://github.com/uv-genai/maya-tp/issues/109);
background and design rationale: [analysis/design](DEEPSEEK-TP2-ANALYSIS-DESIGN.md). This is a
plan, not evidence of implemented support. The user will decide when implementation begins.

## Goal and boundaries

Run DeepSeek V4.1 Flash Maya-XS and Maya-S on two gfx1151 Strix Halo hosts using RDMA, with the
existing DSpark speculative drafter and prompt lookup where validated. The initial production
configuration is one request at a time, text only, with synchronized target inference and
speculation. Keep GLM TP2/MTP and single-host DeepSeek behavior intact. Preserve the user-accepted
production binaries and user-owned processes; use isolated review builds and recorded process IDs.

DeepSeek DSpark uses pack `mtp.*` tensors but is not GLM NextN MTP. Its target verify/rollback,
taps and sampled acceptance require their own protocol. Do not claim multimodal, concurrent
requests, checkpoint reuse or other features until implemented and tested; reject them explicitly
in distributed mode rather than silently changing their behavior. Do not create a TCP tensor data
path. `--tp` already describes local GPU grouping: use distinct, explicitly documented two-host
settings and reject incompatible combinations.

The design's starting preference is **two-rank uneven block-aligned tensor slicing** with a
replicated logical residual and rank 0 owning DSpark/sampling. This is provisional: P1 must confirm
geometry and memory feasibility before P2 commits to it. If it does not fit or is unsuitable,
record evidence and revise the design and affected phase issues for a whole-expert strategy before
continuing; do not silently change the intended architecture.

## Dependencies and gates

P1 → P2 and P3 → P4 → P5 → P6 → P7. P2 and P3 can progress independently after P1's contract is
settled, but P3 must consume the agreed partition identity. P6's failure rules are designed in P3
and enforced from the first transport integration; P6 completes the cross-layer/API audit. No later
phase is accepted merely because an earlier unit test passes. Update its dedicated issue at each
meaningful milestone, attach measurements and commit SHAs, and close it only when its gate passes.

| Phase | Workstream | Depends on |
| --- | --- | --- |
| P1 | [#112](https://github.com/uv-genai/maya-tp/issues/112) — baseline, memory and architecture decision | None |
| P2 | [#115](https://github.com/uv-genai/maya-tp/issues/115) — pack geometry and rank partition | P1 |
| P3 | [#111](https://github.com/uv-genai/maya-tp/issues/111) — RDMA identity, control and lifecycle | P1; partition contract from P2 |
| P4 | [#110](https://github.com/uv-genai/maya-tp/issues/110) — two-rank target inference | P2, P3 |
| P5 | [#113](https://github.com/uv-genai/maya-tp/issues/113) — DSpark and rollback | P4 |
| P6 | [#114](https://github.com/uv-genai/maya-tp/issues/114) — failure and API integration | P3–P5 |
| P7 | [#116](https://github.com/uv-genai/maya-tp/issues/116) — validation, performance and operation | P1–P6 |

## P1 — Establish reference and select distribution (#112)

* Record exact code/pack identities and per-host hardware, context, settings and available memory.
  Capture single-host XS and S output, tokens, prefill/decode timings, DSpark acceptance and
  physical storage reads, using identical prompts/sampling and both cold/warm conditions. Preserve
  existing results as historical controls; do not assume their timing is a reproducible baseline.
* Inventory each target/drafter tensor, block type, size and likely rank ownership, along with
  attention/KV, compressed/index state, Engram, scratch, staging and RDMA buffers. Calculate actual
  per-host peak including unified CPU/GPU memory and headroom at the user's intended contexts.
  Separate page cache from committed allocations and distinguish whole model bytes from hot data.
* Trace all local-group attention/MoE sum, tap, router, load and checkpoint call sites for decode,
  verify and prefill. Confirm which same-host fast paths and quant kernels support uneven shards.
  Measure link payload and latency with existing transport on the available hardware; never probe
  bootstrap sockets with `/dev/tcp`.
* Decide tensor slicing versus expert ownership with a written comparison and measurable fit
  criteria. Define comparison tolerances and unsupported modes; capture baseline GLM TP2/MTP and
  single-host DeepSeek parity tests needed to detect regressions.

**Gate:** reproducible baseline and rank-by-rank peak budget, documented geometry and reduction
map, selected architecture and risk list. If feasibility cannot be demonstrated, keep the issue
open and revise the design instead of assuming two hosts eliminate disk reads.

## P2 — Partition and load ranks (#115)

* Add a pack-derived partition description independent of GLM geometry. Validate tensor types,
  per-layer expert width, head and output-group compatibility, rank ownership, and shared versus
  replicated tensors. Record a stable digest to compare between hosts. Detect different sources,
  quant, geometry, and rank mappings before collectives.
* Implement and unit-test block-aligned rank ranges, including Maya-S's 2304/256 = nine blocks,
  with 1280+1024 and 1024+1280 cases and complete nonoverlapping coverage. Verify tensor-specific
  gate/up/down offsets and layouts against decoded reference tensors, including mixed quant types;
  fail on unsupported formats or non-divisible head/group geometry instead of falling back.
* Allocate and load only each rank's selected shards, plus explicitly replicated pieces and rank-0
  drafter. Account for rank-dependent size, pool and scratch, Engram, 512K context, APU unified
  memory and transport reserve. Report expected/actual bytes and experts by tier for each rank.
  Verify separately that both packs and drafter are the intended XS or S, without config overwrite.
* Keep world-size-one loading and numerical behavior unchanged. Do not rely on the local
  `DS4_TP_GGUF_UNEVEN` switch as proof that remote tensor splits or prefill are correct.

**Gate:** host geometry/property/offset tests and gfx1151 rank-load checks pass for XS and S;
per-rank allocations and disk/CPU fallback are reported, with no unsupported configuration
silently accepted. No inference performance claim yet.

## P3 — Pair setup, control protocol and operation lifetime (#111)

* Add a DeepSeek pair configuration and worker entry point. Wire `maya-ds4` to the existing verbs
  library conditionally without changing GLM's wire compatibility or its local path. Agree on
  rank, model family, protocol/build version, pack fingerprint, geometry/partition digest,
  context, features and maximum row/partial sizes before executing model work.
* Define versioned control messages and sequencing for load/ready, feed, step, speculation
  window/verdict, reset, cancellation and shutdown. Validate lengths, position, sequence and
  feature flags; reject mixed builds, missing DSpark and unsupported operations. Rank 0 is the
  only API scheduler and sampled-token authority.
* Bind every model operation to a communicator scope; establish deterministic collective order and
  message sizes, drain and check all queued work before completion, and preserve first transport
  failure. Define explicit startup timeout and safe teardown for either rank failing. Reuse
  RDMA transport with control/bootstrap sockets only; do not borrow GLM's model-specific state.

**Gate:** serialization and fake-communicator tests cover mismatch, duplicate/out-of-order message,
truncated control, and disconnect; two-host smoke confirms handshake and real scoped RDMA exchange
without executing a model. GLM transport and peer-loss regression suites remain green.

## P4 — Distributed trunk without speculation (#110)

* Implement one synchronized text request: mirrored feed and token step, agreed chunk/window sizes,
  rank-local attention and FFN contributions, reductions before residual updates, and one coherent
  output/head and sampling authority. Make router IDs/weights consistent across ranks and count a
  replicated shared expert only once. Preserve DeepSeek's layer-dependent Engram and KV semantics.
* Wire both window and chunked prefill paths to bounded collectives; negotiate the smallest
  available receive window and memory-safe row count before either rank queues work. Check stream
  completion and propagate failures at every operation boundary. Start with no unproven overlap;
  optimize only after parity is established.
* Keep ring, compressor, indexed/compressed KV, `pos`, token history and any rank-local state in
  agreement. Explicitly reject checkpoint/restore and partial-feed operations until their mirrored
  semantics are implemented, rather than pretending local reuse is safe. Disable unsupported
  vision, concurrent sequences, helper and local multi-GPU combinations with actionable errors.

**Gate:** deterministic unit and two-host gfx1151 layer/output comparisons against the same
single-host XS pack; prefill chunk and token paths agree within recorded tolerances; S can load and
complete prompts when the budget permits. Compare greedy continuations and explain differences;
no GLM TP2/MTP or DeepSeek single-host regression. Quantify per-rank residency and storage I/O.

## P5 — DSpark and speculative state (#113)

* Separate target-window execution, acceptance/sampling, rollback and drafter seeding from the
  current monolithic `Model::verify` as needed, retaining single-host behavior. Rank 0 makes draft
  and lookup decisions and broadcasts ordered token IDs, row count, position and operation ID;
  both ranks execute identical target windows. Capture **fully reduced** trunk taps required by
  DSpark at the correct committed positions.
* Rank 0 retains q and makes authoritative greedy or sampled p/q acceptance with seed, counter,
  penalties, logprobs and output-limit semantics. Send verdict and committed position to rank 1;
  apply the same rollback and reseed on both sides before the next operation. Audit rings,
  compressor snapshots, index/KV, token/Engram history and drafter taps for each rejection position.
* Implement prompt lookup as an explicitly synchronized draft source. If a sampled DSpark or
  lookup combination cannot be supported in an intermediate checkpoint, reject it and document
  the restriction; final DSpark support requires the validated modes defined in P7.

**Gate:** forced zero, partial and full acceptance, every rejection position, no-draft and lookup
paths leave equal committed state and match subsequent target-only continuation. Test greedy and
sampled q/p semantics, seed/counter, penalties, EOS, logprobs and boundary budgets on the actual
pair; check statistical distribution where bit-exact output is inappropriate. Confirm draft
acceptance is real, not merely that a flag is enabled. Validate the existing single-host drafter.

## P6 — End-to-end failure, API and operational safety (#114)

* Integrate pair-down reporting with `serve/server.py`, retaining the first fatal cause and the
  pre-write guard in `_request_error` / `generate` (#88). No communicator poison clearing, rank
  retry, one-rank restart or implicit CPU fallback after a transport failure. Both ranks require a
  coordinated fresh start. Report clear context/model/feature mismatches at startup.
* Specify synchronized cancel, timeout, EOS and shutdown: finish or fail a matched operation before
  reset, never leave ranks on different collectives or commit positions. Exercise loss during
  bootstrap, prefill, queued reduction, target verify, verdict and rollback. Validate health and
  later requests immediately report the original pair-down cause.
* Integrate explicit family-aware launch/config selection and per-quant config isolation, matching
  model/pack on both ranks. Audit restart, worker log and process ownership; do not invoke existing
  name-wide cleanup harnesses against user services. Surface negotiated rank, DSpark status,
  feature restrictions, memory placement and first failure in health/logs.

**Gate:** automated injection and API tests pass with bounded completion and no deadlocks; verify
no later GEN reaches a failed engine. GLM `serve.test_tp2_peer_loss`, mirrored prefill/cancel,
MTP and user-facing endpoints pass unchanged. Review exact ownership/cleanup paths.

## P7 — Complete validation and prepare operation (#116)

* Run the ordered HIP mapping, server and tools suites in `AGENTS.md` on the final source, then
  relevant C++ parity suites and isolated gfx1151 two-host checks. Test both XS and S, DSpark
  on/off, prompt lookup, greedy and sampled requests, prefill/verify chunk boundaries, multiple
  prompt lengths including intended context, cold/warm startup, cancellation and coordinated
  recovery. Cover model/pack/version mismatch and explicit rejection of unsupported features.
* Compare one-host and two-host runs with identical quant, prompts, token limits and sampling.
  Repeat enough to report run-to-run ranges; record generated quality/differences, prefill and
  generation throughput, accepted/offered drafts, rank VRAM/system memory and headroom, physical
  reads/token, Engram I/O, RDMA traffic/latency, GPU utilization and load time. Record whether all
  required experts actually fit; do not claim speedup or eliminated I/O without measurement.
* Update user and agent docs, model/launcher choices and settings; supply reproducible commands,
  feature limitations, recovery instructions and results. Prepare isolated review artifacts and
  an explicit promotion/rollback procedure. The user controls production launches and validation;
  do not overwrite or restart their current pair as a routine test.

**Gate:** all required CI and gfx1151 tests pass; the evidence distinguishes supported functionality
from limitations and reports output differences. The user reviews and tests the result before any
production promotion. If results fail parity, do not publish performance as support.

## Cross-cutting rules

- No independent rank decisions that can change collective order or commit positions. Prefer
  explicit authoritative control and fail closed on uncertainty.
- Preserve original DeepSeek single-host execution and GLM TP2/MTP; do not broaden supported
  backends. Feature flags/defaults must avoid silently changing existing users' execution.
- Keep each phase's GitHub issue current, link commits with `refs #N`, and update the live handoff
  after state changes. The phase issues are work tracking, not permission to begin implementation.
- If a phase discovers the assumptions do not hold, update its issue and design, including
  dependencies and validation, before continuing. Do not skip a failed gate to reach a release.
