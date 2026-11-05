# DeepSeek TP2 continuous batching and DSpark implementation plan

Reviewed on 2026-11-05 against `main` at `69bb0eeb1ea3f893dd3103b46bc53d378dff2b49`.
Feature tracking is #140 and the completed source-review workstream is #142.
This document defines proposed implementation work. Use the
[DeepSeek runbook](DEEPSEEK-TP2-RUN.md) for the accepted runtime.

## Goal and preservation contract

Run 2–4 active XS/S text conversations with continuous admission/retirement,
bounded prompt/decode interleaving and actual DSpark proposals and acceptance
under remote TP2. Rank 0 owns scheduling, draft generation, sampling and
canonical verdicts. Both ranks execute matching target rows and state commits.

Keep the default one-sequence controller, model operations, rollback and API
behavior. Preserve the accepted queued RDMA path, GPU router comparison,
256-row prefill boundary, rank-0 drafter ownership and alternating 256-block
expert partitions. Maya-S's 2304-wide experts retain their 1280/1024 and
1024/1280 layer-dependent bands. Preserve existing shared-expert counting,
head/output-group slices, sampler semantics and first-failure poisoning.

The accepted user engine is the versioned queued build recorded in Handoff.md
§44. Use an isolated `69bb0ee` parent build for causal comparisons and record
its relationship to that accepted executable. The published
[concurrency baseline](benchmarks/deepseek-maya-s-concurrency-2026-11-04/README.md)
records the user's earlier running deployment. New performance decisions use
matched contemporaneous controls.

## Source review

| Component | Findings and entry points |
| --- | --- |
| Supported TP2 configuration | `Model::Impl::load` requires `o.seqs == 1`. `tp2_rank.py` supplies no multi-sequence option. Feature negotiation covers the accepted single-conversation modes. [Model](../src/ds4/ds4_model.cu), [rank launcher](../tools/ds4/tp2_rank.py), [identity](../src/ds4/ds4_tp_identity.hpp) |
| Single-host scheduler | The event-driven `Act`/`Prompt` loop already handles tagged requests, idle slots, piecewise prompt reading, running decode and new arrivals. [Serving loop](../src/ds4/ds4_main.cpp) |
| Sequence state | `alloc_seqs`, `SeqState`, `seq_park/unpark/point/bind` hold per-conversation window/compressor/KV and token/checkpoint state. `MRun` tracks ragged verification windows. [Model state](../src/ds4/ds4_model.cu), [public API](../src/ds4/ds4_model.hpp) |
| Remote guards | `step_runs`, `draft_multi` and `verify_multi` reach `tp_load_only` guards, and public `sample_multi` returns `-1` under TP2. Their single-host presence supplies reusable logic. A new checked controller path is required. [Model entry points](../src/ds4/ds4_model.cu) |
| Control/state | `Control` carries one operation ordinal and position. `WorkerState` and `ModelWorker` hold one pending verify/verdict pair. [Codec](../src/ds4/ds4_tp_control.hpp), [logical state](../src/ds4/ds4_tp_worker_state.hpp), [model adapter](../src/ds4/ds4_tp_model_worker.hpp), [controller](../src/ds4/ds4_tp_controller.hpp) |
| Batched target | `step_multi` builds one-row runs. `step_runs` uses `WinPlan.rseq/rpos`, Engram lookup, per-sequence caches and shared row execution. `plan_window` selects async execution for multi rows. [Target implementation](../src/ds4/ds4_model.cu) |
| Commit reuse | `verify_multi` combines rank-0 sampling and `rollback_run`. Remote workers need a checked deterministic commit entry point that applies the wired accepted count to every run. [Verification and rollback](../src/ds4/ds4_model.cu) |
| Drafter capacity | `kWin` is 16. `can_draft_multi` requires `n*d_block <= kWin` and a second output-scratch bound. For block-size 5, four requests need 20 rows. The existing scheduler clears `tried` for every drafting request when the combined pass fails to fit. [Drafter and bounds](../src/ds4/ds4_model.cu), [scheduler fallback](../src/ds4/ds4_main.cpp), [row limit](../src/ds4/ds4_kernels.cuh) |
| API and launchers | The shared server already routes `rid` and `READY multi=N`. The DeepSeek wrapper already records ownership and uses per-model JSON. Extend its matching feature/config checks. [Server](../serve/server.py), [pair launcher](../tools/ds4/tp2_launch.py), [config generator](../tools/ds4/tp2_config.py) |

These findings come from static source inspection. The review also consulted
GLM row/MTP scheduling and the local Gufo batch control at `78d8a4bf`, described
in the [GLM plan](GLM-TP2-BATCHING-PLAN.md). Share their canonical-plan and
operation-identity rules while retaining DeepSeek's model and state contracts.

## Proposed execution contract

### Opt-in capacities and state ownership

Continue using `--parallel 1` as the default execution mode. Add a shared
launcher JSON `parallel` field and CLI override for the initially validated
TP2 values 2–4. Pass the same count and context to both ranks and bind the
effective batch mode, row capacity and speculation features during preflight.
Advertise `READY multi=N` after both peers validate their allocations and mode.
Reuse the checked model-content manifest and hash the selected executable as
the current launcher does. Batch capacities belong to the execution handshake.
Preserve #138's stable-filesystem identity checks during launcher integration.

Budget every rank's target cache, per-sequence compressed KV/index/Engram state,
window/compressor snapshots, rank-0 drafter rings/taps, shared scratch, pinned
staging and transport buffers before allocating the expert pool. Derive the
requested context/slot fit explicitly. Reject infeasible budgets with effective
limits and headroom details. Four full 262144-token contexts require a D1
capacity result. Keep slot count and context adjustments explicit in the config.

Each slot owns a generation, committed position, token/Engram history,
checkpoint/slot identity and pending target-verification state. Rank 0 also
owns its sampler counter, pending token, reasoning/output limits, request bias,
penalties, proposal distribution, draft confidences and per-request statistics.
Preserve source `SeqState` and `MRun` representations where they fit this
contract. Keep shared expert weights and scratch shared.

### Versioned batch control

Extend the model-specific control/identity handshake with a distinct negotiated
batching capability and explicit slot/context/row limits. Specify its encoded
version and bounded payload before implementation. Preserve the accepted
single-sequence control contract and existing GUFT/GUCH tensor transport.

Use one global monotonic operation ordinal plus per-member slot/generation and
expected committed position. Batch plans carry ordered members, row lengths
and canonical flattened tokens. Verification verdicts carry every member's
retained-input count, accepted tokens and continuation/pending-token decision.
Separate logical request identity from the physical reusable slot.

A batch can contain one-row ordinary members beside longer verify windows.
Define commit semantics for both in the new batch contract and preserve the
accepted legacy operation validation rules.

Validate the whole message and every member before queueing model work. Check
duplicates, stale generations, context/output bounds, token ranges, row sums,
overflow, incomplete verdict sets and pending-window conflicts. Establish peer
plan acceptance before the controller enters a collective. A batch commit has
an explicit completion/acknowledgement contract before slot reuse or success
reporting. Keep tensor traffic on RDMA and control metadata on bootstrap sockets.

Extend `WorkerState`, `Controller` and `ModelWorker` around per-slot state. Keep
scoped `BeginOperation`, queued checks, `EndOperation` and device synchronization
at transaction boundaries. Any partially executed target/commit failure ends
the pair with the original cause retained.

### Target rows and bounded prefill

Provide checked controller wrappers for ordinary rows and ragged target
windows. The standalone implementation and the remote adapter can reuse a
local row executor after the owning path has validated the operation. Keep
the raw remote guards in force for unsupported calls.

Use `WinPlan.rseq/rpos` and the accepted per-layer rank geometry. Match every
flattened row's attention/Engram state and every queued mixer/FFN reduction.
Preserve router-ID/weight checks, padding handling, shared-expert accounting
and the negotiated partial-byte limit. Validate grouped and fallback kernels.

Mirror slot bind/reset and each bounded prompt piece with its slot identity,
position and relevant feed/logit semantics. Preserve the accepted 256-row
boundary and cancellation/yield deferral inside a collective operation. Request
cancellation is observed between completed operations, followed by a mirrored
slot transition.

### DSpark drafting, verification and commit

Draft on rank 0 before sending the target operation. Group eligible requests
according to `can_draft_multi`, including its output-scratch constraint. With
four block-size-5 requests, use fitting groups such as 2+2. A single-member
group can use the existing bound-conversation drafter. Rotate ordering for
fairness and record actual per-request offered/accepted drafts.

Allocate the target verification row budget separately. Every participating
request gets an anchor, and rank 0 distributes the remaining rows subject to
`verify_max`, context, output and proposal limits. Generate four drafter blocks
in fitting groups, then trim the target plan to at most 16 total rows. Draft
capacity and retained target drafts are separate decisions.

Factor the minimum logic needed to separate rank-0 token selection from a
deterministic per-run commit. Reuse `rollback_run` for rings/compressors and
retain the correct row offsets and accepted positions. Both ranks apply the
same kept inputs. Rank 0 seeds its drafter taps from those retained rows and
keeps the newly emitted pending token's consumed/unconsumed distinction.

Keep each pending verify window alive until all its verdicts complete.
Cancellation, EOS or a thinking/output cap still requires a valid verdict for
that member. Preserve greedy and supported sampled-target/proposal semantics,
including the current distinction between point proposals and stored proposal
distributions. Carry compatible per-request proposal metadata through grouped
drafting. A mode must pass its sampler tests before it is advertised.

### Scheduler and API

Reuse `Act`, `Prompt`, waiting requests and the event-driven reader/turn loop.
Admit and retire slots between drained batch transactions, interleave bounded
prompt pieces and decode, and poll new arrivals when one request is active in
batching mode. The default one-sequence configuration keeps its accepted loop.

Complete shared phase C1 before API acceptance. It supplies whole-line command
serialization, a final health check under the writer lock and immediate fatal
fanout to every registered route. Retain request-local cancellation, FIFO
capacity, thread-local usage/timing and paired restart/mutation guards.
[Shared API contract](GLM-TP2-BATCHING-PLAN.md#shared-api-safety-phase-c1).

## Implementation phases

D1–D5 run in order. Shared C1 can progress independently and must pass before
D5's API acceptance. Each phase has its own issue and acceptance evidence.

| Phase | Issue | Dependencies | Deliverable and gate |
| --- | --- | --- | --- |
| C1 | [#143](https://github.com/uv-genai/maya-tp/issues/143) | Review contract | Shared server write serialization, route lifetime and sticky fatal fanout, with mocked concurrent API regressions. |
| D1 | [#149](https://github.com/uv-genai/maya-tp/issues/149) | Review contract | Checked per-slot state/ownership and per-rank XS/S memory budgets. Gate on explicit 1–4-slot context capacities and expert-residency/headroom accounting. |
| D2 | [#150](https://github.com/uv-genai/maya-tp/issues/150) | D1 | Bounded batch codec, identity negotiation, per-slot worker state and operation/commit completion. Gate on malformed/replayed/stale-state host cases and a small owned RDMA protocol smoke. |
| D3 | [#151](https://github.com/uv-genai/maya-tp/issues/151) | D2 | Distributed ordinary rows, rank-0 sampling and sequence-aware 256-row prompt feeds. Gate on row/position isolation, rank-local geometry, router/collective traces and target numerical checks. |
| D4 | [#152](https://github.com/uv-genai/maya-tp/issues/152) | D3 | Fitting rank-0 drafter groups, canonical ragged verification, authoritative per-run verdicts and deterministic worker rollback. Gate on acceptance 0 through all drafts and real DSpark activity for every eligible conversation at four clients. |
| D5 | [#153](https://github.com/uv-genai/maya-tp/issues/153) | C1, D4 | Continuous admission/retirement, per-request cancellation/reuse, matching paired launch controls, benchmarks, docs and user acceptance. |

## Validation and rollout

Start with host codec/state tests next to the current control/controller/worker
fixtures. Reuse the existing sharded control, DSpark, sampled and rollback
harnesses for paired checks. Use the single-host multi-row test infrastructure
for independent sequence references.
[Existing TP2 tests](../tests/ds4_tp_worker_state_test.cpp),
[model adapter tests](../tests/ds4_tp_model_worker_test.cpp),
[sharded DSpark](../tests/ds4_tp_sharded_dspark_test.cpp),
[rollback](../tests/ds4_tp_sharded_rollback_test.cpp).

Run assert-based host fixtures with assertions enabled and verify that failure
cases execute in the selected build configuration.

Required focused cases cover the following.

- Preserve default XS/S one-sequence ordinary and DSpark output/state against
  matched controls, including accepted sampling and lookup settings.
- Batch 1/2/3/4 members at different positions and row lengths. Reject duplicate
  slots, stale generations and omitted/repeated verdicts before new work.
- Test accepted counts 0 through all proposed drafts, EOS and budget endings
  inside a window, penalty/bias isolation, logprobs and reasoning closure.
- Verify windows, compressors, Engram history, cached positions and drafter
  taps after partial rollback and subsequent requests. Pair token agreement
  accompanies independent per-sequence checks.
- Exercise staggered requests, a long prompt beside active decode, retirement
  followed by immediate slot reuse, a disconnected client and surviving streams.
- Inject peer loss during prefill, target verification and commit, then check
  immediate all-route failure, retained first cause and blocked later GEN.

The existing [multi-attention layout test](../tests/ds4_multi_attn_plan_test.cpp)
and [GPU parity test](../tests/ds4_multi_attn_test.cu) are specific reuse points.
D3 must identify whether its chosen attention/prefill paths overlap #68's
packed-KV WMMA failures. Retain the recorded exception and thresholds, and
validate the newly exercised dimensions. Use #110's independent-reference
work to assess arithmetic differences and #112 for new capacity evidence.

Run the ordered HIP mapping/server/tools checks and CPU gates, then matching
gfx1151 HIP builds. Begin live coverage with bounded XS/S fixtures and a small
API smoke. Reuse component evidence for unchanged paths and reserve broader
operational workloads for the user's trial. Future live work uses available
isolated resources and recorded process ownership.

Extend [the existing benchmark](../tools/bench_concurrency.py) with replayable
requests and staggered/replenished load. Compare matched serialized/batched
modes at 1/2/3/4 clients with DSpark off/on. Record aggregate throughput,
client TTFT/completion, server generation, per-request draft counters, memory,
residency and storage reads. The #139 short-prompt burst remains a reproducible
starting workload, followed by bounded larger prompts and sustained admission.

Use the companion plan's proposed performance gate of at least three matched
waves, aggregate gains larger than observed run spread at two/four clients and
a provisional 5% median throughput regression budget for batching mode with
one active client. Apply the strict functional/fidelity and run-variation
control to the default one-sequence branch. Capacity, quality and lifecycle
gates also apply. This proposal has received no new GPU measurement in this
source-review workstream.

Use separate phase PRs with exact-head and post-merge CI. Keep batching opt-in
and retain the accepted versioned runtime/configs as the paired rollback.
Update launch/runbook/settings documentation when the controls are implemented.
Close #140 after operational acceptance with DSpark genuinely active.

## Dependency decisions

The phase plan supplies the new implementation dependencies. #112 contributes
memory/reference evidence, #110 contributes numerical analysis and #68 tracks
the existing WMMA exception. Establish a concrete overlap in D1/D3 before
turning any of those broader investigations into an additional prerequisite.
The current single-host memory issue #117 concerns a different allocation
mode and keeps its own scope.

The GLM and DeepSeek engines retain separate scheduling/state ports. Reuse C1,
the RDMA transport, the published benchmark and appropriate source primitives
while keeping each family's geometry and speculative commit contracts explicit.
