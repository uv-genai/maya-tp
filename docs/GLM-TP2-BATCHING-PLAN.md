# GLM TP2 continuous batching and MTP implementation plan

Reviewed on 2026-11-05 against `main` at `69bb0eeb1ea3f893dd3103b46bc53d378dff2b49`.
Feature tracking is #11 and the completed source-review workstream is #141.
This document defines proposed implementation work. The accepted operating
instructions are in the [TP2 runbook](TP2-RUN.md).

Implementation checkpoint: C1 and G1–G4 passed their phase gates. G5's local
operational and performance gates pass on `codex/tp2-continuous-batching`;
publication/CI is tracked in #148. See the [G5 report](validation/tp2-batching-g5-2026-11-05.md).
The original plan below remains a dated design contract. Combined integration
and user trial status are recorded in Handoff.md §52 and #11.

## Goal and preservation contract

Support 2–4 active text conversations, continuous admission and retirement,
bounded prompt/decode interleaving and actual MTP under remote TP2 on the two
gfx1151 HIP/ROCm hosts. Rank 0 owns scheduling and sampling. Each rank executes
the same ordered target rows and collectives over APU RDMA.

Keep the existing default one-sequence execution branch and its MTP input,
accept, continuation and abort protocol. Preserve compact gate/up/down expert
layouts, shared/dense reductions, rank-local KDA heads, replicated full-width
NextN, prefix reuse, output budgets and sticky first-failure reporting. Keep
GLM vision gated under #16 and retain the current source pin and HIP target.

Use an isolated build of the reviewed parent as the causal control for each
feature comparison. The accepted deployed GLM build is separately recorded in
Handoff.md §33. Record both identities and use explicit configs for validation.
Promote through the existing user acceptance workflow after the gates pass.

## Source review

| Component | Findings and entry points |
| --- | --- |
| Sequence allocation | `seq_layout`, `seq_alloc_arenas` and `seq_bind` already separate recurrent/attention state, snapshots and streaming KV. Extra arenas are allocated before expert placement. The historical `SEQS=2` first-prompt fault requires a controlled reproduction. [Sequence state](../src/core/glm_multi.cu), [load order](../src/core/glm_model.cu) |
| Batched target rows | `rows_layers`, `rows_moe_grouped` and `multi_rows` contain mixer/FFN reductions and sliced dense geometry. Both grouped and fallback execution need auditing with compact TP weights. [Row execution](../src/core/glm_rows.cu), [geometry helpers](../include/strata/core/glm_model.hpp) |
| Scheduler | `Gen`, `Ctx`, pending admission, bound-context reuse and batched row dispatch already exist. `multi` is explicitly gated by `!tp2`. [Serving scheduler](../src/program/generate.cpp) |
| Batch framing defect | `tp_send_vec` sets `h.n` to payload words. A `MultiRows` payload has `2*N` words, while the worker uses `h.n` as the number of rows and checks for `2*h.n` words. Every positive batch fails that test. This is a static finding in the gated path. [Sender and worker](../src/program/generate.cpp) |
| Batched MTP | `mtpw_front/back/picks/commit` and per-conversation buffers exist. The scheduler currently calls them without a matching remote window/verdict protocol. KDA replay allocates using `head_compute()` but contains full `g.n_head` uses and strides. [MTP rows](../src/core/glm_mtp_rows.cu) |
| NextN geometry | `head_compute_for(il)` and `ff_compute_for(il)` distinguish the replicated MTP layer. Grouped row FFN code still has unconditional `tp_.active()` reduction sites. Audit those call paths before using grouped NextN rows under TP2. [Rows](../src/core/glm_rows.cu), [MTP drafting](../src/core/glm_mtp_rows.cu) |
| Existing MTP contract | Both ranks run the replicated drafter, while rank 0 supplies canonical anchors/draft IDs and accepted tokens. Near-tie local proposals never determine peer target inputs. Preserve this authority model. [Accepted MTP loop](../src/core/glm_mtp.cu) |
| Python API | `FairLock`, `READY multi=N`, tagged routes and thread-local timing already support multiple conversations. Complete command writes and pair-wide failure wakeup need explicit concurrency guarantees. [Server](../serve/server.py), [multi tests](../serve/test_multi.py) |
| Launchers | Direct rank scripts already take `SEQS`. The llama-swap wrapper forces `SEQS=1` and uses broad cleanup tracked in #83. [Rank 0](../tools/tp2/rank0.sh), [rank 1](../tools/tp2/rank1.sh), [wrapper](../tools/llama-swap/tp2.sh) |

The GPU fault's cause is still unresolved. Arena pressure, rank-local offsets
and prompt scratch are investigation targets. The framing and replay findings
provide additional work beyond reproducing that fault.

The local Gufo checkout at `78d8a4bf` was reviewed in
`src/cli/serve/tp_control.hpp` and `tp_executor.cpp`. Its `TpBatchMember`,
`kAdvanceBatch` and `kDecodeBatch` carry state identity, budgets, draw state and
rank-0-selected draft widths. Reuse those control invariants and execution
digest ideas while retaining Maya's existing model representations and RDMA
transport ABI. The shared transport contract is in
[tp_wire.hpp](../include/strata/net/tp_wire.hpp).

## Proposed execution contract

### Configuration and memory

Default `--parallel 1` selects the accepted execution path. Explicit TP2
`--parallel 2`, `3` or `4` selects the new negotiated batching mode after its
phase gates pass. Forward the selection through both rank launchers and the
wrapper with the existing model, pack, config, context, MTP and draft controls.

Negotiate the application batch protocol, slot count, per-slot context limits,
target-row cap, MTP mode and maximum collective bytes before model work. Include
the existing `MAYA_GLM_SEQ_CTX` auxiliary-context choice where used. Report
effective capacities. Startup rejects infeasible requests with the calculated
budget and available headroom. A 512K primary context and four full-size arenas
require a measured fit decision.

Account for persistent state, snapshots, MTP backup/replay, temporary windows,
pinned staging, RDMA and expert-pool tail overlap on each rank. Preserve the
accepted one-sequence allocation policy. Initial batching admission requires
its documented expert residency and unified-memory reserve to fit.

### Batch identity and transaction boundaries

Use a monotonically increasing batch operation ID. Each member names a slot,
slot generation, expected position, anchor/input tokens, row count and remaining
output/context budget. Rank 0 sends the complete canonical plan before either
rank enters a target collective. Validate duplicate slots, stale generations,
counts, arithmetic overflow and negotiated limits on the host.

For ordinary row messages, define payload words and logical rows explicitly.
Correct the existing two-words-per-row contract and test sender-to-worker
round trips. Keep the established one-sequence op-10/op-12/op-13/op-14 meanings.
New batch payloads require an agreed model-control version/capability. Reuse
the existing GUFT/GUCH tensor transport.

Complete and check queued work before binding another context, reusing scratch
or changing membership. A target or transport failure poisons the entire pair.
Preserve the first cause and require both ranks to recover together.

### MTP and continuous scheduling

Flatten each conversation's anchor and drafts into a canonical target-row
layout. Rank 0 determines forced tokens, accepted prefixes, emitted tokens,
continuation flags and counters. Send a verdict for every participating slot,
including cancelled/finished slots, before retiring or reusing any member.
Both ranks replay the same retained KDA rows and update their replicated NextN
caches with canonical accepted inputs.
Carry forced tokens inside the batch input/verdict frames. Keep the legacy
global force callback scoped to its existing request protocol.

Use rank-local trunk head counts for backup/replay and full-width NextN
geometry for drafting. Trunk reductions follow the accepted slice contract.
Replicated NextN drafting executes with its own reduction policy. Start from
the working NextN primitives, then enable grouped draft rows after their
geometry and collective audit passes.

Reuse the current scheduler's `Gen`, `Ctx`, pending queue and prefix logic.
Poll admission and STOP/PAUSE between bounded rounds. A batching-mode request
running alone still yields at round boundaries for new arrivals. Preserve the
original complete-request loop for the default one-sequence configuration.
Bound prompt pieces by the agreed receive/scratch capacity and interleave them
with active decode. Respect per-request reasoning, EOS, bias, penalties and
output limits through admission and active-count changes.

## Shared API safety phase C1

This phase is shared with the [DeepSeek plan](DEEPSEEK-TP2-BATCHING-PLAN.md).

- Serialize complete GEN, STOP, PAUSE and lifecycle writes using a per-engine
  writer lock. Recheck pair health under that lock immediately before writing.
  Cover prompts larger than a pipe's atomic-write size.
- Document writer/route lock ordering. Snapshot routes under the route lock and
  wake their queues outside it. Limit the writer lock to health checks and
  complete command writes, releasing it before waiting for engine replies.
- Reuse the existing request router, FIFO places and request-local metadata.
  Atomically register/remove routes around admission and failure.
- On the first fatal tagged error, `TP2_DOWN`, GPU-state `END` or unexpected peer EOF,
  wake every active route with the retained cause and release its FIFO place.
  The current `_tp_pair_down` records health but supplies no route broadcast.
  Reuse the existing `_pump` EOF broadcast and extend cause-aware wakeup to
  fatal notices received while the engine pipe is still open.
- Keep request-local cancellation scoped to its own route. Preserve exclusive
  lifecycle/mutation guards and capacity refresh after a successful restart.
  Invalid requests rejected before model work use request-local errors. A
  malformed remote plan or partially executed batch uses pair-fatal recovery.

The existing `generate` pre-write guard and `_request_error` first-cause logic
are required regression anchors. Extend
[multi tests](../serve/test_multi.py) and
[peer-loss tests](../serve/test_tp2_peer_loss.py).

## Implementation phases

Each row has its own issue. G1–G5 run in order. C1 can progress independently
and must pass before G4's API acceptance.

| Phase | Issue | Dependencies | Deliverable and gate |
| --- | --- | --- | --- |
| C1 | [#143](https://github.com/uv-genai/maya-tp/issues/143) | Review contract | Whole-line API writes, route lifecycle and paired fatal fanout. Fake-engine tests cover concurrent large prompts, request-local STOP, all-route failure and blocked later GEN. |
| G1 | [#144](https://github.com/uv-genai/maya-tp/issues/144) | Review contract | Reproduce the first-prompt GPU fault and prove its fix. Check arena/layout/offset and peak-memory accounting at 1–4 slots. Gate on two-rank binding/prefill checks and the one-sequence MTP control. |
| G2 | [#145](https://github.com/uv-genai/maya-tp/issues/145) | G1 | Repair and bound batch framing, mirror ordinary rows and audit grouped/fallback reductions. Gate on malformed-frame host tests and paired 1–4-row output/state/collective checks. |
| G3 | [#146](https://github.com/uv-genai/maya-tp/issues/146) | G2 | Canonical MTP windows/verdicts, rank-local KDA replay and replicated NextN cache/draft handling. Gate on ragged windows, acceptance 0 through all drafts, forced tokens and genuine proposals at 2–4 active conversations. |
| G4 | [#147](https://github.com/uv-genai/maya-tp/issues/147) | C1, G3 | Continuous admission, drained retirement, bounded prefill and reuse/cancel integration. Gate on staggered arrivals, active-count transitions, slot generations and peer loss with MTP off/on. |
| G5 | [#148](https://github.com/uv-genai/maya-tp/issues/148) | G4 and ownership-safe operations | Add opt-in paired launcher controls, benchmark and validate the user route, update runbook/settings and complete user acceptance. Use #83 for the existing GLM wrapper/harness cleanup prerequisite. |

G1–G4 can use isolated launches with recorded ownership. G5's existing-wrapper
route depends on the ownership fix tracked in
[#83](https://github.com/uv-genai/maya-tp/issues/83).

## Validation and rollout

1. Run the ordered HIP mapping, server and tools suites and CPU policy tests
   required by AGENTS.md. Add host framing/state tests at the phase that changes
   their contract. Reuse the existing row, MTP acceptance and memory fixtures.
2. Protect the default one-sequence path with a fixed-prompt, fixed-draft
   same-topology control. Require raw target-pick agreement and actual MTP
   activity. Record output, proposal, accepted-token and committed-state traces.
3. Validate unequal positions, ragged windows, mixed speculation, forced
   thinking-end tokens, penalties/bias and EOS. Check layer outputs and state
   when batched arithmetic differs. The documented single-host/TP near-tie
   distinction stays part of the numerical assessment.
4. Show two requests active in the engine before either ends. Admit a new
   request while an earlier one decodes. Refill a retired slot, cancel a
   different request and verify the survivors' streams and positions.
5. Begin GPU coverage with Maya-S and short fixtures, then smoke Maya-M/L at
   proven capacities. Reuse unchanged-component evidence and keep the first
   failures. Use owned isolated PIDs/groups and stop server before engine.
6. Extend [the concurrency harness](../tools/bench_concurrency.py) with replayable
   inputs and staggered/replenished traffic. Compare matched serialized and
   batching modes at 1/2/3/4 clients, MTP off/on, short and bounded larger prompts.
   Record aggregate throughput, TTFT, full latency, draft counters, cache,
   physical reads and peak per-rank memory.

The proposed rollout performance gate uses at least three matched waves per
selected level. Require aggregate improvement at two and four clients larger
than the observed run spread. For batching mode with one active client, use a
provisional 5% median throughput regression budget, with an exceeded budget
resolved before rollout. The default one-sequence branch requires its strict
functional/fidelity control and throughput within observed run variation.
The 5% threshold is a planning choice and has received no new measurement here.

Publish one-topic phase PRs with exact-head and post-merge CI. Keep batching
opt-in during validation. Preserve versioned accepted executables/configs for
paired rollback. Finish with the user's operational trial, then close #11.

## Dependencies and review limits

The first-prompt cause, final memory capacities and measured batching gain
remain decisions for G1/G5. The source review establishes static blockers and
specific reuse points. This planning checkpoint validates source references,
phase dependencies and the existing accepted evidence. Hardware fault
reproduction and new inference measurements belong to the implementation gates.

The companion DeepSeek work has a separate engine, state protocol and
speculative implementation. Share the Python API phase and validation tooling,
and maintain each model family's numerical and rollback contracts.
