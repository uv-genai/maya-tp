# DeepSeek two-host parallelism and DSpark: analysis and design

Status: historical design proposal, written before the experimental implementation on 2026-11-03.
Tracking: [#108](https://github.com/uv-genai/maya-tp/issues/108). The code descriptions and
unverified assumptions below describe that earlier checkpoint; see the [review guide](DEEPSEEK-TP2-RUN.md),
[implementation plan](DEEPSEEK-TP2-IMPLEMENTATION-PLAN.md) and [handoff](../Handoff.md) for current status.
Scope: DeepSeek V4.1 Flash on two gfx1151 Strix Halo hosts connected by USB4 RDMA, with the
existing DSpark drafter. This is not an implementation, a measured performance claim, or a phased
implementation plan. GLM TP2/MTP and existing single-host DeepSeek remain supported unchanged.

## Motivation and evidence

The local Maya-S run in [Handoff §36](../Handoff.md) used DSpark (262 of 332 drafts accepted on
one request) but generated about 4.05 tokens/s. Of 15,744 routed experts, startup put 9,779 in
GPU slots and 5,965 (~65.8 GB) on the CPU lane. Another request reported ~370 MB of **total
process storage reads** per token with less than 1 GB system MemAvailable. These are observations,
not proof that every byte was an expert read. With unified CPU/GPU memory, the GPU allocation and
file cache compete for the same physical RAM. The smaller Maya-XS measurements and their caveats
are in [the benchmark report](benchmarks/deepseek-maya-xs-2026-11-02/README.md).

A second host potentially makes the expert working set resident and adds compute bandwidth.
It also introduces per-layer latency, staging and synchronization costs. It is not established that
all of Maya-S, its Engram data, context, drafter, and runtime scratch fit simultaneously across the
pair, or that storage reads will disappear. Capacity and throughput must be measured with the
actual pack and context; do not derive speed from aggregate advertised memory or link bandwidth.

## Code and boundary at the design checkpoint

* `src/ds4/ds4_model.cu` and `ds4_model.hpp` implement the DeepSeek engine independently of GLM.
  A block includes Engram on selected layers, hyper-connection mixing, attention with a sliding
  ring and compressed/indexed KV, then routed plus shared experts. The checkpoint's geometry,
  layer ratios, quantization and available drafter are pack-derived.
* `Options::tp` / `--tp` is **same-process, same-host GPU grouping**, not this fork's two-host
  protocol. `Group`, `ArGroup` and the window/chunk paths use local device pointers, streams,
  host-mapped flags and local collectives (`ds4_model.cu`, around `Group`, `chunk_allreduce`,
  and the attention/MoE reductions around lines 19268 and 19456). The helper GPU and its fallback
  also assume local GPU/host memory; it is not a network peer. Do not treat a remote GPU as an
  additional local device or overload `--tp 2` to silently enable RDMA.
* For GGUF, group formation checks the quant block, attention heads and output groups. The code
  falls back to a smaller local group if the split is invalid. `DS4_TP_GGUF_UNEVEN` is an opt-in
  local whole-block split; its group support and limitations do not establish remote correctness.
* `ds4_main.cpp::round` chooses a DSpark or prompt-lookup draft window and invokes `Model::draft`
  and `Model::verify`. Verification runs the target trunk, accepts a prefix and produces the next
  token; it rolls back rejected state and seeds the drafter with committed trunk data. With sampled
  DSpark enabled the acceptance uses the draft distribution q and target p; argmax/lookup drafts
  use point distributions. Preserve penalties, random seed/counter and the existing sampling rule.
* `Model` also exposes `feed`, `feed_mark`, `step`, checkpoint/restore, partial feed and multi-sequence
  interfaces. Position, tokens, window ring, compressed KV/index keys, compressor snapshots,
  Engram history, taps, drafter state, penalties and RNG state have different lifetimes. Merely
  mirroring tokens or copying the GLM MTP wire is insufficient.
* GLM's `TpContext` and `strata::net::Communicator` supply ordered scoped collectives, RDMA verbs,
  bounded receive windows and failure detection. GLM control mirroring lives in
  `src/program/generate.cpp`, not in `maya-ds4`. `CMakeLists.txt` links the existing TP2 components
  into `maya`; `maya-ds4` needs explicit integration. Reuse and generalize transport where justified;
  do not depend on GLM model internals or regress its protocol, startup, MTP or health handling.

## Candidate distributions

| Design | Benefit | Costs and unresolved constraints |
| --- | --- | --- |
| Tensor-slice each layer across ranks | Uses the existing local-group mathematical pattern: attention and MoE partials are summed. Both hosts participate on every token; roughly halves eligible weight storage. | Requires distributed group abstraction, block-aware quant slices, reductions and deterministic control. Replicated state/dense/drafter and long-context caches reduce the capacity gain; output may differ numerically from single-host. |
| Own whole routed experts on one rank | Avoids splitting quantized experts; each rank stores and computes its owned routed experts, then their weighted contributions are summed. | Router decisions, load imbalance, shared expert exactly once, prefill dispatch, and expert placement must be defined. Attention remains replicated or separately sharded. This is expert parallelism, not automatically tensor parallelism. |
| Split layers between hosts | Fewer network exchanges than per-layer tensor reductions and simpler full-expert storage. | Pipeline handoffs at each boundary, per-stage KV/Engram/tap ownership and sequential decode bubbles; DSpark needs target-layer taps and verified state across stages. Evaluate separately rather than assuming it is equivalent to TP2. |

**Proposed initial architecture for design purposes:** two distributed tensor ranks with a replicated
logical residual and agreed control flow; use whole-block, possibly uneven, expert FFN slices and
split compatible attention dimensions. Keep whole-expert ownership as a serious alternative if
uneven slicing or memory/latency measurements make it preferable. Both choices require separate
correctness and capacity proofs. No configuration should silently fall back to one host.

For the currently observed Maya-S geometry `n_ff=2304`, down-weight blocks of 256 give nine blocks:
`1152` per rank is **not** a 256-aligned split. A 5+4 partition is 1280/1024, potentially rotated
by layer as in the local uneven mechanism. Derive ranges from each actual tensor type and geometry;
validate gate/up row offsets, down-column blocks, heads, grouped output, shared experts and drafter
weights. Reject unsupported layouts before loading. Never assume other DeepSeek packs use the same
width or share XS and S's quantization. A distribution that leaves any expert replicated or in a
CPU/disk tier must report it explicitly.

## Computation and network contract

Rank 0 owns the API, request scheduling, sampling and authoritative user-visible output. Rank 1 is
a worker. Both load a verified matching pack identity, engine/protocol version, model geometry,
partition mapping, context and feature set. Bootstrap sockets are control only; exchange activation
partials through USB4 verbs RDMA, using the existing transport's operation scopes, ordered ordinals,
stream synchronization and bounded buffers. No TCP data path.

For each compatible target layer, each rank consumes the same logical token rows and residual,
computes its own attention projection contribution, exchanges/reduces at the attention output
before the residual/hyper-connection update, then computes its routed/shared FFN contribution and
reduces before the next residual update. Confirm the exact DeepSeek buffer layouts and reduction
points separately for decode window and chunked prefill: same-host `allreduce` call sites are a
starting map, not proof that every path can use the same buffer or schedule. Replicated operations
must not be summed twice. Router IDs and weights should be decided once and distributed, or shown
bit-identical before any data-dependent collective; rank-local routing disagreements must fail
closed, never change the collective sequence.

For a conceptual FP32 output of H=5120, one row is 20,480 bytes per rank per reduction. Two
reductions per layer over 40 layers move about 1.64 MB **per rank** in each direction per token,
excluding headers, other communications and batching. The 37 Gb/s / ~6 µs link targets imply
meaningful synchronization overhead even when payloads are small; these figures are estimates,
not measured end-to-end latency. Prefill/verify rows multiply message size; chunk at the negotiated
`MaxPartialBytes` and agree on row counts and fast-path availability on both ranks before queuing.
Benchmark actual transport including host staging, GPU synchronization and both directions.

## DSpark and state consistency

A conservative proposal keeps the full drafter and sampling authority on rank 0, subject to a
measured memory budget. Both ranks execute the same target verification window. At every accepted
trunk position, rank 0 must receive or already hold the full target-layer taps DSpark requires;
this depends on where the existing tap is captured relative to the reductions. Do not seed from a
rank-local partial. If drafter memory prevents the working set from fitting, evaluate a sharded
drafter or different placement as a separate design, including its own reductions and state.

One logical speculative round should have this ordering:

1. Rank 0 drafts with its current committed state and chooses the window (including lookup policy).
   It sends the base position, input token, ordered draft IDs, row count and operation identity to
   rank 1 before either enters the corresponding target collectives. No draft case is a synchronized
   single-token step; the decision cannot be made independently on the ranks.
2. Both execute the same trunk window and reductions. Rank 0 alone computes the authoritative
   acceptance, next token, log probabilities and RNG advancement, retaining q for sampled DSpark.
   The worker must not independently sample or consume a conflicting RNG stream.
3. Before the next model operation, send the accepted-prefix length, committed position and next
   token to rank 1. Both roll back the same rejected suffix and make the same committed state
   visible. Rank 0 seeds DSpark from the committed target taps only after rollback. Define where
   verdict transfer fits the scoped operation so neither rank waits for a collective the other
   will never enter.

`verify` currently combines target execution, sampling, rollback and drafter seeding; separating
or parameterizing those responsibilities is a design task, not a claim that a ready worker API
exists. `rollback` restores ring buffers and compressor snapshots, updates `pos`, `toks` and
`ctoks`, and relies on later writes for other position-indexed data; audit all state, including
Engram, compressed/indexer state, checkpoints, prompt reuse, vision and DSpark taps. A failure
or cancellation during a window must never leave one rank committed and the other speculative.

Initial supported feature boundary should be explicit and fail closed: one request at a time,
text, matching pack/config, fixed/negotiated prefill chunks, normal decode and DSpark on/off.
Do not silently accept `--parallel`, independent slot reuse, vision, unmirrored checkpoint or
partial-feed operations until their distributed state and protocol are implemented and validated.
Prompt lookup and sampled speculation need explicit coverage; unsupported modes must return a clear
error rather than downgrade correctness. The existing single-host paths remain unchanged.

## Memory, placement and deployment

At startup, compute a per-rank budget using actual physical available memory and explicit headroom;
count GPU allocations and CPU pinned/file-cache consumption together on the APU. Include sliced and
replicated dense weights, routed and shared experts, 512K-context KV and compressed/index state,
Engram tables/working set, drafter and taps, prefill staging, RDMA registered buffers, runtime and
OS headroom. Expose expected and actual resident expert counts, bytes per rank, CPU/disk misses,
Engram reads and communication traffic. Capacity is a per-host and per-tier constraint, not just
a combined total. Fail with a useful explanation if a requested resident mode cannot be met.

Use identical model selections, pack integrity checks and a negotiated partition digest on both
hosts. Retain local model paths and user-owned config conventions; two-host launchers need explicit
DeepSeek family selection and rank-specific settings rather than hardcoded GLM packs. Neither setup
nor launch should overwrite another quant's config. Rollout must use isolated review builds and
explicit configs; never replace the user-accepted GLM binary or change running user processes as a
test setup. RDMA bootstrap/listener roles and device discovery need an explicit DeepSeek interface.

## Failure and recovery

A transport, device, collective or protocol error is fatal for the entire pair. Preserve the first
cause, drain/check queued work at operation boundaries, report health down immediately and reject
later generation before engine writes, matching GLM's sticky `_request_error` guard (#88). Do not
clear poisoned verbs state, retry a collective, silently switch to CPU, or restart a single rank.
Provide coordinated shutdown and a fresh two-rank startup; do not probe bootstrap with `/dev/tcp`.
A cancel must finish or abort a matched distributed operation before synchronized rollback/reset;
peer loss in prefill, verify or rollback must not deadlock the survivor. Detect mismatched pack,
partition, operation ordinal, row count and unsupported features at the earliest safe boundary.

## Evidence needed before claiming support

* Host tests for pack geometry, block-aligned 5+4/4+5 coverage, bounds, quant offsets, ownership,
  capacity, handshake and control serialization; reject unsupported/mismatched configurations.
* Ordered HIP mapping, server and tools CI; tests for GLM TP2 peer-loss and all existing GLM MTP
  regression cases. Validate transport error injection, queued failures and first-cause health.
* gfx1151 per-operation parity against single-host DeepSeek on a fitting pack: layer partials and
  reductions, router decisions, prefill chunks versus decode, ring/compressor/KV and checkpoint
  state; inspect numerical differences and model output, not only absence of crashes.
* Two-host DSpark on/off and lookup cases, forced reject at each window position and full acceptance,
  seeds and counters, penalties, sampled p/q verification, logprobs, EOS, output limits and
  cancellation. Compare greedy tokens where mathematically appropriate; for non-greedy verify the
  sampling distribution and reproducibility contract, not merely identical random tokens.
* API tests for long prompts, chunk boundaries, context limits, restart, mismatched configs,
  disconnects and worker loss at every operation boundary. Explicitly exercise unsupported features.
* Same-prompt/same-settings baseline on one and two hosts for XS and S, cold and warm loads,
  prefill and generation, DSpark on/off, acceptance, storage bytes/token and both ranks' memory,
  CPU/GPU utilization and network latency/traffic. Repeat and report distributions; separately
  compare total startup time, quality/output differences and resident capacity. No throughput target
  or elimination of disk I/O is guaranteed until measured.

## Decisions still requiring evidence

Whether uneven TP or whole-expert ownership wins on this link; the workable DSpark/tap placement;
actual Engram and context residency; whether replicated router decisions are bit-stable; the
prefill scheduling and wire boundaries; how to preserve sampled q efficiently; and which existing
local-group fast paths can be safely reused across processes. Resolve these through implementation
planning and controlled experiments, not by treating this proposal as already validated.
