# DeepSeek TP2 batch control

D2 checkpoint, 2026-11-05, [#150](https://github.com/uv-genai/maya-tp/issues/150).
This protocol is opt-in for 2–4 slots. The production CLI remains gated until
model rows, DSpark transactions and serving are validated. Default one-slot
identity and operations keep their existing meanings.

The control envelope stays version 1. Batch peers negotiate identity protocol
2, codec version 1, identical slot count and a 16-row limit. Legacy feature
bits retain their meanings. All tensors travel over APU RDMA; bootstrap
messages carry bounded metadata and acknowledgements only.

Commands 16/17 bind/admit a slot using its index and 64-bit generation, with
the expected position in the envelope. Admission advances exactly one
generation, including when prompt state is reused. Commands 18/19 carry an
ordinary/verify plan: codec version, member count, total rows, then each
member's slot, generation, position, token count and canonical input tokens.
64-bit values use low/high words. Ordinary members have one input; verify
members may be ragged. Limits are four distinct members, sixteen total rows
and 43 words. The existing Feed limit remains independently bounded.

Command 20 commits the whole verify plan: version, member count, target
operation ordinal, then each member's identity, retained count, continuation
flag and emitted tokens. Its maximum is 48 words. The retained count equals
the emitted count. Earlier emitted tokens must equal the retained canonical
inputs after the anchor; the final emitted token is pending and unconsumed.
The complete ordered verdict set must match the pending target operation.
Binding, resetting, admission and unrelated work are forbidden while a verdict
is pending. Slot reuse waits for completion of that entire verdict.

Both peers validate bounds, vocabulary, context, unique slots, generations,
positions, operation order and complete payloads before model execution.
Rank zero waits for an Accepted acknowledgement before entering its matching
operation, and a Completed acknowledgement after queued checks and operation
completion before publishing state. Acknowledgements carry the operation,
stage and at most 1024 bytes of error text. Any partial operation, malformed
frame or transport failure is sticky; no retry, reset or single-rank restart.
The selected-slot legacy Verify/Verdict remains available inside batch mode
for the accepted single-active DSpark path.

The CPU fixtures test codec/state/controller/worker rejection and ordering.
The pack-free RDMA fixture tests ragged commit, reuse and integer-exact queued
reductions through the production StagePartial/WaitValue/AddRowsBroadcast
path. The dormant WaitAdd helper has a separate defect tracked in
[#154](https://github.com/uv-genai/maya-tp/issues/154); engines do not call it.
