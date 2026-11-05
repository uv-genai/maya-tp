# GLM batching G2 framing and ordinary rows

Validated on 2026-11-05 after G1 at `5f54297`, tracked in #145.
The model was Maya-S-v2-IQ2_XXS at context 32768, with rank 0 on strix-halo-2
and rank 1 on strix-halo-1 over `usb4_rdma5`. The feature branch is
`codex/tp2-continuous-batching`.

## Fault reproduction and correction

The extended `glm_tp2_sequences_test --rows` passed every G1 state check and
then reproduced the historical `mv_rdna_kernel` memory fault at its first
packed-row call. The original process aborted with
`HSA_STATUS_ERROR_MEMORY_FAULT`. This isolates the fault to row execution.

The corrections use per-layer head and FF geometry consistently through KDA
projection consumers, DSA attention/output and shared-expert row projections.
KDA beta and projection rows use their compact strides. A grouped case also
exposed a shared-expert quantized-input stride mismatch. That call now uses
the explicit column strides from the working window implementation.

The corrected fixture passes one through four packed rows at unequal retained
positions against independent ordinary forwards. The fallback path and the
grouped-expert/batched-KDA case both pass. Both peers exit 0. The grouped case
has 318 identical normalized row-collective trace entries per rank.

## Checked serving protocol

The opt-in TP2 multi-sequence path negotiates application protocol version 1,
slot/row limits, per-slot contexts and relevant execution flags before READY.
The shared tensor transport retains its GUFT/GUCH protocol.

Every ordinary batch carries a monotonic operation ordinal, physical slot,
slot generation, expected position and canonical input token. The worker
validates the entire plan and acknowledges it before either rank enters a
target collective. A second acknowledgement follows completed model/queued
work. Shutdown also has a paired acknowledgement.

The legacy two-words-per-row decoder now derives logical rows from payload
words. The host codec bounds its allocation, rejects truncation, extra words,
duplicate slots, invalid tokens/contexts and stale/skipped operations, and
round-trips 64-bit operation/generation values. It is covered by the CPU CI
gate and passes with ASan/UBSan and Release checks enabled.

A fatal batch failure reports pair-down before request errors and ends the
pair. Its failure path releases host routes without resetting uncertain device
state. Shared API admission also formats command metadata before registering
its route, preventing a failed conversion from leaving an orphan route.

## Serving and preservation gates

| Gate | Outcome |
| --- | --- |
| Four simultaneous API-engine requests, 32 tokens each | All four complete with isolated timing/output routes and overlapping streams. Both ranks exit 0. |
| Default one-sequence TP2/MTP | All 64 raw picks match the frozen G1 parent control. MTP accepts 37 of 47 proposals. |
| Requested slots 4 on rank 0, 3 on rank 1 | Startup rejects the mismatch before READY, with the capability error in the engine log. |
| Shared Python suites | 743 server tests, 20 skips. 760 tools tests, 3 skips. |
| Build and CPU gates | HIP mapping, both affected HIP targets, batch codec, GLM memory policy and DeepSeek row/stub checks pass. |

G2 covers ordinary batched target execution. Batched MTP windows/verdicts follow
in G3. The default one-sequence control exercises the accepted MTP protocol.
The short correctness requests here are separate from G5's matched rollout
performance measurements.
[Raw controls, memory values and hashes](tp2-batching-g2-2026-11-05.json).

## Evidence and ownership

The frozen original fixture is under
`/home/ugo/software/project-maya-batching-validation/g2-before/` on both hosts.
The final geometry fixture is under `g2-final/`, and the serving engine under
`g2-serving-final/`. The earlier serving smoke used its recorded prior binary
hash. Final serving and default-control evidence use the same final engine.

Raw logs and owned identities are under `/tmp/maya-batching-validation/` in
`g2-before-rows`, `g2-geometry-rows`, `g2-grouped-rows`, `g2-final-grouped`,
`g2-serving-final-four`, `g2-serving-final-one` and `g2-capability-reject`.
All owned model processes have exited. Python gate logs are
`/tmp/maya-batching-g2-{serve,tools}-tests.log`. The issue records the phase
commit and CI outcome.
