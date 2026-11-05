# DeepSeek continuous serving and integration — D5

2026-11-05, [#153](https://github.com/uv-genai/maya-tp/issues/153),
[PR #155](https://github.com/uv-genai/maya-tp/pull/155).

The opt-in 2–4-slot scheduler admits, cancels and retires between completed
mirrored operations. It limits prompt pieces even with one active client,
keeps request-local sampling/limits, drafts in fitting groups and commits the
complete post-EOS/output/cancel verdict before reuse. The default single-slot
loop/protocol stays separate. Shared launcher capacity/context/speculation
choices have an execution identity independent of cached model contents.
Both ranks use matching current engines and launchers; production files and
unrelated services remain untouched.

XS four-client mixed requests pass per-request DSpark disablement, sampled
targets with penalties, bias and forced thinking closure. Staggered lifecycle
checks pass with three slots/DSpark on and two slots/globally off: a 1210-token
prompt interleaves with an existing 192-token generation, cancellation stays
local, queued/short requests refill freed slots, and continuation reuses 215
tokens. All normal owned pairs exit 0/0. The first mixed/on checks use frozen
`d5-first` SHA256 `d0ea7a636eda43e36354e22d00d37c127ceff88b52ba3bbf5da684ce254c334f`;
the final error-guard candidate used for the off check and all HTTP benchmarks
is `d5-guard`, SHA256
`ecac4769d8bd3d1d7cf1045e35191b343a012b5b24ec4c359e8604db10faf9f0`.

XS/S matched HTTP serving through the host-1 pair wrapper passes all 88
measured response comparisons exactly. Two/four-client median aggregate
throughput improves beyond wave spread with DSpark off/on; every eligible
request proposes and accepts actual drafts. One-client points and medians
including matched warmups improve. Complete routed-expert residency, observed
GPU GTT allocation peaks and remaining per-process disk reads are reported.
[Tables, method, raw requests/responses/configs/telemetry](../benchmarks/deepseek-tp2-batching-2026-11-05/README.md).

S passes three staggered HTTP clients with two requests each, short output
caps and two natural EOS completions. Deliberate owned-worker loss on XS
fans the identical original cause to four HTTP streams in 0.924 seconds;
health reports degraded with peer down and retains that cause. The supervisor
then stops the entire owned pair. Later work requires paired restart.

The final default one-sequence DeepSeek DSpark control matches all 64 raw
parent picks and both requests' proposal/acceptance counters, exiting 0/0.
The final host-1 GLM wrapper check passes two simultaneous clients with real
MTP on both, healthy TP2 and owned supervisor exit 0. Earlier G5 evidence
retains strict GLM default/raw-pick, M/L and numerical/performance gates.

Local gates pass HIP mapping, 743 server tests (20 skips), 770 tools tests
(3 skips), 37 focused launcher/harness checks, CPU memory/row/stub checks and
all seven checked control suites with assertions active and warnings as errors.
HIP engine/fixture builds and D4 ASan/UBSan checks pass. Raw default/lifecycle
results are in the adjacent JSON. Final publication/PR/main CI is recorded
on the phase issue. Implementation source `109aefc` already passed
push CI 42016420751 and PR CI 42016445488. D1–D4 reports establish slot capacity,
checked control, exact ordinary rows and forced speculative rollback/drafter
continuations independently of these serving tests.

Validation is HIP gfx1151/APU RDMA only, with four allocated 262K slots but
short inference contexts. It does not establish filled-context performance,
vision, unsupported sampled-q proposals/lookup in batch mode, or completed
operational user trials. Feature issues #11/#140 retain the latter. The old
single-slot user runtime remains selected until an explicit new-build choice.

The final launcher review retains the three-field legacy preflight identity
for parallel=1; only batching requires the execution digest. Its focused
regression passes. This does not change the tested engine binary or timings.
