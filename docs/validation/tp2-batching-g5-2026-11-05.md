# GLM operational checkpoint — G5

Issues [#148](https://github.com/uv-genai/maya-tp/issues/148) and
[#83](https://github.com/uv-genai/maya-tp/issues/83), 2026-11-05.

The GLM wrapper supports explicit 1–4 slots, matching context/MTP/drafts and
preflight engine/model-index identities. It uses the existing DeepSeek process
ownership supervisor, with model preparation retained in the GLM module. Stop
by port or recorded supervisor PID; a bare stop selects only a sole owned pair.
The acceptance and MTP harnesses no longer clear process names or occupied ports.
An EOF-managed remote child and recorded identities bound benchmark cleanup.

The resident HIP expert-row optimization passes the proposed throughput gates:
[matched measurements and raw requests](../benchmarks/glm-tp2-batching-2026-11-05/README.md).
Selected defaults apply only to opt-in TP2 batching. Explicit settings remain
available, and the default one-sequence branch retains its accepted behavior.

A final audit moved HIP FFN reduction after the complete local row batch and
CPU contributions. Grouped/fallback selection can depend on local residency;
it must never change collective counts or order. A fixture intentionally uses
grouped experts on one rank and fallback on the other. Both pass independent
full-logit, ragged-window and forced partial-commit checks, with 630 identical
normalized row-collective trace entries and no trunk reduction for NextN.

| Gate | Outcome |
| --- | --- |
| Final binary default one-sequence MTP | All 64 parent picks match. |
| Final binary four-slot mixed requests | All 128 ordinary-control picks match; per-request sampling, penalties, bias, thinking closure and speculation preserved. |
| Maya-M and Maya-L | Two clients each, 16 tokens each, actual MTP and clean paired shutdown at 32K context. |
| Host-1 owned HTTP wrapper | Three staggered clients with two successive requests each complete successfully. |
| Owned worker loss during two HTTP streams | Both get the same original cause within 0.95 seconds; health reports degraded/peer down, then the owned pair exits. |
| Host tests | HIP mapping; 743 server tests (20 skips); 765 tools tests (3 skips); 41 focused ownership/launcher/harness checks; GLM memory/framing and DeepSeek row/stub CPU gates pass. |

Final GLM binary on both hosts: `g5-release/maya` under
`/home/ugo/software/project-maya-batching-validation`, SHA256
`c50b12b630766a26dda39f525d48e25e6d9e74d36ac184d75b2b1b0cdec0bbbc`.
[Raw controls, binary identities and owned process metadata](tp2-batching-g5-2026-11-05.json).
Exact-head CI and the phase commit are recorded on the issues. All owned
model processes are stopped. Main integration waits for DeepSeek D1–D5; user
production configs/executables and unrelated services remain untouched.
