# GLM batched MTP checkpoint — G3

Issue [#146](https://github.com/uv-genai/maya-tp/issues/146), 2026-11-05.
This extends G2's ordinary TP2 row protocol with version 2 target windows and
complete verdicts. Default one-sequence control operations remain unchanged.

Each target plan identifies its operation, slot generation, expected position
and canonical inputs. The peer accepts before collective work. A subsequent
complete verdict checks retained prefixes, emitted/pending tokens, continuation
and draft counts. Completion precedes retirement or reuse. KDA rollback uses
rank-local geometry; replicated NextN does not reduce its outputs across ranks.

The four-client Maya-S test accepts 65/108 drafts and matches all 128 raw picks
of G2's ordinary control. A mixed four-client test covers speculation disabled
per request, seeded sampling with penalties, forced thinking closure and bias.
All four token sequences match the same binary's ordinary-batch control.

The paired fixture independently derives full logits and forces retained
prefix lengths 1, 2 and 3 with ragged windows and unequal positions. Both ranks
pass cache reseeding, disabled drafting and subsequent full-logit state replay
at tolerance 1e-4. Grouped experts, row attention and batched KDA are enabled.
The trace contains no trunk reduction for replicated NextN layer 45.

The default one-sequence control matches all 64 parent picks. CPU framing and
commit tests pass in Release and under ASan/UBSan; existing MTP row tests pass.
HIP mapping passes. Exact-head CI is recorded on the issue. This checkpoint
establishes correctness, with lifecycle and rollout performance gates in G4/G5.

[Raw results and owned process identities](tp2-batching-g3-2026-11-05.json).
The frozen engine on both hosts is `g3-first/maya` under
`/home/ugo/software/project-maya-batching-validation`, SHA256
`b8bcd62b42d31fe3bab2f8df17487f969aee36c51c01b9b9515fb1ccfdd2e5a6`.
The forced fixture is in `g3-fixture/`. Logs and identities are under
`/tmp/maya-batching-validation/g3-*`; all owned model processes have exited.
