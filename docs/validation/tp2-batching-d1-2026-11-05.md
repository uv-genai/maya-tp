# DeepSeek slot-state and capacity checkpoint — D1

Issue [#149](https://github.com/uv-genai/maya-tp/issues/149), 2026-11-05.
The model loader permits 1–4 TP2 state arenas for the manual fixture. The
production CLI still rejects multi-sequence TP2 until D2–D5 validate the wire,
row execution, DSpark transactions and scheduler. Default one-sequence control
behavior remains unchanged.

Per-slot rings, partial compressors and compressed KV/index sources allocate
before the complete expert pool. Diagnostics check shapes and cross-slot
pointer isolation and record weights, cache, scratch, complete experts, pinned
staging and post-load free/system memory. Infeasible multi-slot loads retain
the underlying cause and identify requested slots/context and expert budget.

| Model/rank | Routed experts, GB | Cache allocations, GB | Extra sequence caches, GB | Post-load free, GB |
| --- | ---: | ---: | ---: | ---: |
| XS lead, DSpark loaded | 58.32 | 0.987 | 0.743 | 60.38 |
| XS peer | 54.75 | 0.986 | 0.742 | 64.77 |
| S lead, DSpark loaded | 91.73 | 0.987 | 0.743 | 25.91 |
| S peer | 84.51 | 0.986 | 0.742 | 33.96 |

Both allocate four complete 262144-token contexts and all target experts.
The measured maximum allocation and monotonic per-slot layout establish the
1–4-slot capacity bound at this context; separate 2/3-slot peak measurements
are not claimed. Extra sequence caches include the multi-attention allocation
and are already part of the cache column. Free memory is the runtime's
post-load planning value, not a peak-memory sample. Full counters are in the
[raw report](tp2-batching-d1-2026-11-05.json).

The paired fixture feeds 509 tokens and an ordinary token to each slot, checks
identical full logits, then crosses the 512 boundary with a three-row verify.
It retains only the anchor and compares the subsequent full logits and rank-0
DSpark drafts with an independent continuation. Reverse slot switches preserve
all other positions. All seven logit comparisons per rank are exactly equal
on XS and S, and both pairs exit 0/0. This covers ring/compressor rollback at
the boundary; concurrent row execution follows in D3/D4.

The first fixture re-fed slot 0 with a different final prefill/token boundary;
its logits differed while the other slots matched. The corrected fixture uses
the same boundary on both sides of the comparison. No engine arithmetic was
changed to suppress that difference.

Commands use the explicit `ds4_tp_verify_test --sharded-sequences` mode with
rank, lead address, bootstrap port, RDMA device and pack. Set
`DS4_TP_TEST_CONTEXT=262144` and `DS4_TP_TEST_SEQS=4`; checked model, partition
and executable digests are passed through the existing test identity bridge.
Sources: [fixture](../../tests/ds4_tp_sequences_test.cpp).
Frozen tested fixture: `d1-aligned/ds4_tp_verify_test` under
`/home/ugo/software/project-maya-batching-validation` on both hosts, SHA256
`7d811ef051a745a87bdae57442993c547c3b61f42183e9c57f062f634329a9e8`.
The final source additionally improves the allocation-error diagnostic.

HIP engine/fixture builds, HIP mapping and existing CPU row/stub checks pass.
Exact-commit CI is recorded on the issue. Raw logs and owned identities are
under `/tmp/maya-batching-validation/d1-*`; all owned model processes are stopped.
