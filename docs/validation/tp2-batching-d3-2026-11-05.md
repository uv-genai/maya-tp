# DeepSeek checked model rows — D3

2026-11-05, [#151](https://github.com/uv-genai/maya-tp/issues/151).
A checked model adapter now binds/adopts slot generations and executes ordinary
rows through the negotiated controller. Raw unsupported TP2 calls remain
guarded; the production multi-slot CLI is still gated pending D4/D5.

The reused row layer uses rank-local heads, output groups and shared/routed
FFN widths. Every target layer retains the accepted GPU router comparison and
queued attention/FFN reductions. Replicated rank-0 drafter seeding performs no
trunk reduction. Per-slot selected feeds remain bounded to 256 inputs and are
acknowledged before the next bind or rows.

The explicit `ds4_tp_verify_test --batch-rows rank host port rdma pack` fixture
loads four 262144-token slots with DSpark loaded on rank zero. It prepares
unequal 505/507/509/511-token prefixes using matching 256-input feed boundaries,
then compares ordinary row batches of sizes 1, 2, 3 and 4 against independent
selected-slot continuations. All ten complete vocabulary-logit comparisons
are exact on XS and S, and greedy picks and committed positions match. Both
pairs exit 0/0. The fallback attention pass additionally checks peer router
traces: all 320 records match (valid expert IDs and weight bits). Final results are in the adjacent JSON and phase issue.

Frozen fixture on both hosts under the validation root's `d3-first/`:
SHA256 `5715842de3118def04ca5472428ea4c41199f247207259654f8fe0e564b4ea67`.
Owned logs/identities: `/tmp/maya-batching-validation/d3-{xs,s}-rows` and
`d3-xs-fallback`. No production executable/config is changed. This gate covers
ordinary rows and isolation; speculative commits and dynamic API serving have
separate D4/D5 gates. Exact-commit CI is linked on the phase issue.
