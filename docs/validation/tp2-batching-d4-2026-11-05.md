# DeepSeek speculative batch transactions — D4

2026-11-05, [#152](https://github.com/uv-genai/maya-tp/issues/152).
Rank zero drafts in fitting groups, using 2+2 for four block-size-five clients.
The drafter returns five proposals per block; the separate target plan still
fits at most sixteen rows including anchors. All groups use point proposals.
Sampled targets retain the point-proposal rejection rule; shared sampled-q
scratch is never reused across clients. Default one-sequence drafting is
unchanged.

Target windows stay open after token selection. A complete canonical verdict
validates every member before either rank rolls back any member. Rank zero
seeds only retained taps; both peers retain identical ring/compressor/history
positions. The last emitted token remains unconsumed. Continuation checks that
pending token on the next target; terminal members require a new generation.
Acceptance and completion acknowledgements bracket each transaction. Fatal
errors retain their first cause and require discarding the pair.

The explicit `--batch-spec` fixture uses four 262K slots and ragged target runs
of 6/4/3/3 inputs at unequal positions around the 512 boundary. It forces the
first member to retain 1–6 inputs, covering zero through all five proposals,
while other members retain different counts. It compares the later complete
logits and DSpark drafts with independent ordinary continuations. Selection
also exercises sampled targets with bias and penalties, without committing
until the complete verdict. The terminal case checks generation reuse.

XS and S each pass all 24 full-vocabulary comparisons exactly, with matching subsequent
drafts. Fitting groups produce 120 real proposals over the six rounds. Both pairs exit 0/0 and all owned processes are stopped. Exact-head CI
is recorded with the final adjacent JSON and phase issue. Host state/channel gates and ASan/UBSan pass. Production multi-slot
serving remains gated until D5.

Frozen fixture on both hosts: validation root `d4-counted/ds4_tp_verify_test`,
SHA256 `026b73ab6ce6b45a573c5c26b2dff3814d65aeb08f19191a53e5ba06f7ceead0`.
Owned logs/identities: `/tmp/maya-batching-validation/d4-{xs,s}-spec_counted`.
The earlier `d4-first` fixture stopped at an incorrect four-proposal assertion;
the source drafter returns block_size proposals. Its failing evidence is kept.
No model arithmetic was changed to suppress that fixture error.
