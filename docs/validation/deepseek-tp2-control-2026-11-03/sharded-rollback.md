# Sharded DeepSeek worker rollback, no drafter

Issue #113. On two gfx1151 hosts, the isolated test executable (SHA-256
`2434aa83ab462fd3fa59b20d3cdc1454d1a8d86f7d5214516a87c606b215fc11`)
loaded rank shards from matching Maya-XS packs, then repeated on the verified
Maya-S packs. A synthetic handshake identity was used; this is not a production
launcher.

For each acceptance count 0, 1, 2 and 3, the test first computed a target-only
continuation. After a synchronized reset it fed the same prompt, executed a
four-token target verification window on both ranks, sent the forced verdict,
rolled both states back, and evaluated the same probe token. All 129,280
subsequent logits matched the corresponding target-only run exactly for every
acceptance count on both quantizations. The worker control and model positions
agreed at shutdown; all four owned ranks exited. The HIP mapping check passed.

The test uses a short synthetic token fixture and deliberately has no DSpark
or lookup drafter. It does not cover actual draft proposals, sampling, rejection
positions in a longer window, long-context compressor boundaries, failures or
serving integration.

The same forced-verdict test was subsequently repeated with all three DSpark
blocks and their routed experts resident on rank 0; rank 1 had no drafter but
reserved target rollback state. For both XS and S, each acceptance count again
left exactly matching next logits. At each count the subsequent five DSpark
draft IDs matched the target-only path, and confidence values matched within
1e-4. The review binary SHA-256 was
`c0889ebec9251913f838ddfcfd6ab41f2ad609372ae98212399202089c71d715`.
All owned ranks exited. These are forced verdicts; actual acceptance sampling,
long windows and production DSpark serving remain untested.
