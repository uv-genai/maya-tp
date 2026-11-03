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

Actual greedy DSpark proposals and rank-0 decisions were then exercised on the
eBPF chat prompt. The worker executed each target verify window and committed
the authoritative typed verdict. On XS, 22 rounds offered 106 drafts and
accepted 41; on S, 23 rounds offered 113 and accepted 41. Each generated the
same 64-token sequence as its corresponding target-only sharded pair. The
review executable SHA-256 was
`e54e280cc101f26f915fb360af9f0f8a039e6cefabe46ae53e54785788bf0348`.
All owned ranks exited. These are short greedy tests; sampled acceptance,
long-context boundaries, performance and production serving remain open.

Sampled speculative decoding was also checked with a short “Name a fruit”
chat fixture. For each model and mode, a context was selected from a fixed set
using independent plain target draws; point-q mode additionally required the
point draft to have at least 10% observed target support. Each comparison used
1,000 target samples and 50 speculative first-token samples. On XS, sampled-q
gave chi-square p=0.8221 (3 groups, 11 accepted of 250 proposed drafts), and
point-q with penalties gave p=0.7247 (3 groups, 71 accepted). On S, the same
checks gave p=0.7755 (3 groups, 2 accepted) and p=0.1919 (3 groups, 36
accepted), respectively. The first three seed/counter trials in each mode were
replayed and reproduced drafts, confidences and verdicts. The test binary was
`e6d62bbb93631da4a4b248ab01d492033860dd9129f3e3a224558d986787d160`.
An earlier S point-q context had zero observed accepts; it is retained as a
coverage limitation, not treated as a distribution failure. These small samples
cannot prove distributional equivalence. All owned processes exited.
