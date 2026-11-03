# Sharded DeepSeek worker rollback and DSpark

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

An additional XS test used independently verified matching pack, partition and
binary digests and a 127-token synthetic prompt. For forced acceptance counts
0 through 3, the verification and rollback crossed the 128-token ring and
compressor boundary. Each next-token full logit vector matched the target-only
continuation exactly; all five subsequent rank-0 DSpark drafts and their
confidences also matched within 1e-4. Both ranks shut down cleanly. The review
binary SHA-256 was
`478af130683ca0558f12d1471ee913f160ab1c05092ce50c76dfce3a5a432c2f`.
This does not cover longer contexts, Maya-S at this boundary, lookup, or every
failure mode; the phase gate remains open.

A subsequent real-identity XS run used experimental matched-projection chunk
prefill on a valid 545-token eBPF prompt and compared greedy DSpark generation
against a separately recorded target-only pair. It **failed** at output index
19, after six rounds offered 30 drafts and accepted 13: the next DSpark token
differed from the target-only token. The lead exited, and the worker observed
peer close and exited. This may reflect speculative state or numerical
differences in multi-row verification; the cause has not been established.
It blocks the relevant long-prompt DSpark gate.
The diagnostic rerun reproduced the mismatch (actual token 2572, reference
24353). After a synchronized reset and replay of the committed prompt and 19
generated tokens, the next token was again 2572. Feeding the same probe before
and after replay produced identical full logits (max and RMS 0). The worker
acknowledged shutdown. This is evidence that this specific committed state
survived rollback; it does not establish why the separately run target-only
pair chose a different next token or prove all rollback states correct.
Code inspection identified a likely mismatch in the comparison: the direct
545-token target-only feed was balanced into chunks of 182+182+181, while the
controller-based DSpark test fed 256+256+33. Those execution shapes may have
different numerical results. An identical-boundary target-only comparison is
required before interpreting the failed output check. A subsequent controlled
run with the drafter loaded but never invoked generated 64 tokens; its first
19 matched the direct-feed reference, and its next token was 2572, identical
to the speculative and replay result. A no-drafter controlled comparison is
still required to isolate the effect of the feed boundaries. That independent
no-drafter controlled run then produced exactly the same 64 tokens as the
drafter-loaded no-draft run; both first differed from the direct-feed sequence
at token 19 (2572 versus 24353). The original comparison used mismatched
execution setups. A later whole/split test found a 4.2774 maximum final-logit
difference, and whole-feed greedy with no prior teacher-forced phase exactly
reproduced the direct-feed continuation. This isolates chunk shape as the
cause of the reference mismatch. Against the
corrected controlled target-only reference, DSpark then generated all 64
tokens exactly, accepting 41 of 111 drafts over 23 rounds. Both ranks shut down
cleanly. This is one XS prompt and does not close the remaining phase gates.
After the fixed-boundary review change, the same real-identity DSpark test
again matched all 64 reference tokens, with 41/111 drafts accepted over 23
rounds and coordinated shutdown. The review binary SHA-256 was
`072670e7bc1a4c558568697a5aaec903a006afce8d051eecb4a19210ecdc6a46`.
On Maya-S, the same changed binary and real identities matched all 64 tokens
of its controlled 545-token prompt, accepted 40/116 drafts over 24 rounds, and
shut down both ranks. This is one prompt, not the full phase gate.
The S pair also passed at configured context 524288 with this 545-token prompt:
all 64 tokens matched the target-only reference and 40/116 drafts were accepted.
Both ranks shut down. The prompt did not fill a 512K context, so this only
checks allocation and a short execution at that setting.
An XS 3,204-token real-format prompt also passed a real-identity DSpark check:
64 target-only tokens matched, with 38/124 drafts accepted over 25 rounds.
Both ranks shut down. This remains shorter than a full long-context test.
