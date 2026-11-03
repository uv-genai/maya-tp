# Current DeepSeek TP2 window/chunk checks

Issue #110. The earlier [chunk review](chunk-review.md) recorded a 512-token
synthetic fixture where the then-current chunk path chose EOS and the window
path chose token 9857. The window-matched attention input projections from the
subsequent experiment were integrated into the experimental TP chunk path in
`a34d616`. This report rechecks that fixture on the later `de96d64` test build
(SHA-256 `fe9c8c898607db81903cc354fd0409e6e0957aaf845665c690e78cf06fb0d5f5`).

The test independently checked each host's pack, geometry and executable
identities before launch; its test communicator itself used a synthetic wire
identity. Its required reference-file argument was a synthetic placeholder;
the actual window comparison was computed on the same paired model in the
sweep. On XS and S, both ranks produced identical logits and the window and
256-row chunk paths chose the same top token for repeated-token prompts of 17,
128, 129, 256, 257 and 512 tokens. At 512, XS selected token 9857 in both
paths and S selected token 271 in both. A direct whole 512-token feed and two
256-token feeds had identical full logits on both models.

The stricter XS experiment first required direct whole-feed and split-feed
logits to be identical at 257 tokens. It failed: maximum logit delta 1.059103,
though both selected token 1192. The same non-asserting diagnostic measured
3.5900075 on S, again with the same top token. The test's direct 257-token
`Model::feed` executes a one-row remainder within a chunk, whereas the serving
controller limits individual feeds to 256 and executes its final one-row feed
through the window path. Thus this direct-call diagnostic is not a supported
server operation; its difference is retained rather than hidden. The strict
failed test and both owned ranks exited before the non-asserting rerun.
An additional test-only check reset each XS and S pair and replayed the
supported 256+1 feed sequence: its full logits matched the original split
exactly and both ranks agreed. This rebuilt test executable had SHA-256
`f06ad48c9cc471fb6a08f61424504aa740109a07261d828bb0b65eab14badfb3`;
the serving engine was unchanged. Both pairs exited.

Window and chunk **full logits still differ materially** even where the top
ID matches. A match on these selected synthetic cases does not establish
single-host equivalence or general output quality. On a formatted 545-token
eBPF prompt, XS and S each had identical logits between ranks, the same window
and chunk top token (5), and identical whole/split chunk logits. Window/chunk
maximum full-logit deltas were 4.0495863 on XS and 2.9532645 on S. The
formatted 3,204-token fixture also passed the rank, top-token and whole/split
checks on each model: window/chunk maximum deltas were 3.7900484 on XS and
1.8526626 on S. All owned ranks exited.

[XS lead](sweep-current-xs-rank0.txt), [XS worker](sweep-current-xs-rank1.txt),
[XS preflight identity](sweep-current-xs-identity.json),
[S lead](sweep-current-s-rank0.txt), [S worker](sweep-current-s-rank1.txt),
[S preflight identity](sweep-current-s-identity.json),
[strict 257 failure lead](sweep-xs-whole257-failure-rank0.txt),
[strict 257 failure worker](sweep-xs-whole257-failure-rank1.txt),
[XS split replay lead](split-replay-xs-rank0.txt), [XS worker](split-replay-xs-rank1.txt)
and [XS identity](split-replay-xs-identity.json), [S split replay lead](split-replay-s-rank0.txt),
[S worker](split-replay-s-rank1.txt) and [S identity](split-replay-s-identity.json). The 545-token
logs and checked identities are [XS lead](sweep-current-xs-545-rank0.txt),
[XS worker](sweep-current-xs-545-rank1.txt), [XS identity](sweep-current-xs-545-identity.json),
[S lead](sweep-current-s-545-rank0.txt), [S worker](sweep-current-s-545-rank1.txt)
and [S identity](sweep-current-s-545-identity.json). The 3,204-token logs and
identities are [XS lead](sweep-current-xs-3204-rank0.txt),
[XS worker](sweep-current-xs-3204-rank1.txt), [XS identity](sweep-current-xs-3204-identity.json),
[S lead](sweep-current-s-3204-rank0.txt), [S worker](sweep-current-s-3204-rank1.txt)
and [S identity](sweep-current-s-3204-identity.json).
