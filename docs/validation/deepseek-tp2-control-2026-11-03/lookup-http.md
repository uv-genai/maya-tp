# Experimental DeepSeek TP2 prompt lookup

Issue #113. An isolated XS pair with matching checked pack and executable
identities ran 256-row chunk prefill, DSpark off, and prompt lookup with a
minimum match length of 3. `/health` reported the pair up and chunk size 256;
`/metrics` reported Lookup and did not report DSpark.

For one repetitive greedy request, the lookup-off and lookup-on API responses
matched exactly. The server internally ran two engine continuations for the
lookup-on response: the engine reported 90/92 and 81/81 lookup drafts accepted
in 15 and 9 rounds respectively. API output usage reflects the server's final
response, not the sum of those internal engine tokens. The owned server was
stopped before its verified worker, and all processes exited. Logs and saved
responses use `/tmp/ds4-tp2-113-lookup-http-*` and
`/tmp/ds4-tp2-113-lookup-output-{0,1}.txt`.

The same greedy XS prompt then ran with both DSpark and lookup enabled.
Lookup-off/on API responses matched exactly. In the first internal lookup-on
continuation, 82/92 lookup drafts and 15/15 other (DSpark) drafts were kept;
the second kept 81/81 lookup drafts. Metrics reported both modes. All owned
processes exited. Logs use `/tmp/ds4-tp2-113-combined-http-*`.

The same combined test passed on Maya-S: lookup-off/on responses matched,
with 82/92 lookup drafts and 15/15 DSpark drafts accepted in one internal
continuation and 81/81 lookup drafts in the other. Metrics reported both
modes. Logs use `/tmp/ds4-tp2-113-s-lookup-http-*`.

A separate real-identity XS startup deliberately requested lookup minimum 3
on rank 0 and 0 on rank 1. Both ranks rejected the feature mismatch before
inference and exited; logs use `/tmp/ds4-tp2-113-lookup-mismatch-rank*`.

These are narrow greedy examples, not general quality, sampled-distribution,
failure, or throughput validation. Lookup remains experimental and defaults off.

Separately, sampled DSpark after a 19-token chunked prefill passed a small
two-sample distribution smoke test on XS using checked rank identities:
1,000 plain versus 50 speculative draws, 4 grouped bins, p=0.612 for sampled
proposals (35/250 accepted) and p=0.527 for point proposals with penalties
(8/250 accepted). The first three seeds per mode replayed exactly. This checks
DSpark's sampled verifier; it does not establish sampled lookup behavior.
Logs: `/tmp/ds4-tp2-113-sample-chunk-xs-rank{0,1}.log`.
The same Maya-S check passed its preset p>=0.001 threshold: sampled-q
p=0.0129 (26/250 accepted), point-q with penalties p=0.365 (15/250 accepted),
with seed replay. Logs use `/tmp/ds4-tp2-113-sample-chunk-s-rank{0,1}.log`.
The samples are small and do not rule out subtle distribution errors.

At 262,144 configured context on XS, three sampled API requests at
temperature 0.2 with lookup and DSpark enabled each exercised lookup and
accepted drafts. Across the six internal continuations the kept/offered
counts were 82/92, 81/81, 86/87, 81/81, 84/86 and 81/81. A separate
one-token request returned logprobs within its output budget. Responses and
logs use `/tmp/ds4-tp2-116-xs-{sampled-lookup.json,logprobs.json,bench-*}`.
This is an API smoke test rather than a sampled distribution comparison.
The corresponding Maya-S run at the same configured context also accepted
lookup drafts on all three sampled requests: 82/92 in one internal
continuation and 81/81 in each of the other five. Its one-token logprobs
request passed. Logs use `/tmp/ds4-tp2-116-s-bench-*`.
