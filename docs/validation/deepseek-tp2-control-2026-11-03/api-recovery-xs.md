# DeepSeek TP2 API and coordinated recovery

Issue #114. An isolated XS pair used gfx1151, checked pack/build identities,
256-row prefill, DSpark on and lookup configured. Its new review engine
SHA-256 was `d13f478d0702cebb35a40c99cdf0918787fa4079fa8de02cf4166a12663e203e`.
The test used a 3 GiB available-memory guard and recorded the owned worker PID,
start time and executable before stopping it. Both owned ranks exited.

- Non-streaming OpenAI Responses and Anthropic Messages returned completed
  answers to the same question. A streamed chat ended with a normal stop.
- For the same greedy 128-token coding request, `strata_tune.spec=0` with
  lookup disabled offered zero drafts; `spec=1` offered 100 and accepted 88.
  Answers matched exactly. The live monitor reported the corresponding off/on
  status during both requests.
- With DSpark off and lookup enabled for a repetitive request, the live
  monitor remained on and the engine reported 164/166 accepted/offered
  drafts. The server can run internal continuations, so the engine's draft
  counters do not necessarily match final API output-token usage.
- One- and two-token output limits were respected and logprobs returned;
  an image request was rejected explicitly before encoding.
- The test killed its own idle worker. Health marked the pair down and an
  Anthropic request retained `verbs peer closed the bootstrap connection`.
  After stopping the server, it started both ranks afresh and a chat request
  succeeded. The restart took several minutes while the worker read model
  files; this is one restart, not a controlled startup benchmark.
- After recovery, an eBPF tutorial response began coherently and used all
  1,024 requested output tokens, so it is a partial qualitative sample rather
  than a completed answer or a quality-equivalence test.

[Responses and captured answers](api-recovery-xs.json), [server log](api-recovery-xs-server.txt),
[engine log](api-recovery-xs-engine.txt) and [restarted worker log](api-recovery-xs-worker.txt)
contain the raw results. This test complements the earlier live prefill peer
loss and cancellation checks in [server-first-cause](server-first-cause.md).
