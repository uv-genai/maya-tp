# Experimental DeepSeek model controller

Issues #111 and #113. An isolated test build connects a rank-zero controller to
the existing `Model` interface. The controller validates sequence, position,
tokens and speculative verdict before sending each command, scopes transport
operations, checks queued work, preserves the first failure and checks the
worker's shutdown acknowledgement. Checkpoints are explicitly rejected in this
test mode; it uses window prefill only.

On two gfx1151 hosts, the Maya-XS model processed a 19-token eBPF prompt and
generated the same 64 greedy tokens as the earlier typed-control target-only
pair. It reset, replayed the prompt and shut down both ranks cleanly. The
DSpark-enabled XS test then accepted 41 of 106 proposed drafts over 22 rounds,
again matching the target-only sequence. Both used RDMA for partials and the
control connection for commands. The same DSpark test on Maya-S accepted 41
of 113 drafts over 23 rounds and matched its target-only 64-token sequence.
All owned ranks exited.

Host tests exercise invalid control ordering and shutdown acknowledgement,
send, begin, end, model and queued failures, wrong rank and sticky first error.
These tests use the test-only model build and synthetic pair identities; they
do not establish production startup, chunked prefill, long-context correctness
or end-to-end API recovery.
