# TP pair-down first cause

> This report records successive dated checks. Experimental DeepSeek TP2
> serving was added after the initial test-only checkpoint; the
> [current review guide](../../DEEPSEEK-TP2-RUN.md) describes its operation.

Issue #114. The server retains the first fatal TP pair-down message, whether it
arrives as `TP2_DOWN` or a poisoned-communicator `ERR`, and returns it on later
requests before writing another `GEN`. Later errors cannot replace that cause.
This applies to existing GLM TP2 as well as a future DeepSeek TP2 serving path.

`serve.test_tp2_peer_loss` passed its single and multi-request checks, including
an initial pair-down notification followed by a different engine error, an
HTTP health check and verification that no later request reaches the engine.
The first run exposed a trailing newline in the multi-request error; trimming
the received cause fixed it and the suite then passed. This is an API test
with a test engine, not an injected failure in a live DeepSeek serving pair.

In a separate two-host XS model test with real preflight identities, the owned
rank-1 process was terminated during generation. Rank 0 reported the original
`verbs peer closed the bootstrap connection` cause and subsequent model step
and feed calls returned the same cause without entering another operation.
Both owned ranks exited; this exercised model control and RDMA, not HTTP.

An isolated HTTP test then loaded the experimental XS pair with checked
identities, completed a 38-prompt-token, 64-output-token chat request with
30/36 DSpark drafts accepted, and stopped only its recorded idle worker.
Health changed to degraded/down without another request; the next completion
was rejected with the original peer-closed cause. The driver stopped its HTTP
server, and all owned processes exited. Its first attempt completed the HTTP
request but did not inject loss because it looked for a remote PID file on the
local host; the corrected retry produced the failure result above.

In an isolated real-identity XS model test, an owned monitor waited for a
256-row chunk-reduction marker, verified the recorded worker's executable,
and terminated only that worker. The lead retained the original `verbs peer
closed the bootstrap connection` cause, invoked its failure callback once,
and rejected later feed and step operations with the same cause. The lead
test and monitor exited successfully. This checks a chunk-prefill operation;
the subsequent cases below exercise speculative failure points separately.

Three further real-identity XS model tests stopped and terminated their own
worker at the verify command, at the verdict command, and during rollback
after ring restoration was queued but before compressor restoration. In each
case an owned monitor checked the stopped worker's executable and PID before
terminating it. The lead retained the first peer-close cause, notified once,
and rejected later feed and step operations with the same cause. These tests
do not substitute for HTTP cancellation or the remaining operational gates.
The process-stopping model hooks were then placed behind a test-only compile
definition. Both the test binary and regular engine built, the hook strings
were absent from the regular engine, and the guarded rollback test passed again
with the same first-cause and later-operation checks.

An isolated HTTP pair with the experimental 256-row launcher option then
completed a 545-token request with 64 output tokens (25/29 DSpark drafts
accepted). The driver closed a 3,204-token streaming request; the engine
reported cancellation while reading the prompt. A follow-up request completed
and health still reported the pair up. The owned server was stopped before
its verified worker; all owned processes exited. This is one cancellation
smoke, not a complete failure or performance suite.
After adding the negotiated prefill size to health and rebuilding, the same
isolated HTTP check passed again; `/health.tp.prefill_chunk` reported 256,
cancellation occurred during prefill, and a follow-up succeeded with the
pair up. All owned processes exited.
The same rebuilt-engine check also passed on Maya-S: health reported 256,
the 545-token request accepted 29/39 drafts, a 3,204-token request cancelled
during prefill, and a follow-up completed with the pair up. All owned
processes exited.
# HTTP peer loss during chunked prefill

With the generated dedicated config, an isolated XS HTTP pair used 256-row
prefill, DSpark and lookup. The test observed 256 of 3,204 prompt tokens read,
verified its recorded remote worker PID and executable, and killed that worker.
The first request failed; health became degraded and retained the first
`verbs peer closed the bootstrap connection` cause; a subsequent request was
rejected. The initial run returned the bare cause in the first response.
After fixing server handling of a raw bootstrap peer-close, the same test
passed again: the first response and subsequent response both reported the
same sticky TP2 pair-down cause. The five focused peer-loss unit tests passed.
The driver stopped its server and both owned processes exited. Logs use
`/tmp/ds4-tp2-114-active-loss{,-v2}-*`. This is a prefill fault, not a live
HTTP injection at every speculation boundary.

DeepSeek TP2 now opens the HIP Lt tuning store read-only: existing picks are
read, but newly measured picks are kept only in the process. A standalone
gfx1151 test confirmed read-only mode leaves a file empty and the normal mode
still appends. A checked-identity XS pair then tuned 11 real shapes on each
rank with separate empty stores, which both remained zero bytes; the full
16-position rollback/DSpark check also passed. Both ranks exited. Logs use
`/tmp/ds4-tp2-114-lt-{store-test,readonly-rank0,readonly-rank1}.log`.
