# TP pair-down first cause

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
