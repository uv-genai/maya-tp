# Experimental DeepSeek TP2 chunk prefill

Issue #110. This review-only path is not enabled by the launcher. It uses
256-row buffers, disables flow and prompt skipping, compares router choices,
and exchanges attention and MoE partials before residual updates. The first
paired XS attempt failed before logits because an existing load-only guard
still rejected chunk execution; both ranks exited. The guard was narrowed to
the explicitly enabled, communicator-bound review path.

The next paired XS test processed a 19-token eBPF prompt through that chunk
path. At every one of 64 checked rows, all 129,280 logits on the two ranks
agreed within 1e-5. The pair's 64-token greedy continuation matched its earlier
window-prefill continuation. Against the single-host window reference, 63/64
top predictions matched, with maximum absolute logit difference 4.64711 and
RMS difference 0.404073. This is one short prompt: these numerical differences
and the previously documented quantization threshold sensitivity preclude a
claim of exact single-host equivalence.

A separate model-controller test used real preflight identities and the same
19-token prompt. It matched the window pair's 64-token continuation, verified
that a cancellation callback was not invoked inside the chunk, reset/replayed
the prompt and acknowledged shutdown. Both ranks exited. This checks deferral;
an HTTP cancellation followed by a synchronized reset remains to be tested.

With rank-0 DSpark enabled and real preflight identities, the same chunked XS
prompt produced the target-only pair's 64-token continuation and accepted
41 of 106 drafts over 22 rounds. Both ranks shut down cleanly. This checks a
single short greedy prompt; larger prompts and sampled DSpark remain open.

An XS sweep compared chunk and window prefill at 17, 128, 129 and 256 tokens
using a repeated token fixture. At the final row of each prompt, full logits
matched exactly between ranks, and chunk and window selected the same top ID.
The maximum absolute chunk-versus-window logit differences were respectively
2.5441246, 7.7583981, 3.634325 and 3.2555804. The subsequent 64-token eBPF
reference run also passed. These nonzero differences remain material when
assessing numerical equivalence; this small sweep is not a quality study.

An extended sweep passed the 257-token top-ID check but **failed at 512**:
both ranks agreed with each other, yet chunk prefill selected EOS ID 1 while
window prefill selected ID 9857 (maximum logit difference 7.4473). A repeat
showed a single two-chunk call and two separate chunk calls were identical in
final logits and every captured position-511 layer residual. This does not
explain or excuse the window/chunk discrepancy; it remains under investigation.

On the same fixture, an existing single-host window/chunk diagnostic chose
window top 19851 and chunk EOS 1, with maximum absolute logit difference
8.5022. The paired chunk path also chose EOS. This exposes sensitivity in the
inherited paths on a synthetic prompt with repeated conversation tokens; it
does not establish correctness for realistic prompts or clear the TP gate.
Matching-stage traces put the first discrepancy after layer 0 attention
(max 0.01365); it grew to 5.116 after layer 39. A whole two-chunk call and
two chunk calls remained identical at all traced attention and residual stages.
Additional traces show the layer-0 input and post-Engram values are identical
between window and chunk, narrowing the first difference to attention.
The normalized attention input was also identical, but the low-rank query and
key/value projections differed by at most 0.0005835 and 0.0005542 respectively.
The window and chunk implementations use different projection arithmetic;
whether this difference alone explains the later behavior is under test.
An isolated experiment replaced chunk attention-input projections with the
window's 16-row matrix-vector operations. At layer 0 the traced query and KV
projections became bit-identical, and the 512-token synthetic prompt selected
ID 9857 in both window and chunk modes (rather than EOS in chunk mode). The
two ranks agreed, and one-call and two-call chunk results remained identical;
the chunk/window maximum logit difference was still 6.8618. This experiment is
not enabled in the launcher and needs broader correctness and performance tests.
The matched-projection XS sweep subsequently passed top-ID comparisons at 17,
128, 129, 256, 257 and 512 tokens, with identical final logits between ranks.
The 64-row short real-text follow-up matched 63 single-host teacher-forced top
IDs (maximum logit delta 3.8486, RMS 0.3746). Both ranks exited. This does not
resolve the remaining full-logit differences or establish performance.
On the same 545-token eBPF fixture in two separate paired runs, the baseline
chunk took 2.9061 s and the experimental matched-projection chunk took 3.2084 s,
including synchronized operation completion and final logit exchange. The
corresponding window timings were 9.1835 s and 6.7692 s. These are single
samples with substantial run-to-run variation and do not establish reliable
throughput or isolate the cost of the change. Both modes selected the same top
token across ranks and paths on this one prompt. Comparing the two chunk
logit vectors directly gave maximum difference 3.9529, RMS 0.8356, and nine
common top-10 IDs; the arithmetic change can affect model output.
With matched projections and the same 545-token request, the paired 64-token
run matched 61/64 teacher-forced single-host top IDs (max logit delta 4.9847,
RMS 0.4413). Both ranks agreed within 1e-5; independent greedy continuations
shared 21 initial tokens and then diverged into coherent tutorial openings.
The 545-token whole/split diagnostic then showed a maximum logit difference
of 4.2774 and different continuations at token 19; removing the prior
teacher-forced phase reproduced the direct whole-feed continuation. A
review-only change made TP chunk planning use the controller's fixed-size
boundaries. With that change, the same whole and split inputs produced
identical full logits and identical 64-token greedy continuations on XS;
both ranks exited. It remains disabled in the launcher pending further tests.
An additional XS 3,204-token prompt passed full-logit rank and fixed
whole/split comparisons exactly and produced 64 tokens. Its window and chunk
top IDs also matched. This was run with concurrent CPU checks and is not a
performance measurement.
The same 545-token test on Maya-S produced identical whole/split and rank
logits, then generated 64 tokens. Each rank loaded 84,509,982,720 expert bytes;
both exited. The synthetic-identity test was followed by a real-identity
controller run, which matched the 64-token continuation, deferred a cancel
callback, reset/replayed and shut down both ranks. The subsequent S DSpark
test matched all 64 controlled target-only tokens and accepted 40/116 drafts
over 24 rounds; both ranks shut down.
For a separate 545-token properly formatted eBPF request, the existing
single-host window and chunk paths chose the same next token (ID 5); their
maximum logit difference was 3.6831 (RMS 0.8259). The paired test then
completed with identical full logits on the two ranks; its window and chunk
paths both selected ID 5, with maximum logit difference 4.3801. The token
decodes to `#`, a plausible tutorial opening, but one token is not a quality
evaluation. Comparing single-host and paired full logits on this prompt gave
window max/RMS 3.9863/0.8676, with all ten top IDs in common; chunk max/RMS
3.9327/0.7978, with eight top IDs in common. These are substantial numerical
differences; further continuation and quality checks are required.
The request text and DeepSeek chat-format token IDs are in
`tests/fixtures/ds4-ebpf-long.txt` and `tests/fixtures/ds4-ebpf-long.tokens`.
The token IDs were produced with this XS pack's tokenizer and
`serve.ds41_encoding.encode_messages(..., thinking_mode="chat")`.
Using that prompt and the chunk path, the single-host reference and two-host
pair each generated 64 tokens. The ranks' full logits matched within 1e-5 for
all compared rows; 59/64 teacher-forced top IDs matched the single-host
reference (max logit delta 7.9736, RMS 0.4767). Their independently generated
greedy outputs shared 12 initial tokens and then diverged, with both forming
coherent eBPF tutorial openings. This is one prompt, not a general quality
assessment or an exact-equivalence result.

This does not yet validate HTTP cancellation, sampled DSpark,
long-context state, Maya-S, HTTP serving or prefill throughput. Rank-local
cancellation/yield callbacks are deferred until the current bounded chunk
finishes so that one rank cannot abandon a collective independently.
