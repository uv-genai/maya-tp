# Experimental DeepSeek TP model numerics

Issue #110. The review test executable can bind a sharded XS model to the TP2
RDMA communicator. This is **test-only** and uses a synthetic identity. It sums
attention and shared/routed expert partials before their residual updates, and
fails if the two ranks' router IDs or weights differ. Production `maya-ds4`
cannot enter the experimental execution path.

For synthetic input tokens `[5, 6, 7]`, followed by `[8]`, `[9]`, `[10]`,
both ranks produced identical logits at each checked position. The full
129,280 logits nevertheless differed from a single-host XS reference: maximum
absolute differences were 18.67, 17.17, 14.52 and 15.24. The first three top
IDs matched, but the fourth was 438 for the pair and 223 for the reference.
This strict synthetic exact-logit/greedy comparison failed.

The first material residual difference occurred at prompt row 2 after layer 2,
with matching selected experts. Attention residuals through layer 2 agreed
within 1.8e-7; the routed expert hidden values and summed partial outputs at
that boundary agreed within 2.4e-7 and 6e-8 respectively. The shared Q6_K
projection dynamically quantizes activations in blocks of 32. For one input,
a 4.4e-8 difference changed the value divided by its common scale from
3.4999928 to 3.5000079, rounding the Q8 code from 3 to 4. Its shared output
then differed by up to 0.00036; router order and later selections diverged.

As a controlled test, the paired layer-2 input was substituted into the
single-host reference at that point. Row 2's post-layer-2 residual difference
fell from 0.0003301 to 1.79e-7 and remained below 3e-7 through layer 4,
before another difference arose. This confirms the initial threshold crossing;
it does not establish acceptable model quality or make later differences safe.
The traced runs, reference and pair used the same verified XS files and the same
gfx1151 review binary. Both ranks and the controlled reference process exited.

The same isolated build was then tested with the actual chat prompt “Write an
eBPF tutorial to explain how to use eBPF.” A single-host XS reference produced
64 greedy tokens. The paired model consumed those reference tokens in order:
63 of 64 top predictions matched the reference, with maximum absolute logit
difference 3.7627 and RMS difference 0.3754 across all checked logits. At each
step the two ranks' logits agreed within 1e-5. After a paired reset, rank 0
chose 64 greedy tokens and sent them to rank 1; both ranks recorded the same
sequence. The pair and reference shared the first 55 tokens and then produced
different, coherent continuations of the eBPF introduction. This is one short
text prompt, not a benchmark or a quality equivalence claim. Both ranks exited.

With the typed DeepSeek worker protocol, a separate XS run processed the prompt
in two feed messages, generated the same 64-token pair sequence through 63 step
messages, reset, replayed the prompt and shut down with matching positions. An
equivalent test with the SHA-256-verified Maya-S pack on both hosts generated a
coherent 64-token eBPF introduction and passed reset, prompt replay and shutdown.
These tests used the review executable with SHA-256
`06c04d60ae57e39390c5a7840b6c159edd11cb0393d572c89d57ad2454b36325`;
they did not use a production serving launcher or DSpark. All owned ranks exited.

Neither these tests nor the standalone expert and projection checks establish
DSpark, multi-request state, long-context behavior, production identity
negotiation, failure recovery, or end-to-end serving readiness.
