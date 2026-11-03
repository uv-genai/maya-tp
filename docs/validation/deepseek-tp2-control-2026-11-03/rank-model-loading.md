# DeepSeek experimental rank-model loading

Issue #110. The internal experimental model mode loads rank-local attention,
shared-expert and complete routed-expert slices but **refuses inference and
state restoration**. It is not exposed by the production CLI. It requires a
single GPU, window prefill, `DS4_STREAM=0`, read-in-place experts, no cache,
vision, helper, or drafter, and a read-only profile.

On two gfx1151 hosts, both Maya-XS model ranks loaded concurrently, each with
all 15,360 local expert slices. Rank 0 reported 4.22 GB dense weights and
54,814,310,400 expert bytes; rank 1 reported 4.22 GB and 54,751,395,840.
Each used 4,249,600 pinned staging bytes. Calls to feed, step, worker verify,
sampling, logits and restore were rejected, without advancing the position.
Invalid rank, reserve and speculative options were rejected before loading.
The existing single-host XS DSpark rollback test passed for zero, partial and
full acceptance. HIP mapping passed.

Both Maya-S ranks loaded sequentially on the source host, each with 5.22 GB
dense weights, 84,509,982,720 expert bytes and 7,014,400 staging bytes; about
35 GiB was left free in each run. Load time and memory vary with cache state
and the running S file copy.
This checkpoint does not test distributed inference, DSpark on TP, full context
memory, the copied S pack, or production launchers.
