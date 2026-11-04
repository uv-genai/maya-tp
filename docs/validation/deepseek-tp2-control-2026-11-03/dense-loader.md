# DeepSeek rank-local dense matrix loading

Issue #115. The isolated pack-backed loader handles complete matrices, halves of
output rows, and halves of input columns aligned to their quantization blocks.
It validates matrix metadata, mapped file bounds and a caller-supplied GPU budget
before copying a rank's slice. It exposes the device pointer and encoding only
after the copy succeeds. It is not yet connected to model execution.

On gfx1151, rank-local GPU bytes matched independent file reconstruction for
XS Q6_K and S Q8_0 layer-0 `wq_b` output halves and `wo_b` input halves. A whole
XS `wo_a` group and an S BF16 matrix also matched. The tests rejected invalid
ranks, too-small budgets, unsupported dtypes and out-of-file offsets. An isolated
single-host XS DSpark verification regression passed after the loader was added.
These tests do not cover full-rank dense allocations or attention computation.
