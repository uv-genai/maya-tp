# Attention arithmetic investigation

Issue #110. An isolated gfx1151 test build from `8efd8ab` traced the same
absolute position on the checked XS pair after 16-row window prefill and
256-row chunk prefill. Each host's pack and executable identities were checked
before launch. The captured build and model identity binds the baseline vectors
to that diagnostic executable and XS pack. The model test used its documented
synthetic communicator identity. All owned diagnostic ranks exited.
[Captured identity](attention-substage-identity.json).

At position 16 in a 17-token fixture, normalized attention input and normalized
low-rank query were identical. The next `wq_b` projection differed by maximum
0.0547868 and RMS 0.0067531. After layer-0 attention, the residual streams
differed by maximum 0.0013485 and RMS 0.0001920. Differences grew through later
layers. The window uses per-32 Q8 activations against the quantized matrix,
while the chunk path reconstructs the weights and inputs as FP16 for GEMM.

An independent NumPy/gguf-py CPU calculation dequantized the same rank-0
Q6_K matrix slice, then accumulated its products in FP64. The captured input
and GPU outputs are in [vectors](query-projection-vectors.npz). Against the
FP64 product of dequantized weights and original FP32 input, the chunk result
was closer on this vector.

| Comparison | Maximum absolute error | RMS error |
| --- | ---: | ---: |
| Window versus dequantized-weight FP64 product | 0.0546641 | 0.00674365 |
| Chunk versus dequantized-weight FP64 product | 0.00148460 | 0.000242679 |
| Window versus independent per-32 Q8-input product | 0.000001818 | 0.0000000968 |
| Chunk versus independent FP16-input/weight product | 0.000009308 | 0.000000439 |

This isolates and explains one early projection difference. The reference
uses the pack's dequantized weights. Its scope covers one layer, rank slice
and activation vector. Broader numerical and output-quality review remains
open.

Three isolated experiments progressively substituted window arithmetic for
`wq_b`, then `wo_a`, then `wo_b`. On the 17-token fixture, their layer-0
reduced attention-output maximum differences were 0.0253255, 0.0074887 and
0.0022570, compared with 0.0262634 in the baseline. Their final maximum logit
differences remained 2.6801, 2.4911 and 2.3613, compared with 2.6234 in the
baseline. Every run's two ranks agreed exactly and selected the same top token
11992 on that fixture.

| Chunk arithmetic on 512 repeated tokens | Maximum window/chunk logit difference | Chunk top | Window top |
| --- | ---: | ---: | ---: |
| Current serving path, earlier validated run | 6.86175 | 9857 | 9857 |
| Matched `wq_b` | 6.30192 | 9857 | 9857 |
| Matched `wq_b` and `wo_a` | 7.02586 | 9857 | 9857 |
| Matched `wq_b`, `wo_a` and `wo_b` | 6.50303 | 1, EOS | 9857 |

The three-projection experiment failed the existing top-token assertion on
512 tokens. Its 545-token formatted eBPF prompt selected token 5 on both
paths, with maximum logit difference 3.93233. These substitutions remain in
the isolated diagnostic worktree. The recorded EOS failure and the CPU
reference give reasons to retain the existing serving arithmetic while
investigating later numerical effects. Single-sample timings include tracing
and synchronization overhead.

Reproduce the independent projection calculation with the original XS pack
and the repository's pinned gguf-py checkout available.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  docs/validation/deepseek-tp2-control-2026-11-03/query-projection-reference.py \
  --pack /path/to/Maya-XS/maya-ds4 \
  --vectors docs/validation/deepseek-tp2-control-2026-11-03/query-projection-vectors.npz
```

The [reference script](query-projection-reference.py) and
[recorded result](query-projection-reference.json) accompany the vectors.
Raw rank-0 logs cover the [baseline](attention-substage-rank0.txt),
[query](attention-query-rank0.txt), [query and output-a](attention-woa-rank0.txt),
[all three projections](attention-wob-rank0.txt),
[512 query](attention-query512-rank0.txt),
[512 query and output-a](attention-woa512-rank0.txt),
[512 failure](attention-wob-512-rank0.txt) and
[545 formatted prompt](attention-wob-545-rank0.txt). Matching rank-1 logs
have the same filenames ending in `rank1.txt`.
