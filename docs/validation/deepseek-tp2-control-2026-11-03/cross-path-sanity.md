# DeepSeek cross-path sanity checks

Issue #110. The same build (engine SHA-256
`1200fbaff5c69ca27312fa4321149e08f260848c1f0b6d1776cbc04c29e209ec`),
model files, 32,768-token context, prompts, greedy sampling and 64-token
output limit were used on single-host and TP2 for XS and S. DSpark was on,
lookup and vision were off. TP2 used one GPU per host and the established
rank-sharded placement; single-host used 8,192 MiB reserve and capped expert
read-ahead at zero for memory headroom. These placement and reserve differences
remain part of the comparison.

Eight short prompts checked addition, hexadecimal conversion, multiplication,
a Python set/sort result, sorted JSON, a JSON object, a simple logical
inference, and identification of the eBPF verifier. A simple evaluator
accepted JSON inside a code fence and any verifier answer containing
`verifier`; the exact prompts, answers and checks are in the raw files.

| Model | Single-host passed | TP2 passed | Text-identical answers |
| --- | ---: | ---: | ---: |
| XS | 8/8 | 8/8 | 5/8 |
| S | 8/8 | 8/8 | 7/8 |

XS differences were JSON whitespace/code fencing and `The BPF verifier`
versus `Verifier`; S differed only in `Verifier` versus `The verifier`.
All owned processes exited. [XS single](quality-xs-single.json),
[XS TP2](quality-xs-tp.json), [S single](quality-s-single.json),
[S TP2](quality-s-tp.json).

These are eight easy, short cases, one run each. They can reveal obvious
failures but do not establish general quality or exact numerical equivalence.
The earlier synthetic 512-token top-token discrepancy was not reproduced in
the [current window/chunk sweep](window-chunk-current.md); full-logit and
longer-answer differences remain in [chunk review](chunk-review.md).
