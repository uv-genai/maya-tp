# DeepSeek rollback across position 4096

Issues #113 and #116. The paired gfx1151 XS and S tests used a 4,095-token
synthetic prefix, 256-row prefill, a 16-row target verification window and
32,768-token context. Rank 0 loaded DSpark, while rank 1 reserved target rollback
state. The launcher prepared the model, partition and executable identities
from each host's checked manifest before the pair exchanged them.

For each acceptance count from 0 through 15, the test reset and replayed the
same prefix. Its target-only reference fed the retained verification prefix,
generated five DSpark drafts, and stepped a probe token to capture logits.
A separate reset/replay then verified the full window, committed the forced
verdict, and generated the same drafts and probe logits. Verification crossed
position 4096 with a populated compressed-KV and index history.

| Model | Acceptance counts passed | Maximum next-logit difference | Draft token IDs |
| --- | ---: | ---: | --- |
| XS | 16/16 | 0 in every case | All five match in every case |
| S | 16/16 | 0 in every case | All five match in every case |

Draft confidences passed the existing absolute tolerance of 0.0001. Both ranks
exited for each model. These are forced-verdict state checks on a synthetic
prefix. Sampled acceptance and natural requests have separate validation
records.

The C++ test now accepts prefixes up to 8,192 tokens while reserving 32 context
positions for verification and probes. Its default prefix remains three tokens.
The tested executable SHA-256 is
`ab836d6725ba444d41f6894d0794bc8670e158a8f3a7827a67f933e3fa753118`.
The checked launcher records `DS4_TP_TEST_PROMPT_LEN=4095`,
`DS4_TP_TEST_VERIFY_ROWS=16`, `DS4_TP_TEST_CHUNK=256`, and invokes
`ds4_tp_verify_test --sharded-rollback-dspark` on each rank.

The extended target built successfully, and the ordered HIP mapping check,
618 server tests with 11 skips, and 707 tools tests with 3 skips passed after
the test-source change. These functional tests overlapped host checks.

Raw results are [XS lead](rollback-4k-xs-rank0.txt),
[XS worker](rollback-4k-xs-rank1.txt), [S lead](rollback-4k-s-rank0.txt),
and [S worker](rollback-4k-s-rank1.txt). The captured
[launcher](rollback-4k-launch.py) documents the machine-specific invocation.
