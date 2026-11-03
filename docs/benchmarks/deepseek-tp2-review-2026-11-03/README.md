# DeepSeek TP2 review measurements, 2026-11-03

Experimental work under #116; this is not a release or production promotion.
The XS pair used matching checked pack and executable identities, gfx1151 on
each host, USB4 RDMA, context 262,144, 256-row bounded prefill, DSpark on,
lookup enabled, vision off, and one active request. The earlier one-host
reference used the same quant, context, prompts and output limits, but enabled
vision and used the earlier one-host execution and memory placement. It is a
useful reference, not a controlled isolation of TP overhead.

| XS measurement (tok/s) | Earlier single-host runs | Experimental TP2 runs |
| --- | --- | --- |
| Python decode | 35.7, 39.9, 41.0 | 25.2, 26.4, 24.0 |
| Prose decode | 26.0, 29.2, 30.0 | 25.5, 29.7, 25.2 |
| HTML decode | 42.8, 46.9, 47.3 | 29.1, 24.6, 29.2 |
| 2K fresh prefill | 121.7, 237.6, 249.1 | 192.3, 198.6, 205.5 |
| 8K fresh prefill | 376.6, 388.8, 390.0 | 201.0, 199.9, 203.5 |

The first single-host prefill trial was relatively cold. All nine matching
short prompts had different answer hashes between the two runs; differences
are consistent with, but do not by themselves explain, the numerical
differences documented in the [chunk review](../../validation/deepseek-tp2-control-2026-11-03/chunk-review.md).
The hash comparison does not evaluate answer quality. No general speedup or
exact output equivalence is claimed.

Raw TP2 data: [short prompts](xs-tp-reference.json), [fresh prefill](xs-tp-prefill.json),
[local sampled telemetry](xs-tp-telemetry.jsonl). The earlier one-host raw
data and measurement conditions are in the [P1 report](../deepseek-tp2-p1-2026-11-03/README.md).
During a 190-second portion of the TP2 run, sampled local GPU utilization
ranged from 34% to 81%; local USB4 RDMA device counters increased by about
12.7 GB in each direction. These device-wide counters are not per-request
measurements. Over a separate partial interval, process `read_bytes`
increased by about 1.37 GB on each rank; the saved `*-io-{mid,late}.txt`
snapshots include all physical reads by those processes and do not isolate
Engram or give an exact per-token rate. Neither routed-expert residency nor
these observations establishes that storage I/O has been eliminated.

Maya-S then ran with the same TP2 settings and harness:

| S measurement (tok/s) | Earlier single-host runs | Experimental TP2 runs |
| --- | --- | --- |
| Python decode | 5.6, 6.0, 7.7 | 25.1, 27.3, 26.4 |
| Prose decode | 2.6, 5.3, 7.3 | 24.3, 24.2, 25.3 |
| HTML decode | 3.0, 5.3, 7.2 | 27.5, 30.0, 29.9 |
| 2K fresh prefill | 92.0, 125.6, 140.7 | 186.1, 200.0, 192.0 |
| 8K fresh prefill | 265.1, 286.0, 258.9 | 190.9, 187.7, 189.8 |

The prior S single-host run had substantial storage reads and memory pressure;
TP2 placed 84.51 GB of routed experts on each rank, so the decode comparison
includes that placement difference. All nine matching-prompt answer hashes
differed from the earlier single-host runs. The captured TP2 answers begin
coherently, but each reached the 256-token limit before completing, so this is
not a comprehensive quality comparison. Raw TP2 data: [short prompts and
answers](s-tp-reference.json), [fresh prefill](s-tp-prefill.json) and
[local sampled telemetry](s-tp-telemetry.jsonl). Local GPU utilization ranged
from 39% to 78% in 27 samples over 260 seconds; device-wide RDMA byte counters
rose about 13.5 GB in each direction. Partial-interval process physical reads
increased by approximately 366 MB and 365 MB on ranks 0 and 1 respectively;
these aggregate counters do not isolate Engram or give per-token costs.

Both owned servers were stopped before their workers; all ranks and telemetry
samplers exited. The later same-binary DSpark comparison follows below; fuller
quality review, filled long contexts and same-build one-host comparisons remain.

## Same-binary Maya-S DSpark comparison

The rebuilt engine `31b2e21d529f5de3016c7ce58a3bab704f052ab5d22beea62b64829680a81fb1`
then ran the same S pair at context 262,144 and 256-row chunk prefill with
lookup **off** in both runs. All nine answers matched exactly between DSpark
on and off. DSpark accepted 185/211, 155/175 and 206/215 drafts respectively
for the Python, prose and HTML prompts in each repetition.

| Measurement (tok/s) | DSpark off | DSpark on |
| --- | --- | --- |
| Python decode | 17.6, 19.3, 19.7 | 22.7, 26.6, 24.6 |
| Prose decode | 18.0, 19.8, 19.4 | 21.8, 27.3, 25.7 |
| HTML decode | 19.7, 19.9, 19.2 | 27.1, 29.9, 30.4 |
| 2K fresh prefill | 181.5, 193.5, 193.5 | 191.8, 198.0, 193.7 |
| 8K fresh prefill | 192.6, 190.5, 192.7 | 192.1, 192.3, 190.6 |

The six prefill prompt hashes also matched. The fixed short answers all hit
the output budget; exact equality here is a scoped greedy check, not proof
for all prompts or sampled requests. Raw data: [off decode](s-tp-off-reference.json),
[on decode](s-tp-on-reference.json), [off prefill](s-tp-off-prefill.json),
[on prefill](s-tp-on-prefill.json), [off telemetry](s-tp-off-telemetry.jsonl)
and [on telemetry](s-tp-on-telemetry.jsonl). Both pairs and their telemetry
samplers exited. XS same-binary comparisons and long-context gates remain open.
