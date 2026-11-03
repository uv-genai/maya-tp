# DeepSeek utilization and Engram observations

Issue #116. XS and S were measured once on single-host and TP2 with the same
engine built from source `52cf519` (SHA-256
`a0dcc1cb91e9e52fa365291d68eef8203af714d8222e77e49d72741a4331ddc2`),
model files, 32,768-token context, greedy CSV-coding prompt and 256-token
output limit. DSpark was on, lookup and vision off. Each run had a short warmup.
The single-host runs reserved 8 GiB and capped expert read-ahead at zero for
headroom; TP2 used rank-sharded experts and 256-row prefill. Configurations and
cache states therefore differ. Output continuations can differ.

GPU busy was sampled approximately every 0.8 seconds via each host's
`gpu_busy_percent` during the entire measured request. Each number is a
device-wide sample, not process-attributed; the first may include the request
boundary, local and remote samples are sequential, and prefill and decode are
combined. The sampling thread and remote SSH add overhead. One run per
configuration is not a benchmark distribution.

| Model / path | GPU busy min / median / max, rank 0 | Rank 1 | Samples |
| --- | ---: | ---: | ---: |
| XS single | 34 / 96 / 98% | — | 10 |
| XS TP2 | 13 / 38 / 53% | 10 / 33 / 45% | 15 |
| S single | 49 / 88 / 99% | — | 26 |
| S TP2 | 16 / 43.5 / 61% | 12 / 38 / 48% | 14 |

Rank-0 monitor snapshots just before and after the request provide these
cumulative Engram lookup deltas. `eg_lookup_reads` counts logical requested
rows after cache hits and deduplication; it does **not** measure physical disk
reads. The reported Engram wait is the final monitor value for the request,
in milliseconds per model-fed token; speculative verification can feed more
tokens than the API emits. Both prefill and generation contribute to the
counter deltas. Rank-1 Engram counters are not exposed by this server monitor.

| Model / path | Lookup rows | Cache hits / duplicates | Logical row reads | Engram wait ms / fed token |
| --- | ---: | ---: | ---: | ---: |
| XS single | 15,168 | 2,080 | 13,088 | 1.255 |
| XS TP2 | 15,264 | 2,048 | 13,216 | 0.858 |
| S single | 15,216 | 2,400 | 12,816 | 3.195 |
| S TP2 | 14,832 | 2,336 | 12,496 | 0.720 |

For context only, the engine process's Linux `read_bytes` delta over the
request and adjacent measurement calls was 443.4 MiB for XS single versus
155.5 / 155.6 MiB for TP ranks 0 / 1; S single read 49,240.7 MiB versus
157.1 / 155.6 MiB for TP ranks 0 / 1. These are **whole-process physical
reads**, including but not isolated to Engram, and reflect different placements
and cache states. Corresponding reported generation rates were 35.8, 21.5,
13.8 and 22.3 tok/s respectively. They are single observed values, not a
controlled general speedup or a claim that disk I/O is eliminated.

All owned servers and workers exited. Raw metrics, counters, samples,
answers and timing records: [XS single](xs-single-util-engram.json),
[XS TP2](xs-tp-util-engram.json), [S single](s-single-util-engram.json),
[S TP2](s-tp-util-engram.json). Broader limitations and earlier repeated
measurements are in the [review report](README.md).
