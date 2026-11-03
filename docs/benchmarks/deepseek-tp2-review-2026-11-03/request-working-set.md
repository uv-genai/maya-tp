# Repeated-request cache observations

Issues #112 and #116. XS and S each ran three identical greedy requests in a
single loaded process, on single-host and TP2 configurations. The engine
SHA-256 was `a0dcc1cb91e9e52fa365291d68eef8203af714d8222e77e49d72741a4331ddc2`.
Each configuration used a 32,768-token context, DSpark on, lookup and vision
off, and a 256-token output limit. The prompt requested a typed CSV parser
with a docstring. A separate short greeting request preceded the measurements.
The first measured request therefore has an unspecified source-cache state.

Single-host reserved 8 GiB and capped expert read-ahead at zero. TP2 used
rank-sharded placement and 256-row prefill. Each configuration returned
identical answer text on all three measured requests. Answers can differ
between configurations. The engine monitor snapshots and Linux process
`read_bytes` were captured before and after each request.

| Model / path | Generation tok/s, requests 1 / 2 / 3 | New rank-0 logical Engram reads, requests 1 / 2 / 3 | Prompt tokens reused on requests 2 / 3 |
| --- | --- | --- | ---: |
| XS single | 36.1 / 36.1 / 36.7 | 13,088 / 0 / 0 | 27 / 27 |
| XS TP2 | 24.2 / 30.3 / 28.0 | 13,216 / 0 / 0 | 0 / 0 |
| S single | 15.5 / 14.0 / 15.4 | 12,816 / 0 / 0 | 27 / 27 |
| S TP2 | 21.8 / 24.4 / 28.0 | 12,496 / 0 / 0 | 0 / 0 |

| Model / path | Physical process reads MiB, request 1 | Request 2 | Request 3 |
| --- | --- | --- | --- |
| XS single | 474.73 | 36.76 | 208.46 |
| XS TP2, rank 0 / rank 1 | 153.02 / 155.60 | 0 / 0 | 0 / 0 |
| S single | 50,054.56 | 39,189.66 | 38,899.31 |
| S TP2, rank 0 / rank 1 | 157.20 / 155.41 | 0 / 0 | 0 / 0 |

The two repeated requests on each TP2 pair had zero new rank-0 logical Engram
reads and zero recorded physical-read bytes on both ranks, while generating
the same answers. This establishes a warmed request working set for those
observations. The single-host processes continued physical reads while their
logical Engram misses were zero. Physical counters cover the whole engine
process, and rank-1 logical Engram counters are unavailable through this monitor.

Prompt reuse differed between single-host and TP2. The request histories,
placement and reserve settings also form part of this comparison. Sampling
GPU busy through sysfs and SSH added monitoring overhead. The timings cover
three requests per configuration, with two warmed repetitions of one workload.
Startup-cache residency, arbitrary new prompts, larger contexts and peak-budget
validation retain their separate scope.

All owned servers and workers exited. Raw requests, answers, metrics, samples
and counter snapshots are retained below. Duplicate monitor history arrays were
omitted. [XS single](xs-single-requestwarm.json),
[XS TP2](xs-tp-requestwarm.json), [S single](s-single-requestwarm.json),
and [S TP2](s-tp-requestwarm.json). Earlier startup and context measurements
are in the [review report](README.md).
