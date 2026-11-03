# DeepSeek TP2 review measurements, 2026-11-03

Experimental work under #116; this is not a release or production promotion.
For eight short objective cross-path checks on both models, see the
[sanity report](../../validation/deepseek-tp2-control-2026-11-03/cross-path-sanity.md).

## Filled-context functional check

An isolated Maya-S pair at checkpoint `a34d616` used context 524,288, 256-row
prefill, DSpark on and lookup off. A fresh 32,772-token prompt generated 32
tokens and accepted 15 of 18 drafts. A fresh 524,004-token prompt also
generated 32 tokens and accepted 12 of 13 drafts. Its recorded prefill was
136.1 tokens/s and generation 19.9 tokens/s, but host tests and an isolated
build ran concurrently, so these are functional-check timings, not a clean
performance comparison. The later sampled-lookup fix was not in this build.

For the sampled interval from 23,552 to 522,496 prompt tokens, process
`read_bytes` rose 15.50 GiB on rank 0 and 15.04 GiB on rank 1. These are all
physical reads attributed to the engine processes, not an Engram-only or
complete whole-request measurement. Minimum sampled host `MemAvailable` was
21.1 and 29.4 GiB respectively. Device-wide RDMA counters rose approximately
193 GiB in each direction on each host; those counters are not per-process.
The sampler and both owned ranks exited. Raw [request results](s-tp-long-prefill.json)
and [telemetry](s-tp-long-telemetry.jsonl) retain the measurements.

## Cold and warmer-cache startup

An isolated XS pair at context 32,768 used the same engine (SHA-256
`bf1adc9fcf81f5478f5774b1fb961e3c950d0816fd28fa055128c70c548236f4`), DSpark,
256-row prefill, lookup off and the same 256-token Python request for two
consecutive launches. Before the first, `POSIX_FADV_DONTNEED` was applied only
to the manifest-verified model source files on each host. A `mincore` sample
found 0/448 resident pages on each host afterward, down from 93/448 and 95/448.
These samples do not prove every page was evicted. After the first pair exited,
92/448 and 102/448 sampled pages were resident; the second run therefore had a
warmer, but far from fully resident, cache.

| XS pair | Pair startup | Rank 0 process reads by ready | Rank 1 process reads by ready | Decode |
| --- | ---: | ---: | ---: | ---: |
| Cold sample | 46.17 s | 112.54 GiB | 108.53 GiB | 25.4 tok/s |
| Warmer sample | 46.67 s | 95.83 GiB | 108.45 GiB | 26.3 tok/s |

The answers matched exactly, with 178/213 DSpark drafts accepted in each run.
The startup interval starts before worker launch and ends when health reports
the pair ready. Process `read_bytes` covers all physical reads charged to each
engine, not only model shards or Engram, and request-time reads were another
0.15–0.18 GiB per rank. Each condition has one run. The result is not proof
that a warmer cache cannot improve other workloads or that model I/O is gone.
All owned processes exited. [Raw XS results](xs-tp-coldwarm.json),
[cold worker log](xs-tp-cold-worker.txt), [warmer worker log](xs-tp-warm-worker.txt).

The same check on Maya-S found 0/512 sampled pages resident on either host
after targeted eviction; after the cold pair exited, 22/512 and 15/512 sampled
pages were resident. Cold and second-launch startup were 91.30 and 67.20 s.
Process reads by ready were 174.30/166.03 GiB and 173.47/165.94 GiB for
ranks 0/1 respectively. The answers matched, with 185/211 drafts accepted
each time; decode was 20.8 and 21.9 tok/s. The second launch's shorter time
does not establish why it was faster, especially with so little sampled cache
resident and nearly unchanged physical reads. Both pairs exited. [Raw S results](s-tp-coldwarm.json),
[cold worker log](s-tp-cold-worker.txt), [warmer worker log](s-tp-warm-worker.txt).

For a guarded XS single-host control with the same model, prompt, sampling,
context, engine and DSpark setting, the reserve was 8,192 MiB and expert
read-ahead was capped at zero to preserve memory headroom. Startup took 25.03
and 23.03 s; process reads by ready were 110.71 and 110.47 GiB, with another
0.54–0.55 GiB during each request. Decode was 35.9 and 36.0 tok/s, and both
single-host answers matched with 175/208 drafts accepted. After the first
process exited, 0/448 sampled source pages were resident, so the second run
is not a fully warm-cache measurement. Its answer differed from the paired
answer despite a matching opening. The single-host and pair configurations
also differ in expert placement and reserve, so this is not an isolated TP
overhead measurement. The single-host engine reported 14,784 GPU expert slots
and 576 CPU experts. [Raw single-host XS results](xs-single-coldwarm.json).

The same guarded single-host control on S took 27.05 and 27.03 s to start,
with 129.90 and 129.33 GiB of engine process reads by ready. The 256-token
requests incurred another 48.83 and 45.78 GiB of process reads, respectively,
and decoded at 15.3 and 16.0 tok/s. Answers matched between these two S
single-host runs, with 185/219 drafts accepted each time; their answer differed
from the paired answer. Targeted eviction left 0/512 sampled pages resident
before the first start; after it exited, 17/512 were resident, so the second
start was not a fully warm-cache test. In the paired S runs above, each rank
incurred approximately 0.18–0.19 GiB of process reads during the same request.
The different reserve, placement and memory pressure are part of this
comparison; the counters do not identify which specific data was read. Both
owned single-host processes exited. The single-host engine reported 9,063 GPU
expert slots and 6,297 CPU experts. [Raw single-host S results](s-single-coldwarm.json).

## Earlier XS and S comparisons

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
samplers exited. Later same-binary and same-build comparisons follow below;
the filled-context functional check is above. Fuller quality review remains.

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
samplers exited.

## Same-binary Maya-XS DSpark comparison

The same rebuilt engine, context and lookup-off settings were used on XS.
All nine generated answers and all six prefill prompt hashes matched between
DSpark on and off. The first off repetition was slower than later ones.

| Measurement (tok/s) | DSpark off | DSpark on |
| --- | --- | --- |
| Python decode | 17.1, 22.7, 23.0 | 26.0, 27.9, 27.8 |
| Prose decode | 19.4, 23.0, 23.4 | 26.6, 27.2, 27.1 |
| HTML decode | 19.3, 22.5, 22.7 | 24.6, 29.1, 35.7 |
| 2K fresh prefill | 205.7, 208.5, 201.1 | 195.5, 205.6, 203.6 |
| 8K fresh prefill | 200.0, 203.3, 204.0 | 200.3, 199.3, 201.6 |

Raw data: [off decode](xs-tp-off-reference.json), [on decode](xs-tp-on-reference.json),
[off prefill](xs-tp-off-prefill.json), [on prefill](xs-tp-on-prefill.json),
[off telemetry](xs-tp-off-telemetry.jsonl) and [on telemetry](xs-tp-on-telemetry.jsonl).
Both owned pairs and samplers exited. These are bounded greedy checks; the
filled-context functional check is above and same-build one-host comparisons follow.

## Same-build XS single-host control

One repetition of each fixed prompt used the same rebuilt engine and context,
with lookup, vision and flow off, 256-row configured prefill, and a copied
temporary Lt store. This is a controlled comparison, not the fastest default
single-host setup. Single-host DSpark on/off produced identical answers for
all three prompts; each answer differed from its TP2 counterpart. The
single-host and TP2 answer openings were coherent, but the 256-token budget
truncated the outputs and does not establish quality equivalence.

| Measurement (tok/s, one run) | Single-host off | Single-host on |
| --- | --- | --- |
| Python decode | 20.3 | 38.0 |
| Prose decode | 20.3 | 28.9 |
| HTML decode | 20.3 | 45.6 |
| 2K fresh prefill | 176.2 | 174.8 |
| 8K fresh prefill | 182.0 | 181.7 |

Raw data: [on answers](xs-single-on-reference.json), [off answers](xs-single-off-reference.json),
[on prefill](xs-single-on-prefill.json), [off prefill](xs-single-off-prefill.json),
[on telemetry](xs-single-on-telemetry.jsonl) and [off telemetry](xs-single-off-telemetry.jsonl).
The engine's single-host chunk planner can choose different boundaries than
the TP2 planner even with the same configured cap. Both servers and samplers
exited.

## Safety-constrained Maya-S single-host control

The initial controlled S startup planned 62.1 GB of host expert read-ahead
and was stopped when available memory fell to roughly 127 MB; memory recovered.
The separate issue [#117](https://github.com/uv-genai/maya-tp/issues/117) tracks
that behavior. A retry set `MAYA_DS4_WARM_CAP_GB=0`, an 8192 MiB reserve and
a 3 GiB available-memory guard, with flow, vision and lookup off. Its 2K/8K
prefill ran at 10.6/10.1 tok/s, far below the earlier normal single-host
reference. This is a safety-constrained diagnostic, not the optimized setup.

| Decode (tok/s, one run) | Single-host off | Single-host on | TP2 off | TP2 on |
| --- | --- | --- | --- | --- |
| Python | 10.1 | 15.7 | 17.6–19.7 | 22.7–26.6 |
| Prose | 7.5 | 8.9 | 18.0–19.8 | 21.8–27.3 |
| HTML | 8.0 | 10.9 | 19.2–19.9 | 27.1–30.4 |

TP2 values are ranges across the three lookup-off runs above; S single-host
values are one sample, with a larger reserve and no host expert warm-up.
The single-host on/off runs produced different answers for all three prompts;
they ran in separate processes with different drafter allocations and expert
placement, so this does not isolate speculative decoding. A same-process
parity check then kept one S engine and its placement loaded, forced fresh
19-token and 545-token eBPF prompts, and compared 64 greedy tokens with
`spec=0` and `spec=1`. Both sequences matched exactly; the DSpark requests
accepted 38/47 and 30/35 drafts respectively. This limits the concern for
those two prompts but does not prove the separate-launch differences were
caused by placement. [Token results](../../validation/deepseek-tp2-control-2026-11-03/s-single-spec-parity.json)
are retained; logs use `/tmp/ds4-tp2-113-s-single-spec-parity-*`.
Raw data: [on answers](s-single-safe-on-reference.json),
[off answers](s-single-safe-off-reference.json), [on prefill](s-single-safe-on-prefill.json),
[on telemetry](s-single-safe-on-telemetry.jsonl) and [off telemetry](s-single-safe-off-telemetry.jsonl).
The off run did not repeat prefill. Both owned servers and samplers exited.
