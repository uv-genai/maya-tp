# DeepSeek TP2 P1: initial pack and RDMA evidence

2026-11-03, issue #112. Preliminary evidence; P1 is not complete and the distribution decision has
not yet passed its capacity and parity gates. Measurements use the existing single-host DeepSeek
binary and an isolated build of the existing two-host transport test, not a DeepSeek TP2 engine.

Both hosts are gfx1151 Strix Halo, each with ~129.5 million KiB physical RAM, and usb4_rdma5 was
active. `--pack-info` reports the following decimal GB (pack inventory, not resident allocations):

| | Maya-XS | Maya-S |
| --- | ---: | ---: |
| Trunk routed experts | 109.57 | 169.02 |
| Drafter experts | 3.50 | 7.22 |
| Dense and head | 6.72 | 8.46 |
| Drafter blocks, excluding expert weights | 0.75 | 0.75 |
| File-backed Engram tables | 161.28 | 161.28 |

Both packs report 40 layers, 384 experts per layer, six routed per token, width 2304 and three
DSpark blocks tapping layers 37–39. Their expert quant types differ; see the raw pack reports.
Rotating 5+4 quant blocks across equally sized layers suggests a rough half of routed expert bytes
per rank, but the actual per-layer formats, rank placement, scratch, context, OS headroom and
Engram cache must be measured. The complete Engram data and all model memory are not shown to fit.

`tools/ds4/estimate_tp2_experts.py` uses each indexed trunk expert tensor's actual byte count and
rotates a 5+4 split by layer, assuming 256-value blocks and proportional tensor bytes. It reports
Maya-S 84.510 GB per rank (169.020 GB total) and Maya-XS 54.814/54.751 GB (109.566 GB total).
This is a logical-data estimate, not a tested loader: verify each quant format's block/layout and
include padding and all non-expert allocations before treating it as a capacity result. Synthetic
checks covered even rotated totals and rejection of a missing tensor.

An isolated `tp_comm_rdma_test` was built from commit `231a48b` source using HIP gfx1151 and
`STRATA_ENABLE_TP2_RDMA=ON`. Both hosts ran the same executable with `--hidden 5120 --exchanges 100
--bench` on port 18672; see the rank logs. Both reported success. For a 20,480-byte row, queued
all-reduce p50 was 39.9 µs and p99 ~44.5–44.7 µs; 48 MiB blocking throughput was 27.37 Gb/s
one-way. Eight rows showed p50 ~76.5 µs and p99 ~950 µs. These are one-run transport results, not
model latency; the ~40 µs single-row measurement makes an illustrative 80-reduction serial
budget about 3.2 ms/token before model compute or other synchronization. Overlap and actual
schedules must be measured; numbers do not predict DeepSeek tokens/s.

An owned rsync is copying Maya-XS plus shared Engram, DSpark and vision files to the previously
absent matching models directory on strix-halo-1. It is a preparation step, not verified pack
integrity or an inference result. Single-host comparisons must wait until this disk/network load
ends. User-owned llama-swap daemons were not stopped. Long-running copy details are in the handoff.

`tools/ds4/bench_tp2_reference.py` is the reference API harness: it requires an explicit server,
model label, DSpark state and new output path, and checks the server is idle before each request.
It records prompt/answer hashes, timings, usage and engine/tier snapshots. Its three fixed short
prompts are not a substitute for the planned long-prefill or paired-model measurements. A mock
HTTP check exercised sample capture and rejection of an existing output and DSpark mismatch.

After the copy completed, an owned isolated Maya-XS server with the review binary, 262144 context,
DSpark on and profile writes disabled ran three repetitions of each reference prompt. The nine
samples are in `xs-single-reference.json`. Decode rates in tokens/s were Python 35.7/39.9/41.0,
prose 26.0/29.2/30.0 and HTML 42.8/46.9/47.3. Python and HTML answer hashes matched across repeats;
prose had two distinct hashes. These are warm-up/short-prompt samples, not a cold-load study,
long-prefill result or bit-exact determinism guarantee. The owned server and engine were stopped
after the run, without touching user-owned services.
