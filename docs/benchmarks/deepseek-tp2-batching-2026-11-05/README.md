# DeepSeek TP2 batching — matched gfx1151 HTTP measurements

2026-11-05, [#153](https://github.com/uv-genai/maya-tp/issues/153),
[PR #155](https://github.com/uv-genai/maya-tp/pull/155).
Two Strix Halo HIP/ROCm hosts, tensor traffic over usb4_rdma5. All launches
use isolated source/configs/executables and the owned pair wrapper, initiated
on strix-halo-1.local. Production runtime files are unchanged.

The parent is `72d9260` (`maya-ds4` SHA256
`70891055ff14a722bfcd3f6fcccc2ff06e4f610e3c50a4eea7d277c12f525359`).
The candidate engine source is `109aefc`, SHA256
`ecac4769d8bd3d1d7cf1045e35191b343a012b5b24ec4c359e8604db10faf9f0`.
Both hosts use identical bytes per case. Parent capacity is one sequence;
candidate capacity is four, with actual concurrency varied below. Context is
262144 per slot, expert budget 92160 MiB, reserve 1024 MiB, 14 CPU threads,
256-row prefill, DSpark loaded on rank zero and lookup off in every case.

The same `tools/bench_concurrency.py` uses the detailed eBPF tutorial prompt,
96 output tokens, temperature zero, thinking off and fixed matched request
identifiers. `--spec 0` disables proposals per request without changing model
residency; `--spec 1` enables DeepSeek DSpark. There is one measured wave at
one/three clients, three at two/four, plus a warmup per group. Each mode/build
has 22 measured requests. Every healthy measured request succeeds. All 88
paired response comparisons across both models and modes match exactly.
This compares response texts, not arbitrary tokenizer encodings of that text.

Values are **aggregate end-to-end output tok/s**, including prompt time and
queueing. Two/four-client columns are medians of three waves; one/three are
single measured waves. Full per-request generation rates, TTFT, completion
latencies, proposal/acceptance counters, request bodies and responses are in
the raw JSON. Per-stream speed falls while streams share the GPU; aggregate
throughput and first-token latency improve.

| Model | Mode | Build | 1 client | 2 clients | 3 clients | 4 clients |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| xs | ordinary | parent | 20.65 | 20.89 | 20.61 | 20.7 |
| xs | dspark | parent | 25.13 | 25.26 | 25.03 | 25.13 |
| xs | ordinary | batch | 21.31 | 32.11 | 38.02 | 40.15 |
| xs | dspark | batch | 26.09 | 33.97 | 37.42 | 40.05 |
| s | ordinary | parent | 17.85 | 17.6 | 17.74 | 17.73 |
| s | dspark | parent | 21.18 | 22.27 | 21.73 | 22.44 |
| s | ordinary | batch | 18.63 | 27.76 | 33.69 | 35.74 |
| s | dspark | batch | 21.81 | 29.66 | 33.15 | 36.08 |

| Model/build | Lead peak GTT, GB | Peer peak GTT, GB | Lead/peer inference reads, MB |
| --- | ---: | ---: | ---: |
| xs/parent | 66.16 | 61.78 | 303.5/215.7 |
| xs/batch | 66.96 | 62.57 | 293.6/224.2 |
| s/parent | 100.63 | 92.58 | 270.4/252.1 |
| s/batch | 101.42 | 93.38 | 280.8/217.7 |

The two/four-client gains exceed the observed wave spreads on both models,
with or without DSpark. All measured one-client points improve, satisfying
the provisional 5% regression budget. The raw warmups provide two additional
matched one-client observations per mode/build; their medians also improve.
DSpark proposes and accepts tokens for every eligible measured request.

Memory counters sample global GPU GTT allocation every 0.2 seconds. Lead
sampling covers load and inference; peer sampling starts after READY. These
are observed allocation peaks, include other fixed display/driver allocations,
and are not exclusive per-process physical-memory measurements. Model logs
confirm complete routed-expert residency on both ranks. Reads are per-engine
`/proc/PID/io` deltas after READY and include warmups; they are not classified
by file. Resident routed experts do not imply all Engram/model I/O is absent.
Raw counters, ownership identities, launcher settings and engine logs are kept.

The XS candidate additionally streams four requests, deliberately kills its
recorded worker, and receives the identical first fatal cause on all four
HTTP routes within 0.924 seconds. Health reports degraded/peer down with that
cause, and the owned pair shuts down. This expected negative test follows
all timed measurements. S additionally passes three staggered clients with
two requests each, short output caps and two natural EOS completions; the
raw refill request bodies/results are included.

Reproduce with the isolated pair wrapper and the commands recorded in each
`ownership.json`. Run `tools/bench_concurrency.py` with `--concurrency 1 3
--rounds 1` and separately `--concurrency 2 4 --rounds 3`, `--max-tokens 96`,
`--run-id ds4-xs-matched` (or `ds4-s-matched`) and `--spec 0`/`1`. Keep model,
context, engine, tuning, residency and prompt/identifier choices matched.
Use an owned dedicated API/bootstrap port and stop that supervisor only.

These are short-context, 96-token text workloads. They do not establish
filled-context throughput, arbitrary sustained concurrency, answer quality,
vision support or completed operational user acceptance. New batching remains
opt-in; the accepted deployed single-sequence runtime is unchanged. See the
[D5 validation report](../../validation/tp2-batching-d5-2026-11-05.md) for the
lifecycle, default-mode and CI gates and their scoped limits.
