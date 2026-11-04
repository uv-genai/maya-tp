# DeepSeek Maya-S concurrent API benchmark

Issue #139. Measured on 2026-11-04 against the user’s running TP2/DSpark pair through `http://strix-halo-1.local:8080/v1`, using the exact llama-swap alias `maya-deepseek-s-mtp`.

## Workload

One 256-token warmup preceded two measured waves at each concurrency level, in order 1, 2, 3, 4. Each wave released all clients through a barrier. The prompt requested a detailed eBPF tutorial, with a fresh request identifier near the start. Requests used streaming, temperature 0, top_p 1, thinking disabled and a 256-token output cap.

All 20 measured requests completed exactly 256 tokens with finish reason `length`, giving 5,120 output tokens. Every prompt contained 89 tokens, with zero cached prompt tokens throughout. DSpark offered drafts in every measured request and accepted 2,741/3,484 drafts overall. The warmup is excluded from the table.

## Results

| Concurrent requests | Measured requests | Aggregate output tok/s | Median first token, s | Median completion, s | Median server decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 27.93 | 0.74 | 9.17 | 30.35 |
| 2 | 4 | 27.23 | 5.30 | 14.09 | 29.20 |
| 3 | 6 | 27.37 | 10.22 | 18.60 | 29.65 |
| 4 | 8 | 27.94 | 14.27 | 22.85 | 30.30 |

Aggregate throughput is total successful completion tokens divided by the sum of the two complete wave durations. First-token and completion latencies are client-side medians across that level’s requests, including queueing. Server decode rates come from each response’s own `timings`.

The current TP2 setup serves one active sequence. Additional API requests wait in the queue. The per-wave first-token timings show this ordering, and aggregate throughput stays around 28 tok/s as concurrency rises. Active generation stays around 29–30 tok/s.

These are two short-prompt burst waves per level. The 256-token cap produces partial tutorials. The workload establishes bounded queueing performance on this running deployment. Long-context and sustained-load performance require separate workloads.

## Reproduce

Run against an already running model and choose a fresh output path.

```sh
.venv/bin/python tools/bench_concurrency.py \
  --url http://strix-halo-1.local:8080/v1 \
  --model maya-deepseek-s-mtp \
  --concurrency 1 2 3 4 --rounds 2 --max-tokens 256 \
  --output /tmp/maya-deepseek-s-concurrency.json
```

The tool accepts `--prompt-file`, `--timeout` and `--api-key`. Its output captures the warmup, request bodies, generated responses, errors, per-request usage/timings and per-wave summaries. It saves completed waves incrementally and stops after a failed wave.

Four focused regressions passed for streamed usage/first-token accounting, incomplete/error streams, synchronized client release and aggregate throughput calculation.

[Benchmark script](../../../tools/bench_concurrency.py) · [Raw requests, responses and timings](results.json)
