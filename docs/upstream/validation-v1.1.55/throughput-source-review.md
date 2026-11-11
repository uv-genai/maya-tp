# #165 read-only source/evidence audit

2026-11-11. Source checkpoint cc5ea59 versus v54 ad7f71f. No new GPU processes, source edits or unrelated-service inspection performed.

## What the recorded S2 evidence establishes

Files: `/tmp/maya-upstream-20261111-validation/{v1.1.54,v1.1.55}-ds4-s-slots2/results.json` and their rank logs.

| Case | v54 prompt/decode ms | v55 prompt/decode ms | v55 proposals accepted/offered |
| --- | --- | --- | --- |
| short-off | 754.1 / 2998.8 | 1205.0 / 5405.4 | 0/0 |
| short-on, identical prompt/output | 454.7 / 2134.7 | 455.1 / 2766.5 | 34/44 |
| prefill-off | 6617.3 / 3116.2 | 11729.4 / 10091.5 | 0/0 |
| prefill-on, identical prompt/output | 6377.9 / 1965.5 | 6575.9 / 2033.2 | 36/37 |

The large slowdown is already present with one active ordinary request in a two-slot server, before mixed admission. It is not explained only by different queue order. The later normal DSpark case in the same process also prevents interpreting it as a uniform whole-run slowdown. Mixed arrivals differ in admitted order, which affects TTFT, but all four 64-token clients and proposal counts are equal; aggregate makespan still slows from ~8.89s to21.83s.

All ordinary 64-token cases report hits=lookups=15120. Startup records84509982720 routed expert bytes resident on each rank,0 CPU-lane experts, and rank0~24.2GiB free. Thus routed-expert misses/CPU computation do not fit the recorded slowdown. They do NOT rule out Engram reads, CPU scheduling, memory bandwidth/power or transport latency.

v54 XS's slow mixed wave followed by a normal warmed repeat, and v55 XS2's normal mixed wave, are independent evidence that a fixed v55 arithmetic regression has not been established. Exact output/proposals and clean peer recovery are correctness evidence, not latency attribution.

## Source review

The v54..v55 DS4 production changes are the opt-in AI_RUN prerequisite (`Chunk.AIM`, iq/bm allocations and prompt slab widths), feature identity, a comment-preserving LOAD_INTERLEAVE default, and an NVIDIA-only AR2 default. AI_RUN-off assigns AIM=AI and retains original scores/strides/slabs. HIP AR2 stays0. No DS4 batch scheduling, Engram reader, slot/control ACK, RDMA worker or GPU wait source changed in this interval.

Relevant unchanged boundaries:
- `src/ds4/ds4_main.cpp`: the scheduler rotates eligible rows, emits buffered DSpark tokens, and interleaves prompt pieces. Different client arrival order changes which two requests share rows and waiting time; it cannot by itself explain the earlier serialized ordinary slowdown.
- `src/ds4/ds4_model.cu::step_runs`: each multi window issues Engram reads, plans/runs layers, finishes reads, then synchronizes the head stream. `tp_batch_rows`/`tp2_commit_multi` require full paired acceptance/completion; `tp_worker_synchronize` drains the device and checks queued collectives before ACK. Keep all these correctness boundaries.
- Engram remains file-backed unless DS4_ENGRAM_RAM is explicitly enabled; no such mode is in these run envs/logs. Each token does48 logical lookups over two layers. `eg_issue` uses a host row cache and submits misses to EgReader; `eg_finish` waits, copies/cache-fills and updates timing. On Linux reader threads are32-ish by hardware count (16..64), each owning file handles. New request/topic and cold page-cache states can change reads even with all routed experts resident.
- `warm_after` is an expert-page-cache prefetcher restricted to MXFP4 file experts, and skips already-resident slots. It does not provide a strong candidate for these fully resident native-GGUF TP2 routed experts.
- `src/net/verbs.cpp`: control sockets have TCP_NODELAY. Each small RecvControl spins500us then blocks; RDMA worker waits for GPU-staged partials then CQ completion (5ms initial spin, completion-channel wait/fallback poll). `src/net/tp_kernels.cu::WaitValueKernel` waits on host arrival flags. Host worker delays can extend GPU waits without causing transport failure. GPU busy alone does not separate useful kernels from GPU flag waits.

## Minimal diagnostics, in priority order

1. Continue the currently authorized matched original-prefix + wave replay with no toggles, as validator is doing. Preserve per-case wall/first-token times, request IDs and GPU clocks/power/busy on both hosts; owned thread CPU time/context-switch/blocked/runnable counters plus per-process read_bytes/faults; memory/IO pressure.
2. Persist existing `StrataEngine.stat_history` at each prefix case/wave (or timestamped `engine.stat` samples). `serve/server.py::_parse_stat` already records120 entries. DS4 STAT exposes engram_ms, eg_lookup_rows/hits/reads, eg_prefetch_wait/io, disk_mb, draft_ms/verify_ms, vram_hit and CPU counts. Snapshot per-wave before the120-entry ring drops initial evidence. This needs only test-harness collection, no engine change.
3. Attribute before testing another mode: high Engram wait+read/IO deltas supports reader/page-cache investigation; low Engram wait but low clock/high pressure supports APU resource contention; normal compute clocks with long owned RDMA-worker waits supports transport/host scheduling. Lack of transport errors alone does not eliminate transport latency.

Avoid misleading existing debug options: DS4_ROUND_LOG explicitly omits --parallel multi rounds, so it does not measure these stalls. DS4_PROF profiles T==1 windows and is disabled for multi-row step_runs; it will not provide complete batch breakdown. Do not disable ACKs/synchronization, change thread counts, flush caches, pin entire Engram tables or alter production configs as an unmeasured fix.
