# TP2 MTP on/off benchmark — 2026-10-30

> Historical measurement: this run predates the MTP attention and collective fixes in v1.1.38.
> It does not establish the MTP performance of v1.1.38; repeat the controlled comparison for current timings.

Issue [#14](https://github.com/uv-genai/maya-tp/issues/14), engine `5b547ce`, binary MD5
`75a180099201dbc0abdf29808a536022` verified on both hosts. No engine changes.

On this fixed continuation, MTP reduces throughput by **26.76%**
and increases latency per token by **36.55%**. The simple loop remains the default.

| Mode | Median ms/token | Tokens/s (reciprocal of median latency) | Run range, ms/token |
| --- | ---: | ---: | ---: |
| MTP off | 38.255 | 26.14 | 38.163–38.301 |
| MTP on, 2 drafts | 52.235 | 19.14 | 52.169–52.302 |

## Method

- Two gfx1151 Strix Halo APUs, HIP/ROCm Linux, Maya-S-v2 IQ2_XXS pack; tensor transport is Thunderbolt RDMA
  `usb4_rdma5`. Rank 0 strix-halo-1, rank 1 strix-halo-2. Context 4096, one sequence, greedy.
- Same resident pair, MTP loaded in **both** modes; only `GEN spec=1` (off) versus `spec=0` (on) changes.
  MTP opt-in `MAYA_TP_MTP=1`, fixed draft width 2. No engine reloads between warm or measured requests.
- Generate a 256-token reference with the simple TP2 loop, then load it using `--spec-follow` for every request.
  Exact prompt IDs: `154822,25062,287,29905,371,25,7487,785,6722,315,9621,374` (12 tokens).
- Four excluded warmup requests: off/on/on/off. Four measured requests per mode (256 output tokens each),
  balanced order **off/on/on/off/off/on/on/off**. Statistic: engine `DONE.decode_ms / 256`.
  Model loading, prefill and warmups are excluded from decode timing; timing covers the emitted first token too.
- All eight measured runs emit the exact reference (SHA256 `6cde8783f6a05cb51b3feaf63fb6f9070ebfeb0f44d01320cf02aeff6337d738`).
  Snapshot reuse is enabled on both paths; measured requests reuse 11 prompt tokens.
- All measured runs report **100% VRAM hit**, **zero disk reads/waits/time**, and no CPU inference lane.
  No unit tests or profilers were run alongside the timing.
- Fixed-length microbenchmark: EOS stopping disabled using unreachable EOS ID 2147483647, loop guards disabled.
  The reference crosses natural turn boundaries at zero-based positions 51, 125 and 187. This measures this exact
  token workload; it is not a claim about every natural chat prompt or acceptance rate.

## Why MTP loses here

Every measured MTP request accepts **87/335 drafts (25.97%)**, requiring **168 verification rounds**,
only **1.52 output tokens per round**. Per round: drafts ~9.95 ms, verify ~69.56 ms, commit/cache ~0.09 ms.
That work costs more than advancing those output tokens through the simple loop. No default was changed.

The excluded first simple warmup took 75.471 ms/token, including 36.91 ms/token of disk time;
its second warmup was 37.804 ms/token with zero disk time. MTP warmups were 57.081 then 52.053 ms/token.
Keeping the pair resident removes this cold-I/O confound from the measured repetitions.

## Reproduce

From strix-halo-2, with matching binaries on both hosts:

```sh
.venv/bin/python tools/tp2/bench_mtp.py --output /tmp/maya-mtp-comparison --tokens 256 --runs 4
```

Use a fresh output directory. The harness stops existing Maya engines and the remote server, generates a reference,
saves all results and rank logs, and stops the pair afterward. Re-running regenerates the reference; check its hash
when comparing results across sessions. The MTP allocations stay loaded during the off mode for a fair switch.

[Raw results including warmups](results.json) · [Exact reference token IDs](reference.txt)

Original detailed logs remain under `/tmp/maya-mtp-bench-14/` on strix-halo-2.
