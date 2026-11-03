# DeepSeek queued TP2 performance

2026-11-04, issue #120. The user requested focused correctness checks and
generation above the same single-host build before handing back testing.

## Change

DeepSeek reductions now follow GLM's QueuePartial/StagePartial/WaitValue path.
Router IDs and weights use one queued exchange per layer and a GPU bitwise
comparison over active picks. A sticky mapped disagreement flag is read after
the operation drains, with queued transport failures checked first. This
removes the previous per-layer host copies, comparisons and blocking exchanges.

## Conditions

Both modes used the same Release HIP gfx1151 executable, SHA-256
`a950e7e9350be4fa49a7063c67ef901ebbc1bd2a9b583d07250eb016bcf72102`,
DeepSeek Maya-XS, context 262144, prefill chunk 256, DSpark on, lookup off,
vision off and temperature 0. The single-host arm and TP2 API rank ran on
strix-halo-2; the TP2 worker ran on strix-halo-1 over usb4_rdma5. The user's
staged launcher retains strix-halo-1 as API rank.

The single-host arm used its existing config with the candidate executable,
14 CPU threads, reserve 1024 MiB, mmap experts and no expert cache. Its original
large default prefill buffers triggered the memory guard during loading, so
both arms were configured with 256-row buffers for the completed comparison.
Single-host retained threads on; the TP2 controller requires threads off.
TP2 used an expert budget of 92160 MiB per host and reserve 1024 MiB.
The saved engine configs record the exact arguments and environment.

Both arms reported all 15,360 routed experts resident and zero CPU experts.
The second measured repeat reported zero disk traffic for each prompt in
both modes. No prompt tokens were cached in any measured response.

## Results

Each prompt received a 256-token warmup, then two measured 512-token responses.
The table uses API generation timings in tokens per second.

| Prompt | Single repeat 1 | Single repeat 2 | TP2 repeat 1 | TP2 repeat 2 | Median gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python CSV module | 33.1 | 33.7 | 42.0 | 40.9 | 24.1% |
| eBPF tutorial | 28.0 | 28.2 | 35.2 | 33.9 | 23.0% |

The prompts were “Write a complete Python module that parses a CSV file into a
list of dictionaries, with type hints, detailed docstrings, error handling,
usage examples, and a unittest suite. Explain each part.” and “Write an eBPF
tutorial to explain how to use eBPF.” They encoded to 44 and 19 prompt tokens.

These are end-to-end DSpark comparisons. Single-host and TP2 responses and
draft acceptance differed, so the measured gain includes that effect.
The short prompts measure generation and provide limited prefill evidence.
The user's longer workloads remain part of ongoing review.

## User-reported Maya-S result

On 2026-11-04, the user reported Maya-S eBPF generation slightly above 30 tok/s,
prefill similar to XS and a larger relative speedup. This records their
llama-swap test of the staged candidate. Exact prefill rates, response length,
repeat count and a matched single-host baseline were not supplied, so no
Maya-S speedup percentage is calculated.

## Focused validation

- Release engine and focused test target built successfully.
- HIP name mapping check passed.
- GPU route comparison passed padding exclusion, active mismatch detection
  and retention of the first failure tag.
- All five `serve.test_tp2_peer_loss` tests passed.
- Paired XS DSpark off/on returned identical greedy 256-token text. Draft
  counters were zero with speculation off and positive with speculation on.
- Every measured TP2 generation rate exceeded its corresponding single-host rate.

The full server/tools suites belong to the earlier launch checkpoint. This
change follows the user's request to limit validation and hand back testing.
Both owned API servers and the owned remote worker exited after measurement.

## Evidence and staged runtime

The raw JSON files contain requests, responses, API timing and residency
metrics. The TP2 file also contains the DSpark off/on correctness responses.

- [Single-host results](single-results.json)
- [TP2 results](tp-results.json)
- [Single-host engine config](single-config.json)
- [TP2 engine config](tp-config.json)

Both hosts select the measured binary at
`/home/ugo/software/project-maya-ds4-tp2-review/build-hip/queued-a950e7e9350b/maya-ds4`.
The two existing aliases `maya-deepseek-xs-mtp` and `maya-deepseek-s-mtp` load
their distinct packs through this binary. Launcher config backups have suffix
`.before-queued-120-20261104`. The user launches through llama-swap on
strix-halo-1. Issue #120 is complete with the user's Maya-S feedback; #116
retains broader numerical/quality and resource review.
