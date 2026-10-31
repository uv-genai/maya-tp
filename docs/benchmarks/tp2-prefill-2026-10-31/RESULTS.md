# TP2 prefill: sliced projection calibration and worker configuration

Issue: [#33](https://github.com/uv-genai/maya-tp/issues/33). Two gfx1151 Linux hosts over
Thunderbolt RDMA. **Maya-S median fresh-prompt throughput improves 31.9–37.2%** at approximately
1K/4K/8K tokens. MTP remains fixed at two drafts.

## Change and cause

`rank1.sh` did not apply the base config's `env`, while the rank0 server did. The worker used
256-row mixer sub-batches instead of the configured 1024 rows and did not load the hipBLASLt
calibration. Applying the same settings alone improved these samples by approximately 4–6%.

The existing gfx1151 table also omitted six sliced TP2 projection shapes: KDA projections/output,
shared expert gate/up/down, dense lead FFN, and DSA query projection. Those GEMMs fell back to
hipBLASGemmEx. Eighteen measured entries cover six shapes at T=64/256/1024 for hipBLASLt **100401**.
Other library versions are unchanged. The full-model comparison includes both fixes.

The initial issue description misread the stage log. A “copy lane” span includes idle mixer windows:

```text
1052 tokens, wall 4782 ms
copy lane 4380 ms: actual copies 124, ring wait 1072, mixer windows 3182
```

Detailed baseline rank1 event profile: KDA projections 30.4%, resident experts 15.7%,
mHC plus mixer-collective wait 12.8%, DSA projections 10.0%, staged experts 4.0%.
The transport and expert copy code are unchanged.

## Controlled Maya-S comparison

| Prompt tokens | Baseline median ms | Fixed median ms | Baseline tok/s | Fixed tok/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|
| 990 | 4732.35 | 3587.95 | 209.2 | 275.9 | +31.9% |
| 4062 | 16189.25 | 12015.75 | 250.9 | 338.1 | +34.7% |
| 8158 | 31416.20 | 22897.25 | 259.7 | 356.3 | +37.2% |

These are 24–27% reductions in prompt time. They are synthetic prompt-processing measurements;
they do not establish the gain on the user's exact 11th request or long-context workload.

Method:

- Same pack (`Maya-S-v2-IQ2_XXS`), same deployed binary MD5
  `5a52cd27167523b6ea4b0f0bde93f503`, same two hosts, context 32768, one sequence, greedy,
  MTP fixed/cap 2. No engine rebuild. Baseline source/launchers: `5067ed9`.
- Base JSON: prefill auto, KV streaming on, 6144 MiB prompt budget, 1024-row configured sub-batches,
  16 GiB RAM headroom, 1024 MiB reserve. The old worker ignored these config env entries.
- Identical API request order: 990/4062/8158 tokens, then repeat. Every request has zero reuse,
  an early distinct case number, a repeated-word system document, and the same framing question.
  Eight generated tokens keep decode short; its time is excluded from prompt metrics.
- All experts loaded before serving. Startup excluded; no request warmups excluded. Two samples per
  length; arm order baseline → matched settings → tuned. Arms restart both ranks. This is a small
  controlled comparison, not an estimate of run-to-run confidence. No background CPU suites during timing.
- `MAYA_GLM_PREFILL_PROF=1` for both baseline and fixed runs; default stage lines enabled.
  Fixed run also enables hipBLASLt verbose logging to verify dispatch.
- [baseline-api.json](baseline-api.json), [matched-settings-api.json](matched-settings-api.json),
  [tuned-api.json](tuned-api.json) contain every API response. Full engine logs are
  [baseline-rank0.txt](baseline-rank0.txt), [baseline-rank1.txt](baseline-rank1.txt),
  [tuned-rank0.txt](tuned-rank0.txt), [tuned-rank1.txt](tuned-rank1.txt).
  The fixed logs include subsequent correctness checks, so their aggregate phase totals are not
  directly comparable to the six-request baseline profile.

All six short greedy assistant messages match the baseline, including reasoning content. GEMM
accumulation order changes; general bit-exact outputs are not claimed. The eight-token response
comparison is a smoke check, not a quality evaluation or a raw-pick audit of long generation.

## Calibration and reproduction

[calibration.json](calibration.json) records measured matrix times and accuracy;
[calibration-full.log.gz](calibration-full.log.gz) contains the full tuner output. All 18 selected
kernels produce finite results, write no padding, and meet relative L2 ≤1.91e-5 against hipBLASGemmEx
(gate 1e-4). Matrix speedups span **1.85–7.25×**; these are not whole-model speedups.

```sh
cmake --build build-hip --target tune_hipblaslt -j 4
build-hip/tune_hipblaslt \
  --shapes-file docs/benchmarks/tp2-prefill-2026-10-31/shapes.json \
  --tokens 64,256,1024 --tuning-out /tmp/tp2-calibration.txt
```

Run the API probe from rank1 against an idle matching pair. Select the same pack and config on
both ranks, start rank1 first, set context 32768 and MTP fixed/cap 2. Restart both ranks between arms
and save their logs. The probe does not manage processes or use the llama-swap wrapper.

```sh
.venv/bin/python tools/tp2/bench_prefill.py --output /tmp/prefill-arm.json
```

## Validation

- Launcher tests: configuration reaches rank1, matches server override precedence, preserves literal
  values with whitespace/newlines, and does not evaluate shell substitutions. Existing launcher tests pass.
- Maya-S MTP acceptance PASS: three-turn reuse, thinking-budget forcing, cancellation/follow-up,
  output limits 1/2/3, peer up. [Log](mtp-acceptance.txt).
- Maya-S MTP collective matrix PASS: 990/7290/14550 prompt tokens, follow-up with 7289 cached tokens,
  streamed DONE, healthy TP2 peer. [Responses](regression-mtp.json).
- Maya-M and Maya-L fresh 990/4062/8158-token smoke probes PASS. These are single samples with no
  matched baseline for those packs, so they establish functionality, not a comparative speedup:

  | Pack | 990 tokens | 4062 tokens | 8158 tokens |
  |---|---:|---:|---:|
  | Maya-M | 200.1 tok/s | 299.2 tok/s | 339.6 tok/s |
  | Maya-L | 25.6 tok/s | 192.5 tok/s | 344.6 tok/s |

  [M responses](maya-m-api.json), [L responses](maya-l-api.json),
  [validation transcript](model-validation.txt). The first Maya-L request took 38.6 seconds;
  the main prefill chunk itself took 9.56 seconds. There is substantial time outside that chunk
  (request setup/lending and other prompt work). Rank1's first chunk also spent 6.05 seconds in
  actual expert copies at only ~0.1 GiB/s; later chunks return to normal transfer rates.
  This startup/lending outlier is unresolved. See [L rank0](maya-l-rank0.txt) and
  [L rank1](maya-l-rank1.txt) stage logs; do not describe all Maya-L requests as fixed.
- Twelve sequential Maya-L conversation turns PASS, MTP 2, context 32768. Turns 2–12 reuse the
  conversation; each reads 35 new tokens. Turns 11/12: **603.2/612.0 ms, 58.0/57.2 prompt tok/s**,
  with 338/372 cached tokens. The original 9.6 tok/s on request 11 is not reproduced here. This is
  a short conversation, not a 524288-token-context validation. [All responses](maya-l-12-turns.json).
- Test engines/server stopped; three binary hashes unchanged; llama-swap daemon/YAML untouched.
- Required CPU gates PASS, in order: HIP mapping, **464 serve tests / 11 skips**,
  **630 tools tests / 31 skips**. Launcher subset: 11 tests PASS. Shell syntax and report JSON/links
  checked. [Gate summary](cpu-gates.txt). GitHub CI/review links are tracked in #33.

## Scope remaining in #33

The ~3008-row transport cap and unimplemented `MAYA_TP_PREFILL_ROWS` remain. Short follow-ups pay fixed
per-layer work; the reported 9.6 tok/s request has not been reproduced by the fresh-prompt comparison.
The calibration is scoped to gfx1151 / hipBLASLt 100401. No new release, engine kernel, decode default,
RDMA protocol, llama-swap daemon, or user YAML change is part of this work.
