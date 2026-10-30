# TP2 MTP review and controlled measurements — 2026-10-30

Issue [#22](https://github.com/uv-genai/maya-tp/issues/22), with independent fixes
[#23](https://github.com/uv-genai/maya-tp/issues/23),
[#24](https://github.com/uv-genai/maya-tp/issues/24),
[#25](https://github.com/uv-genai/maya-tp/issues/25) and
[#26](https://github.com/uv-genai/maya-tp/issues/26).

The review found three numerical defects and one missed HIP optimization. After fixing them,
MTP gains **10.8% on the technical prompt with one draft**, and **29.6% on the code prompt with two**.
The remaining difference is explained by measured draft acceptance and verification cost: the technical
prompt needs more verification rounds for the same 256 output tokens. TP2 also accelerates ordinary
single-token decoding more than multi-row verification, reducing MTP's incremental percentage benefit.
A universal 30% gain is not supported by these measurements.

These are **post-v1.1.38 fixes on main**. The immutable v1.1.38 tag at `c322b3e` predates them.

## Review findings and fixes

| Issue / commit | Defect | Correction / evidence |
| --- | --- | --- |
| #23 / `dc4c484` | NextN draft weights are replicated, but its mixer/MoE outputs were all-reduced, doubling full results. Three shared-expert GEMVs also used the trunk's half FF width. | Skip reductions on the replicated NextN layer; use `ff_compute_for(il)` and include its full shared output. Isolated one-draft acceptance rises from 58.75% to 80.85%; draft cost falls from about 5.3 to 4.1 ms. |
| #24 / `56b9755` | The verifier omitted rank 1's shared-expert contribution even though trunk shared weights were sliced. | Include both local partials in all five window paths. Raw greedy-pick discrepancies against ordinary decoding fall from 55/256 to 18/256. |
| #26 / `c8758cb` | Ordinary token decoding omitted the dense lead FFN reduction. Prefill and verification already summed the partials. | Reduce before the residual. A fresh corrected ordinary reference now agrees with both paths at **0/256** raw picks. The remaining 18 discrepancies were a real numerical error. |
| #25 / `ec22f4d` | Packed inputs with padded output strides bypassed the optimized RDNA dense GEMV. | Honor output stride in the HIP kernel and allow padded outputs through dispatch. Thirty-two new Q6_K/BF16 parity/guard cases pass bit for bit; targeted real-shape kernels improve 1–52%. |

The changes can change generated answers. The pre-fix forced continuation is useful for isolated
ablations, but is not a correctness oracle for the corrected model.

Review also covered MTP hidden-state seeding, head selection, rejection replay, verification batching,
shared/expert geometry and TP control/collective ordering. The pinned llama.cpp GLM-DSA implementation
also seeds NextN from the post-output-normalization hidden state, as Maya does. Disabling multi-expert
batching, rejection replay, batched MLA attention or tiled projections did not eliminate the old
18-pick discrepancy; the missing dense reduction did. No new runtime tuning knob was introduced.
Multi-sequence control remains gated (#11); its separate `glm_rows.cu` audit is still required.

## Method and provenance

- Twin Strix Halo Linux hosts, HIP gfx1151, `usb4_rdma5` tensor transport; no TCP tensor path.
- Maya-S-v2 IQ2_XXS pack at `/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack`.
- Final source `c8758cb3e304e8d2ad33777504ddbe6db85e64fa`; engine MD5
  **`5dbc93560e0eb024dcbdc0d9eb60ea57`**, identical at all three deployed paths.
- `tools/tp2/bench_mtp.py`: MTP remains loaded in both modes; `GEN spec=1` selects the simple loop,
  `spec=0` selects MTP. Count and capacity both equal the requested width on both ranks.
- Greedy, one sequence, context 4096, fixed 256 output tokens, loop guards off, unreachable EOS ID.
  Model load and prefill are excluded from `DONE decode_ms / 256`.
- A fresh ordinary reference per workload is reused across widths. All measured requests have identical
  emitted IDs **and zero raw model-pick differences**, not merely forced output identity.
- Four warmups (simple/MTP/MTP/simple) precede balanced ABBA measured blocks. Four measured runs per
  mode at widths 1/2; two at widths 3/7 and in the single-host control. Reported throughput is
  `1000 / median(ms/token)`; throughput gain is `simple_latency / MTP_latency - 1`.
- All final measured requests: **100% VRAM hits, zero disk reads/waits, zero CPU expert work**.
  No concurrent tests, compilation, profilers or other GPU jobs during performance timing.
- Proper chat framing uses the repository tokenizer/template. Technical: 64 prompt tokens, thinking high;
  code: 77, thinking off. Saved rendered prompts, token IDs and references are adjacent to this report.
  The technical answer contains one `</think>` marker; neither continuation crosses EOS/turn boundaries.

Raw per-run timing, spread, acceptance, diagnostics, hashes and settings are in `tp2-*.json`,
`single-*.json` and [summary.json](summary.json). This is a small workload study, not a broad model evaluation.

## Corrected-model TP2 results

| Workload | Drafts | Simple tok/s | MTP tok/s | Throughput gain | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Technical | 1 | 25.82 | 28.60 | **+10.79%** | 105/149 (70.47%) |
| Technical | 2 | 25.97 | 27.79 | +6.99% | 133/241 (55.19%) |
| Technical | 3 | 25.95 | 24.78 | −4.53% | 139/343 (40.52%) |
| Technical | 7 | 25.51 | 15.67 | **−38.58%** | 146/746 (19.57%) |
| Code | 1 | 25.64 | 31.30 | +22.07% | 118/136 (86.76%) |
| Code | 2 | 25.86 | 33.53 | **+29.64%** | 155/199 (77.89%) |

One draft is the measured choice for the technical prompt; two for code. The launcher can select these
with `--mtp --drafts 1` or `--mtp --drafts 2`. Runtime defaults and the user's llama-swap YAML were not changed.
The YAML's existing seven-draft entry remains expensive for the measured technical workload.

## Why the gain varies

Median phase costs below come from ordinary uninstrumented MTP diagnostics. Tokens/round include the
initial/final partial rounds; phase means in each engine diagnostic cover full drafting rounds, so these
are an explanatory approximation rather than an exact reconstruction of total request time.

| TP2 workload / width | Draft ms/round | Verify ms/round | Useful tokens/round | Commit ms/round |
| --- | ---: | ---: | ---: | ---: |
| Technical / 1 | 4.09 | 55.64 | 1.71 | 0.03 |
| Technical / 2 | 8.37 | 67.37 | 2.10 | 0.06 |
| Technical / 7 | 25.83 | 124.96 | 2.35 | 0.10 |
| Code / 1 | 3.70 | 56.13 | 1.87 | 0.03 |
| Code / 2 | 7.49 | 68.82 | 2.56 | 0.05 |

Two-draft code and technical verification have similar costs, but code emits 2.56 useful tokens per
round versus 2.10. At seven drafts the technical answer yields only 2.35 tokens per round while
paying about 151 ms for draft plus verification. Rejected suffixes consume substantial work.
Commit/control bookkeeping is small in these measurements; tensor reductions remain part of verification.

The same technical prompt on **one local gfx1151** provides a scaling control:

| Drafts | Simple tok/s | MTP tok/s | Gain | Accepted / proposed | Verify ms/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.55 | 23.40 | +26.11% | 111/144 | 71.75 |
| 2 | 18.56 | 23.62 | +27.26% | 141/226 | 85.90 |

Ordinary token latency improves about 54→39 ms with TP2; one-draft verification improves about 72→56 ms,
a smaller relative improvement. Draft work remains replicated and near 4 ms. This supports reduced
amortization under TP2 as a contributor to the smaller incremental MTP gain. Each topology uses its
own ordinary reference because reduction order can change greedy text, so this is a scaling control,
not an exact decomposition with identical outputs across topologies. TP2 remains faster in absolute tok/s.
No experiment here establishes how much a different draft quantization would change acceptance.

## Isolated history: do not confuse with final speed claims

The release-engine baseline used the old 12-token prompt,
`[gMASK]Reasoning Effort: MaxThe capital of France is`, which lacks proper chat framing and crosses
answer boundaries in a fixed 256-token run. Its results establish implementation effects on an identical
forced workload; the final natural prompts above are the current performance evidence.

| Stage (old reference) | Width | Simple tok/s | MTP tok/s | Gain | Raw verifier pick differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1.1.38 engine `d6cd72ab` | 1 | 25.65 | 26.43 | +3.06% | 55/256 |
| v1.1.38 engine `d6cd72ab` | 2 | 25.94 | 20.65 | −20.42% | 55/256 |
| #23 only `3d20e9d6` | 1 | 26.07 | 30.91 | +18.56% | 55/256 |
| #23 only `3d20e9d6` | 2 | 25.69 | 29.89 | +16.35% | 55/256 |
| #23 + #24 `f021a5a2` | 1 | 26.20 | 31.64 | +20.78% | 18/256 |
| #23 + #24 `f021a5a2` | 2 | 25.82 | 32.79 | +26.97% | 18/256 |

Original width-3/7 baselines were −33.07%/−59.12%. All ablation raw JSON is included as `ablation-*.json`.
Ablations were measured before their patches were committed, so some JSON `source_head` fields still
name the release commit; the binary hashes and stage labels above identify the actual tested changes.
After #26, a new ordinary reference is necessary: keeping the old buggy ordinary reference would itself
create discrepancies. `--spec-follow` forces emitted IDs; it never proved numerical agreement by itself.
The harness now rejects raw-pick differences by default; four unit tests cover the former false-pass condition,
explicit tolerance and missing/ambiguous diagnostics.

## HIP padded-output optimization

[padded_mv_bench.cu](padded_mv_bench.cu) measures the original tiled fallback and the new RDNA dispatch
on the same library/device. Seven timing samples, 20 calls/sample, rotating at least 256 MiB of distinct
weights, median GPU-event time. Actual TP projection shapes, padded outputs; valid rows and padding are
bit-identical. Full data: [padded-mv-results.txt](padded-mv-results.txt).

| Q6_K input / output rows | NT=2 speed ratio | NT=3 | NT=8 |
| --- | ---: | ---: | ---: |
| 4096 / 4096 | 1.013× | 1.139× | 1.457× |
| 1536 / 8192 | 1.092× | 1.459× | 1.310× |
| 4096 / 6144 | 1.052× | 1.516× | 1.234× |

These are **per-kernel** improvements, not whole-model gains. The separate GPU parity suite tests Q6_K
and BF16, NT=1…8, packed and padded outputs, guard sentinels and partial row tiles (32 new cases),
plus the existing parity matrix. All pass; [mv-parity.txt](mv-parity.txt) retains the output.

Compile the microbenchmark after building the HIP kernel/core libraries:

```sh
/opt/rocm/lib/llvm/bin/clang++ -x hip -O3 -DNDEBUG -std=c++20 \
  --offload-arch=gfx1151 -DSTRATA_USE_HIP=1 -DSTRATA_HAS_WMMA=1 \
  -Ithird_party/ggml -Iinclude -Iinclude/strata/hip_compat \
  -include include/strata/hip_compat/cuda_runtime.h \
  docs/benchmarks/tp2-mtp-review-2026-10-30/padded_mv_bench.cu -x none \
  build-hip/libstrata_kernels.a build-hip/libstrata_core.a \
  -L/opt/rocm/lib -lamdhip64 -latomic_asneeded -o /tmp/padded-mv-bench
/tmp/padded-mv-bench
```

## Reproduce the end-to-end comparison

Run on rank 1 with matching binaries on both hosts, both GPUs available. The harness stops the existing
Maya engines and remote API server. It leaves the llama-swap daemon/config alone. Use a new output directory.

```sh
.venv/bin/python tools/tp2/bench_mtp.py --output /tmp/technical-d1 \
  --drafts 1 --tokens 256 --runs 4 \
  --prompt-file docs/benchmarks/tp2-mtp-review-2026-10-30/technical-prompt.txt
.venv/bin/python tools/tp2/bench_mtp.py --output /tmp/technical-d2 \
  --drafts 2 --tokens 256 --runs 4 \
  --prompt-file docs/benchmarks/tp2-mtp-review-2026-10-30/technical-prompt.txt \
  --reference /tmp/technical-d1/reference.txt
```

Replace the prompt with `code-prompt.txt` for code; use widths 3/7 for the wider sweep. Add `--single-host`
for the local control and generate its own first reference. `--window-profile` is diagnostic and must
be excluded from speed comparisons. The default legacy prompt is retained for historical reproduction;
use an explicit framed prompt for workload conclusions.

## Final correctness and regression gates

CPU gates pass in required order: HIP name mapping; **464 server tests (11 skips)**;
**625 tools tests (31 skips)**, including the four new benchmark fidelity tests.
Source-fix GitHub CI at `c8758cb` also passes (run 41081377055); final tooling/docs CI is recorded in #22.

Simple and seven-draft API matrices pass 990-, 7290- and 14550-token prompts, cached follow-up, streaming and peer health.
Seven-draft cancellation/follow-up and output limits 1/2/3 pass. Trace audit confirms eight rank-1-only
forced clashes triggered matching global replays and **94/94 round-end records agree** between ranks.
See [trace-audit.json](trace-audit.json), [API simple](api-simple.json), [API MTP](api-mtp.json) and
[MTP acceptance](mtp-acceptance.txt).

An initial simple-mode readiness limit of 120 seconds expired during disk-backed expert loading,
without an engine error. The repeat with a longer loading allowance passed. Functional API checks ran
after performance timing; their cold-load and diagnostic overhead are excluded from speed claims.

CLI output identity passes **32/32** in simple and seven-draft MTP modes. Captured final hidden states
also match byte for byte between ranks: 32 ordinary-token rows in simple mode and the initial ordinary
head row in MTP mode (the debug dump does not instrument the batched verifier). See [cli-parity.json](cli-parity.json).

Seven-draft peer kill/recovery passes **8/8**: request failure and degraded peer health after rank1 death,
then healthy pair and successful chat after restarting both. See [kill-recovery.txt](kill-recovery.txt).
The first harness invocation was invalidated by editing its shell file during execution; the stable final
script passed shell syntax validation and the complete rerun above. Readiness now allows cold disk-backed
expert loading and waits explicitly for peer up; fixed draft count/capacity/context propagate to both ranks.

Final operations: test API server and engines stopped on both hosts; llama-swap daemon remains running,
its configuration unchanged. Engine hashes rechecked at all three paths. Remote dirty checkout preserved;
only binaries were deployed there. The final report/tooling commit and GitHub CI outcome are tracked in #22.
