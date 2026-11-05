# GLM TP2 batching measurements

2026-11-05, [#148](https://github.com/uv-genai/maya-tp/issues/148).
Two gfx1151 APUs, `usb4_rdma5`, Maya-S-v2-IQ2_XXS, 32K context, fixed two-draft
MTP. HTTP streamed requests use the eBPF tutorial workload, 96 output tokens,
three measured waves per selected concurrency and identical request identifiers.
Rates include prompt work and queueing, excluding model load and the warmup.

| Mode | One client | Two clients | Four clients |
| --- | ---: | ---: | ---: |
| Serialized, proposals disabled per request | 21.82 | 21.78 | 21.68 |
| Batched, proposals disabled per request | 21.49 | 29.71 | 36.45 |
| Serialized MTP | 24.04 | 24.38 | 23.86 |
| Batched MTP, fused expert rows | 25.77 | 29.95 | 34.04 |

Numbers are median aggregate output tok/s. Two/four-client gains exceed the
observed wave spreads in both modes. The one-active-client regression budget
of 5% passes (ordinary: about 1.5% slower; MTP: faster). All 21 matched response
texts per comparison are identical. Every eligible MTP conversation offers
real drafts; no-proposal controls offer zero. Three-client ordinary batching
measures 35.03 tok/s. See raw files for individual TTFT, latency and draft counts.

The initial grouped-member implementation did not meet the performance gate:
19.88/23.27 tok/s at two/four clients. Its data is retained. The selected HIP
row path uses the accepted fused multi-token expert launch for resident regular
quants. Grouped-member/tiered/EXL3 fallback remains available; collective shape
is fixed by the canonical batch regardless of local dispatch.

[Serialized MTP](serial-mtp.json), [selected MTP candidate](fused-mtp.json),
[initial grouped MTP](grouped-mtp.json), [serialized ordinary](serial-ordinary.json),
[batched ordinary](fused-ordinary.json), [host-1 staggered refills](host1-replenish.json).
The G5 [validation report](../../validation/tp2-batching-g5-2026-11-05.md)
records binary identities and later final-source preservation checks.

The serialized control uses the accepted one-sequence branch of the same source
line. Parent raw-pick controls also pass. Throughput measurements precede the
final uniform-fallback reduction/conditional compilation cleanup; the resident
fused kernel is unchanged, and final binaries pass default/mixed API controls.
These are bounded Maya-S results, not a speed claim for all contexts or quants.
Maya-M/L have two-client functional smokes at 32K. Four simultaneous maximum
contexts on those models have not been validated. The fixture records GPU
free memory and complete expert residency; OS physical-read bytes were not
sampled during the throughput waves. No production runtime was promoted.
