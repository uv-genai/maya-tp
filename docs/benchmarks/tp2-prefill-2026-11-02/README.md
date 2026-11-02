# TP2 prefill, 2026-11-02

The user tested this candidate through llama-swap using “write an eBPF tutorial to explain how to use eBPF”. They reported working prefill and generation for all models:

| Model | Prefill tok/s | Generation range tok/s | Average generation tok/s |
| --- | ---: | ---: | ---: |
| Maya-S | 131–308 | 29–46 | 30 |
| Maya-M | 187–299 | 31–45 | 32 |
| Maya-L | 194–310 | 25–33 | 28 |

They preferred Maya-L's output quality and requested promotion. These are user-observed measurements, not a controlled benchmark.

The balanced S/M comparisons contain four warm samples per length and variant, with the first cycle excluded. All 18 messages and usage records matched per model. The L comparison failed its exact-response gate: the second baseline arm's first 8K response differed from its later two and all candidate responses; all usage values matched. Raw L responses and the failed sweep output are retained here. The strict S historical reference failed at 12 different raw picks out of 256; the failure log is retained. These observations are not resolved by the user acceptance, and full candidate API and CLI regression was not repeated.

The HIP APU complete pool avoids expert staging when the entire layout fits within the memory budget. TP2 startup scratch uses the existing receive-window bound. Three DSA tuning entries are scoped to gfx1151/hipBLASLt 100401. See [settings](../../SETTINGS.md) and [handoff](../../../Handoff.md) for controls and deployment state.
