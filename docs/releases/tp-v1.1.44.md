# Maya-TP 1.1.44 — Continuous batching for GLM and DeepSeek

Opt-in **2–4-slot TP2 continuous batching** now runs GLM with MTP and DeepSeek
XS/S with DSpark. The user tested the integrated work and reports that it is
working really well, completing acceptance of #11/#140. The implementation is
[PR #155](https://github.com/uv-genai/maya-tp/pull/155). The default remains one sequence.

## Concurrent serving

- Checked slot generations, positions, canonical target plans and complete speculative verdicts preserve conversation isolation.
- Dynamic admission, bounded prompt interleaving, cancellation, EOS/output limits and slot reuse run between completed paired operations.
- DeepSeek drafts in fitting groups independently of the target-row budget, keeping DSpark proposals active at four clients.
- Fatal errors retain their first cause, wake every active API route and require restarting both ranks.
- Owned launchers stop only recorded processes. Legacy one-sequence preflight remains compatible.

## Measured throughput

Four-client **aggregate end-to-end output tok/s**, serialized → batching:

| Model and speculation | Context per slot | Throughput |
| --- | ---: | ---: |
| GLM Maya-S / MTP | 32,768 | 23.86 → 34.04 |
| DeepSeek XS / DSpark | 262,144 | 25.13 → 40.05 |
| DeepSeek S / DSpark | 262,144 | 22.44 → 36.08 |

These are matched, short-context text workloads with 96 output tokens. The
context column is allocated capacity; it does not describe filled-context
throughput. Individual streams share GPU capacity. One-client measurements
meet the provisional 5% regression budget. See the
[GLM report](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.44/docs/benchmarks/glm-tp2-batching-2026-11-05/README.md)
and [DeepSeek report](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.44/docs/benchmarks/deepseek-tp2-batching-2026-11-05/README.md)
for wave spreads, memory/I/O telemetry, requests, responses and validation limits.

## Other changes since the `tp-v1.1.43` checkpoint

- DeepSeek two-host TP2/DSpark support, queued GPU/RDMA reductions and checked model identities. [PR #122](https://github.com/uv-genai/maya-tp/pull/122).
- Stable filesystem identity caches prevent DeepSeek startup rejection after Btrfs device renumbering. [#138](https://github.com/uv-genai/maya-tp/issues/138).
- Applicable upstream v1.1.50–v1.1.53 installer, API, catalog, dashboard and packing improvements, preserving fork-owned documentation and paired safety. PRs [#126](https://github.com/uv-genai/maya-tp/pull/126), [#129](https://github.com/uv-genai/maya-tp/pull/129), [#131](https://github.com/uv-genai/maya-tp/pull/131), [#134](https://github.com/uv-genai/maya-tp/pull/134).
- HIP setup fixes for split ROCm compiler installations. [PR #99](https://github.com/uv-genai/maya-tp/pull/99).

This is the first published `tp-v…` release following legacy `v1.1.42`;
`tp-v1.1.43` was an unpublished checkpoint. The changelog also retains the
earlier checkpoint's accepted prefill and TP2 safety improvements.

## Upgrade and validation

Target: **HIP/ROCm, gfx1151 Linux, two Strix Halo APUs connected by USB4 RDMA**.
Select matching current engines and launchers on both hosts before opting in
with `--parallel 2`, `3` or `4`. Context is per slot. DeepSeek batch mode requires
lookup off and point proposals; sampled targets are supported, while
`DS4_DSPARK_SAMPLE=1` proposals remain unsupported in batch mode. Follow the
[GLM runbook](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.44/docs/TP2-RUN.md#opt-in-glm-tp2-continuous-batching)
and [DeepSeek runbook](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.44/docs/DEEPSEEK-TP2-RUN.md#opt-in-continuous-batching).

Validation includes paired HIP/RDMA numerical and lifecycle checks, all 88
matching DeepSeek HTTP response pairs, all 64 matching default-mode raw picks,
GLM MTP regression checks, 743 server tests and 770 tools tests. Feature-head,
PR and merged-main CI passed for the implementation. Release metadata CI and
publication are tracked in [#156](https://github.com/uv-genai/maya-tp/issues/156).

This is a **source release**, with no prebuilt binary assets. Publishing it does
not replace existing production binaries/configs or change running processes.
Filled-context performance, vision, other platforms and broader numerical/
capacity follow-ups keep their documented scope.

[Changes since the previous checkpoint](https://github.com/uv-genai/maya-tp/compare/tp-v1.1.43...tp-v1.1.44).
