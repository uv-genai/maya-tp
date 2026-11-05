# Maya-TP documentation

This fork targets **HIP/ROCm on gfx1151 Linux**, with two Strix Halo APUs connected over
Thunderbolt/USB4 RDMA. GLM Maya-S, Maya-M and Maya-L have TP2/MTP validation.
DeepSeek Maya-XS and Maya-S have TP2/DSpark validation and user acceptance.
Broader platform and model descriptions retained from Project Maya are upstream references.

## Current guides

| Reader/task | Guide |
| --- | --- |
| Project scope and measured results | [README](../README.md) |
| Pair setup, model/executable selection, MTP and recovery | [TP2 runbook](TP2-RUN.md) |
| DeepSeek two-host operation | [DeepSeek TP2 and DSpark guide](DEEPSEEK-TP2-RUN.md) |
| DeepSeek TP2 measurements | [Queued-collective comparison and user results](benchmarks/deepseek-tp2-queued-2026-11-04/README.md), [earlier review benchmark](benchmarks/deepseek-tp2-review-2026-11-03/README.md) |
| DeepSeek concurrent requests | [Opt-in XS/S batching throughput](benchmarks/deepseek-tp2-batching-2026-11-05/README.md), [earlier serialized Maya-S baseline](benchmarks/deepseek-maya-s-concurrency-2026-11-04/README.md) |
| Continuous-batching implementation plans | [GLM TP2 and MTP](GLM-TP2-BATCHING-PLAN.md), [DeepSeek TP2 and DSpark](DEEPSEEK-TP2-BATCHING-PLAN.md) — phase issues and dated design contracts |
| GLM batching validation | [Lifecycle](validation/tp2-batching-g4-2026-11-05.md), [rollout checkpoint](validation/tp2-batching-g5-2026-11-05.md), [matched throughput](benchmarks/glm-tp2-batching-2026-11-05/README.md) |
| DeepSeek batching state/capacity | [D1 XS/S four-slot checkpoint](validation/tp2-batching-d1-2026-11-05.md) and [D2 checked protocol](validation/tp2-batching-d2-2026-11-05.md) and [D3 rows](validation/tp2-batching-d3-2026-11-05.md), [D4 speculative commits](validation/tp2-batching-d4-2026-11-05.md) — [D5 serving and rollout](validation/tp2-batching-d5-2026-11-05.md) |
| Installer and clean-install updates | [Setup](SETUP.md) |
| Model catalog and fork validation limits | [Models](MODELS.md) |
| Runtime settings, thinking budgets and default-off options | [Settings](SETTINGS.md) |
| Responses API | [Responses](RESPONSES.md) |
| Agent-shaped API benchmark and coverage | [Agent benchmark](AGENT_BENCH.md) |
| Contributor workflow and validation | [Contributing](../CONTRIBUTING.md) |
| Coding-agent rules | [AGENTS.md](../AGENTS.md) |
| Coding-agent current state and restart prompt | [Handoff](../Handoff.md), [Handoff-Prompt](../Handoff-Prompt.md) |
| Current release and changes | [tp-v1.1.44 notes](releases/tp-v1.1.44.md), [changelog](../CHANGELOG.md) |
| Upstream integration decisions and evidence | [v1.1.53](upstream/v1.1.53.md), [v1.1.52](upstream/v1.1.52.md), [v1.1.51](upstream/v1.1.51.md), [v1.1.50](upstream/v1.1.50.md), [v1.1.49 merge and fresh live validation](upstream/v1.1.49.md), [earlier index](upstream/INTEGRATION-2026-11-01.md), [earlier completion](upstream/COMPLETION-2026-11-02.md) |

The current source incorporates applicable upstream v1.1.50–v1.1.53 changes as
individual validated checkpoints, completed through PR #134 at `de184d9` with
merged-main CI 41834772034 passing. The reports above record per-tag decisions,
paired results and CI tracking. New v53 engine experiments are deferred in #133.

The current `main` includes the user-accepted DeepSeek TP2/DSpark implementation
merged in PR #122 at `75e18e8`. Both feature-head and merged-main CI passed.
[Merge and validation](https://github.com/uv-genai/maya-tp/pull/122).

At the 2026-11-02 checkpoint, `main` includes the completed semantic integration through upstream
v1.1.49 (v1.1.47–v1.1.49 in one cumulative merge). The latest release at that checkpoint was legacy v1.1.42. The current
release is [tp-v1.1.44](releases/tp-v1.1.44.md), after the unpublished
`tp-v1.1.43` checkpoint. The user-accepted prefill candidate was promoted to the canonical binary
on both hosts after the v1.1.49 integration; earlier runtime files are backed up. See [executable selection](TP2-RUN.md#development-pair-production-and-review-builds)
before launching from an existing development checkout.

The user launches their own processes. Agents clean up only their recorded validation processes;
the [runbook](TP2-RUN.md#process-ownership-and-validation) identifies legacy harness limitations
tracked in [#83](https://github.com/uv-genai/maya-tp/issues/83).

## Historical and upstream references

[TP2-ANALYSIS](TP2-ANALYSIS.md), [TP2-DESIGN](TP2-DESIGN.md), [TP2-PLAN](TP2-PLAN.md) and the
[original prompt](../initial-prompt.md) describe the initial plan. Their original no-process
instructions, MTP restrictions and proposed settings do not describe the completed implementation.
Use the current guides above and GitHub issue states for ongoing work.
The [DeepSeek analysis and design](DEEPSEEK-TP2-ANALYSIS-DESIGN.md) similarly records its initial
pre-implementation assumptions. The [implementation plan](DEEPSEEK-TP2-IMPLEMENTATION-PLAN.md)
retains its original phase gates. Use the [DeepSeek operations guide](DEEPSEEK-TP2-RUN.md)
for the accepted implementation and current runtime paths.

`benchmarks/`, `validation/`, `upstream/v1.1.*` and `../bench/results/` contain dated measurements,
controls, failures and per-tag evidence. They are historical records. A report's old build hash,
test count, pending milestone or cleanup transcript is not a current operating instruction.
In particular, the early [MTP comparison](benchmarks/tp2-mtp-2026-10-30/RESULTS.md) is superseded by
the [corrected review](benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md).

[AMD_MAYA](AMD_MAYA.md), [AMD_RDNA4](AMD_RDNA4.md), [EXL3_HIP](EXL3_HIP.md), the Windows/DGX Spark
guides and other engine/kernel notes retain upstream bring-up and experiment details. Their
non-gfx1151 targets, local multi-GPU arrangements, vision paths and speed claims are outside this
fork's current TP2 validation scope. Full DeepSeek WMMA failures
[#68](https://github.com/uv-genai/maya-tp/issues/68), the deferred tail-drain port
[#75](https://github.com/uv-genai/maya-tp/issues/75), and default-off chunk-cap measurement
[#79](https://github.com/uv-genai/maya-tp/issues/79) remain open.

The post-v53 documentation audit is tracked in
[#135](https://github.com/uv-genai/maya-tp/issues/135). The preceding audit is
[#123](https://github.com/uv-genai/maya-tp/issues/123). The earlier upstream-integration
audit is recorded in [#82](https://github.com/uv-genai/maya-tp/issues/82).
The active guides carry current fork instructions. Dated evidence retains its recorded conditions.
