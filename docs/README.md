# Maya-TP documentation

This fork targets **HIP/ROCm on gfx1151 Linux**, with two Strix Halo APUs connected over
Thunderbolt/USB4 RDMA. GLM Maya-S, Maya-M and Maya-L have TP2/MTP validation. Broader platform and
model descriptions retained from Project Maya are upstream references, not fork support claims.

## Current guides

| Reader/task | Guide |
| --- | --- |
| Project scope and measured results | [README](../README.md) |
| Pair setup, model/executable selection, MTP and recovery | [TP2 runbook](TP2-RUN.md) |
| Experimental DeepSeek two-host review | [DeepSeek TP2 review guide](DEEPSEEK-TP2-RUN.md) |
| Experimental DeepSeek TP2 measurements | [Queued-collective XS comparison](benchmarks/deepseek-tp2-queued-2026-11-04/README.md), [earlier review benchmark](benchmarks/deepseek-tp2-review-2026-11-03/README.md) |
| Installer and clean-install updates | [Setup](SETUP.md) |
| Model catalog and fork validation limits | [Models](MODELS.md) |
| Runtime settings, thinking budgets and default-off options | [Settings](SETTINGS.md) |
| Responses API | [Responses](RESPONSES.md) |
| Contributor workflow and validation | [Contributing](../CONTRIBUTING.md) |
| Coding-agent rules | [AGENTS.md](../AGENTS.md) |
| Coding-agent current state and restart prompt | [Handoff](../Handoff.md), [Handoff-Prompt](../Handoff-Prompt.md) |
| Published versus unreleased changes | [Changelog](../CHANGELOG.md) |
| Upstream integration decisions and evidence | [v1.1.49 merge and fresh live validation](upstream/v1.1.49.md), [earlier index](upstream/INTEGRATION-2026-11-01.md), [earlier completion](upstream/COMPLETION-2026-11-02.md) |

At the 2026-11-02 checkpoint, `main` includes the completed semantic integration through upstream
v1.1.49 (v1.1.47–v1.1.49 in one cumulative merge). The latest published fork release remains legacy v1.1.42; future fork tags use
`tp-vMAJOR.MINOR.PATCH`. The user-accepted prefill candidate was promoted to the canonical binary
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
pre-implementation assumptions; the DeepSeek review guide describes the experimental implementation.

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

Documentation consistency was audited under [#82](https://github.com/uv-genai/maya-tp/issues/82).
Historical raw evidence is retained unchanged; the active guides carry current fork instructions.
