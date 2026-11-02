# Contributing to Maya-TP

Maya-TP targets **HIP/ROCm on gfx1151 Linux**, with two Strix Halo APUs connected by
Thunderbolt/USB4 RDMA. Development and measurements belong in
[uv-genai/maya-tp issues](https://github.com/uv-genai/maya-tp/issues).
Use one issue per workstream, comment at meaningful milestones, link commits/PRs with `refs #N`
or `closes #N`, and explicitly close or leave the issue open with an outcome.

## Reporting a bug or performance result

Include the source commit and binary hashes on both hosts, model/quant and pack, ROCm version,
context, draft count, config/environment, and relevant server/engine logs. `./maya.sh --report`
collects local diagnostics. Run `./maya.sh --bench` only when the model is stopped and the hosts
are available for your benchmark. Do not stop another user's processes to prepare a report.

For comparable numbers, use the [controlled TP2 harness](tools/tp2/bench_mtp.py) and
[saved methodology](docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md): the same prompt and
continuation, excluded warmups, balanced run order, raw greedy-pick agreement, cache/disk counters
and retained outliers. Select the same model/pack on both ranks; see
[Choosing a model](docs/TP2-RUN.md#choosing-a-model). The launch/cleanup limitations of the older
harnesses are documented in [the runbook](docs/TP2-RUN.md#process-ownership-and-validation).

## Pull requests

- Base on `main`, keep one topic per PR, and push only to `origin` (`uv-genai/maya-tp`).
- For output/performance changes, measure before/after on the gfx1151 pair with the same harness.
  Explain any answer changes; expert placement and reduction order can change floating-point results.
- New settings use `MAYA_*`, preserving `STRATA_*` aliases. Keep current defaults unless measured
  better and document the setting in [SETTINGS.md](docs/SETTINGS.md).
- Preserve inherited CUDA behavior in shared sources, using HIP guards where needed. New CUDA-name
  runtime/cuBLAS calls need a mapping in `include/strata/hip_compat/`. This fork's validation target
  remains HIP/gfx1151; no NVIDIA build or measurement is required here.
- Run the CPU checks in order: HIP mapping, server tests (`discover -s serve -p "test_*.py" -t .`),
  then tools tests (`discover -s tools -p "test_*.py"`). Use a venv and the pinned gguf-py checkout;
  [AGENTS.md](AGENTS.md) and [the CI workflow](.github/workflows/tests.yml) give the exact commands.
- Require green CI before merge. Engine changes also need affected HIP builds/parity and live TP2/MTP
  validation, preserving strict raw-pick checks and paired cancellation/lifecycle behavior.
- Update applicable user docs, changelog and [Handoff.md](Handoff.md). Merge upstream README/AGENTS
  changes semantically; those documents belong to this fork.

## Releases

An upstream merge does not itself publish a fork release. When a release is requested, use an
immutable **`tp-vMAJOR.MINOR.PATCH`** tag, keep `VERSION` numeric, and describe measured behavior and
known limits. Existing published legacy `v*` tags stay unchanged. Upstream is read/merge only;
fetch its tags as `upstream/v*` and never push there.
