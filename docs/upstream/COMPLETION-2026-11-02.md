# Sequential upstream integration complete — 2026-11-02

All requested upstream tags **v1.1.37–v1.1.46** are ancestors of main. Each checkpoint was reviewed
semantically and validated before the next. Final PR81 merged as `1139fa27d7731941b187de57093c3abe424a5e1f`;
exact-head `b5e46b2` CI41350489382/41350494526 and post-merge mainCI41351082604 pass. Per-tag reports
and the [integration index](INTEGRATION-2026-11-01.md) retain failures, controls and CI provenance.
The final documentation checkpoint `dede882111be994b3e913463c8fa6b211f34633d` also passed
[main CI41352750402](https://github.com/uv-genai/maya-tp/actions/runs/41352750402).
Integration #77 and audit #36 are closed; no integration closeout steps remain.

Final46 checks pass: HIP mapping, 605 server tests (11 skips), 679 tools tests (33 skips), 99 focused
checks, gfx1151 HIP DeepSeek build and 18 host checks. MTP7/ordinary large/reuse/stream/guard/idle
checks and real18Kpartial-prefill cancel/cache0retry/follow-up pass;132rank-matched rounds and eight
asymmetric retries agree. A20K-token text-parts API history retains20020cached tokens, then ordinary
and MTP follow-ups pass. Direct CPU regression isolates text-parts counting; the OpenAI path flattens
text-only parts, so the live format check is not an independent before/after performance claim.
GLM binary/GPUimage remain exact validated43; DeepSeek GPUimage also remains identical after host
policy changes. Prior strict S/M/L and kernel evidence is reused only for unchanged executed paths.

README/AGENTS remain fork-owned; conflicts and applicable documentation were merged semantically.
VERSION stays numeric1.1.42. Legacy fork tag v1.1.42 still points at6471995; future fork tags use
**tp-vMAJOR.MINOR.PATCH**. No release or tag was published during this integration.

## Runtime restored and preserved

| Stack | GLM binary MD5 | State |
|---|---|---|
| Production, both canonical build paths and rank0 `/tmp/maya-tp2` | `5a52cd27167523b6ea4b0f0bde93f503` | Original production binaries restored atomically from verified backups |
| Review, both hosts | `b398d4a0711430edd6a28b95264c7293` | Validated GLM build retained, engine code6a2fbe1 |
| Review DeepSeek, both hosts | `51749e860763aea34847b183de95a933` | HIP host-policy build retained, code5b13c83; no full-model performance claim |

The protected rank0 canonical checkout remains777cc92 with its user edits and two configuration
backups. User diffs, selected model configs, backups, YAML and daemon175392 match exactly before and
after restoration. YAML checksum remainsc5392d8c696f5ff8869510057427675b18f8a3cb482d0a136012495151297a20;
its six-entry configuration and Maya-L context524288 were preserved. These are checkpoint snapshots,
not current service status. All agent-started validation engines/server ended; the user launches
their own processes. No protected source pull/reset/rebuild occurred. The user's later clarification
limits agent cleanup to processes the agent started; do not inspect/manage unrelated services.

The existing Sreviewconfig points at the review binary on both hosts. M/L validation used explicit
transient configs; future review launches must select the desired model/pack/config on both ranks
and use the review executable: rank0 uses config.exe, while rank1 uses ENGINE or its checkout's
build-hip/maya. Review source can
track the final main checkpoint without rebuilding unchanged engine code. Accepted46 backups are
`/tmp/maya-upstream-v46-baseline/maya` and `maya-ds4-accepted` on both hosts; the local `maya-ds4`
in that directory is the older pre46control retained for the ELF comparison.

## Explicitly open work

- #75: v1.1.45's functional DS4tail-drain fix was deferred because its automatic NVIDIA path is disabled on HIP and it depends on excluded lending machinery. No partial port or DeepSeek long-prompt stability claim.
- #68: full DS4 WMMA GPU stress still fails on the pre-existing accepted42control; no threshold was relaxed. The unchanged kernel image does not fix it.
- #79: optional gfx1151 DeepSeek4Kcap remains default0 pending measurements. No new fork5%speedup claim; configured local model store has GLM S/M/L only.
- #33/#60 and the other documented deferred work remain open. Streamed fetch #46, vision #16 and multisequence #11 remain deferred/gated.

[Completion JSON and raw snapshots](completion-2026-11-02/completion.json) record ancestry, binary
restoration and protected-state comparisons. Completed documentation CI and closeout are also
recorded in [#77](https://github.com/uv-genai/maya-tp/issues/77) and
[#36](https://github.com/uv-genai/maya-tp/issues/36). The later documentation consistency audit is
[#82](https://github.com/uv-genai/maya-tp/issues/82); current operations are in
[Handoff](../../Handoff.md).
