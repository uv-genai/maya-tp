# Maya-TP 1.1.45 — Upstream refresh and validated Flash TP2

This release integrates **upstream v1.1.54–v1.1.58**, reviewed and validated one
tag at a time, while preserving **GLM-5.3-Flash TP2/MTP** and
**DeepSeek-V4.1-Flash XS/S TP2/DSpark**. Default single-sequence operation and
opt-in continuous batching retain their checked paired control contracts.
[Integration PR #168](https://github.com/uv-genai/maya-tp/pull/168).

The validated engines were installed on both hosts at the **same existing
executable paths**, with verified backups and unchanged launcher/config files.
The user reports that the installed builds are working fine, completing
[#169](https://github.com/uv-genai/maya-tp/issues/169).

## Changes since tp-v1.1.44

- **GLM state correctness:** rank-local KDA geometry is preserved across arena
  carving, sequence contexts, snapshots, checkpoints and slot copies. Canonical
  DSA, residual and replicated NextN geometry stays intact.
- **GLM drafting and kernels:** Q5_K multi-column GEMV support, whole-vocabulary
  MTP fallback for non-Latin output, and correct handling of false-valued NO_*
  switches. HIP EXL3 kernel updates include bounded model-free parity coverage.
- **DeepSeek streaming/staging:** per-chunk skip decisions, routing publication,
  sliced staging offsets and pooled helper loading preserve the TP2 expert layout.
- **Paired safety:** complete speculative verdicts and acceptance/completion ACKs,
  slot generations, cancellation/reuse boundaries and sticky first-fatal handling
  remain enforced. Unmirrored request tuning and new ablation controls are refused
  before paired mutation; ablation remains unpaired only.
- **Serving and tools:** Responses compatibility and image-detail handling,
  resumable web answers, inline document inputs, loop-guard fixes, and applicable
  installer/catalog/packing updates from the five upstream tags. Text extraction
  follows the existing paired text-only limits.
- **Operations and documentation:** accepted same-path binary installation,
  exact hashes/backups, unchanged launcher/config fingerprints, current user and
  coding-agent guides, and the rule to retain stable executable paths for future
  authorized installations.

Per-tag decisions and evidence:
[v54](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/upstream/v1.1.54.md),
[v55](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/upstream/v1.1.55.md),
[v56](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/upstream/v1.1.56.md),
[v57](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/upstream/v1.1.57.md),
[v58](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/upstream/v1.1.58.md).

## Validation

Target: **HIP/ROCm on gfx1151 Linux**, two Strix Halo APUs connected by USB4 RDMA.

- At each tag, six configurations—GLM Flash Maya-S and DeepSeek Flash XS/S,
  default1/batch2—matched **36 fixed responses / 2,304 output tokens** against
  the preceding control, with real MTP/DSpark proposals.
- Final source passed HIP mapping, **979 server tests** (32 skips), **937 tools tests** (5 skips),
  workflow CPU protocol/state suites and builds of all three HIP engines.
- **568 bit-exact GEMV cases**, Chinese whole-vocabulary MTP, retained-prefix
  reuse, cancellation/refill and deliberate peer-loss fanout passed.
- Final four-slot public HTTP checks passed for **GLM 32K / DeepSeek XS 8K**:
  ordinary/non-streaming versus speculative/SSE token and text parity, four
  active requests, cancellation and overlapping refill. All recorded validation
  processes exited.
- Per-tag source CI, final-head push/PR CI and merged-main CI passed for the
  implementation. Release metadata CI and final publication are recorded in
  [#170](https://github.com/uv-genai/maya-tp/issues/170).

These are bounded compatibility/performance checks, not filled-context or
release-wide speedup claims. The reports retain contexts, cache state, timings,
admission-order caveats and the unsuccessful experimental numerical probe.

## Runtime selection and qualification

The installed binaries include batching support. Existing entries keep their
single-sequence settings, and protected older runtime launcher/server files were
preserved. Opting into batching separately requires matching current source and
an explicit parallel setting. Do not redirect existing launchers as an automatic
upgrade step. See the
[GLM runbook](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/TP2-RUN.md),
[DeepSeek runbook](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/DEEPSEEK-TP2-RUN.md)
and [installation record](https://github.com/uv-genai/maya-tp/blob/tp-v1.1.45/docs/deployments/v58-existing-paths-2026-11-11/README.md).

Full GLM-5.3 source/catalog is imported, but its remote TP2 is refused before
bootstrap. No model was downloaded and no actual full 744B model was run during
this integration. Vision and broader platform/model paths remain outside the
fork's paired qualification.

`MAYA_DS4_AI_RUN` remains **experimental/default-off**: the strict long-prompt
probe changes answers (#166), so no bit-exactness or quality claim is made.
`DS4_INDEX_TC` remains opt-in. Intermittent stalls #165, full-GLM RAM-slot risk
#167, remaining experiments #133 and other scoped follow-ups stay open.

This is a **source release with no prebuilt assets**. Publication bumps source
VERSION to 1.1.45; it does not rebuild or replace the already accepted binaries,
which retain their recorded build identities/hashes. No launcher/config or
running-process change accompanies tagging.

[Full comparison](https://github.com/uv-genai/maya-tp/compare/tp-v1.1.44...tp-v1.1.45).
