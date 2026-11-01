# Sequential upstream integration, 2026-11-01

Requested by the user: review new upstream tags, integrate one at a time, and verify/validate each
before starting the next. Fork-owned documentation requires semantic merging, especially README.md
and AGENTS.md. Audit #36; first integration #37. No new fork release requested.

Baseline: fork main `df735db` (v1.1.42), common upstream base **v1.1.36 `f031a23`**. Published fork
and upstream tags overlap in name but differ in content: upstream tags are fetched under
`refs/tags/upstream/v…`; never overwrite/push fork release tags. Raw commit/file inventory:
[tag-audit-2026-11-01.json](tag-audit-2026-11-01.json).

User naming requirement (#40): all future fork release tags use **`tp-vMAJOR.MINOR.PATCH`**.
Keep `VERSION` numeric, published legacy `v*` tags immutable and fetched upstream tags namespaced.
The updater accepts both fork-prefixed and legacy tags; merging an upstream tag does not publish a release.

| Upstream tag | Exact commit | Commits/files since preceding tag | Status |
|---|---|---|---|
| v1.1.37 | `7ee8f2777c754229605a8c19ca985cbb68540419` | 18 / 86 | #37 accepted; PR #38 / `7ea9d71`, final CI PASS |
| v1.1.38 | `cf6472cc84489765791ad97bb55b395e8f442088` | 4 / 13 | #43 accepted ad96053; #47 fixed f479b91, main CI PASS |
| v1.1.39 | `376b4b87402bc992af7b4a8dec089ef7cba7801c` | 33 / 48 | #45 active review; #46 stream hazard identified |
| v1.1.40 | `cbaaec5d70fde2730e251dec6b418e4b454c01bd` | 16 / 55 | queued |
| v1.1.41 | `171077787036ccc2784250ca644760f654e8d4a3` | 50 / 82 | queued |
| v1.1.42 | `cd48762b6d4888205c130238dbc0c7f7904993cf` | 22 / 49 | queued |
| v1.1.43 | `acdab34bd58eec957715392a1d822777c3c579c3` | 18 / 41 | queued |
| v1.1.44 | `5d20b0e8fd7c15fde2a9f2b4eac5e9776883690b` | 4 / 6 | queued |
| v1.1.45 | `6ecbd3cbc5330b57f70cd5cf8a5c42eb8ef950cc` | 2 / 3 | queued |
| v1.1.46 | `3589d6f12a3f77e813b184b51a586f8fc3de2185` | 5 / 8 | queued |

## Preservation and validation decisions

- Preserve fork VERSION 1.1.42 and published changelog/release history; integration notes go under
  Unreleased with the upstream tag and checkpoint issue. A merge is not a new fork release.
- README.md and AGENTS.md stay fork-owned. Review documentation hunks individually; carry accurate
  applicable settings/API information while retaining TP2/MTP scope, identity, credit and operations.
  Do not restore upstream donation banners/funding files or replace fork-specific handoffs/runbooks.
- Preserve GLM TP2 slicing (including shared/dense widths), replicated NextN computation, canonical
  draft/accept/continuation wire, peer-down signaling, mirror order and global scratch/chunk decisions.
  New upstream snapshot/cancellation paths must not call unmirrored state operations on one rank.
- Keep multi-sequence TP2 and vision synchronization gated until their own acceptance is met.
  Upstream HIP vision alone does not close #16 (RDMA image synchronization still required).
- Keep RDMA as GLM TP2's tensor path and gfx1151 Linux as the target. Exclude new Metal backend/macOS
  workflow additions at v1.1.40. Do not add new backend abstractions or NVIDIA-specific development.
- Preserve the #33 rank1 config inheritance and measured sliced hipBLASLt 100401 calibration entries
  when upstream changes table lists/presets. New upstream default expert/snapshot behavior needs a
  same-harness check, not an assumption that it improves this fork.
- v1.1.37 proposes leaving the thinking cap unchanged for output limits above 1024. Review against
  our answer-reserve tests and recent timeout report; do not silently change established behavior.
- Each checkpoint: semantic diff review; HIP mapping, server and tools suites in order; gfx1151 HIP
  build and relevant host/shared-kernel checks; TP2/MTP CLI/serve/reuse/cancel/large-prompt validation;
  CI and issue-linked commit/PR. Do not start the next tag on a failed or incomplete checkpoint.

## Host state at audit

The audit started with user Maya-L context524288 (rank0 API :5812, rank1 local). User then explicitly
authorized stopping the pair for per-tag validation. First-tag tests are complete; both engines/server
are stopped, daemon/YAML untouched. See Handoff §29 and the per-tag checkpoint for current state.
Rank0 checkout `777cc92` has user `maya.py` changes/backups and deployed #33 files: preserve them;
no pull/reset/rebuild there. Daemon/YAML unchanged. Use an isolated integration worktree and retain
baseline binaries/reference data before candidate deployment. Three baseline engine MD5s:
`5a52cd27167523b6ea4b0f0bde93f503`.
