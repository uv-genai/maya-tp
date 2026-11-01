# Handoff-Prompt — Maya-TP restart

Read `/home/ugo/software/project-maya/Handoff.md` in full (especially §§18,21,23,27–29 and the latest §29.9), then `AGENTS.md`.

Maya-TP runs GLM-5.3-Flash across two Strix Halo hosts. **HIP/ROCm gfx1151 Linux only**, GPU inference, **RDMA tensors only**; sockets carry bootstrap/control/token metadata. No new backend abstractions or CUDA/NVIDIA work. No subagents unless explicitly authorized.

## Active task

Integrate upstream **v1.1.37–v1.1.46 one at a time**, verifying each checkpoint and post-merge main CI before the next. Preserve TP2/MTP and semantically merge fork-owned README/AGENTS/docs. Audit: #36 /`docs/upstream/INTEGRATION-2026-11-01.md`. Tags are fetched as `upstream/v1.1.N`, never over fork tags.

Accepted:

- **v1.1.37:** #37 /PR #38 /`7ea9d71`; evidence `docs/upstream/v1.1.37.md`. Full S/M/L API/CLI, parity, retries, kill/recover and controlled prefill passed.
- **v1.1.38:** #43 /PR #44 /`ad96053`; evidence `docs/upstream/v1.1.38.md`. Its real post-merge CI failure was fixed separately by #47 /PR #48 /`f479b91`, fixed main CI41303961934 PASS. Preserve that failed-start liveness fix; do not retry away the historical failure.
- **v1.1.39:** #45 /PR #49 /`d2708b2`; final b6483f2 CI41310209384/41310204043 and **main41311902227 PASS**. #45 closed. Evidence `docs/upstream/v1.1.39.md` and adjacent directory.

**Next: #50 / upstream v1.1.40 `cbaaec5d70fde2730e251dec6b418e4b454c01bd`**,16 commits/55 files. No candidate yet. Review worktree `/home/ugo/software/project-maya-upstream-review` should branch `merge/upstream-v1.1.40` from updated origin/main. Exclude the new Metal backend/macOS workflow under fork scope. Preserve #46 stream deferral, all-limit answer reserve, replicated NextN/shared/dense widths/ARs, mirror order, canonical draft/accept/continuation wire, global scratch retries/chunks and primary100401 tuning/config inheritance. No v1.1.41 work until40 passes.

v39 retains MMQ row-ID padding, ordered Lt fallback with measured primary first, bounded sentence-end thinking closure inside the answer reserve, cancel draining, malformed-call recovery and installer fixes. Streamed fetch is **deferred entirely** (#46); buffered reader/header/GPU fetch kernels are byte-identical, even with `MAYA_GLM_STREAM_FETCH=1`. DS4 prefix/tail-auto/link-sized rings stay opt-in. Multi/vision remain gated.

## Current hosts / binaries

User explicitly authorized: **“Hosts are free; you may stop the pair for validation.”** This persists throughout per-tag tests; do not ask again. The earlier user Maya-L context524288 pair was stopped under this authorization.

You are **strix-halo-2 (rank1)**. SSH **only `strix-halo-1.local` (rank0)**; never nested SSH or invoke the llama-swap wrapper via SSH. Both review worktrees exist. Remote canonical `/home/ugo/software/project-maya` stays at **777cc92**, with user-modified `maya.py`, two Maya-L config backups and deliberate #33 launcher/tuning deployments. **No pull/reset/rebuild there; preserve user work.** Local canonical main tracks origin/main.

Accepted v39 engine MD5 **c3132ee406429195545f145e04e7aadd**, runtime source27fefcc, matched on local canonical/review, rank0 canonical/review and `/tmp/maya-tp2`. Test engines/server stopped. Llama-swap daemon **175392** and YAML untouched (six model entries, two drafts on MTP, L context524288).

Baselines on BOTH hosts:

- Original released engine **5a52cd27167523b6ea4b0f0bde93f503**: `/tmp/maya-upstream-baseline-2026-11-01/`.
- Accepted38 **cff75b48a0f1e0b2d2355c73ce302d94**: `/tmp/maya-upstream-v38-baseline/maya`.
- Accepted39 **c3132ee406429195545f145e04e7aadd**: `/tmp/maya-upstream-v39-baseline/maya`.

At task end leave a coherent source/binary pair; do not leave an unvalidated old-server/new-engine combination after later protocol changes. Restoring original canonical binaries is available while retaining validated builds in review worktrees. Never overwrite remote user source to accomplish deployment.

## Latest validation / reusable evidence

v39: HIP map, **542 serve/11 skips**, **651 tools/31 skips**,11 budget tests,13 host CTest checks, all three gfx1151 HIP engines,196 fused cases, MMQ/attention and real calibrated/forced-fallback GEMM checks PASS. S MTP7/simple large-prompt matrices,105 matching round tuples/8 asymmetric retries,CLI32/32 both modes matchingv38, real sentence-close resume and kill/recover8/8 PASS. S/M/L API/prefill and32 framed ordinary/MTP records all raw-pick0.

Controlled S prefill vs38: nine paired messages/usage equal, one warmup cycle then two measured cycles; throughput **−0.01/+0.18/−1.21%** at1K/4K/8K. No material warm regression/new speed claim. M startup32/375s and L806/49s on the same candidate are retained cold-I/O outliers, not new speed claims. L's ordinary reference completed within the existing900s harness limit; the prepared longer-timeout wrapper was never used.

Same-pack/same-prompt benches can use existing `bench_mtp.py --reference` with strict **ordinary and MTP raw-pick0** checks. This proves agreement with the accepted trajectory while avoiding a redundant cold reference startup. Investigate any difference; never weaken the gate or treat forced IDs alone as parity.

- S256 references/prompts: `docs/benchmarks/tp2-mtp-review-2026-10-30/`.
- M/L64 references: `docs/upstream/v1.1.39/{Maya-M,Maya-L}/reference.txt`.
- Temp live harness: `/tmp/maya-upstream-live.py`; CLI: `/tmp/maya-upstream-cli.py`.
- Completed v39 batch: `/tmp/maya-upstream-v39-remaining.sh`, outputs `/tmp/maya-upstream-v39/`. Do not rerun completed gates without reason. Update output paths/config logs per new tag.

## Open issues

- **#33 prefill:** measured rank1 config inheritance +18 sliced GEMM rows improve S32–37%. Preserve `9bc0ec9` /PR #34 /`b63f8af`. Cold/long-context L and exact request11 at9.6tok/s remain unresolved; ~3008-row RDMA cap and unimplemented `MAYA_TP_PREFILL_ROWS` remain. Copy-lane spans include idle mixer windows; do not repeat the old “90% copying” interpretation.
- **#42 CLI warmup readiness:** pre-existing rank0/rank1 warm skew poisons the first collective. L paired CLI needs `STRATA_GLM_WARM=0` on both ranks and is slow; no fix yet.
- **#46:** future chunk-aware compact TP streamed-fetch port; complete deferral is not implementation.
- **#11:** multi remains gated; full-width shared FFs bug in `glm_rows.cu` remains.
- **#16:** vision deferred; HIP vision alone does not supply RDMA embedding synchronization.
- #1 umbrella, #10 optional remain open. #29 M launcher work, #43/#45 integration and #47 restart fix are closed.

## Operations

1. Stop rank0 server **before** engine: `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'`. Local `pkill -9 -x maya`. Never broad `pkill -f server.py`.
2. After every engine build, deploy to both rank0 paths (`build-hip/maya`, `/tmp/maya-tp2`) and review copy, then verify hashes. Start rank1 locally before rank0 via direct SSH scripts. Never probe18515 with `/dev/tcp`. Restart both after a poisoned communicator.
3. Same model/pack/config on both ranks: `--model`/`--pack`/`--config` or `MAYA_MODEL`/`MAYA_PACK`/`MAYA_CONFIG`; resolver `tools/tp2/model_env.sh`. No hardcoded S-only launchers.
4. MTP: `MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=N MAYA_GLM_MTP_DRAFT=N`, unset `MAYA_GLM_NO_MTP`. Simple: `MAYA_GLM_NO_MTP=1`. Match contexts (32768 for API checks unless specified).
5. `.venv/bin/python`; in review tools set `STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py`. Pinned llama.cpp is3cf03257f219afbe7334045ff7c6a06ac68c627d; do not change the pin.
6. Build in the isolated review tree: HIP ON, CUDA OFF, MMQ ON, TP2_RDMA ON, gfx1151, `/opt/rocm/lib/llvm/bin/clang++`; use canonical pinned llama.cpp via `STRATA_GGML_DIR`. Ordered CPU gates are in AGENTS.md.
7. Track each workstream on `uv-genai/maya-tp`, milestone comments, issue-linked commits/PRs. Push **origin only**. Keep Handoff current. Stop test engines/server after checks; preserve daemon/YAML.

## Releases / tag naming

Future fork release tags are **`tp-vMAJOR.MINOR.PATCH`** (#40); keep VERSION numeric. Existing legacy `v*` tags are immutable; upstream tags stay `upstream/v*`. Updater accepts both. Legacy v1.1.42 installs need one manual clean-main `git pull --ff-only` to acquire prefix support; mention in next release notes. **Merging tags does not authorize a new release.**

Published v1.1.42 is **64719953312df98c9fdf53a10198c106cb195345**, CI41203815519, public/latest source release with0 assets; #35 /Handoff §28. Do not republish/move. Earlier v1.1.41=a6d15a7, v1.1.40=d9248e8, v1.1.39=20ff03a, v1.1.38=c322b3e remain unchanged.
