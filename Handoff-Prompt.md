# Handoff-Prompt — Maya TP2 restart

Read `/home/ugo/software/project-maya/Handoff.md` in full (especially §18, §21, §23, §24, **§27–§29**), then `AGENTS.md`.

Project: **Maya-TP**, GLM-5.3-Flash TP2 across twin Strix Halo hosts. HIP/ROCm **gfx1151 Linux only**;
GPU inference, RDMA tensor path only. Bootstrap/control/token metadata may use TCP.
Repository `uv-genai/maya-tp`, local branch **main**, tracking origin/main, including prefill commit
`9bc0ec9` (PR #34 merged as `b63f8af`) and release commit `6471995`.
Rank0 is at `777cc92` with user-modified `maya.py` and two untracked Maya-L config backups; preserve them.
No subagents unless explicitly authorized.

## Current state

**Active task: sequential upstream integration** (#36 audit complete; #37 accepted; #45 active). User requests upstream
v1.1.37 through v1.1.46 merged one at a time, each fully validated before the next. Common upstream
base is v1.1.36 `f031a23`; use namespaced `upstream/v1.1.N` tags (fork tag names collide). See Handoff
§29 and `docs/upstream/INTEGRATION-2026-11-01.md`. Preserve fork-owned README.md/AGENTS.md through
semantic review, all TP2/MTP invariants and #33 tuning. No new fork release requested.

**Tag naming (#40):** all future fork releases use **`tp-vMAJOR.MINOR.PATCH`**; `VERSION` stays
numeric. Existing published `v*` tags stay immutable. Upstream tags stay under `upstream/v*` locally.
The updater accepts both the new prefix and legacy tags. Do not publish a release just for a merge.

The previous user **Maya-L context 524288** pair was stopped with explicit authorization:
"Hosts are free; you may stop the pair for validation". This persists for all per-tag tests; do not ask
again. Baseline binaries are saved on both hosts in `/tmp/maya-upstream-baseline-2026-11-01/`.
Accepted v1.1.37 baseline MD5 is **6e59558bdd421fa521040263978b2bd6**. Current v1.1.38
engine MD5 is recorded below on all canonical paths; engines/server stopped, daemon/YAML untouched. Remote user source preserved.

**Upstream v1.1.37 accepted:** PR #38 merged as **`7ea9d71`** after final `bedc611` CI
**41298854143 / 41298857613 PASS**. #37/#39 closed. All three HIP engines, CPU 530/637 suites,
shared parity, S/M/L API/CLI tests, asymmetric retries and kill/recover pass. Release-vs-candidate CLI
IDs match 32/32 in both modes. Reversed warm prefill deltas **+0.66/−1.77/−0.73%** at 1K/4K/8K;
all 9 paired messages match. Initial cold outliers preserved; #33 remains open. Evidence:
`docs/upstream/v1.1.37.md` and its adjacent directory. Do not repeat completed checks without reason.

**Accepted upstream tags:** v1.1.37 (PR #38 /7ea9d71) and v1.1.38 (PR #44 /ad96053).
The v1.1.38 post-merge CI failure was a real pre-existing failed-start liveness race, fixed separately
in **#47 / PR #48 /f479b91**. Fixed **main CI41303961934 PASS**, plus both a6f1ab5 feature CI runs
41303387525/41303664047. #43/#47 closed; do not skip this historical failure or repeat the fix.
Local current CPU gates **532 serve/11 skips**, **637 tools/31 skips**, HIP map PASS.

**Active: #45 / upstream v1.1.39 `376b4b8`**,33 commits/48 files; candidate merge in progress in the isolated review worktree on
`merge/upstream-v1.1.39`, based on b30514b. Runtime source27fefcc built/deployed; MD5 c3132ee406429195545f145e04e7aadd matches both hosts. Preserve server fix #47 and all TP2/MTP invariants. #46 tracks the newly
identified stream-ready contract mismatch with compact TP blobs: defer the entire streamed-fetch feature; preserve the buffered reader/header/kernels byte-for-byte. Preserve all-output-limit answer reserves,
TP chunk/retry agreement, replicated NextN and measured100401 calibration priority. See Handoff
§29.7. No v1.1.40 work until v1.1.39 checkpoint passes.

Current candidate engine **c3132ee406429195545f145e04e7aadd** is deployed/matched; authorized test pairs run from the review worktrees. Remote canonical user source/daemon/YAML preserved. Future fork tags use tp-v.

**v1.1.42 published** from **`64719953312df98c9fdf53a10198c106cb195345`**, release CI **41203815519 PASS**.
Annotated tag verified locally/remotely; public/latest/non-prerelease GitHub source release, 0 assets.
Both handoffs updated first; publication outcome recorded afterward. **#35** / Handoff §28 has the
release audit. **Do not republish or move the tag.**
Previous **v1.1.41** = `a6d15a7`, CI41191735148; #32 closed (§24).
Older tags: v1.1.40 = `d9248e8`, v1.1.39 = `20ff03a`, v1.1.38 = `c322b3e`.
Do not repeat completed measurements or start a release without an explicit request.

**Validated TP2 + MTP packs:** **Maya-S, Maya-M, Maya-L** (API matrices, acceptance, CLI 32/32,
kill/recover, `bench_mtp`). Reports:

- `docs/benchmarks/tp2-mtp-review-2026-10-30/` (S)
- `docs/benchmarks/tp2-maya-m-mtp-2026-10-31/`
- `docs/benchmarks/tp2-maya-l-mtp-2026-10-31/`

**Open work:**

- **#33** — TP2 **prefill**: rank1 config inheritance + 18 sliced GEMM calibration rows (gfx1151,
  hipBLASLt 100401) improve Maya-S 1K/4K/8K median throughput **32–37%**. Report:
  `docs/benchmarks/tp2-prefill-2026-10-31/RESULTS.md`; handoff §27. M/L smoke and 12 L turns PASS.
  The “copy lane” includes idle mixer windows: initial staging diagnosis was incomplete.
  Cold L startup/lending remains slow; user **9.6 tok/s** request 11 not reproduced (test turn 11: 58).
  ~3008-row RDMA cap and unimplemented `MAYA_TP_PREFILL_ROWS` remain. Decode/MTP generation is fine.
- **#29** — closed; launchers + Maya-M full suite completed (§21).
- **#42** — existing CLI readiness/warmup skew; tracking only. Cold L WARM=0 MTP7 passed32/32
  but needs126s prompt +432s decode; no runtime fix in this merge.
- **#1** umbrella, **#10** optional, **#11** multi gated, **#16** vision deferred.

**Launchers:** `--model NAME` / `--pack PATH` / `--config PATH` or `MAYA_MODEL` / `MAYA_PACK` /
`MAYA_CONFIG` (resolver `tools/tp2/model_env.sh`). Same flags on **both** ranks.
Runbook: `docs/TP2-RUN.md` “Choosing a model”.

**llama-swap on strix-halo-1** (`~/.config/llama-swap/config.yaml`, user-owned — do not edit unless asked):
six entries `maya-glm5-{s,m,l}-tp2` and `-tp2-mtp` with explicit `--model`; drafts **2** on MTP rows.
Maya-L uses **`--context 524288`** (also in `maya-maya-l-hip.json`). Daemon `--watch-config` running.
Backups sit beside the YAML.

**Baseline binary:** `5a52cd27167523b6ea4b0f0bde93f503`, saved on both hosts. Current deployed
engine is `c3132ee406429195545f145e04e7aadd` (see active task above); TP2 RDMA is enabled.

**Engines:** the old release-check Maya-M runtime is superseded by the authorized GPU window above.
The subsequent Maya-L pair was stopped with explicit permission; candidate test pairs may be restarted.
Daemon/YAML untouched.
**#33 gates:** HIP map + 464 serve / 11 skips + 630 tools / 31 skips PASS. No engine rebuild;
rank1 launcher + hipBLASLt 100401 tuning table deployed to rank0 with matching SHA256s.
Prefill commit **9bc0ec9**, [PR #34](https://github.com/uv-genai/maya-tp/pull/34); both CI runs
**41202760879 / 41202777839 PASS**. Do not repeat the completed fresh-prompt comparison.

## Conclusions / evidence

Maya-S corrected MTP workloads (§6/§18): technical +10.8% @1 draft, code +29.6% @2; report
`docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md`. Do not reuse #14 numbers.

Maya-M: pair bench 32/32, bench_mtp +8.11% @2 drafts (41.12→38.03 ms/tok).
Maya-L: pair bench 32/32 ~64 ms/tok, bench_mtp **+31.24%** @2 drafts (42.81→32.62 ms/tok).
CLI on Maya-L needs **`STRATA_GLM_WARM=0`** both ranks (rank0 warm skew → first-AR poison); serve is fine.

CPU gates green at last run (HIP map + serve + tools). CPU CI passed for release commits.

## Operations / constraints

1. You are **on strix-halo-2** (rank1). SSH **only strix-halo-1.local** (rank0). **Never nested SSH**.
   Do not invoke the llama-swap wrapper over SSH (it SSHes back). Use direct rank scripts.
2. Kill remote server **before** engine:
   `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'`.
   Local: `pkill -9 -x maya`. Never `pkill -f server.py`.
3. After every engine build, copy to **both** rank0 paths (`build-hip/maya` and `/tmp/maya-tp2`) and
   verify **three** hashes. `/tmp/maya-tp2` can disappear — redeploy if missing.
4. Rank0 has user work (above), plus the deployed rank1 launcher and tuning table from #33. **No pull/reset/rebuild there.**
   Local main includes `9bc0ec9` and v1.1.42 (`6471995`); authoritative source is origin/main.
   Candidate binaries are deployed as recorded above. Do not leave worktrees dirty without saying so.
5. Use `.venv/bin/python`, jq. One issue per workstream on `uv-genai/maya-tp`; comment at milestones;
   `refs #N`; push only origin; close or leave open explicitly. Keep Handoff current.
6. Start rank1 locally before SSH-launching rank0. Pass the **same** `--model`/`--pack`/`--config` (or
   `MAYA_*`) to both ranks. MTP: `MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=N MAYA_GLM_MTP_DRAFT=N`; unset
   `MAYA_GLM_NO_MTP`. Simple: `MAYA_GLM_NO_MTP=1`. Match contexts (32768 for API tests unless told).
   Never probe bootstrap 18515 with `/dev/tcp`. Restart **both** ranks after poisoned communicator.
7. Stop your test engines/server after finishing; the user authorized stopping the previous pair.
   Leave llama-swap **daemon and YAML** alone unless the user
   asks. Live model ids: `maya-glm5-{s,m,l}-tp2` and `-tp2-mtp`.
8. #16 vision deferred; #11 multi gated (`glm_rows.cu` shared-width/attention audit first); #10 optional.
9. Do not hardcode model paths in launch scripts (`AGENTS.md` convention 8).
10. **Prefill performance (#33)** is the open perf topic — not decode. Quote stage lines if diagnosing.

Finish any explicitly pending release step only if asked, then follow the next user request.

Accepted v1.1.38 cff75b48 binary is additionally saved on BOTH hosts at `/tmp/maya-upstream-v38-baseline/maya` for controlled v1.1.39 comparisons. Candidate retains MMQ ID padding, measured primary table priority, all-limit answer reserve with bounded sentence grace, and #47 failed-restart fix. New DS4 prefix/tail-auto/link rings stay opt-in. CPU/build/live gates pending; do not proceed to v1.1.40.

#45 checkpoint progress: draft PR #49, code CI41305518760/41305514302 PASS atf475729 (test-only follow-ups; runtime27fefcc). Ordered542 serve/11skips,651tools/31,11budget,13host checks,3HIP engines,196fused/MMQ/attention/Lt primary+forced-fallback parity PASS. S MTP7 acceptance, MTP/simplelarge matrices,105rounds/8asymmetricretries,CLI32/32 each PASS; CLI exactlymatchesv38. Real sentence-close probes PASS at16+2/16+24 tokens with answer/resume/follow-up. Initial math probes only missed coverage and are preserved in /tmp. Remaining batch /tmp/maya-upstream-v39-remaining.sh: kill/recovery,2framedrawpick benches,prefillA/B vs38,M/LAPI+prefillsmoke/rawpicks. No v40 before39acceptance/mainCI. Evidence docs/upstream/v1.1.39/.

#45 milestone: peer kill/recover8/8 PASS. Framed S technical/code256-token benches16/16 raw-pick0; both ordinary references match historical corrected references256/256. ccab795 CI41306661283/41306656736 PASS. Prefill A/B vs acceptedv38 running; M/L gates and final evidence/main CI pending. Current script /tmp/maya-upstream-v39-remaining.sh, outputs /tmp/maya-upstream-v39. No v40 yet.

#45 prefill A/B vs acceptedv38 complete: all9messages/usage match; warm1K/4K/8K throughput -0.01/+0.18/-1.21% (two measuredcycles afteronewarmup, bothrawarmsretained). No material warm regression/new speedclaim; #33 remainsopen. M/L checks now running; no nexttag before39acceptance/mainCI.

#45 M/L milestone: Maya-M APIacceptance+fresh1K/4K/8K+8raw-pick requests0differences PASS (warmup32s vs375s retained as cold-I/O outlier). Maya-L APIacceptance+fresh1K/4K/8K PASS; last framed raw-pick gate running. Both use measured primary100401 table; final evidence/mainCI pending. Candidate includesmainhandoff e9663f8 via4971b37; latter CI41308044569/41308039304 PASS.

#45 GPU checkpoint complete: S/M/L API/prefill/raw-pick gates PASS; S16+M8+L8 framed requests all raw-pick0. L warmups806s then49s on the same candidate, no timeout/retry/override (temporary long-start wrapper prepared but never used). All5binary hashes remainc3132ee4; both engines/server stopped, daemon/YAML untouched. Final evidence CI, PR#49 merge and mainCI pending; no nexttag before those pass. #33/#42/#11/#16/#46 remainopen.
