# Handoff-Prompt — Maya TP2 restart

Read `/home/ugo/software/project-maya/Handoff.md` in full (especially §18, §21, §23, §24, **§27–§28**), then `AGENTS.md`.

Project: **Maya-TP**, GLM-5.3-Flash TP2 across twin Strix Halo hosts. HIP/ROCm **gfx1151 Linux only**;
GPU inference, RDMA tensor path only. Bootstrap/control/token metadata may use TCP.
Repository `uv-genai/maya-tp`, local branch **main**, tracking origin/main, including prefill commit
`9bc0ec9` (PR #34 merged as `b63f8af`) and release commit `6471995`.
Rank0 is at `777cc92` with user-modified `maya.py` and two untracked Maya-L config backups; preserve them.
No subagents unless explicitly authorized.

## Current state

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
- **#29** — launchers + Maya-M; work done, issue still open (close when user agrees).
- **#1** umbrella, **#10** optional, **#11** multi gated, **#16** vision deferred.

**Launchers:** `--model NAME` / `--pack PATH` / `--config PATH` or `MAYA_MODEL` / `MAYA_PACK` /
`MAYA_CONFIG` (resolver `tools/tp2/model_env.sh`). Same flags on **both** ranks.
Runbook: `docs/TP2-RUN.md` “Choosing a model”.

**llama-swap on strix-halo-1** (`~/.config/llama-swap/config.yaml`, user-owned — do not edit unless asked):
six entries `maya-glm5-{s,m,l}-tp2` and `-tp2-mtp` with explicit `--model`; drafts **2** on MTP rows.
Maya-L uses **`--context 524288`** (also in `maya-maya-l-hip.json`). Daemon `--watch-config` running.
Backups sit beside the YAML.

**Binary:** `5a52cd27167523b6ea4b0f0bde93f503` on local `build-hip/maya`, rank0 `build-hip/maya`,
and `/tmp/maya-tp2` (built with `STRATA_ENABLE_TP2_RDMA=ON`; installer always passes that flag now).

**Engines:** tests finished/stopped, then **user started Maya-M TP2 under llama-swap** during release
preparation. At the read-only check: context 32768, rank0 API :5814 (dynamic), status ok / peer up,
rank1 worker running locally. **Leave this user pair running.** Daemon/YAML untouched.
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
   The binary is unchanged. Do not leave worktrees dirty without saying so.
5. Use `.venv/bin/python`, jq. One issue per workstream on `uv-genai/maya-tp`; comment at milestones;
   `refs #N`; push only origin; close or leave open explicitly. Keep Handoff current.
6. Start rank1 locally before SSH-launching rank0. Pass the **same** `--model`/`--pack`/`--config` (or
   `MAYA_*`) to both ranks. MTP: `MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=N MAYA_GLM_MTP_DRAFT=N`; unset
   `MAYA_GLM_NO_MTP`. Simple: `MAYA_GLM_NO_MTP=1`. Match contexts (32768 for API tests unless told).
   Never probe bootstrap 18515 with `/dev/tcp`. Restart **both** ranks after poisoned communicator.
7. Stop only your own test engines/server after finishing; preserve the current user-started Maya-M pair.
   Leave llama-swap **daemon and YAML** alone unless the user
   asks. Live model ids: `maya-glm5-{s,m,l}-tp2` and `-tp2-mtp`.
8. #16 vision deferred; #11 multi gated (`glm_rows.cu` shared-width/attention audit first); #10 optional.
9. Do not hardcode model paths in launch scripts (`AGENTS.md` convention 8).
10. **Prefill performance (#33)** is the open perf topic — not decode. Quote stage lines if diagnosing.

Finish any explicitly pending release step only if asked, then follow the next user request.
