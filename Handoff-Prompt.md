# Handoff-Prompt — Maya TP2 restart

Read `/home/ugo/software/project-maya/Handoff.md` in full, especially §18 and §21, then `AGENTS.md`.

Project: **Maya-TP**, GLM-5.3-Flash TP2 across twin Strix Halo hosts (Maya-S default; Maya-M packed and
pair-benched). HIP/ROCm **gfx1151 Linux only**; GPU inference, RDMA tensor path only.
Bootstrap/control/token metadata may use TCP.
Repository `uv-genai/maya-tp`, local branch main. No subagents unless explicitly authorized.

## Current state

**v1.1.40 is published** (`d9248e8`, CI41173208751); #31 closed (Handoff §22) — do not republish.
v1.1.39 remains `20ff03a`; v1.1.38 remains `c322b3e`. Do not repeat completed measurements or start a
new release without an explicit request.

**#29 — model-selectable TP2 launchers + Maya-M (open for leftover notes):**
- All operational launchers take `--model NAME` / `--pack PATH` / `--config PATH` or `MAYA_MODEL` /
  `MAYA_PACK` / `MAYA_CONFIG` (resolver `tools/tp2/model_env.sh`). Same selection on both ranks.
- Docs: `docs/TP2-RUN.md` “Choosing a model”, README, MODELS, SETTINGS, Handoff §9, AGENTS.md #8.
- Maya-M packed on **both** hosts; `run_bench.sh --model Maya-M` → **32/32 parity**, 42.57/42.83 ms/tok.
- **Maya-M full MTP suite PASS** (same as Maya-S): `docs/benchmarks/tp2-maya-m-mtp-2026-10-31/RESULTS.md`
  — API MTP/simple matrices, mtp_acceptance, CLI 32/32 both modes, kill/recover 8/8,
  `bench_mtp` +8.11% @2 drafts (41.12→38.03 ms/tok).
- **Maya-S + Maya-M are the validated TP2/MTP models.** **#30 open for Maya-L** (same suite when
  packed on both hosts). README scope, MODELS, TP2-RUN, CHANGELOG updated accordingly.
- `run_bench.sh` scps rank0’s token save before parity (`36b38ef`).
- Live llama-swap YAML still Maya-S entries.

Current engine MD5 **`5dbc93560e0eb024dcbdc0d9eb60ea57`** matches local `build-hip/maya`, rank0
`build-hip/maya` and `/tmp/maya-tp2`. Never rebuild on the remote dirty checkout.
Test engines/server stopped; GPUs free; llama-swap daemon running.

## Conclusions / evidence

Maya-S corrected MTP workloads (unchanged claims): technical +10.8% @1 draft, code +29.6% @2; report
`docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md`. Do not reuse #14 numbers.
Maya-M: pair bench only (above) — not an MTP speedup claim.
CPU gates: HIP map; serve and tools suites green (629 tools tests at last run).

## Operations / constraints

1. You are **on strix-halo-2** (rank1). SSH **only strix-halo-1.local** (rank0). **Never nested SSH**.
   Do not invoke the llama-swap wrapper over SSH, since it SSHes back. Use direct rank scripts.
2. Kill remote server **before** engine to prevent respawn:
   `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'`.
   Local engine: `pkill -9 -x maya`. Never use `pkill -f server.py`.
3. After every engine build, copy to BOTH rank0 binary paths (`build-hip/maya` and `/tmp/maya-tp2`) and
   verify all three hashes. `/tmp/maya-tp2` can disappear — redeploy if missing.
4. Rank0 checkout is older `7ee8f27` with staged/unstaged work. Preserve it: no pull/reset/rebuild there.
5. Use `.venv/bin/python`, jq. Track each workstream through its own GitHub issue, comment at milestones,
   link commits `refs #N`, push only origin, and explicitly close or leave open. Keep Handoff current.
6. Start rank1 locally before directly SSH-launching rank0. Pass **the same** `--model`/`--pack`/`--config`
   (or `MAYA_*` env) to both ranks. MTP env on both:
   `MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=N MAYA_GLM_MTP_DRAFT=N`; unset MAYA_GLM_NO_MTP.
   Simple: `MAYA_GLM_NO_MTP=1`. Match contexts (32768 for API tests).
   Never probe bootstrap 18515 with /dev/tcp; restart BOTH ranks after any poisoned communicator/peer failure.
7. Stop test engines/server after finishing, leave llama-swap daemon/config alone. User's YAML entries:
   `maya-glm5-s-tp2` and `maya-glm5-s-tp2-mtp` (currently `--mtp --drafts 7`).
8. #16 vision is deferred; #11 multi-sequence stays gated. #10 gather bit-exact reduction remains optional.
9. Do not hardcode model paths in launch scripts (AGENTS.md convention 8). Document new flags in
   `docs/TP2-RUN.md`.

Check **#30** if the user wants Maya-L TP2/MTP validation (pack first, then the suite in the issue).
Otherwise follow the next user request. Do not repeat completed measurements or publish a new release
automatically.
