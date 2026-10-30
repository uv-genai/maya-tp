# Handoff-Prompt — Maya TP2 restart

Read `/home/ugo/software/project-maya/Handoff.md` **in full**, especially §16, then `AGENTS.md`.

Project: GLM-5.3-Flash/Maya-S TP2 across twin Strix Halo hosts. HIP/ROCm **gfx1151 Linux only**;
GPU inference, RDMA tensor path only. Bootstrap/control/token metadata may use TCP.
Repo `uv-genai/maya-tp`, local branch main, release version **v1.1.38**.

## Current state

**#18–#21 fixes are implemented and fully validated** (Handoff §16). The user then explicitly resumed
release #17. VERSION is 1.1.38; release notes are in CHANGELOG.md. Source release with an annotated v1.1.38
tag and no binary assets; publication/CI outcome is tracked in #17 and Handoff §17.

Binary MD5 **`d6cd72ab0376c8df1d58cd1c2a20441b`** matches local `build-hip/maya` and both rank0
`build-hip/maya` + `/tmp/maya-tp2`. Test engines/API server stopped, both GPUs free. Llama-swap daemon running.
Live YAML model names now `maya-glm5-s-tp2` (simple) and `maya-glm5-s-tp2-mtp` (`--mtp --drafts 7`),
set by the user; YAML untouched by us. Canonical launcher supports `run PORT --mtp --drafts N` and propagates
MTP fixed count, capacity and context to both ranks. Do not launch extra inference after finishing this work
unless needed for a new user request.

## Fixes / evidence

- #18: globally agree before MTP scratch retry, so both ranks replay with matching collective shapes.
- #19: prefill cap 3008 rows / 49283072 bytes below 50331584, RDMA agreement on local scheduling decisions;
  dense lead FFN partials also summed before residual.
- #20: RAM experts already compact; event-ordered copy replaces erroneous double-slicing.
- #21: window attention uses local 32 heads; HIP scratch capacity tracks head/KV dimensions.
- Both simple and seven-draft MTP pass 990/7290/14550-token API prompts, large reuse and stream DONE.
- Eight rank1-only forced scratch retries confirmed rank0 global replay; 150/150 round traces match.
- Seven-draft cancellation/follow-up, limits 1/2/3, CLI 32/32 pair parity, 8/8 kill/recovery pass.
- HIP map, 464 serve tests (11 skips), 621 tools tests (31 skips) pass. Host boundary + gfx1151 attention parity pass.
- Durable report/raw API data: `docs/validation/tp2-collectives-2026-10-30/RESULTS.md`.
- Historical MTP comparison #14 predates these correctness changes; rerun before current perf claims.

## Operations / constraints

1. You are **on strix-halo-2** (rank1). SSH **only strix-halo-1.local** (rank0). **Never nested SSH**.
   Do not invoke the llama-swap wrapper over SSH, since it SSHes back to rank1. Use direct rank scripts.
2. Kill remote server **before** engine to prevent respawn:
   `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'`.
   Local engine: `pkill -9 -x maya`. Never use `pkill -f server.py`.
3. After every engine build copy to BOTH rank0 paths above, after stopping processes; verify MD5 all three.
4. Rank0 checkout is older `7ee8f27` with staged/unstaged work: preserve it, do not pull/reset/rebuild there.
   Current authoritative source is this checkout/origin; deployed binaries are current on both hosts.
5. Use `.venv/bin/python`, jq. Track one GitHub issue per workstream, comment while progressing,
   commits `refs #N`, push only origin, explicitly close/leave open; keep Handoff.md current.
6. Start rank1 locally first (`CONTEXT=32768 bash tools/tp2/rank1.sh`), then rank0 script directly over SSH.
   MTP raw engine env: `MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=7 MAYA_GLM_MTP_DRAFT=7` on both;
   leave MAYA_GLM_NO_MTP unset. Simple: MAYA_GLM_NO_MTP=1 on both. Diagnostic retry default off.
   Never probe bootstrap 18515 with /dev/tcp. Restart BOTH ranks after any poisoned communicator/peer failure.
7. No subagents authorized unless requested. #16 vision remains tracking-only/deferred; #11 multi-seq gated.
   `glm_rows.cu` still needs its own shared-width/attention audit before re-enabling multi-sequence work.

Follow the next user request. Do not redo completed diagnosis or publish another release automatically.
