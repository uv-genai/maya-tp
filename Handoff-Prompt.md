# Handoff-Prompt — Maya TP2 restart

Read `/home/ugo/software/project-maya/Handoff.md` in full, especially §18, then `AGENTS.md`.

Project: GLM-5.3-Flash/Maya-S TP2 across twin Strix Halo hosts. HIP/ROCm **gfx1151 Linux only**;
GPU inference, RDMA tensor path only. Bootstrap/control/token metadata may use TCP.
Repository `uv-genai/maya-tp`, local branch main. No subagents unless explicitly authorized.

## Current state

The user authorized autonomous MTP performance investigation and code review, and will not use the
GPUs until it is finished. **#22 diagnosis, measurements and hardware/CPU validation are complete**.
Final report/tooling commit, GitHub CI and issue disposition are tracked in #22; check that issue before
repeating work. See Handoff §18. Test engines/server are stopped, GPUs free, llama-swap daemon running.

Source release **v1.1.38** is already published at `c322b3e`; #17 is closed. Do not redo or retag it,
or publish another release automatically. Subsequent fixes on main are recorded under Unreleased:

- #23 `dc4c484`: keep replicated NextN outputs local and use its full shared-expert width.
- #24 `56b9755`: verifier includes both sliced shared-expert partials.
- #26 `c8758cb`: ordinary token path reduces dense lead FFN partials before residual.
- #25 `ec22f4d`: HIP RDNA GEMV handles padded window output strides.

Current engine MD5 **`5dbc93560e0eb024dcbdc0d9eb60ea57`** matches local `build-hip/maya`, rank0
`build-hip/maya` and `/tmp/maya-tp2`. Source fixes are committed/pushed. The final tooling/report commit and CI outcome are tracked in #22.
Never rebuild on the remote dirty checkout.

## Conclusions / evidence

Corrected natural TP2 workloads, common 256-token references, all raw picks agree (0/256), resident experts,
zero measured disk/CPU expert work, balanced ABBA measurements with excluded warmups:

| Workload | Drafts | Simple tok/s | MTP tok/s | Gain |
| --- | ---: | ---: | ---: | ---: |
| Technical | 1 | 25.82 | 28.60 | +10.79% |
| Technical | 2 | 25.97 | 27.79 | +6.99% |
| Technical | 3 | 25.95 | 24.78 | −4.53% |
| Technical | 7 | 25.51 | 15.67 | −38.58% |
| Code | 1 | 25.64 | 31.30 | +22.07% |
| Code | 2 | 25.86 | 33.53 | +29.64% |

One-host technical control gains 26–27%; TP2 accelerates ordinary decoding more than verification.
Draft acceptance differs by workload (one draft: technical 70.5%, code 86.8%); wide rejected suffixes waste work.
No evidence establishes a quantization contribution or a universal 30% gain. Defaults/YAML unchanged.
Report/raw data: `docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md`.
Old #14 and intermediate ablations predate numerical fixes; do not present them as current gains.
Forced emitted IDs alone hid defects; benchmark now requires raw model-pick agreement by default.

Required CPU gates pass: HIP map; 464 serve tests/11 skips; 625 tools tests/31 skips.
Source CI41081377055 passes. Thirty-two new padded-output GPU parity cases plus existing MV suite pass.
Simple/seven-draft API matrices pass 990/7290/14550-token prompts, reuse, stream DONE and peer health.
MTP cancel/follow-up and limits1/2/3 pass; eight forced rank1-only retries coordinate; 94/94 round records match.
CLI pair outputs 32/32 and captured hidden-state bytes match in both modes (32 simple rows, 1 initial MTP row).
Seven-draft peer kill/recovery passes 8/8. Final tooling/report CI and issue closure are tracked in #22.

## Operations / constraints

1. You are **on strix-halo-2** (rank1). SSH **only strix-halo-1.local** (rank0). **Never nested SSH**.
   Do not invoke the llama-swap wrapper over SSH, since it SSHes back. Use direct rank scripts.
2. Kill remote server **before** engine to prevent respawn:
   `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'`.
   Local engine: `pkill -9 -x maya`. Never use `pkill -f server.py`.
3. After every engine build, copy to BOTH rank0 binary paths above and verify all three hashes.
4. Rank0 checkout is older `7ee8f27` with staged/unstaged work. Preserve it: no pull/reset/rebuild there.
   Current authoritative source is this checkout/origin. Only runtime binaries were deployed remotely.
5. Use `.venv/bin/python`, jq. Track each workstream through its own GitHub issue, comment at milestones,
   link commits `refs #N`, push only origin, and explicitly close or leave open. Keep Handoff current.
6. Start rank1 locally before directly SSH-launching rank0. MTP env on both:
   `MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=N MAYA_GLM_MTP_DRAFT=N`; unset MAYA_GLM_NO_MTP.
   Simple: `MAYA_GLM_NO_MTP=1`. Match contexts (32768 for API tests). Diagnostic retry defaults off.
   Never probe bootstrap 18515 with /dev/tcp; restart BOTH ranks after any poisoned communicator/peer failure.
7. Stop test engines/server after finishing, leave llama-swap daemon/config alone. User's YAML entries:
   `maya-glm5-s-tp2` and `maya-glm5-s-tp2-mtp` (currently `--mtp --drafts 7`).
8. #16 vision is deferred; #11 multi-sequence stays gated. `glm_rows.cu` needs its own shared-width/attention
   audit before enabling it. #10 gather-based bit-exact reduction remains optional future work.

Check #22 for final CI/issue status; finish any still-running final gate, then follow the next user request.
Do not repeat completed measurements or publish a new release automatically.
