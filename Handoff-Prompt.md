# Handoff-Prompt — restart prompt for a fresh session

Read `/home/ugo/software/project-maya/Handoff.md` **in full** first; it is the live state. Also read `AGENTS.md`.

**Project:** TP2 for GLM-5.3-Flash on twin gfx1151 Strix Halo hosts over Thunderbolt RDMA. Repo
`/home/ugo/software/project-maya`, origin `uv-genai/maya-tp`, branch `main`, release v1.1.37.

**Rules:**
1. HIP/ROCm gfx1151 Linux only. GPU inference; CPU only for slice-math unit tests.
2. RDMA for tensors. Bootstrap TCP is handshake/control/token IDs only.
3. One GitHub issue per workstream; comment during work; commits `refs #N`. Push only `origin`, never `upstream`.
4. You are **on strix-halo-2**. SSH only to strix-halo-1; **never nested SSH**.
5. Use `.venv/bin/python` and `jq`.
6. Kill engines with `pkill -9 -x maya`. Remote server kill: `pkill -9 -f "[s]erver.py --engine"`.
   Kill the server **before** its engine to prevent respawn. Never use `pkill -f server.py`.
7. After every build, deploy to rank 0 **both** `build-hip/maya` and `/tmp/maya-tp2` after killing processes;
   verify MD5 across hosts. Current binary **`75a180099201dbc0abdf29808a536022`** on all three paths.
8. Keep `Handoff.md` current; it is the restart source of truth.

**Completed — do not redo:** slicing + RDMA, serve mirror/health/kill-recover, single-conversation snapshot reuse,
pack-derived geometry (#12), and **#13 MTP protocol** (engine commit `5b547ce`; prior WIP preserved in `74a47d8`).
There is **no uncommitted engine WIP to rescue**. Both GPUs are free (engines/servers stopped).

**#13 outcome:** shared-expert window OOB fixed. The wire desync was **thinking-budget force injecting op 11
inside MTP before its op-12 accept frame**. Rank 0 now owns anchor/draft tokens (op 13), acceptance **and stop**
(op 12); no nested force instructions during MTP. Incomplete decode sends abort (op 14); worker exits on bad/truncated
frames. Force-reset and snapshot mirror failures send `ERR`, so HTTP does not wait silently.

**MTP is opt-in**: `MAYA_TP_MTP=1` on rank 0 for serve, both ranks for paired CLI; leave `MAYA_GLM_NO_MTP` unset.
Default TP2 stays the simple loop. Greedy verification, fixed draft width default 2. DFlash and multi-seq stay gated.
Verified: 3-turn reuse, think-budget force, stream cancel + next request, output limits 1/2/3; all 8 peer kill/recover
checks; pair parity 32/32; near-ties 6/32 vs single-host MTP; single-host output identical to its saved baseline.
CI trio passes (HIP map, 464 serve tests / 11 skips, 614 tools tests / 31 skips).
Regression command against an opt-in pair: `.venv/bin/python tools/tp2/mtp_acceptance.py`.
Kill/recover: `MAYA_TP_MTP=1 bash tools/tp2/acceptance.sh`.

**Perf:** short free-greedy pair runs simple 37.81 / MTP 44.06 ms/tok; verify 75.52 ms/round.
Same-reference follow run had cold-I/O noise (44.20 / 58.50); do not claim a precise delta. MTP has no demonstrated
speed win. Overlapped AR (gufo `StartPartial`/`FinishPartial`) is future work; **no `MAYA_TP_OVERLAP` knob exists**.

**Open:** #1 umbrella, #10 optional gather design, **#16 vision deferred (tracking only until user asks)**, #11 multi-seq still gated (`SEQS>1` prompt `mv_rdna`).
`glm_rows.cu` has the same full-width shared `FFs` bug the window had; fix when explicitly resuming #11.
**#14 benchmark completed (Handoff §13):** same resident pair, fixed 256-token continuation, two warmups per
mode, four measured runs/mode: MTP off **38.255 ms/token, 26.14 tok/s**; MTP on **52.235 ms/token, 19.14 tok/s**.
MTP throughput −26.76% on this workload, 25.97% draft acceptance. All measured runs have zero disk I/O.
Harness `tools/tp2/bench_mtp.py`; durable data/report `docs/benchmarks/tp2-mtp-2026-10-30/RESULTS.md`.
Fixed-length microbenchmark ignores EOS (reference crosses turn boundaries); default gate unchanged.
**#15 deployment + launchers verified (Handoff §14):** MTP-capable HIP target built and matching binary copied to
both hosts, same MD5. Canonical rank-0 wrapper `tools/llama-swap/tp2.sh` now accepts
`run PORT --mtp --drafts N` (1..8) and forwards both fixed count and capacity to rank 1; contexts match.
Live rank-0 YAML `~/.config/llama-swap/config.yaml` model `maya-glm5-tp2` already points to this wrapper,
**unchanged**, so still MTP off until the user appends `--mtp --drafts N` to its `cmd`.
Remote wrapper backed up as `tp2.sh.pre-mtp-15`; new scripts deployed, six wiring tests and a two-turn four-draft
GPU smoke pass. CI trio: HIP mapping, 464 serve / 620 tools tests. User will run tests themselves now;
**leave both GPUs free and do not start more inference without their request**.
Follow the next user request; don't reopen completed #13 diagnosis.

**Operations:** start rank 1 first (`bash tools/tp2/rank1.sh`), then rank 0 on strix-halo-1
(`bash tools/tp2/rank0.sh 18080`). Never probe bootstrap port 18515 with `/dev/tcp`.
For remote background launch, use `cd ... || exit; setsid nohup ... >/tmp/log 2>&1 </dev/null &`;
`cd ... && server &` leaves a subshell holding SSH pipes open and hangs the caller.
Logs: rank 1 `/tmp/tp2-r1.log`, rank 0 engine repo `maya-maya-s-v2-iq2_xxs-hip.log` (append), server `/tmp/tp2-r0.log`.
An idle worker holds ~45 GB VRAM, normal. Protocol/RDMA failure requires restarting **both** ranks.
Runbook: `docs/TP2-RUN.md`. Full session evidence and caveats: Handoff §11.
