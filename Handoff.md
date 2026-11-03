# Handoff — Maya-TP (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-11-03 · Current state: **§43**, DeepSeek launch scripts ready for user testing; §42 records implementation and staged validation; §§38–41 record GLM startup and DeepSeek design/plan; §33 records the promoted GLM runtime.
> PR #94 merged as `6f85d30`; #33/#90/#91/#93 are closed. The user launches their processes.
> §§10–32 are dated history, including superseded status, deployment paths and pending actions.
> Current process-ownership rules in §2 and the promotion state in §33 override those snapshots.

## 1. What we are building

Two-way **tensor parallel (TP2)** for **GLM-5.3-Flash** (**Maya-S, Maya-M and Maya-L** passed the full
TP2+MTP suite on earlier builds and user trials on the promoted build; geometry is pack-derived via #12) across two twin hosts:

| Host | Role | IP |
| --- | --- | --- |
| `strix-halo-1.local` | rank 0 (API server) | 192.168.1.22 |
| `strix-halo-2.local` | rank 1 (worker; usual agent workspace) | 192.168.1.92 |

- **Platform (hard rule):** HIP/ROCm **gfx1151 (Strix Halo) Linux only**. No CUDA/NVIDIA work.
- **Interconnect (hard rule):** **RDMA only** over Thunderbolt `usb4_rdma*` (gufo's path). No TCP data path (bootstrap TCP: QP handshake + serve instruction mirror + MTP accept wire only).
- **Reference implementation:** gufo at **`/home/ugo/projects/gufo`**. See `docs/models/qwen3.8-flash-next/TP2.md`, `src/cli/serve/tp_executor.cpp` + `tp_control.hpp`. MTP pattern: rank 0 `PlanDecode` → wire drafts → rank 1 `DecodeStepPlanned(forced_drafts)`.

## 2. Rules (must follow)

1. **Track all work through GitHub issues on `uv-genai/maya-tp`.** Comment while progressing. Link commits (`refs #N` / `closes #N`).
2. **Push only to `origin` (`uv-genai/maya-tp`). Never push to `upstream` (`mw00/project-maya`).**
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. Perf claims need **controlled** measurements. Use `--spec-follow` for near-tie counts.
5. **NEVER use nested SSH.** The usual workspace is on strix-halo-2; SSH directly to strix-halo-1 from there.
6. **Python uses `.venv/bin/python`.** Shell uses `jq`.
7. **Stop only validation processes you started**, using recorded PIDs/process groups. The user launches their own processes. No name-wide `pkill`, port-wide killing, or unrelated service inspection/control; prior validation permission is not standing permission to stop user processes.
8. **Binaries are per-host.** Select matching review builds on both hosts and verify hashes before testing. Preserve the promoted canonical production binaries and the protected remote source; do not overwrite them as routine test setup. `run_bench.sh` uses rank0 `/tmp/maya-tp2`; verify its executable rather than assuming it matches production.
9. **Stop an owned rank0 server before its owned engine** to prevent respawn and release its executable. Track those processes at launch. Do not clear a busy environment by name or port. Older harness cleanup is tracked in #83 and must not run against user processes.

## 3. Git / GitHub

| | |
| --- | --- |
| `origin` | `https://github.com/uv-genai/maya-tp.git` |
| `upstream` | `https://github.com/mw00/project-maya.git` (read/merge only) |
| `main` | tracks `origin/main` |
| Release version | **v1.1.42** published from `6471995` ([#35](https://github.com/uv-genai/maya-tp/issues/35)); previous v1.1.41 = `a6d15a7`, v1.1.40 = `d9248e8`, v1.1.39 = `20ff03a`, v1.1.38 = `c322b3e` |

## 4. Issue map

| # | Workstream | Status |
| --- | --- | --- |
| 1 | Umbrella (index) | open |
| 2–8 | Partition/queue/parity/fast-path/serve/calibration/RDMA | closed |
| 9 | MTP / spec decode TP | closed by **gating**; superseded by #13 |
| 10 | Gather bit-exact variant | open (optional design) |
| 11 | Multi-seq + snapshot reuse | **partial — snapshot/reuse + rows ARs done; multi gated (`seq_count>1` prompt `mv_rdna`). NOTE: `glm_rows.cu` still has the full-width `FFs` bug (see §11.4) — fix when re-enabling multi.** |
| 12 | Pack-derived TP geometry | closed |
| 13 | MTP under TP2 | **Complete (§11). Opt-in `MAYA_TP_MTP=1`; simple loop remains default.** |
| 14 | Historical MTP on/off benchmark | complete; predates numerical fixes, superseded by #22 (§18) |
| 15 | MTP deployment + llama-swap launcher verification | complete; runtime flags and both-rank propagation (§14) |
| 16 | GLM vision on HIP + RDMA TP2 embedding synchronization | open, deferred by user (§15) |
| 17 | Release v1.1.38 | published from c322b3e; closed (§17) |
| 22 | MTP limited speedup: sweep and code review | complete (§18); final report CI/status tracked in #22 |
| 23 | Replicated NextN reduced / shared width truncated | fixed and validated `dc4c484` (§18) |
| 24 | Rank1 shared partial omitted in verifier | fixed and validated `56b9755` (§18) |
| 25 | Padded outputs bypass HIP RDNA GEMV | fixed `ec22f4d`, GPU parity/microbench pass (§18) |
| 26 | Ordinary token dense lead FFN missing AR | fixed and validated `c8758cb`, raw picks 0/256 (§18) |
| 27 | Maya-TP README identity, scope and upstream attribution | completed/closed at 427b5bb; CI41091273791 passed (§19) |
| 28 | Release v1.1.39 | published from `20ff03a`; tag/CI/publication verified; closed (§20) |
| 29 | Maya-M TP2 + model-selectable launchers | **closed** — launchers parameterized; Maya-M packed + full suite PASS (§21); closure recorded during #37 |
| 30 | Maya-L TP2 + MTP validation | **complete** — suite PASS, report `docs/benchmarks/tp2-maya-l-mtp-2026-10-31`; docs updated |
| 31 | Release v1.1.40 | published from `d9248e8`; tag/CI/publication verified; closed (§22) |
| 32 | Release v1.1.41 | published from `a6d15a7`; tag/CI/publication verified; closed (§24) |
| 33 | TP2 prefill throughput | **closed by user acceptance** — tuning and pool improvements promoted; row cap and cold variability remain (§33) |
| 35 | Release v1.1.42 | **published** — `6471995`, CI41203815519 PASS, annotated tag/public latest release verified (§28) |
| 42 | TP2 CLI readiness after expert warmup | **open** — existing Maya-L warmup skew needs a peer-ready handshake; WARM=0 cold CLI is very slow; tracking only during #37 |
| 18 | MTP rank-local scratch retry diverges collective shapes | fixed and validated (§16) |
| 19 | Large prefill exceeds 48 MiB RDMA window | fixed and validated (§16) |
| 20 | Large-prompt cache double-slice | fixed and validated (§16) |
| 21 | MTP window attention head width / HIP scratch capacity | fixed and validated (§16) |
| 36–81 | Sequential upstream v1.1.37–v1.1.46 integration | completed; per-tag evidence in `docs/upstream/`, final #77/#36 closed; deferred #46/#52/#54/#60/#68/#75/#79 remain open |
| 82 | User/agent documentation audit | documentation refreshed; final checks, commit/CI and outcome recorded on GitHub |
| 83 | Validation harness cleanup ownership | open; replace name-/port-wide cleanup with owned PIDs/process groups |
| 85 | Direct cumulative upstream v1.1.49 integration | closed; v47–49 reviewed together (§31) |
| 90 / 91 | Prefill pool sizing and DSA tuning | closed after user acceptance; promoted in PR #94 (§33) |
| 93 | Cold Maya-L baseline output variation | closed as not planned at user request; cause unresolved (§33) |

## 5. What works (Maya-S pack)

- **RDMA pair:** `usb4_rdma5`, RoCE v2, `gid_index=1`, `mtu=4096`, write striping.
- **Full TP2 slicing:** mixer (KDA 64→32 heads, DSA head-slice), experts (2048→1024), shared expert, dense FFN (12288→6144), attn_output col-slice. Two ARs per layer.
- **Bench** (`tools/tp2/run_bench.sh`): both ranks `--tokens`; pair parity **100%**.
- **Serve** (`tools/tp2/rank1.sh` then `rank0.sh`): rank 0 mirrors `TpMirrorHdr` before every collective; rank 1 worker executes them. `curl /v1/chat/completions` works; multi-turn reuse works (simple loop).
- **Peer health:** `/health` → `tp.peer: up|down`, `status: degraded` after first failed mirror send. Sticky `TP2_DOWN` until both ranks restart.
- **TP2 + MTP:** opt-in `MAYA_TP_MTP=1` (leave `MAYA_GLM_NO_MTP` unset). Three-turn serve with restore/reuse, thinking-budget force, and cancel/follow-up verified (§11). Rank 0 owns inputs, acceptance and continuation; tensors stay on RDMA.
- **Acceptance:** `tools/tp2/acceptance.sh` is a legacy broad-cleanup harness (#83); agents must not run it against user processes.
- **Decode under TP2:** simple loop is the default; MTP is opt-in and greedy (see #13).

## 6. Performance (Maya-S IQ2_XXS)

**Historical 2026-10-30 corrected engine**, framed chat, warm resident 256-token common references, all raw picks agreed on that build:

| Workload | Drafts | Simple tok/s | MTP tok/s | Gain |
| --- | ---: | ---: | ---: | ---: |
| Technical | 1 | 25.82 | 28.60 | +10.79% |
| Technical | 2 | 25.97 | 27.79 | +6.99% |
| Technical | 3 | 25.95 | 24.78 | −4.53% |
| Technical | 7 | 25.51 | 15.67 | −38.58% |
| Code | 1 | 25.64 | 31.30 | +22.07% |
| Code | 2 | 25.86 | 33.53 | +29.64% |

One-host technical control: 18.55→23.40 tok/s (+26.11%, one draft), +27.26% with two.
Acceptance and verification amortization govern the gain; TP2 accelerates ordinary tokens more than verification.
All experts resident, zero measured disk/CPU expert work. No defaults/YAML changed.
Durable report/raw data: **docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md**.

Historical #14 and early 32-token results in §13/§11 predate numerical fixes and are not current claims.
Link measurements: 6.04 µs tiny WRITE RTT, 37.4 Gb/s at 1 MiB; ~9 Gb/s at 10–48 KiB AR rows.
Blocking AR ~58 µs; `MAYA_TP_OVERLAP` is NOT implemented.

## 7. Architecture

```
include/strata/core/tp_partition.hpp   TpPartition + tp_slice_expert
include/strata/core/tp_context.hpp     TpContext (create from pack widths — #12)
include/strata/net/{communicator,verbs}.hpp + src/net/{verbs.cpp,tp_kernels.cu}
src/core/{tp_context,glm_model,glm_window,glm_rows,glm_mtp}.cu  slice + AR hooks + MTP accept wire
src/core/glm_prefill.cu                prefill ARs (+ TP_AR_TRACE sites)
src/program/generate.cpp               TpMirrorHdr (+ op 12 MtpRound), tp_send, worker, serve gates, MTP wire
serve/server.py                        TP2_DOWN → /health peer down; child_env passes env to the engine
tools/tp2/{rank0.sh,rank1.sh,run_bench.sh,acceptance.sh}
build-tp2/tp_comm_rdma_test            --bench --hidden N (needs STRATA_ENABLE_TP2_RDMA=ON)
docs/TP2-RUN.md                        ops runbook
```

**Call mirroring:** send immediately **before** each collective `prefill`/`forward` (not inside `forward()`). Rank 1: `--tp-rank 1` without `--tokens`.

**MTP mirror:** op 10 starts decode with sampler/request metadata. Op 13 wires the anchor and each draft window before collective verification. Op 12 wires accepted token IDs and `next_token=more` after rank 0's emit callback (also for the first anchor and T=1). Op 14 aborts an incomplete decode. `force_next` remains mirrored outside MTP; inside MTP no op 11 is injected, since accepted tokens already carry forced results. Same wire works for paired CLI requests. Bounds/op mismatch or truncated frames stop the worker; restart both ranks after failure.

## 8. Critical gotchas

0. **6 µs / 37 Gb/s ≠ AR cost.** 6 µs = 2 B WRITE RTT; 37 Gb/s = 1 MiB plateau. Blocking AR ~58 µs; overlapped residual = gufo's 6 µs form.
1. **Do not probe bootstrap with `/dev/tcp`** — steals rank 0's `accept()`.
2. **Start rank 1 first** (or within rank 0's 120 s accept window).
3. **Shard stride:** GGUF expert stride is full `gu_row * n_ff_exp`; `Ly.gu_bytes` is sliced.
4. **`NOSLICE=1` + AR doubles the FFN** (both ranks full compute then AR). Isolation = `NOSLICE=1` + `NOAR=1`.
5. **Drain:** `hipStreamSynchronize` before `CheckQueued`/`EndOperation`.
6. **Raw picks:** use `--spec-follow` diagnostics as well as emitted IDs. Same-topology ordinary/MTP picks must agree; forced IDs alone hide defects. Cross-topology reduction order can change greedy text.
7. **GCC 16 + HIP `<format>`:** `-include format` (already in CMake for TP targets).
8. **`ibv_create_cq_ex` / `IBV_SEND_INLINE` unsupported** on Thunderbolt.
9. **`RecvControl` must loop on EAGAIN.**
10. **Name-based cleanup can kill unrelated work.** Use only recorded owned PIDs/process groups (§2.7–9); never copy historical broad cleanup commands into a new run.
11. **SETTINGS:** only implemented `MAYA_TP_*` knobs are real (see `docs/SETTINGS.md`). `MAYA_TP_OVERLAP` is NOT one.
12. **Per-host binaries + scripts** — match selected validation paths after a build, preserving production (§2.8/2.9).
13. **Gufo MTP pattern:** rank 0 plans and wires inputs/decisions; Maya uses fixed draft width (default 2), canonical input tokens (op 13), and accepted tokens/continuation (op 12).
14. **Idle TP worker holds ~45 GB VRAM with zero kernels** — normal (persistent worker waiting on the mirror socket). Memory residency alone is not a reason to terminate a user's worker.
15. **rank 0 logs are split:** engine stderr → repo `maya-maya-s-v2-iq2_xxs-hip.log` (**append mode** — multiple runs; segment by the last `glm prefill: CUDA0 chunks` line); server narration → `/tmp/tp2-r0.log`. rank 1 has everything in `/tmp/tp2-r1.log`.
16. **Engine restart during a request wipes TP INFO** — `/health` can briefly report `world_size: 1`; wait for `INFO tp_world=2` before judging. **Poisoned verbs communicator is sticky** — restart both ranks after any `size mismatch`/`timed out`.

## 9. How to run

### Model selection (all launchers)

Flags or env (resolver: `tools/tp2/model_env.sh`; details: `docs/TP2-RUN.md` “Choosing a model”):

| Input | Meaning |
| --- | --- |
| `--model NAME` / `MAYA_MODEL` | `$MAYA_MODELS_DIR/NAME/pack` (default dir `/home/ugo/software/Maya-data/models`) |
| `--pack PATH` / `MAYA_PACK` | explicit pack (wins over model) |
| `--config PATH` / `MAYA_CONFIG` | base `maya-*.json` for serve scripts |

Default pack remains Maya-S. Same flags/env on **both** ranks. `acceptance.sh` uses env only;
`llama-swap/tp2.sh` forwards to both ranks. Maya-M is packed on both hosts (#29).

### Benchmark (pair, controlled)
```sh
bash tools/tp2/run_bench.sh                      # default Maya-S
bash tools/tp2/run_bench.sh --model Maya-M       # or --pack PATH / MAYA_MODEL=…
# explicit single-host reference:
P=/home/ugo/software/Maya-data/models/Maya-M/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P \
  --tokens $TOKS --max-new 32 --greedy --max-context 4096
```
`run_bench.sh` scps rank0’s `--spec-follow-save` before parity (`36b38ef`). TP2+MTP variant: same as §13
but pass `--model`/`--pack` if not Maya-S.

### Serve (rank 1 first, no `/dev/tcp` probe)
```sh
bash tools/tp2/rank1.sh --model Maya-M     # local; --pack/--config or MAYA_* also work
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya || exit; \
  setsid nohup bash tools/tp2/rank0.sh 18080 --model Maya-M >/tmp/tp2-r0.log 2>&1 </dev/null &'
curl http://192.168.1.22:18080/health   # expect tp.peer=up
```
For an agent-started validation pair, record exact PIDs at launch and stop only those processes,
server before engine. User-started processes are outside agent cleanup scope.

Runbook: **`docs/TP2-RUN.md`**. Settings: **`docs/SETTINGS.md`**.

### Debug knobs (MTP / #13)
- `STRATA_GLM_TP_AR_TRACE=1` — every window/prefill/rows `all_reduce` site/size; `decode_mtp` round lines; **MTP wire lines**: `wire-send keep=… p0=…` (rank 0) and `wire-recv wait`/`wire-got keep=…` (rank 1). Must be exported **before** launching rank scripts (server.py passes its env to the engine via `child_env`; a respawned engine loses it).
- `STRATA_GLM_MTP_FIXED=<n>` / `STRATA_GLM_MTP_ADAPT=0` — fixed draft width.
- Logs: rank1 `/tmp/tp2-r1.log`; rank0 engine `maya-maya-s-v2-iq2_xxs-hip.log` in the repo (append — see §8.15); rank0 server `/tmp/tp2-r0.log`.

## 10. Serve mirroring (how it works)

Rank 0 `tp_send` before each collective: `{op, n, next_token}` + `int32` tokens on bootstrap TCP.

| op | meaning |
| --- | --- |
| 0 | `model.reset()` |
| 1 | `model.prefill` (+ `next_token`) |
| 2 | `model.forward` |
| 3 / 4 | snapshot / restore (reuse) |
| 5 | bind context id |
| 10 | `decode_mtp` + `TpMtpSp` POD (+ optional penalty history) |
| 11 | `force_next` |
| 12 | MTP accept wire: `n=keep`, `next_token=more`, payload = accepted token IDs (also initial anchor and T=1) |
| 13 | MTP inputs: canonical anchor or draft-window token IDs before collective verification |
| 14 | MTP abort: incomplete decode; worker stops and both ranks must restart |

Sticky pair-down prints `TP2_DOWN` on stdout; server sets `status=degraded` / `tp.peer=down`.

## 11. #13 completed — protocol fix and opt-in MTP (session 3)

### 11.1 Preserved work and root cause

Prior WIP committed first as **`74a47d8`** (window shared-expert `FFs` slice fix, accept-wire groundwork, AR traces, diagnostic gate).
The traced old binary `e668184c` reproduced failure at the thinking budget on **turn 1**:
rank 1 waits for op 12 at `p0=44 / produced=30`, rank 0 accepts two tokens and advances.
The handoff's assumption that rank 0 never forces under TP2 was false: the **thinking-budget branch of
`on_token` calls `tp_force(think_end)`**, injecting op 11 before the pending op-12 accept frame.
The worker consumes the wrong header, then resumes in the middle of the protocol.

### 11.2 Fix implemented

- Rank 0 owns the anchor and draft window tokens (op 13), accepted tokens and continuation (op 12).
  Canonical draft inputs also prevent draft-head near-ties from giving the two collective windows different inputs.
- No nested force op while MTP is decoding; forced tokens already ride the accept frame.
- Continuation includes EOS, cancellation, output/context limit, and rank-0-only loop-guard decisions.
  Initial-anchor exit and T=1 use the same wire. Synchronized counters permit safe output/context clamps again.
- Bounds and expected op checked **before** reading a token payload. Protocol failures, truncated payloads,
  or disconnect stop the worker. Rank 0 sends op 14 on incomplete decode, marks pair down, rejects subsequent requests.
- Failed force-reset and snapshot save/restore mirror sends terminate the request with `ERR` (previously restore
  could silently return while HTTP waited forever). Accepted recovery for a failed pair remains restart both ranks.
- **MTP gate decision: opt-in `MAYA_TP_MTP=1`.** Default stays simple loop; MTP's prior 44.2 ms/tok vs 41–43
  simple is not a speed win. Serve flag on rank 0; paired CLI flag on both ranks. Greedy only for opt-in MTP.
- `tools/tp2/mtp_acceptance.py` exercises three turns/reuse, thinking-budget forces, stream cancel/follow-up,
  and output limits 1/2/3. Existing `acceptance.sh` propagates MTP opt-in remotely; its SSH launch now backgrounds
  only the redirected server command (the old `cd ... && server &` held SSH pipes open and stalled the harness).

### 11.3 Final verification (engine commit `5b547ce`)

- Opt-in MD5 `9e1ed096`: turn outputs 64/64/35 (last EOS), cached tokens **14 / 85** on turns 2/3.
- Stream disconnect confirmed server `cancel=True`; next 16-token request works. Limits 1/2/3 pass; peer remains up.
- Traces: **113/113** round-end `(p0, produced, keep)` tuples identical across ranks, **110 sends / 110 receives**.
- CI trio: HIP map **PASS**; **464 serve tests** (11 skips), **614 tools tests** (31 skips), both PASS.
- Latest binary MD5 **`75a180099201dbc0abdf29808a536022`** includes silent-ERR peer-loss fix. Deployed to local
  `build-hip/maya` and rank 0 `build-hip/maya` + `/tmp/maya-tp2`; hashes match.
- **Final binary: all 8 kill/recover acceptance checks pass.** After killing rank 1, chat fails cleanly and health
  is degraded/peer down; restarting both ranks restores health and chat.
- Final `mtp_acceptance.py`: three 64-token turns, reuse 14/85 tokens on turns 2/3; stream cancel and follow-up,
  limits 1/2/3 all pass; peer up. (First turn also reused 14 tokens from acceptance's smoke request.)
- CLI **simple and MTP pair parity 32/32**, both free generation and `--spec-follow`.
- Single-host MTP **32/32 identical to the saved pre-session baseline** `/tmp/single-mtp.txt` (42.01 ms/tok).
  MTP TP2 follows that reference with **6/32 near-tie differences**; simple TP2 has 2/32 on this run (prior 4/32).
- Short free-greedy runs: simple **37.81 ms/tok**, MTP **44.06 ms/tok**; MTP verify **75.52 ms/round**.
  These are different free-greedy continuations. The same-reference follow runs were **44.20** simple / **58.50**
  MTP ms/tok, with unequal cold I/O (4.66 / 8.67 ms/tok; MTP prefill took 10 s after the 89 GB single-host run).
  Do not claim a precise performance delta from those cold samples. They provide no reason to enable MTP by default.
- Logs: `/tmp/maya-13-{acceptance,final-mtp-acceptance,final-gates,serve-tests,tools-tests}.log`,
  `/tmp/maya-13-{simple,mtp,single}-*.log` (single log is `/tmp/maya-13-single.log`).
- Final state: both engines/servers stopped; both GPUs free; MD5 verified on both rank-0 paths after final build.

### 11.4 Remaining scope and cautions

- `glm_rows.cu` still has full-width shared `FFs` (same latent OOB as window had). Multi stays gated (#11);
  fix when explicitly resuming that issue.
- Overlapped AR (`StartPartial`/`FinishPartial`, gufo) is future performance work; **no `MAYA_TP_OVERLAP` knob**.
- Protocol changed: always deploy matching binaries on both ranks, killing the remote server **before** its engine.
- #13 work is complete in `5b547ce`. #14 measured MTP on/off performance (§13). #1 stays open as the umbrella,
  #10 is optional, #11 remains open with multi-sequence decode gated. Use a separate issue for new optimization work.

## 12. Key commits

`a4c10d5` Handoff (pre-#13 restart) · `bd4607d`/`af0c398` AR traces · `63d3966` emit/greedy/n=2 sync · `57c30ce` window/rows TP dims · `64ba3c6` whq/n=0 drafts · `aec2829` peer health · `805ee0b` serve mirroring · `0b9f5c4` v1.1.37 · earlier TP2 slice/transport.
**`74a47d8`** preserves the prior verified WIP. **`5b547ce`** completes the protocol fix, failure handling, MTP opt-in and regression harness (§11).

## 13. #14 controlled warm MTP on/off benchmark (session 4)

- User requested a fresh direct comparison. **No engine changes/rebuild**; MD5 remains `75a180099201dbc0abdf29808a536022`.
- Reproducible stdlib harness: `tools/tp2/bench_mtp.py`. Same resident pair, MTP loaded for both modes;
  toggle `GEN spec=1` / `spec=0`, following the identical 256-token TP2 simple-loop reference via `--spec-follow`.
- 12-token prompt, context 4096, MTP fixed width 2; two excluded warmups per mode, four measured runs per mode,
  balanced ABBAABBA order. All measured outputs identical; all measured VRAM hits 100%, zero disk time.
- **MTP off: median 38.255 ms/token, 26.14 tok/s** (range 38.163–38.301).
- **MTP on: median 52.235 ms/token, 19.14 tok/s** (range 52.169–52.302).
- MTP: **26.76% lower throughput**, **36.55% higher token latency** on this workload; 87/335 drafts accepted
  (25.97%), 168 rounds, 1.52 tokens/round; drafting 9.95 + verify 69.56 + commit/cache 0.09 ms/round.
- First excluded simple warmup was 75.47 ms/token (36.91 ms/token disk wait), then 37.80 warm; this confirms
  why the older cold 32-token samples were noisy. Only the warm repetitions enter the result.
- Fixed-length microbenchmark: EOS sentinel 2147483647 and loop guards off. Reference crosses turn boundaries
  at positions 51/125/187; do not generalize this acceptance rate to all natural chats. Default gate unchanged.
- Durable report + raw results + exact reference: **[docs/benchmarks/tp2-mtp-2026-10-30/RESULTS.md](docs/benchmarks/tp2-mtp-2026-10-30/RESULTS.md)**.
- Logs: `/tmp/maya-mtp-bench-14-progress.log` and `/tmp/maya-mtp-bench-14/{reference,benchmark}-rank{0,1}.log`.
- Re-run: `.venv/bin/python tools/tp2/bench_mtp.py --output /tmp/maya-mtp-comparison --tokens 256 --runs 4`.
  Use a fresh output directory; harness stops both hosts afterward. Both GPUs are free.
- Local CI trio PASS: HIP mapping, 464 serve tests (11 skips), 614 tools tests (31 skips). Benchmark helper
  `--help` and compilation pass. Test suites ran only after both GPUs were free. No engine defaults changed.

## 14. #15 MTP deployment and llama-swap verification (session 5)

- User is taking over manual testing. **Leave both engines/servers stopped**; do not launch further GPU tests
  without a new request. HIP gfx1151/MMQ/RDMA `maya` target builds successfully; MTP is compiled in (no special
  MTP build flag). Target was already current. Binary deployed to both hosts and rank-0 `/tmp/maya-tp2`,
  MD5 remains **`75a180099201dbc0abdf29808a536022`** on all three paths.
- Actual rank-0 `/home/ugo/.config/llama-swap/config.yaml` model **`maya-glm5-tp2`** points to
  `/home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT}`; cmdStop uses the same wrapper's `stop`,
  proxy `http://127.0.0.1:${PORT}`. **Live YAML unchanged**; it still defaults to the simple loop.
- Canonical wrapper now accepts **`run PORT --mtp --drafts N [--context N]`**, or `--no-mtp`. N=1..8, default 2.
  It sets `MAYA_TP_MTP`, **both** `MAYA_GLM_MTP_FIXED` and `MAYA_GLM_MTP_DRAFT`, and the context on **both** ranks.
  Setting FIXED alone above 3 would otherwise silently hit the default capacity of 3. Raw engine calls use env vars,
  not native `--mtp` / `--drafts` flags. MTP under TP2 remains greedy.
- Wrapper starts `tools/tp2/rank1.sh` remotely on strix-halo-2 and execs `tools/tp2/rank0.sh` on rank 0.
  Supervisor killed before engine, bracket-safe server pattern; SSH startup errors propagate; remote launch is a
  fully redirected `setsid nohup ... &` with no backgrounded `cd &&` subshell. No bootstrap socket probes.
- Rank scripts now match context: explicit `CONTEXT`, otherwise the base JSON's `--max-context` (currently 32768).
  Wrapper fixes SEQS=1 while #11 remains gated. Old `tools/tp2/tp2-swap.sh` delegates to the canonical wrapper.
  Older `maya-rank0.sh` / `maya-rank1.sh` are not the llama-swap path; use the canonical scripts above.
- Remote wrapper had divergent changes; preserved as `tools/llama-swap/tp2.sh.pre-mtp-15` before deployment.
  New wrapper and both rank scripts are deployed to both systems; SHA256 matches.
- Verification: six no-GPU launcher tests (SSH/process commands mocked); directly orchestrated two-turn GPU smoke
  with **four drafts** and context 4096 passes, both 16 tokens, turn-2 cached_tokens=14; peer up. Logs prove rank1
  `n=4`, `mtp=1`. No nested SSH: rank1 local, rank0 launched directly over one hop; full llama-swap wrapper SSH
  propagation is tested with mocks, not by invoking the wrapper through nested SSH.
- Gates: HIP map PASS, 464 serve tests (11 skips), **620 tools tests** (31 skips). Scripts `bash -n` PASS.
- To test on **strix-halo-1**:
  `/home/ugo/software/project-maya/tools/llama-swap/tp2.sh run 18080 --mtp --drafts 4`.
  For llama-swap edit just that model's `cmd` to append `--mtp --drafts 4`, then restart the model.
  Stop: same wrapper `stop`. Full examples: `docs/TP2-RUN.md`, settings in `docs/SETTINGS.md`.

## 15. #16 deferred vision work

User requested tracking only; **do not implement until asked**. GLM-5.3-Flash has a compatible vision tower/projector
in the quant repo's `vision/` folder (`mmproj-GLM-5.3-Flash-F16.gguf` and `GLM-5.3-Flash-vocab.gguf`).
Current HIP installer skips GLM vision; neither host's deployed config enables it. Future work must enable GPU
encoding on gfx1151 and synchronize image positions/embedding tensors across TP2 (tensors over RDMA, not TCP).
Scope/evidence/acceptance: https://github.com/uv-genai/maya-tp/issues/16 . Issue intentionally left open and deferred.
No model files downloaded, no runtime configuration changed, no inference processes started/stopped for this task.

## 16. #18–#21 regressions — fixed and validated

**These regressions blocked the original release attempt.** User stopped Jen testing and explicitly authorized
fixes and GPU validation, then resumed release #17 after validation (see §17). Original errors were MTP-seven
`outgoing=0/539/16384 incoming=0/539/131072` and non-MTP large-prompt
`all-reduce payload of 118521856 bytes exceeds the receive window of 50331584`.

Root causes / fixes:
- **#18 `aee442e`:** rank-local MTP window scratch retry changes AR shapes from T rows to one row/token.
  `TpContext::agree_min` exchanges/drains/checks two-float RDMA votes before accepting/retrying. Any clash makes
  both ranks restore and replay. Diagnostic `MAYA_TP_TEST_WINDOW_RETRY=N` defaults off.
- **#19 `9623e7e`:** cap prefill by `MaxPartialBytes()` and pack hidden width (3008 aligned rows at 4096,
  49283072 bytes < 50331584). Agree on readiness, chunk and layout retries despite differing local budgets
  (25088 vs 32768). Dense lead FFN `continue` previously skipped AR; now sum its sliced outputs too.
  Pure host boundary helper/test covers exact 7234 rows/118521856 bytes, registered in CMake and unittest.
- **#20 `a732362`:** prefill prestage (T>=1024) double-sliced RAM blobs already compacted by preload/readers,
  causing nonfinite routes / CPU memcpy SIGSEGV. Copy compact Ly.blob over the event-ordered H2D stream instead.
- **#21 2e44a8a:** window q_b/k_b/v_b are sliced to 32 heads but MLA calls still computed 64;
  correct all calls to local heads, preserve allocation strides. HIP partial scratch now tracks head/KV dimensions,
  avoiding reuse of 80x32-head capacity for 65x64-head work. New HIP parity regression covers that transition.

Validation on binary **`d6cd72ab0376c8df1d58cd1c2a20441b`**, gfx1151/MMQ/RDMA:
- Identical hashes on local `build-hip/maya`, rank0 `build-hip/maya` and `/tmp/maya-tp2`.
- Context 32768, MTP fixed/cap 7: prompt 990 ->64 output; prompt 7290 ->32; cached follow-up 7333 ->24
  (7289 cached); prompt 14550 ->32; streamed follow-up DONE; peer up throughout. Same matrix passes simple mode.
- **Eight rank1-only forced retries**: rank0 local-clash=0/retry=1 at matching positions; no shape divergence.
  All **150/150** MTP round-end `(p0,produced,keep)` records identical between ranks.
- Seven-draft three-turn reuse, stream cancellation after 8 text chunks + successful follow-up, limits 1/2/3 PASS.
- CLI simple and seven-draft MTP **32/32 pair parity** each.
- Peer death/restart with seven drafts: **8 passed, 0 failed**. Chat errors cleanly/health degraded after rank1 kill;
  restarting both recovers peer and chat. Used acceptance orchestration with draft env forwarded explicitly.
- HIP map PASS; serve **464 tests / 11 skips** PASS; tools **621 tests / 31 skips** PASS.
- Host helper passes unittest and CMake target. `glm_attn_bench --parity-only` on gfx1151 PASS, including
  32->64-head shared-output bit parity for all/tail/chunk masks at 2051 selected cells. build-tp2 cache corrected
  from old gfx1100 to gfx1151 before executing. Engine build-hip was already gfx1151 throughout.

Durable report/raw API data: **docs/validation/tp2-collectives-2026-10-30/RESULTS.md**.
New live harness: `.venv/bin/python tools/tp2/regression_collectives.py --mode mtp|simple --output PATH`.
Logs: `/tmp/maya-21-{mtp,simple}-{tests,r1}.log`, `*-results.json`, `*-r0-engine.log`,
`/tmp/maya-21-{mtp-acceptance,cli-checks,kill-recover,attn-parity}.log`, and CI logs
`/tmp/maya-20-serve-tests.log` / `/tmp/maya-21-tools-tests.log`.

Final operations: **test engines and API server stopped, both GPUs free**. Llama-swap daemon remains running.
Live rank0 YAML (user-requested 2026-10-31, applied **after** deploying model-aware launchers to rank0):
four models — `maya-glm5-s-tp2` (`--model Maya-S-v2-IQ2_XXS --no-mtp`), `maya-glm5-s-tp2-mtp`
(`… --mtp --drafts 2`), `maya-glm5-m-tp2` (`--model Maya-M --no-mtp`), `maya-glm5-m-tp2-mtp`
(`… --mtp --drafts 2`). Pre-edit config backup: `config.yaml.bak-2026-10-31-pre-model-flags`.
**Deployed to strix-halo-1 (file copy only, checkout still `7ee8f27`):**
`tools/llama-swap/tp2.sh`, `tools/tp2/{model_env.sh,rank0.sh,rank1.sh}` — originals under
`/tmp/maya-tp2-launchers-pre-2026-10-31/`. Remote `tp2.sh --help` now includes `--model|--pack|--config`.
`--watch-config` lists all four `maya-glm5-*` models. Do not edit this YAML unless the user asks.
Rank0 checkout has older HEAD `7ee8f27` and staged/unstaged user work; **do not pull/reset or rebuild there**
without preserving it. Fixed binaries are deployed directly; the authoritative fixed source is this checkout/origin.
Prior #14 perf numbers predate these output-affecting fixes; rerun a controlled benchmark before new claims.

## 17. v1.1.38 source release (#17)

The user explicitly resumed release work after #18–#21 passed GPU tests and GitHub CI at `08e242b`.
`VERSION` is 1.1.38; `CHANGELOG.md` records all changes since v1.1.37, leading with MTP/large-prompt correctness.
Annotated tag/release: **v1.1.38**, title **v1.1.38 — TP2 MTP and large-prompt fixes**.
Source release matching the prior format; no binary assets. Publication/commit/CI outcome is recorded in
[issue #17](https://github.com/uv-genai/maya-tp/issues/17) and the
[GitHub release](https://github.com/uv-genai/maya-tp/releases/tag/v1.1.38).

At release time, preparation changed version metadata and documentation only. The deployed gfx1151 binary hash was
`d6cd72ab0376c8df1d58cd1c2a20441b`; GPU validation is in §16 and the durable report.
That metadata-only release required no engine rebuild/deployment or rank0 checkout/YAML edits.
Release-commit CI41072147262 passed before publication. Post-release engine changes and current hashes are in §18.
Historical #14 performance is explicitly marked as predating the correctness fixes; no current MTP speedup claimed.

## 18. #22 MTP performance/code review — completed diagnosis and validation

User authorized all testing/review; will not run workloads until completion. No agents used.
Published v1.1.38 remains at `c322b3edaf3380457ec98662a9d17c95ae415a22`, release CI41072147262 passed.
New fixes below are committed/pushed to main, not included in that tag. VERSION remains 1.1.38.

### Findings and changes

- **#23 `dc4c484`:** NextN layer is fully replicated but fast_dsa/fast_moe wrongly reduced full outputs.
  Keep NextN local and use `ff_compute_for(il)` at all three draft shared GEMVs; include full draft shared output.
  Isolated old-reference one-draft acceptance 58.75→80.85%, draft~5.3→4.1ms; speed+3.06→18.56%.
- **#24 `56b9755`:** all five trunk window inclusion paths now include rank1 sliced shared partial.
  Old-reference ordinary/verifier differences 55→18/256.
- **#26 `c8758cb`:** ordinary fast_layers dense lead FFN skipped AR (prefill/window already sum).
  Reduce before residual. Fresh corrected ordinary reference gives **0/256 raw picks differences in both modes**.
  The remaining18 differences were a numerical bug, not harmless reduction-order near ties.
- **#25 `ec22f4d`:** HIP RDNA dense GEMV honors padded y stride and dispatch permits packed inputs/padded outputs.
  Thirty-two new Q6_K/BF16 NT1…8 output/guard tests plus existing GPU MV parity all pass.
  Targeted real-shape before/after kernel microbench: 1–52% improvement; not a whole-model gain claim.

Current engine **5dbc93560e0eb024dcbdc0d9eb60ea57**, all three paths hash-matched; source c8758cb.
No source edits since build/deploy. Baseline backup `/tmp/maya-22-baseline-maya` (d6cd72ab);
#23-only `/tmp/maya-23-only-maya`, #23+#24 `/tmp/maya-24-correctness-maya` (f021a5a2).

### Measurement conclusions

§6 lists final corrected-model results; report/raw data saved in
`docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md`.
Technical one-draft acceptance 105/149; code one-draft 118/136. With two drafts useful tokens/round are 2.10 vs 2.56,
while verification is ~67–69 ms, explaining technical +7% vs code +30%. Seven technical drafts yield only 2.35 tokens/round
for ~151 ms draft+verify. Single-host technical +26–27% vs TP +7–11%: ordinary 54→39 ms scales more than verify 72→56 ms.
Own common reference per topology; not identical outputs across topologies or exact causal decomposition.
No experiment establishes a draft-quantization contribution. One draft technical/two code are measured choices.

Old #14/#22 prompt decoded as `[gMASK]Reasoning Effort: MaxThe capital of France is`, lacks chat framing and
crosses answer boundaries. Pre-fix ablations prove implementation effects only, not final speed claims.
Forced emitted-ID equality hid raw pick differences; benchmark now rejects raw-pick disagreements (default 0),
with four unit tests. New flags: --drafts, --reference, --prompt-file, --single-host, --window-profile
(instrumented results excluded), --max-pick-differences (explicit diagnostic tolerance only).
Pinned llama.cpp post-output-norm NextN seeding matches Maya. No alternate batching/replay toggle fixed the 18 pick differences.

### Final validation status / operations

- Required CPU gates PASS in order: HIP map, 464 serve / 11 skips, 625 tools / 31 skips.
  Logs `/tmp/maya-22-{hip-map,serve,tools}-tests.log`; source CI41081377055 at c8758cb PASS.
- Context 32768 **simple and seven-draft MTP** API matrices PASS: 990/7290/14550 prompt tokens,
  large reuse, stream DONE, peer up. MTP cancel/follow-up and limits 1/2/3 PASS.
- Eight forced rank1-only clashes have matching global retry shapes; **94/94 round records match**.
  `/tmp/maya-22-final-trace-audit.json`; no transport/nonfinite/GPU errors in captured API engine logs.
- CLI **32/32** pair tokens agree in simple/seven-draft MTP. Captured hidden bytes match (32 ordinary
  rows in simple, one initial ordinary row in MTP; batched verifier is not covered by the dump).
  `/tmp/maya-22-final-cli-results.json` and `*-cli-progress.log`.
- Seven-draft peer death/restart **8 passed, 0 failed**; final stable `acceptance.sh` forwards fixed count,
  capacity/context to both ranks, stops server before engine and allows cold disk-backed loading.
  `/tmp/maya-22-final-kill-recover-rerun.log`; durable copy `kill-recovery.txt` in the report directory.
- Final report, all raw performance/API data, microbenchmark source/results, trace audit and CLI parity:
  **docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md**. Final tooling/report CI and issue closure
  are tracked in #22; check that issue for the final run/commit. Source-fix CI already passes.
- **Test engines and API server stopped; both GPUs free.** Llama-swap daemon still running on 8080,
  user YAML unchanged. All three binary hashes rechecked after testing. Remote dirty checkout preserved.
- #23–#26 are fixed/validated; #22 has the final diagnosis. No new release created automatically.
  At the end of #22 these changes were unreleased; they are included in v1.1.39 (§20).
  v1.1.38 remains immutable and predates these fixes.

Never nested SSH, preserve rank0 checkout 7ee8f27 dirty work, llama-swap daemon/YAML. Stop server before engine.
Local build tests are ON temporarily; gfx1151. No background CPU work during performance timing; timings are complete.

## 19. #27 Maya-TP README and project identity

User requested a cosmetic rewrite presenting the fork as its own derived project, Maya-TP, focused on
HIP/ROCm gfx1151 Linux tensor parallelism across dual Strix Halo systems over Thunderbolt RDMA.
README now explains speed and larger-model capacity, implemented TP2/MTP/serve/reliability work, current
scope and setup. Current #22 measurements replace inherited NVIDIA benchmark tables; the single-host
technical control is 18.55 vs TP2 25.82 tok/s (~39% gain), with the separate-reference caveat preserved.
MTP examples use one draft; code measurements include two. Larger GLM packs are capacity targets, with
published validation still Maya-S; other inherited engines and deferred vision/multi-sequence are explicit.

Peasant Smith (@mw00), @needmorevram, Maya contributors and foundation/dependency credits are retained.
Donation banners, the GitHub funding entry and historical donation-promotion bullet removed; copyright
and license files preserved. Clone/issues/release links point to uv-genai/maya-tp. Runtime names unchanged.

Documentation/metadata only: no engine build, deployment, inference or live configuration changes.
Checks pass: README local links/anchors, shell syntax of examples, source flags, benchmark facts and
repository-wide donation-reference scan. Inherited platform guides point their removed README sections
back to the pinned upstream README; CONTRIBUTING's benchmark link/methodology points to current TP2 data.
Final commit/CI outcome and closure are recorded in issue #27; no new tag/release.


## 20. v1.1.39 release (#28) — published

User explicitly requested commit, tag, push and release after #27. VERSION is 1.1.39; CHANGELOG has
dated notes covering all changes since v1.1.38: #23/#24/#26 numerical corrections, #25 padded-output
HIP RDNA GEMV, #22 benchmark fidelity/results and #27 Maya-TP identity/credits.

**Outcome:** release commit **`20ff03ad081decfe91d6456fcbfa7818779f9393`** on origin/main.
Release-commit CPU CI [41092302606](https://github.com/uv-genai/maya-tp/actions/runs/41092302606)
passed before tagging. Annotated tag **v1.1.39** resolves to that commit.
[GitHub release](https://github.com/uv-genai/maya-tp/releases/tag/v1.1.39) is public, latest and
non-prerelease, with explicit source notes and no binary assets. **v1.1.38 stays unchanged** at
`c322b3edaf3380457ec98662a9d17c95ae415a22`. Issue #28 closed with the same outcome.

Engine code was already built/deployed/validated in #22. Metadata-only release: no engine rebuild,
GPU run, deployment or live-config change. Functional binary remains
`5dbc93560e0eb024dcbdc0d9eb60ea57` on all three paths; llama-swap daemon/config and remote dirty
checkout preserved. Do not republish; treat the tag as immutable.

## 21. #29 model-selectable TP2 launchers (Maya-M prep)

Operational TP2 scripts no longer hardcode Maya-S. Shared resolver `tools/tp2/model_env.sh`:

| Input | Meaning |
| --- | --- |
| `--model NAME` / `MAYA_MODEL` | `$MAYA_MODELS_DIR/NAME/pack` (default dir `/home/ugo/software/Maya-data/models`) |
| `--pack PATH` / `MAYA_PACK` | explicit pack directory |
| `--config PATH` / `MAYA_CONFIG` | base `maya-*.json` |

Updated: `rank0.sh`/`rank1.sh`, `llama-swap/tp2.sh` (forwards to both ranks), `run_bench.sh`/`run_prof.sh`/`run_noar.sh`,
`acceptance.sh`, `bench_mtp.py --pack/--model`. Serve launchers rewrite the base JSON’s `--glm-pack`/tokenizer
when the model differs from the config’s pack. Docs: `docs/TP2-RUN.md` “Choosing a model”.
Launcher unit tests extended; tools suite **629 tests / 31 skips** PASS; HIP map PASS.

**Maya-M status:** packed on **both** hosts (`iq_pack.py --compat-bf16`; v4 native experts, IQ2_S/IQ3_S);
pack MD5s match across ranks; `audit_split.py --world 2` all block-safe (n_ff 2048→1024, dense 12288→6144, trunk 0..44).
Controlled **`run_bench.sh --model Maya-M`**: pair parity **32/32**, decode **42.57 / 42.83 ms/token** (r0/r1),
prefill 11 tokens ~470 ms. Binary `5dbc9356…` on all three paths (rank0 `/tmp/maya-tp2` was missing once and was
redeployed). `run_bench.sh` now **scps rank0’s `--spec-follow-save`** before parity (`36b38ef`) — a stale local
`r0.txt` had produced a false 3/32 FAIL. NOSLICE logs also showed identical T-lines on both ranks.
Single-host Maya-M is a weak reference (disk streaming, ~79 ms/tok). Engines stopped after the run;
llama-swap YAML and rank0 checkout untouched.

**Maya-M MTP suite (same as Maya-S #22/#16), all PASS** — report
`docs/benchmarks/tp2-maya-m-mtp-2026-10-31/RESULTS.md`: `mtp_acceptance.py`; MTP and simple
`regression_collectives` matrices (990/7290/14550, reuse 7289, stream DONE); CLI pair **32/32**
simple and 7-draft MTP; `acceptance.sh` MTP kill/recover **8/8**; `bench_mtp.py --model Maya-M`
raw picks **0**, simple median **41.12 ms/tok (24.32)** vs 2-draft MTP **38.03 (26.29)** → **+8.11%**
throughput on the 256-token reference. Context 32768 API / 4096 CLI; binary `5dbc9356…` unchanged.
Cosmetic: `/health` `model_label` still “Maya-S” when only the pack is overridden.
Engines stopped; issue thread #29. Maya-L suite is **§23** / issue #30 (closed).

**llama-swap rank0 YAML (user-requested; do not edit unless asked):** six entries —
`maya-glm5-{s,m,l}-tp2` (`--model … --no-mtp`) and `maya-glm5-{s,m,l}-tp2-mtp` (`--model … --mtp --drafts 2`).
Maya-L also passes **`--context 524288`** (matches `maya-maya-l-hip.json` `--max-context 524288`; earlier
32768 was setup default and rejected long prompts). Watch-config lists all six. Backups beside the YAML.
Wrapper/scripts on rank0 are origin/main (supports `--model`/`--context`). Daemon left running.

## 22. v1.1.40 release (#31)

User explicitly requested tag and release with release notes. `VERSION` is 1.1.40; CHANGELOG dated section
covers #29 launcher flags, `run_bench` parity fix, Maya-M pack/bench/MTP suite and S/M scope docs (Maya-L #30).

**Outcome:** release commit **`d9248e8eb55210c5e0b0c425d0b098be7c2d0288`** on origin/main.
Release-commit CI [41173208751](https://github.com/uv-genai/maya-tp/actions/runs/41173208751) passed before
tagging. Annotated tag **v1.1.40** peels to that commit. GitHub release
[v1.1.40](https://github.com/uv-genai/maya-tp/releases/tag/v1.1.40) is public, latest, non-prerelease, with
explicit CHANGELOG-based notes and **no binary assets**. **v1.1.39** stays at `20ff03a`; **v1.1.38** at
`c322b3e`. Issue #31 closed with the same outcome.

Metadata/tooling only: no engine rebuild or deploy. Functional binary remains
`5dbc93560e0eb024dcbdc0d9eb60ea57`. Do not republish; treat tags as immutable.

## 23. #30 Maya-L TP2 + MTP suite — PASS (2026-10-31)

Pack on both hosts (identical MD5s). `audit_split.py --world 2` **all block-safe** after adding
Q3_K/Q4_K/Q5_K/IQ4_NL/BF16 to the tool (`99e1f8a`). Report:
**`docs/benchmarks/tp2-maya-l-mtp-2026-10-31/RESULTS.md`**.

| Test | Result |
| --- | --- |
| `run_bench.sh --model Maya-L` | 32/32, 64.34 / 64.42 ms/tok |
| `mtp_acceptance.py` | PASS |
| MTP + simple API matrices | PASS |
| CLI simple / 7-draft MTP | 32/32 / 32/32 (CLI MTP: `STRATA_GLM_WARM=0` both ranks) |
| `acceptance.sh` MTP | 8/8 |
| `bench_mtp` 2-draft | PASS, raw picks 0; simple **42.81** vs MTP **32.62** ms/tok → **+31.24%** |

**Gotcha:** Maya-L rank0 cold expert warm can be minutes longer than rank1 → CLI first-AR timeout and
poisoned verbs. Serve is fine (rank1 waits for mirrors). For paired CLI: `STRATA_GLM_WARM=0` both ranks.
`bench_mtp` READY timeout now **900 s**. Installer `62ee2e7` always configures `STRATA_ENABLE_TP2_RDMA=ON`.
Binary **`5a52cd27167523b6ea4b0f0bde93f503`** on all three paths. Engines stopped. Issue #30 closed.

## 24. v1.1.41 release (#32)

User requested tag and release. `VERSION` 1.1.41; CHANGELOG dated section covers #30 Maya-L suite,
`audit_split`/`bench_mtp` tooling, installer `STRATA_ENABLE_TP2_RDMA=ON`, pack-override log/label.

**Outcome:** release commit **`a6d15a7a64775f005e9840f6c4fd0a0818ef1dc4`**.
CI [41191735148](https://github.com/uv-genai/maya-tp/actions/runs/41191735148) passed before tagging.
Annotated tag **v1.1.41** peels to that commit; release public/latest/non-prerelease, **0** assets.
**v1.1.40** stays `d9248e8`; v1.1.39 `20ff03a`; v1.1.38 `c322b3e`. Issue #32 closed.
Metadata only — no engine rebuild. Do not republish.

## 25. #33 TP2 prefill performance — open (2026-10-31)

User: decode/MTP generation is fine (2 drafts); **prefill** is not — **9.6 tok/s on the 11th request**.
Issue [uv-genai/maya-tp#33](https://github.com/uv-genai/maya-tp/issues/33).

Initial interpretation was corrected in §27: the **copy-lane span includes idle mixer windows**;
it does not mean 90% copying (4782 ms chunk → 4380 ms span, only 124 ms actual copies).
Long prompts cap at **~3008 rows**
(RDMA 48 MiB window). Short prompts: fixed lead 0.1–1.2 s + staging → 55–150 tok/s; ≥4K → ~240–270.
Saved matrices (S/M/L): ~1K often 55–65 tok/s (some M ~186); ≥7K ~90–260.

**`MAYA_TP_PREFILL_ROWS`** is **not** a separate issue — same ticket (knob for TP2 prompt rows; still
“not implemented” in SETTINGS, with `MAYA_TP_OVERLAP`). Next: larger window / implement rows knob /
overlap stage+AR; re-measure 1K/4K/8K with MTP 2 drafts.

## 26. Historical snapshot before #33 (superseded by §§27–28)

- Trees: both hosts clean at `origin/main` (`e4c11c3` at handoff time; check `git log -1`).
- VERSION **1.1.41**; releases through v1.1.41 published.
- Packs present both hosts: Maya-S-v2-IQ2_XXS, Maya-M, Maya-L (identical pack MD5s when built).
- Configs on rank0: `maya-maya-s-v2-iq2_xxs-hip.json`, `maya-maya-l-hip.json` (context 524288).
- No GPU processes expected at rest; llama-swap pid listening on :8080 rank0.

## 27. #33 prefill optimization — resumed (2026-10-31)

User requested speeding up prefill. Initial inspection: local main `5067ed9` clean; rank0 now at
`777cc92` with user-modified `maya.py` and two untracked Maya-L config backups. **Preserve rank0's
work; do not pull/reset/rebuild there.** Three deployed binary MD5s still `5a52cd27167523b6ea4b0f0bde93f503`.
Test engines/server stopped at startup. Llama-swap daemon/YAML untouched. Progress is tracked in #33.

**Diagnosis:** the copy-lane label was misleading: in the saved 1052-token chunk, actual copies were
124 ms, ring wait 1072 ms, idle mixer windows 3182 ms. Detailed baseline profile: rank1 KDA projections
30.4%, resident experts 15.7%, mHC/mixer-AR 12.8%, DSA projections 10.0%, staged experts 4.0%.
Rank1 ignored config.env (256-row sub-batches versus rank0's 1024; no Lt tuning). Existing gfx1151
calibrations also omitted six TP2 projection shapes, so these used slow hipBLAS GEMM fallbacks.

**Changes:** rank1 safely inherits config.env, matching server.child_env; 18 gfx1151/hipBLASLt **100401**
calibration entries (six sliced shapes × T=64/256/1024), all measured/accuracy checked. Other versions
unchanged. No engine build/deploy needed; no kernel, decode or transport changes. New fresh-prompt
probe `tools/tp2/bench_prefill.py`. Report/raw data: `docs/benchmarks/tp2-prefill-2026-10-31/`.

**Controlled Maya-S comparison:** context 32768, MTP fixed/cap 2, same six fresh 990/4062/8158-token
prompts, two samples per length. Median prefill ms: **4732→3588 / 16189→12016 / 31416→22897**;
throughput **+31.9% / +34.7% / +37.2%**. Six 8-token greedy messages unchanged; arithmetic is not
bit-exact (calibration relative L2 ≤1.91e-5). Config parity alone was only ~4–6%.
MTP acceptance and 990/7290/14550 + reuse/stream matrix PASS. Maya-M 1K/4K/8K smoke PASS;
Maya-L 1K/4K/8K smoke and 12 cached turns PASS. Turns 11/12: 35 fresh tokens in 603/612 ms
(58.0/57.2 tok/s, 338/372 cached); context 32768, not the live YAML's 524288. First cold L request
still slow: 38.6 s API prompt time, 9.56 s main chunk; rank1 actual copies 6.05 s at ~0.1 GiB/s.
Keep investigating startup/lending and the user's exact long-context workload; no general L speedup
claim without its own A/B. Final CPU gates PASS in order: HIP map, **464 serve / 11 skips**,
**630 tools / 31 skips**; 11 launcher tests PASS. Engines/server stopped after tests; all three binary
MD5s unchanged. Llama-swap daemon (rank0 pid 736) and YAML left alone. Source/review:
**`9bc0ec90ae66e501d597e661facc5a2ebc319711`**, [PR #34](https://github.com/uv-genai/maya-tp/pull/34).
Both GitHub CI runs PASS: [push 41202760879](https://github.com/uv-genai/maya-tp/actions/runs/41202760879),
[PR 41202777839](https://github.com/uv-genai/maya-tp/actions/runs/41202777839).
User subsequently requested release v1.1.42; published outcome is in #35 / §28.

The 3008-row cap and `MAYA_TP_PREFILL_ROWS` remain unchanged/unimplemented. Original user request-11
9.6 tok/s is not reproduced by the controlled fresh-prompt benchmark; #33 stays open for that workload
and further short-request/window work. Rank0 launcher + tuning file deployed; their SHA256s match
local files. Existing user edits preserved. Remote tree is intentionally dirty with those deployments
beside the pre-existing `maya.py` edit/backups; no pull/reset was performed.

## 28. v1.1.42 release (#35) — published

User explicitly requested updating **both handoff documents**, then committing, pushing, tagging and
publishing. Release workstream: [#35](https://github.com/uv-genai/maya-tp/issues/35).
Both handoffs were updated first. PR #34 merged as **`b63f8af381bca7b1228d0e954faacfd69bbe1aa5`**
after both feature CI runs passed at `9bc0ec9`. VERSION/CHANGELOG/README record the scoped prefill
gain, version-specific tuning and remaining limitations.

**Outcome:** release commit **`64719953312df98c9fdf53a10198c106cb195345`** on origin/main (`refs #35; refs #33`).
Release-commit CI [41203815519](https://github.com/uv-genai/maya-tp/actions/runs/41203815519) passed
before tagging. Annotated **v1.1.42** peels to that commit locally and remotely. GitHub release
[v1.1.42 — Faster TP2 prefill](https://github.com/uv-genai/maya-tp/releases/tag/v1.1.42) is public,
latest, non-prerelease, published **2026-10-31 13:07:10 UTC**, **0 binary assets**, with explicit
CHANGELOG-based notes. Older tags, including v1.1.41 at `a6d15a7`, are unchanged. Do not republish.
Issue #35 tracks final verification; **#33 remains open** for cold Maya-L/request-11/row-cap work.

No engine rebuild, binary redeploy or repeated GPU measurements. Binary remains
`5a52cd27167523b6ea4b0f0bde93f503`. Local checkout is **main**, tracking origin/main; the publication
outcome in these handoffs is a documentation follow-up and does not move the release tag.

**Current runtime differs from the completed test snapshot:** user started **Maya-M TP2** under
llama-swap during release preparation. Read-only check: rank0 API **:5814** (dynamic), context **32768**,
`status=ok`, `tp.world_size=2`, `peer=up`; rank1 worker running locally. **Do not stop/restart this user
pair.** Daemon pid 736 and YAML unchanged. Rank0 remains at `777cc92` with user `maya.py` edit,
two config backups, and the deliberately deployed #33 launcher/tuning file; no pull/reset/rebuild there.

## 29. Sequential upstream integration — started (2026-11-01)

User requested new upstream tags merged **one by one**, each verified and validated before the next,
with extreme care for TP2/MTP. Fork-owned README.md/AGENTS.md and other docs are semantic merges only.
Audit **#36**; first tag **#37**, upstream v1.1.37 `7ee8f27`. Common upstream base v1.1.36 `f031a23`;
ten tags through v1.1.46, fetched as **upstream/v1.1.N** to preserve our colliding release tags.
Inventory/guard decisions: `docs/upstream/INTEGRATION-2026-11-01.md` and its JSON commit/file audit.
Do not start v1.1.38 until v1.1.37 passes the full checkpoint. No fork release authorized in this task.

**User tag naming requirement (#40):** future fork releases use **`tp-vMAJOR.MINOR.PATCH`**
(e.g. `tp-v1.1.43`), while `VERSION` remains numeric. Keep published legacy `v*` tags immutable;
fetch upstream tags into `upstream/v*`. The dashboard updater now accepts the fork prefix, preserves
the exact git tag during update and compares/displays its numeric version. Legacy tags still work.
Tests cover release discovery, numeric ordering and an actual fast-forward from a temporary origin.
Legacy installs at v1.1.42 or earlier need one manual `git pull --ff-only` on a clean main checkout
to acquire the prefix-aware updater; document this in the next release notes (docs/SETUP.md).

Current user pair is **Maya-L**, context **524288**, rank0 API **:5812** (dynamic), rank1 local.
User explicitly answered **"Hosts are free; you may stop the pair for validation"**. This authorizes
stopping the pair for per-tag validation throughout this task; do not ask again. Pair stopped safely
(server before engine), baseline binaries preserved under `/tmp/maya-upstream-baseline-2026-11-01/`
on both hosts. Candidate `maya` MD5 **6e59558bdd421fa521040263978b2bd6** deployed to all three
canonical paths and hash-verified for v1.1.37 tests. Daemon/YAML untouched; rank0 dirty user checkout
still protected (no pull/reset/rebuild).

First-tag worktree `/home/ugo/software/project-maya-upstream-review` exists on both hosts; branch
`merge/upstream-v1.1.37` locally, detached candidate on rank0. Merge commit **884a257**, draft PR **#38**;
HIP host-test dependency fix **0892d77** under **#39**. HIP map, **527 serve / 11 skips**, **637 tools /
31 skips**, six budget tests and 11 CMake host checks PASS; maya/maya-ds4 and HIP parity binaries
built for gfx1151. Shared fused-expert parity (196 runs) and attention parity PASS. Seven-draft
MTP acceptance (reuse/cancel/limits, rank1-only forced retry), MTP and simple large-prompt matrices
and CLI simple/MTP pair parity (32/32 each) PASS. All eight benchmark requests have zero raw-pick differences; kill/recover8/8 PASS. M/L API
acceptance and1K/4K/8K smoke PASS; CLI simple/MTP7 pair parity32/32 each on both M and L.
Combined530 serve/11 skips and637 tools/31 skips PASS; both efa221a CI runs PASS. Saved-release
CLI/prefill comparison is running before final evidence/CI and acceptance.
Later tags remain unmerged; do not proceed until the complete #37 checkpoint passes.


### 29.1 Naming and issue maintenance

#40 completed in `dd8e483`, PR #41 merged as `988f834`; both CI runs passed. The future release
prefix is `tp-v`; no release/tag was created. #29 is now closed: its launchers and complete Maya-M
validation were already finished, with evidence in §21. This does not substitute for current-tag tests.

#42 tracks the pre-existing paired CLI warmup skew (§23). The existing WARM=0 workaround avoids
first-collective poisoning but makes cold Maya-L CLI validation disk-bound (simple CLI32/32 PASS:
134.9 s prompt +145.8 s decode). Seven-draft MTP also exercises global scratch retry/replay under
cold I/O. No transport/control/default change in this upstream checkpoint; #42 stays open separately.

### 29.2 v1.1.37 validation complete — final evidence/CI pending

All S/M/L CLI simple and MTP7 pairs32/32; S MTP/simple large-prompt matrices, acceptance,105/105
wire records,8 asymmetric retries, kill/recover8/8 PASS. M/L normal two-draft API acceptance and
1K/4K/8K smoke PASS. Six budget tests,530 serve/11 skips,637 tools/31 skips, HIP map,11 host tests,
all three HIP engine builds,196 fused parity and attention parity PASS. Source controls/launchers/
calibration remain fork-identical; AGENTS adds only user tp-v policy. Evidence docs/upstream/v1.1.37/.

Saved release vs candidate CLI: both modes32/32 identical. Prefill A/B has cold-copy outliers on both
engines; reversed warmup+two-cycle comparison yields1K/4K/8K throughput deltas **+0.66/−1.77/−0.73%**;
all9 paired messages/usage identical (initial6/6 also). No material warm regression observed in this
small check; #33 remains open for cold/API-request11/row-cap work. No general speedup claim.

Both GPUs free; canonical deployed MD5 remains6e59558bdd421fa521040263978b2bd6 on all three paths.
Engine built0892d77; combined sourceefa221a CI41296064467/41296070808 PASS. Final evidence commit
must pass CI, then merge PR #38 with a merge commit (retain upstream ancestry). Do not start38 yet.
Remote canonical source remains777cc92/user edits; new server tested in isolated worktree. At task
end preserve a coherent deployed pair/source state; do not leave an unvalidated old-server/new-engine
combination after later tags. No pull/reset/rebuild of the remote user checkout; baseline binaries
remain available for restoration while validated new builds stay in the review worktrees.

### 29.3 v1.1.37 accepted; v1.1.38 review next (#43)

**PR #38 merged as `7ea9d713ed7765136c081e871eee6648fcf04554`**, retaining upstream v1.1.37
ancestry. Final evidence head `bedc611` passed CI **41298854143 / 41298857613** before merge.
All validation in §29.2 passed; #37 and #39 are closed. #33/#42/#11/#16 remain open. No fork
release/tag created. Local main includes the merge; both review worktrees retain the accepted candidate.

Next checkpoint **#43**: upstream v1.1.38 `cf6472c` (four commits / 13 files), beginning only after
v1.1.37 acceptance. Review Linux GGUF expert direct I/O and headroom, setup context advice, UI model
card/answer information and home-screen redirects. Keep VERSION 1.1.42, fork docs and all TP2/MTP
paths/tuning. New tag is not merged yet. Review worktree will use branch `merge/upstream-v1.1.38`.

### 29.4 v1.1.38 candidate (#43)

Review worktree branch `merge/upstream-v1.1.38`; upstream tag merged there only. Keep raw-GGUF
direct I/O **opt-in** pending gfx1151 measurements; default buffered behavior preserved. This source
is separate from GLM packed TP2. New direct/headroom knobs documented in SETTINGS. VERSION remains
1.1.42; README/AGENTS, GLM control/core, RDMA, launchers and calibration unchanged.

HIP map, **531 serve / 11 skips**, **637 tools / 31 skips** PASS. All three HIP engines build. Extended
file-source host parity: eight configurations PASS with real O_DIRECT, unaligned offsets, variable
formats/shards, ring eviction, queued close and MAYA/STRATA precedence. Changed JS syntax and real HTTP redirect test PASS.
Live TP2/MTP gates and CI remain pending; do not start v1.1.39. Candidate report docs/upstream/v1.1.38.md.

### 29.5 v1.1.38 validation complete — final evidence/CI pending (#43 / PR #44)

Merge9636b1d + alias follow-upfbb5897; direct I/O stays opt-in, both new knobs honor MAYA-over-STRATA.
Host staging **8 configs PASS** with actual O_DIRECT; HIP map,531 serve/11 skips,637 tools/31 skips,
all three HIP builds, JS syntax/real HTTP redirects PASS. Code CI41300406757/41300412067 PASS.

Final engine **cff75b48a0f1e0b2d2355c73ce302d94** deployed and hash-matched on all canonical/review
paths. S MTP7 acceptance, MTP/simple990/7290/14550+reuse/stream matrices, **110** matched wire rounds
and8 asymmetric retries PASS. CLI simple/MTP7 **32/32 each**, both identical to v1.1.37; all eight
resident benchmark requests have zero raw-pick differences. Executed GLM/GPU/TP2 source and defaults
remain byte-identical to the full S/M/L v1.1.37 checkpoint; raw-GGUF changes tested separately on host.

Evidence docs/upstream/v1.1.38.md and directory. Both engines/server stopped; daemon/YAML and remote
canonical user source preserved. Final evidence CI then PR #44 merge required; do not start39 yet.
#33/#42/#11/#16 remain open. No release/tag publication; future fork tags use tp-v.

### 29.6 Post-merge v1.1.38 CI hold — failed restart #47

PR #44 merged as **ad96053f2aaecb284d1f5fe3c165a1d34a36146f** after final1f51079 CI
**41301067233 / 41301072413 PASS**. Post-merge main CI **41301502943 failed** at
FailedRestartKeepsContext: EOF before READY can arrive while the failed process still appears alive.
This pre-existing server lifecycle bug is separate **#47**. #43 reopened/held; **do not import39 yet**.

Review branch `fix/failed-engine-restart` marks a failed start ended, kills/reaps its child and
preserves context/metadata/routes. A deterministic stdout-EOF/lingering-process test reproduced
the old failure; focused restart/watchdog tests6 PASS. Full ordered CPU gates/CI pending. No C++
engine build/deploy; all canonical engine hashes remain **cff75b48a0f1e0b2d2355c73ce302d94**.

#45 is v1.1.39 review only (no candidate merge). #46 tracks its confirmed stream-fetch mismatch:
TP compact reads only write on role0/chunk0, but upstream would mark every no-op job's chunk
ready. Gate TP streaming before any candidate deployment; preserve existing compact reader.
New streaming defaults must stay off until gfx1151 validation. #33/#42/#11/#16 remain open.

### 29.7 Fixed main green; v1.1.39 review resumes (#45)

#47 fixed in PR #48 / **f479b91940c938b6fe9edee3b3d9e6c2402c6737**. Both head a6f1ab5 CI
**41303387525 / 41303664047** and fixed **main41303961934 PASS**. Local ordered532 serve/11 skips,
637 tools/31 skips, HIP map and6 focused restart/watchdog tests PASS. #47 closed; #43's hold is
cleared/closed. Do not retry away historical main failure41301502943. No C++ or binary changes.

Next #45: upstream v1.1.39 `376b4b8` (33 commits/48 files). Still review only, no candidate merge yet.
#46 records confirmed premature readiness with TP compact reads: only role0/chunk0 fills the blob,
but upstream streams every job's bit before that full gather finishes. Keep TP streaming gated off
and preserve buffered defaults pending gfx1151 validation. Preserve answer reserve, chunk/retry
agreement, NextN widths and #33 primary calibration priority. No v1.1.40 work yet.

### 29.8 v1.1.39 semantic candidate (#45)

Review branch `merge/upstream-v1.1.39`, based on b30514b; merge in progress, not accepted/deployed.
Accepted v1.1.38 cff75b48 binary saved on BOTH hosts at `/tmp/maya-upstream-v38-baseline/maya`.
Defer upstream streamed expert fetch as a whole (#46): host reader/header/GPU fetch kernels remain
byte-identical; setting MAYA_GLM_STREAM_FETCH does nothing. Keep MMQ ID padding and ordered Lt fallback,
our primary100401 tuning first. Preserve answer reserve for ALL limits; sentence grace fits within it.
Retain #47 restart fix, TP mirror/MTP wire/NextN/AR/chunk/retry gates. DS4 prefix, tail-auto and link rings
are opt-in until measured. Exclude funding, Intel guide, Windows telemetry/SDK/pinning extensions and
CUDA-native architecture changes; README/AGENTS/VERSION remain ours. Import walker prevents directory
symlink cycles. Focused CPU checks, full ordered gates, HIP builds and live S/M/L checkpoint pending.
No v1.1.40 work before this checkpoint passes. No release/tag; future tags tp-v.

29.8 progress:27fefcc candidate runtime, HIP test harness follow-upsf475729. Local542 serve/11skips,
651tools/31,11budget tests, HIP map,3HIPengines,13host checks,196fused/MMQ/attention and actual gfx1151
Lt primary/forced-fallback numerical+guard checks PASS. Code CI41305518760/41305514302 PASS.
Candidate MD5 c3132ee406429195545f145e04e7aadd deployed/matched on all5canonical/reviewpaths.
S MTP7 acceptance, MTP/simplelarge matrices,105matchedrounds/8asymmetricretries, CLI32/32 each PASS;
CLI continuations matchv38. First math soft-close probes ended naturally, missing coverage; prose
probes running. Kill/recover,rawpicks,prefillA/B,M/L gates pending. No nexttag yet.

Real prose soft-close probes now PASS (16+2/16+24 tokens, total outputs840/1412 tokens plus follow-up).
Remaining batch `/tmp/maya-upstream-v39-remaining.sh` runs kill/recovery,framed raw-pick benches,
reversed3-cycle prefill A/B vs savedv38 (firstcyclewarmup),M/LAPI/prefill smoke andrawpicks.
Evidencecheckpoint in docs/upstream/v1.1.39/. No nexttag until all gates and mainCI pass.

#45 milestone: peer kill/recover8/8 PASS. Framed S technical/code256-token benches16/16 raw-pick0; both ordinary references match historical corrected references256/256. ccab795 CI41306661283/41306656736 PASS. Prefill A/B vs acceptedv38 running; M/L gates and final evidence/main CI pending. Current script /tmp/maya-upstream-v39-remaining.sh, outputs /tmp/maya-upstream-v39. No v40 yet.

#45 prefill A/B vs acceptedv38 complete: all9messages/usage match; warm1K/4K/8K throughput -0.01/+0.18/-1.21% (two measuredcycles afteronewarmup, bothrawarmsretained). No material warm regression/new speedclaim; #33 remainsopen. M/L checks now running; no nexttag before39acceptance/mainCI.

#45 M/L milestone: Maya-M APIacceptance+fresh1K/4K/8K+8raw-pick requests0differences PASS (warmup32s vs375s retained as cold-I/O outlier). Maya-L APIacceptance+fresh1K/4K/8K PASS; last framed raw-pick gate running. Both use measured primary100401 table; final evidence/mainCI pending. Candidate includesmainhandoff e9663f8 via4971b37; latter CI41308044569/41308039304 PASS.

#45 GPU checkpoint complete: S/M/L API/prefill/raw-pick gates PASS; S16+M8+L8 framed requests all raw-pick0. L warmups806s then49s on the same candidate, no timeout/retry/override (temporary long-start wrapper prepared but never used). All5binary hashes remainc3132ee4; both engines/server stopped, daemon/YAML untouched. Final evidence CI, PR#49 merge and mainCI pending; no nexttag before those pass. #33/#42/#11/#16/#46 remainopen.

### 29.9 v1.1.39 accepted; v1.1.40 review next (#50)

PR #49 merged as **d2708b2a6094fda976f1c4b4122d63f16184f3d2** after finalb6483f2
CI **41310209384 /41310204043 PASS** (evidencef3d8f40 CI41309834801/41309830439 also PASS).
Post-merge **main41311902227 PASS**; #45 closed. All §29.8 CPU/build/parity/S/M/L/API/prefill/
fidelity gates passed. S16+M8+L8 framed requests raw-pick0; S CLI32/32 both modes matchesv38.
Warm S prefill A/B vs38: −0.01/+0.18/−1.21%, all9messages/usage equal; no material regression/new
speed claim. Same candidate warmups M32/375s, L806/49s retained; #33/#42 remain open.

Runtime27fefcc, binary **c3132ee406429195545f145e04e7aadd** matched on all5canonical/review paths,
saved on BOTH hosts at `/tmp/maya-upstream-v39-baseline/maya`. Original release5a52 backup and
accepted38cff backup remain. Engines/server stopped; llama-swap daemon175392/YAML untouched.
Remote canonical source still777cc92 with user edits/backups and #33 deployments; no pull/reset/rebuild.

Next **#50 / upstream v1.1.40 cbaaec5** (16commits/55files), not merged yet. Review worktree will
branch `merge/upstream-v1.1.40` from updated origin/main. Exclude new Metal backend/macOS workflow;
keep HIP/gfx1151-only scope, owned docs, #46 complete stream deferral, existing defaults and all
TP2/MTP invariants/calibration. No v1.1.41 work until40 passes its full checkpoint/main CI.

Durable M/L64-token reference IDs now accompany docs/upstream/v1.1.39/{Maya-M,Maya-L}/reference.txt.
Future same-pack/same-prompt benches may use `--reference` with strict ordinary AND MTP raw-pick0
checks, avoiding a second cold startup while proving agreement with accepted outputs. S256-token
references remain in docs/benchmarks/tp2-mtp-review-2026-10-30/. Investigate any difference; never
weaken the fidelity gate to hide it. No release/tag created; future fork tags tp-v.

### 29.10 v1.1.40 semantic candidate (#50)

Review branch merge/upstream-v1.1.40 fromcb86c4c; merge in progress. Exclude all Metal/macOS
source/build/workflow/setup/telemetry/fixtures/docs. README/AGENTS/CMake/launchers and GLM/TP/RDMA/MTP/
primary tuning unchanged; accepted39c3132ee4 engine retained. New server/context/monitor/UI fixes,
request socket timeout (300s per operation, not thinking deadline), Origin/stream errors/update-lock
and installer download recovery retained. DS4 tail heuristic remains opt-in off, corrected to wall
time per token.13 focused socket/config/long-answer/Origin/concurrent-update/prefixed-tag tests PASS.
OrderedCPU/build/host/Node/liveS API gates pending; no v41 before40acceptance/mainCI. Reuse39S/M/L
kernel/CLI/fidelity/performance evidence where source/binary/defaults are unchanged.

#50 update: first builds relinked GLM via archive parent-GGML metadata (#52); fresh binary8facc62c09b1138b6787beda7d2f371d deployed on all5paths, GPUfatbin identical to39. ELF inspection rewrote inspected copies; local39backup restored from untouchedpeer and candidate relinked before GPUtests. Original c31339baseline preserved bothhosts.
Initial551serve/11skips+653tools/31,all3HIPbuilds/DS4host/JS PASS, de0795f CI41313615184/41313611278 PASS. LiveMTP7 acceptance+large matrix PASS. Probe then found pre-existing empty Models catalog (#54), not inference failure. More critical #53 guards uncoordinated context/model restart underTP2;5focusedtests PASS, oldsetter202 failure reproduced. Updated fullCPU/liveS rejection+simple/fidelity gates pending; no41 yet.

#50 runtime gates complete:554serve/11skips+653tools/31/HIPmap/3HIPtargets/DS4host/Node/JS PASS. InitialMTP7 acceptance+large matrix and guardedSource simple matrix PASS;109/37matchedrounds,8asymmetricretries each. Real#53 rejection400 preserves bothPIDs/context32768/peer, then MTPstream97chunks;77metrics samples69generating andOrigin403/actualcontext1M PASS. Models fixture used only in runtime/tmp and removed; default emptycatalog#54 remains. Technical/code256 ordinary/MTP16/16 raw-pick0. Fresh8facc62c binary all5paths verified;39c313backupbothhosts verified after inspection recovery. Engines/server stopped. Final evidence/mainCI pending; no41 yet.

### 29.11 v1.1.40 accepted; v1.1.41 review next (#55)

PR #51 merged as **0a10f5c042df06bd4aede22ea5e62deda14ef2d8** after finalf1799ad
CI **41316570180 /41316566065 PASS**. Post-merge **main41316911427 PASS**; #50/#53 closed.
554serve/11skips,653tools/31,HIPmap,3HIPtargets/DS4marks/Node/JS PASS. S MTP7/simple matrices,
109/37matchingrounds and8asymmetricretries perphase, real#53 rejection with unchanged PIDs/context/
peer and subsequent inference,16framed256-token ordinary/MTP requests raw-pick0 PASS.

Keep #53 safeguards in later merges: Service.paired_restart_reason/_require_single_rank_restart,
context reload_supported/reload_reason and read-only context UI; no rank0-only context/current-model
restart before paired orchestration exists. docs/TP2-RUN.md explains relaunching both ranks.

Fresh executable **8facc62c09b1138b6787beda7d2f371d** matched all5canonical/reviewpaths, saved
on BOTH hosts at `/tmp/maya-upstream-v40-baseline/maya`;39c313 baseline restored/preserved both.
#52 remains open: archive GGML header inherits parentgit metadata (b305→cb86), relinking otherwise
unchanged GLM; GPU image identical. ELF-inspection rewriting was corrected before GPUtests; no
rewritten artifact used. #54 remains open: TP transient config hides catalog. #33/#42/#11/#16/#46 open.
Remotecanonical777cc92/userwork and daemon175392/YAML preserved; test engines/server stopped.

Next **#55 /upstream v1.1.41 1710777** (50commits/82files), no candidate yet. Isolated branch
merge/upstream-v1.1.41 from updatedmain. Audit every new GLM state/snapshot/cancel/routing operation
for mirror order, pool size-class bytes against compact TP blobs, replicated NextN/full widths/ARs,
MTP wire/global retry/chunks, measured tuning/defaults and owned docs. Keep unsupported backends
excluded, #46 stream deferral and #53 reload guard. No42 work before41acceptance/mainCI.

### 29.12 v1.1.41 candidate live validation (#55 / PR #58 / #56 / #57)

Review worktree merge/upstream-v1.1.41, runtime06aff18 plus loop-limit fix76b21b0, final test/source headc127dec. Tag41 remains UNACCEPTED; no42 work yet. New cancellation checkpoint restore is single-host-only/default off; TP2 always mirrors reset. Optional SNAP_GEN retains paired snapshot/forward order. Route rank/fast variants default off; NextN/AR/MTP wire/chunk/retry/primary100401 unchanged. Exclude unsupported platforms, CUDA-only MMQ loader patch (HIP uses original pinned MMQ), new EXL3 GPU variants, balanced RAM planner and DS4 same-host multi-GPU/helper/placement/lazy/fair-copy additions. Retain applicable GLM/CPU/IQ/server fixes and DS4 size-class calibration/timers.

#56 guards new model load/unload plus idle unload/automatic dead-rank restart before single-rank mutation. #57 fixes loop-guard internal continuation beyond total max_tokens: deterministic pre-fix10>9 and134>133 failures; fixed checks total remaining budget. Final STAT regression was racy (reader already saw30 before assertion10); synchronized fake engine with STOP, not runtime timing/retry workaround.

Ordered HIPmap/580serve11skips/654tools31skips PASS. All3gfx1151HIP engines,396 strict IQ GEMV shape/bit/guard cases,196fused cases,MMQ/attention,15host checks, CPU native PDEP0/1 and EXL3 tiers, actual Node monitor PASS. Initial unbuilt host target was then built/passed; CUDA-only glm_mmq_glu target correctly absent. GLM MD5 **13a7d73bd0ae4eac59b4c5c705595beb** deployed/matched all5paths, backups throughaccepted40 preserved bothhosts.

Live batch **/tmp/maya-upstream-v41-batch.sh**, session61826 at handoff, output **/tmp/maya-upstream-v41/**. State harness /tmp/maya-upstream-v41-state.py tests default MTP7/simple matrices, true partial-prefill cancellation with KEEP_CANCELLED=1 (TP must reset), SNAP_GEN=1 reuse/cancel/limits, lifecycle400 without PID/context/peer change. Then SCLI32/32,kill-recover8/8,default+route-opt-in256-token strict raw fidelity, candidate-first3cycle prefill A/B vs saved40 and M/LAPI+prefill+64token strict references. S MTP acceptance passed; rest still running. Do not rerun completed gates without reason. Engines currently running only for this authorized batch; stop at finish. Remote canonical777cc92/userwork and daemon175392/YAML protected. Final evidence/CI/PRmerge/mainCI required before42.

### 29.13 v1.1.41 HOLD — true prefill cancel divergence #59

Batch61826/4407/24411 stopped on failures; do not resume old scripts blindly. Candidate13a7d73b is NOT accepted. Default MTP7 acceptance/matrix and live lifecycle400/PID preservation passed. First cancel fixture36001tokens exceededcontext; second omitted return_progress (completed normally, coverage assertfailed). Corrected alpha18001/progress-enabled test atPP7104/18030, CKPTS16/EVERY4096/KEEP_CANCELLED1, exposed REAL verbs poison: outgoing0/459/8 vs incoming0/459/17825792 after cancellation. Both ranks stopped. Logs /tmp/maya-upstream-v41/cancel-validated/failed-engine.txt and state-summary.txt. #59 tracks it; #55/PR58 held, no42.

Root: rank0's progress callback can abandon a mirrored prefill call while rank1 completes remaining chunk collectives. Fix **32650e8**: finish the entire mirrored call on both ranks, honor STOP at a completed call boundary, then mirror reset; single-host early cancellation preserved. Added tools/tp2/prefill_cancel.py durable live regression. Build95200 completed, fixed MD5 **a6170c07503e139d73789184d9a93971** deployed/hash-matched all5paths; remote review source32650e8. GPUimage95821268… identical to failed13a (read-only audit). Current batch session **15734**, /tmp/maya-upstream-v41-batch.sh; outputs /tmp/maya-upstream-v41/state-summary.txt and fixed-default/,fixed-snap-gen/. Fixed MTP7 acceptance/matrix/lifecycle+true partial-prefill cancellation/cache0 retry/followup PASS;135roundtuples and8asymmetricretries match. Simple matrix+cancellation PASS. SNAP_GEN1 acceptance+large matrix PASS (103matchingrounds,8asymretries plus1naturalboth-rank clash, all9retryshapes match). CLI32/32simple+MTP7 match accepted39/40; kill/recover8/8 PASS. Default+routeopt technical/code256-token32records raw-pick0. Prefill A/B vs40all9messages/usage match, warmthroughput+0.57/-0.97/-1.84% (no newclaim). MAPI/prefill+8rawpickrecords0 PASS; LAPI/prefillPASS, Lstrictfidelitywarming now. After batch15734 succeeds, explicitly run /tmp/maya-upstream-v41-snap-bench.py technical256 with NO --reference (own snapshot-mode reference), outputnew /tmp/maya-upstream-v41/snap-own-technical. This is NOT in the batch. Then final registration/compile of HIP-only iq_mmvq_parity_test after test-only8db5689 changes; enginea617 must remain hash-matched. Final evidence/CI/PRmerge/mainCI before42. No other GPU process may run concurrently. Kernel source unchanged by host-only fix; earlier GPU parities reusable after read-only GPUimage comparison. FinalCI/mainCI pending.

Current CPU580serve11skip/654tools31skip and3HIP/kernel gatespassed beforecontrolfix; c127dec CI41319041457/41319036198 PASS. Fixed32650e8 CI41320039898 PASS; evidence93eaf5f CI41320390172/41320383817 PASS. Review head1bffc8c includes this recorded hold, no newruntimecode. No release. Preserve usercanonical777cc92 edits, daemon175392/YAML and allbaselinebackups. Never treat previous CPU/CI success as acceptance after live failure.

### 29.14 v1.1.41 HOLD — L MTP timing investigation #60 (2026-11-02)

Batch15734 finished successfully: allS/M/L correctness, including32S+8M+8L strict raw-pick0. L's final64tokenbenchwarmedrank0649s/rank1249s within900. BUT L MTP76.7156ms/token vsordinary42.9219; accepted39same-referenceMTP36.2344/ordinary41.1961. Draft acceptance36/52,27rounds identical;VRAM100%,disk0. Drafting102–112ms/round vsold8ms,verify76–83similar. Do not dismiss the timing: #60, #55/PR58 held. No41acceptance/no42. Initial L data /tmp/maya-upstream-v41/Maya-L/bench-mtp/results.json andlogs.

**ACTIVE A/B session61183**, script /tmp/maya-upstream-v41-l-ab.sh, output /tmp/maya-upstream-v41/l-mtp-ab/. Unmodified bench_mtp.py L64/runs2/drafts2/sameaccepted39reference. Deploys savedaccepted40 **8facc** to both review/canonical andrank0/tmp/maya-tp2 forbaseline arm, thenfixedcandidate41 **a617**. Actualperarmhashes recorded. Candidatea617 savedBOTHhosts **/tmp/maya-upstream-v41-candidate/maya**. EXITtrap restoresalltestcopies toa617evenonfailure. Inspectactivearm/filesbeforeanynewGPU/build/deploy action; do not startconcurrentpair. Historicaltimingaloneisnotcausal; comparecontrolledarms, then investigate draftingifregressionreproduces. Allordinary/MTPrawpickchecks staystrictzero. Originalbaselines through40/usercanonical777cc92edits/daemon175392/YAML preserved.

After resolving60, own-reference SNAP_GENtechnical256 via /tmp/maya-upstream-v41-snap-bench.py (NO --reference), finalHIPIQtest registration/build/parity aftertest-only8db5689, finalsourceevidence/CI/PRmerge/mainCI. Reviewhead093d205 (enginecode32650e8). M/Ldata/lateA-B/snapshotdata stillneed durablecopy/finalreport. No release;futuretagstp-v.

### 29.15 v1.1.41 runtime complete — final CI/merge/main pending

All live runs finished; **no active GPU/test batch**. Candidate **a6170c07503e139d73789184d9a93971**, runtime32650e8, matched localreview/canonical, remotereview/canonical,/tmp/maya-tp2. Review branch merge/upstream-v1.1.41 head **7ab98c8** /PR58. Final evidence/report docs/upstream/v1.1.41.md + directory/validation.json. Code includes56/57/59 fixes; those issues stayopen untilacceptance. Final7ab98c8 CI41326427259/41326422499 running, then finalheadCI/PRmerge/postmainCI required. No42sourcework yet.

580serve11skip/654tools31skip/HIPmap/3HIPengines/15host/196fused/MMQ/attention/CPUexactrows/PDEP0/1/EXL3/Node pass. New HIP-only IQtarget registration rebuilt,396strictcasesPASS, enginehashunchanged. Default/SNAP_GEN state suites, true prefillcancel/cache0retry/followup bothsimple+MTP7, lifecycle400withoutPIDmutation, CLI32/32each matchingaccepted39/40, killrecover8/8 allPASS. Default135roundtuples/8asym retries; SNAP103rounds/8asym+1naturalbothrankclash allmatched. **64candidate framed records raw-pick0**: Sdefault16+routeopt16,M8,L8,Lrepeat8,SNAPown8. Saved40 compatibility8also0. Snapshot benchmark generated its own ordinaryreference and provedordinary/MTPrawpicks, no forced-ID-only claim.

Prefill vs40all9messages/usageequal; warmthroughput+0.57/-0.97/-1.84% at990/4062/8158, no newgainclaim. #33 staysopen. #60controlledLtimingholdcleared forintegration: initial41MTP76.23/77.20, saved40**71.02/35.34**, repeated41**37.06/37.65**ms/token, allrawpicks/acceptanceidentical. Candidate returnedtohistorical34.82–37.65range; accepted40also hastransient. **Keep60open** forvariability/rootcause; do not claimfixedornewMTPgain. StartupR0/R1initial649/249,old40=414/33,repeated41=617/159seconds, allwithinunchanged900s. Rawarmsandphasebreakdownscommitted. A/Bsession61183andSNAP96027 completed; candidate restored allpaths, noactiveruns.

Next: reconcile this final livehandoff into review (docs only), update PR58 finalvalidationbody, finalCI, ancestry-preservingmerge, mainCI. Then close55/56/57/59withfinalcomments, updateaccepted41handoff/report andsavea617baselineBOTHhosts beforeopening42issue. Open33/42/46/52/54/60/11/16remain. Originalbaseline5a52,38cff,39c313,40=8facc andunacceptedcandidatebackup /tmp/maya-upstream-v41-candidate/maya(a617) preservedbothhosts. Remotecanonical777cc92/userwork/daemon175392/YAMLuntouched. No release/tag publication.

### 29.16 v1.1.41 accepted; v1.1.42 review next (#61)

PR58 merged as **0fea27a0243b360fa700139d48a81529664ec42b** after final39523ae CI **41326842619/41326839560 PASS**; post-merge **main41327238265 PASS**. #55/#56/#57/#59 closed with final evidence comments. All §29.15 gates pass. #60 remains OPEN for variability also reproduced on accepted40; no persistent41regression/newgainclaim. #33/#42/#46/#52/#54/#11/#16 remain open.

Runtime32650e8, engine **a6170c07503e139d73789184d9a93971** all5paths matched; accepted41 backup saved on BOTHhosts at **/tmp/maya-upstream-v41-baseline/maya**. Candidatebackup a617 also retained. Rejected13a/pre-fix failure retained separately. Engines/server stopped; no active batches. Daemon175392/YAML and remote canonical777cc92 user work preserved. Report docs/upstream/v1.1.41.md /validation.json markedaccepted; upstream ancestry preserved. No release/tag publication.

Next **#61 /upstream v1.1.42 cd48762b6d4888205c130238dbc0c7f7904993cf**,22commits/49files; no source candidate yet. Branch merge/upstream-v1.1.42 from updated origin/main. Audit HIPvision/APU/idle/env/chunk/pin changes semantically. Keep TP2vision gated (#16 requires RDMA embeddings), #59 completed-call cancellation, #56/#53 lifecycle guards, #57 totalbudget, #47 liveness/all-limit reserves, #46 streamdeferral/#11multi, NextN/fullwidth/AR/MTPwire/globalchunks/retries/primary100401 tuning. Preserve fork docs/defaults; exclude unsupported backend/platform/localmultiGPU work. No43 before42 full acceptance/postmainCI.

### 29.17 Active v1.1.42 semantic candidate (#61/#62)

Review branch merge/upstream-v1.1.42 has an in-progress merge from cd48762. Retain idle gates with explicit TP worker/mid-MTP wake-up, alias exports mirrored in rank1, GLM monotonic auto-chunk search fix, telemetry/setup idle reductions, DS4 dense-memory guard and legacy-stream link calibration, MiniMax bad-router rejection. Defer all new vision enablement/lending/vendor patches to16 and same-host DS4 partial-read/multi-restore/lending/DMA changes. Keep all41 TP2/MTP/cancel/lifecycle/loop-limit safeguards, tuning, owned README/AGENTS/VERSION/pin. #62 explicitly rejects local-only GENI/VLEND/VRECLAIM/LOGP and media before state mutation/encoding; startup skips TP vision config. Focused tests initially found only an upstream source-count assumption (>30 MAYA settings; retained fork has29), corrected to test actual named settings and all discovered aliases. Full ordered CPU/build/live/perf/fidelity/CI/main gates pending; no43 before42 acceptance. Accepted41 a617 backups remain bothhosts. No active live batch/build yet.

42 build milestone: runtime d59ea06 engine **d8b4252a17fc09623ef0677051ee1a47** matched all5paths. All3HIP/16host/196fused/MMQ/attention/396strictIQ pass. First CPU583server had2 fixture errors (excluded WINDOWS symbol); ba4c34e fixes mocks, focusedSampler pass, orderedCPU session85089 rerunning. **Active live session86028**, /tmp/maya-upstream-v42-batch.sh, output /tmp/maya-upstream-v42/. DraftPR63, no42acceptance/no43. Do not start concurrentGPU/build/deployment. Test-only/doc commits afterd59 do not change engine; remote review d59.

42 HOLD #64: force-deep-idle1s/3000ms exposed23.683s MTP32 decode after14.55K prefill (initiald8 candidate), versus~60–66ms/token short probes. Initial port woke rank1 only at mirror/MTP; rank0 needed wake-up again after long prefill. Pre-fix logs/binary preserved, batch86028stopped. **837a679** wakes both lanes at mirrored-call/nested-MTP boundaries without wire changes; newengine **bc4e90a607b87fe609398fbdba0fb98c**. Active fixedbatch19655 same /tmp/maya-upstream-v42-batch.sh. Verify deployedfivehashes/GPUimageaudit completed before live. OrderedCPU583serve11skip/662tools33skip/HIPmap/Node PASS; no Python changes after those. No42acceptance/no43.

42 fixedstate milestone: same14.55K MTP probe decode23.683→2.2415s, prefill41.463→41.399s, long/short12.26x→1.14x; durable4x gate rejectsactualpre-fix/passesfixed. BOTHsimple/MTP7 matrices/lifecycle/media/idlefollowup/trueprefillcancelcache0retry pass,132roundtuples/8asym retries match,CLI32/32both pass. Initialaudit249/132wasappendlogmixing; segmentedlaststartup(line784), rawpreserved. Final584serve11skip/662tools33skip/HIPmap/Node pass; 456Pythonlast-worldflagguardremoteupdatedbeforelaterphases; engine837/bc4unchanged. RuntimeCI41330559259/PythonCI41331029193PASS; finalheadCIstillpending. Activebatch19655 recovery/fidelity/prefillAB/M-L; no43.

42 recovery/fidelity milestone: kill/recover8/8; SCLI32/32bothranks/modesidenticalaccepted41; technical/code256framed16/16ordinaryANDMTPrawpick0 withall4directunsupportedcommandsrejectedbeforedecode. Current779headCI41331428973/41331432952PASS. Candidate-prefill9/9done, accepted41baselinearmloading; M/Lnext in active19655batch. No42acceptance/no43.

42 prefillABcomplete: all9messages/usageequal/cache0; warm1K/4K/8Kthroughput−0.18/+0.30/+1.57% vs41, no materialregression/newgainclaim; #33openforcold9.6case/~3008rowcap. Active19655Maya-MAPI/prefill/fidelity, thenL. **After19655 completes successfully**, run /tmp/maya-upstream-v42-idle-ab.py > /tmp/maya-upstream-v42/idle-ab.txt (normaldefault65squiet+10sCPU/wake sample eacharm, thenMTPfollowup). NeverconcurrentGPU. Finalevidence/CI/mainbefore43.

42Maya-M complete: API/reuse/cancel/limits/fresh1K/4K/8K pass;8framedraw-pick0, ordinary40.62/MTP31.90ms/token. Coldsmokeprefill11.32/39.66/24.97s retainedwithoutperfclaim. Lrank1warm73s/ready, rank0progress90% at156s (900boundunchanged), active19655. Afterbatch0 runnormalidleABscriptasabove. FinalCI/mainrequired; no43.

42LAPI milestone: startupR0186/R173s, API/reuse/cancel/limits/fresh1K/4K/8K pass; raw64benchmarkwarmingnow (coldloadnewpair), active19655. Requiredafterbatch0: normal-idleAB /tmp/maya-upstream-v42-idle-ab.py -> /tmp/maya-upstream-v42/idle-ab.txt. Notrunyet. Updatedrestartpromptcontainsallpendinggates. No43untilfinal42evidence/headCI/merge/mainPASS.

42 fullbatch19655 EXIT0: LAPI/prefill+8framedrawpick0 PASS; LfinalwarmR0731/R1219s within900 unchanged; ordinary42.82/MTP35.16ms/token (historicalrange), no persistentMTPregression. TotalS16+M8+L8=32strictframedrecordsraw0. IOsnapshotsretained(#60), startupvariabilityunresolved. **ACTIVE normal-idleAB session26121**, /tmp/maya-upstream-v42-idle-ab.py, output /tmp/maya-upstream-v42/idle-ab.txt andidle-ab/results.json. Do notstartconcurrentGPU. Thisislastplannedruntimegate; aftersuccess finalreview/evidence/headCI/PR63/mainCIbefore43.

42 ALLRUNTIMEPASS: idleAB26121 EXIT0, default65squiet+10ssamplesCPU baselineR0/R1 .88/.78%→candidate .20/.00%; voluntaryswitches422/428→10.94/10.00persec; botharmsMTPfollowup/peerpass. All32strictS/M/Lraw0, L731/219warm/35.16MTPwithinrange. Finalbc4all5hashesmatch; bothengines/serverstopped, remotecanonical777userdirtyfiles/backups/daemon/YAMLprotected. Reportrewritten/validation.jsonpendingonlyfinalCI/merge/main. No43beforethose.

### 29.18 v1.1.42 runtime complete — final CI/merge/main pending

All planned42gates PASS; no active GPU/server/test batch. Engine837a679/bc4e90a607b87fe609398fbdba0fb98c all5paths matched; Python456b4a5. Fullbatch19655/idleAB26121 exited0; do not rerun. Report docs/upstream/v1.1.42.md andvalidation.json finalpendingonlyCI/merge/main. PR63/#61/#62/#64 openuntilmainPASS. NormalidleCPU/switchbenefitmeasuredonbothgfx1151ranks; 32strictS/M/Lrecordsraw0, allcancel/reuse/limit/recovery/mirror/perfgatespassed. L731/219s startupwithin900,35.16MTPms;60variabilityremainsopen. Allfailedfixture/prefix-log-audit/pre-fixidledataretained.

Next: finalheadCI, readyPR63, ancestry-preservingmerge, post-mainCI, explicitclose61/62/64, acceptedreport/handoffs/index andsavebc4v42baselineBOTHhosts. Onlythen43issue/branch; continuethrough46. NumericVERSION1.1.42/legacyreleasesunchanged; futuretp-v tags; noreleaseauthorized. Protectedcanonical777cc92useredits/backups/daemon175392/YAMLunchanged.

### 29.19 v1.1.42 accepted; v1.1.43 next

PR63 merged as **b1867c8bfa0e7576754208abb9f2ff66303af1b3** after final aad4e03 CI **41334558152/41334563528 PASS**. Post-merge **main41334938602 PASS**; #61/#62/#64 closed with evidence comments. All §29.18 runtime gates pass. Report docs/upstream/v1.1.42.md /validation.json is accepted. Engine source837a679, binary **bc4e90a607b87fe609398fbdba0fb98c**, all5paths matched and saved on BOTH hosts at **/tmp/maya-upstream-v42-baseline/maya**. No active batches;19655/26121 completed.

Preserve42's paired idle wake-up at mirrored calls/nested MTP, alias parity and explicit unmirrored GENI/VLEND/VRECLAIM/LOGP/media/startup guards alongside41's cancellation/lifecycle/total-budget safeguards. Vision16/multi11/stream46 remain deferred. All32S/M/L framed records raw-pick0;132rounds/8asymmetric retries match; CLI/recovery/state checks pass. Warm prefill vs41−0.18/+0.30/+1.57%, all9responses/usage equal. Normal idle engine switches422/428→11/10 per second with lower sampled CPU and successful MTP follow-up. L731/219s startup within900, MTP35.16ms;60 variability remains open.

Next upstream43 **acdab34bd58eec957715392a1d822777c3c579c3** (18commits/41files): create its own issue/branch from updatedmain, semantic review and affected-path validation, final/mainCI before44. Continue44/45/46 in order. README/AGENTS/VERSION/pin/tuning remain ours; no release. Remote canonical777cc92 user edits/backups/daemon175392/YAML preserved. At fulltask end restore coherent production pair as documented; keep validated review builds.

### 29.20 Active v1.1.43 semantic merge (#65/#66/#67)

Review branch merge/upstream-v1.1.43 from39486fe; upstreamacdab34 merge in progress, conflicts resolved in working files but not yet staged/committed. All42 gates/main passed before43. Retain loop/thinking continuation fixes,32K-character collapse cap, monitor draft/tier stats, unattended calibration behavior, DS4 quant/Engram/packing support. New drive probe applies only to raw-GGUF O_DIRECT; packed TP2 FileExpertSource is unchanged. HIP probe is opt-in/off until measured, preserving stagers and DIRECT defaults; no streamed-fetch fragments imported. VERSION/owned docs/pin/tuning preserved.

237 initial focused tests passed. Deterministic boundary probes then found #66: server note can exhaust context yet max(1,room) sends oversized GEN (prompt2612/context1815). Guard no-room as length before next GEN, retain57 total-budget while guard. #67: collapse-limit JSON setting ignored by server; add whitelist/both-alias support and legacy-shell fallback. Added real tests: pre-fix2failures+1error; corrected2tests pass, same standalone reproducer now length/no second engine call and JSON=shell behavior. Full updated focused run239+ tests running (90457); no GPU/build/batch active.

Need complete semantic review, ordered CPU gates,3HIP builds including ds4_gguf_test syntheticQ5_K and FileExpertSource host checks; live S MTP7/simple/cancel/reuse/large/retry/fidelity plus new continuation/context and long-loop prefix reuse probes. Prove unchanged GLM GPU image to reuse42 M/L/kernel evidence only where executed paths are unchanged. FinalCI/PR/main required before44. #65/66/67 open, no43 acceptance yet.42acceptedbc4 baseline saved bothhosts.

### 29.21 Active43 live gates (#65/#66/#67/#68/#69)

Candidate engine6a2fbe1 **b398d4a0711430edd6a28b95264c7293** deployed/matched5paths; remote review6a2fbe1. GLM GPUimage95… identical42;3HIPengines/17host/602serve11skip/676tools33skip/HIPmap/Node pass. Initialruntime4ec487 superseded by explicit MTPbias guard69; no sampling change. Small-context normal/forced probes46313 exited0, PIDs/peer preserved, context guard+MTP follow-ups pass. Forced EOS is branch coverage, not raw fidelity. Failed initial bias fixture retained.

Broader DS4GPU failed2WMMA checks; independentlybuilt42control b186 also fails.3alternatingKVruns each: control[1,2,1], candidate[1,1,2]; both failures pre-existing, WMMAfunction source unchanged. **68OPEN**, do not claim full DS4 suite passed or relax thresholds. New/old Engram-only suite0fail. Control worktree /home/ugo/software/project-maya-v42-ds4-control; no production source changed.

**ACTIVE batch8965**, /tmp/maya-upstream-v43-batch.sh, output /tmp/maya-upstream-v43/. Runs S normal MTP7/simple state/cancel, long-loop collapse0/default32000 cache A/B, CLI/recovery, strict Stechnical/code256 and3cycle prefillAB vs42. No concurrentGPU/build/deploy. M/L/kernel42evidence reused only for identical executed math; serverchanges covered CPU/newliveS paths. Afterbatch success auditrounds/limits/cache/fidelity/perf, final evidence/headCI/PR/mainCI before44. Close65/66/67/69 onlyaftermainPASS; keep68/33/42/46/52/54/60/11/16 open.43notaccepted.

43 state milestone: both simple/MTP7 matrices, lifecycle/media/idle and true18Kprefillcancel/cache0retry/followup PASS.133round tuples/8asymmetric retries match. Short exaggerated-idle MTP491.19ms/token outlier retained; long74.78, simple41.64/43.48; #60open, no speedup inferred. Batch8965nowlong-loop cache A/B: collapse0arm finished initial20021/followcache0; default32000arm loading. ThenCLI/recovery/strictS16/prefillAB remain. PR70draft. Smallboundary artifacts committed23ca97f; report/validationJSON in worktree stillpending. No44.

43 long-loop A/B PASS: sameengine/JSON0vs32000, initial20021/cache0, semanticmodelcontent+usageequal (onlyrandomservertoolcallIDsnormalized; rawretained). Follow unlimitedcache0/prefill61761.3ms versusdefaultcache20020/fresh108/prefill1141.8ms (~54.1x forfixtureonly). #33openforbroadercold9.6case/3008cap. CLI32/32bothmodes/rankspass.8965nowrecoverythenstrictS16/prefillAB. No44untilfinal+mainCI.

43 HOLD/fix71: stricttechnicalbenchmark mergedstdout/stderr; stderrfragment splicedinside DONE (enginefinished/raw0, harnesswaited).8965interrupted130; partialevidenceretainedbench-technical-interleaved. Newseparatestreamreaders/queues/logs, realOSpipe2schedules/tworequestsregression2errors→PASS. Noengine/sourcebinarychange. **ACTIVE13476** /tmp/maya-upstream-v43-finish.sh runsorderedCPUafterharnessfix, correctedstrictS16, thenprefillABvs42. Priorstate/CLI/recovery/cachegatescomplete, doNOTrepeat. #71openalongside65/66/67/69untilmainPASS;68exceptionremainsopen. No44.

### 29.22 v1.1.43 runtime complete; final CI/merge/main pending

13476EXIT0. CorrectedstrictS16ordinaryANDMTPraw0;CLI32/rank/modeexact42;recovery8/8;133rounds/8retries;bothstate/cancel/guards/contextgatesPASS. Longloop0→20020cache/61.761→1.142sprefillforfixtureonly. NormalprefillABvs42all9message/usageequal/cache0; warm+0.94/−0.36/−0.37%, no materialregression/newgain. Both liveexe hashesprovedperarm. CPU602/677/HIPmap/Node,3HIP/17host/Engrampass. FullDS4WMMA68FAILpre-existingcontrol, explicitlyopen;60outlierretained. Engine6a2f/b398all5match. Bothengines/serverstopped;protectedcanonical777userwork/daemon175392/YAMLpreserved. Finalreport/validationJSONnowpendingonlyexactheadCI/merge/postmainCI. Close65/66/67/69/71onlyaftermainPASS;68/33/42/46/52/54/60/11/16remainopen. No44yet.

43 PR70 merged **c2ff7198e8587d9e027bb4f4454e27b1ae784325** after final f862e3b CI41343230502/41343235345 PASS. **Post-main CI41343606174 pending; no44 and65/66/67/69/71 remainopen.** Runtime6a2f/b398 preservedbothhosts /tmp/maya-upstream-v43-candidate/maya; remotereviewfastforwardedtof862(noenginechange). Canonicalremote777userwork/backups/daemon/YAMLunchanged. Bothengines/serverstopped.

### 29.23 v1.1.43 ACCEPTED; v1.1.44 next

PR70 merged **c2ff7198e8587d9e027bb4f4454e27b1ae784325** after final f862e3b CI **41343230502/41343235345 PASS**. Post-merge **main41343606174 PASS**. #65/#66/#67/#69/#71closed with final evidence comments; **#68remainsOPEN** for pre-existing DS4WMMA failures reproduced on42control. All §29.22 targetedruntimegatespass; report docs/upstream/v1.1.43.md/validation.json accepted. Engine6a2fbe1 **b398d4a0711430edd6a28b95264c7293** all5paths matched and savedBOTHhosts **/tmp/maya-upstream-v43-baseline/maya**; candidatebackupalso retained. Noactivebatches; engines/serverstopped. Remotereviewf862, protectedremotecanonical777cc92/userwork/backups/daemon175392/YAMLunchanged.

43preservespairedwake/media/cancel/lifecycle/totalbudget/NextN/fullwidth/MTP/RDMA/tuning. Adds safecontinuationcontextbounds, collapseJSONaliasparity, explicitunsupportedTPMTPbiasrejection(spec1works), and separatebenchmarkstreams. Longloopfixturecache0→20020/prefill61.76→1.14s; notuniversalthroughputgain. FreshprefillABvs42all9response/usageequal/cache0,warm+0.94/−0.36/−0.37%. StrictS16raw0, allCLI32streamsidentical42, recovery8/8,133rounds/8retries. CPU602/677,3HIP/17host/Engrampass; fullDS4suiteFAIL68explicitexception. ShortMTP491msoutlierretained60open. M/L/math42evidencereusedonlyforidenticalGLMGPUimage/executedpaths.

Next44 **5d20b0e8fd7c15fde2a9f2b4eac5e9776883690b** (4commits/6files), ownissue/branchfromupdatedorigin/main. Then45=6ecbd3c,46=3589d6f. Eachsemanticreview+affectedgates+finalheadCI+postmainCI before next. Continue through46. NumericVERSION1.1.42/legacyreleaseunchanged; futuretp-v tags, no releaseauthorized. Keep33/42/46/52/54/60/11/16/68open. Atfulltaskendrestorecoherentproductionruntimewithoutoverwritingprotectedremoteuser source/YAML; retainvalidatedreviewbuilds.

### 29.24 Active v1.1.44 installer-only semantic merge (#72)

Branchmerge/upstream-v1.1.44 fromdffea08; upstream5d20b0e mergedadc54a8. Installer/catalog/docs/tests only; ALLtrackedengine/server/tests/TP2/tuning/README/AGENTS/VERSION/pinexact43 (canonicalGitblobs, intentionalbatCRLFhandled).73focusedtestsPASS; independentHFmetadata18files×2revisions sizes/SHAexactmanifest. Nohugeweightdownload. ACTIVE64704 /tmp/maya-upstream-v44-gates.sh orderedHIPmap/serve/tools thenfreshS MTP/reuse/spec1API/PID/peer. Engine6a2f/b398unchanged; remotereviewadc54a8. Reuse43GPU/math/strict/M-L onlyforprovenunchangedpaths. FullDS4WMMA68remainsOPEN. Finalreport/headCI/PR/mainCI requiredbefore45. No43batchactive;protectedcanonical777userwork/daemon/YAMLpreserved.

44ALLTARGETEDPASS:64704EXIT0,HIPmap/602serve11skip/679tools33skip,73focusedinstaller,HF18files×2revisionsSHA/sizesexact. FreshS MTP/reuse/spec1ordinary64-budget/PIDs/peerPASS; engines/serverstopped.615runtimefiles+all5binarypaths exact43b398, noenginebuild/mathchange. Reused43GPU/strict/M-Levidenceonlyforunchangedpaths. DS4WMMA68open. Report/validationJSONnowpendingonlyexactheadCI/PR73merge/postmainCI before45.

44PR73 merged842d6e67c11ed702d1c7f54d61b8eb69e5212ed6 after final5e920c4 CI41344769559/41344774298 PASS. Post-mainCI41345166897pending; #72open andno45. Bothengines/serverstopped, remote review5e920c4, engine6a2f/b398 unchangedall5paths. Protectedcanonical777userwork/daemon/YAMLpreserved.

### 29.25 v1.1.44 ACCEPTED; v1.1.45 next

PR73/ **842d6e67c11ed702d1c7f54d61b8eb69e5212ed6** afterfinal5e920c4 CI **41344769559/41344774298 PASS**; post-main **41345166897 PASS**. #72closedwithfinalcomment. Report docs/upstream/v1.1.44.md/validation.json accepted. Installer/catalog/docs/tests only:615runtimefiles/README/AGENTS/VERSION/pin/tuningexact43; engine6a2f/b398sameall5paths, savedBOTHhosts /tmp/maya-upstream-v44-baseline/maya.73focusedtests,HF18files×2revisions names/sizes/SHAexact,HIPmap/602serve11skip/679tools33skip,freshS MTP/reuse/spec1ordinary64budget/PIDs/peerPASS;64704EXIT0. Noactivebatches; bothengines/serverstopped. Reuse43fullGPU/strict/M-L/state/perfevidenceonlyforunchangedpaths. FullDS4WMMA68remainsOPEN/failing, no newquality/perfclaim. Remotereview5e920c4; protectedcanonical777userwork/backups/daemon175392/YAMLunchanged.

Next45 **6ecbd3cbc5330b57f70cd5cf8a5c42eb8ef950cc** (2commits/3files), ownissue/branchfromupdatedmain; then46=3589d6f. Eachsemanticreview+affectedgates+finalheadCI+postmainCI before next. Continue through46. NumericVERSION1.1.42/legacyreleaseunchanged, futuretp-vtags,no release. Keep33/42/46/52/54/60/11/16/68open.

### 29.26 Active v1.1.45 scoped semantic review (#74/#75)

Upstream6ecbd3c merged6862700withALLruntimeexact44; onlyCHANGELOGfunctionaldecisionnote. Its c968b94autoNVIDIAoneGPUtailfixdepends ondeferredcopy_flush/strictboolrc_drain/rc_ticklentguard; retainedHIPautomaticpathisdisabledandourlegacydrain5sfallbackcannotacceptpartialrc_in_tailpatchsafely. Explicitfunctionaldeferral **#75OPEN** forHIPvalidation/completeport; donotclaim45crashfixactiveor32K/64K/128KDS4promptsvalidated. SeparateDS4WMMA68alsoOPEN. Dependencyproof/rawpatchunder docs/upstream/v1.1.45/.

ACTIVE2813 /tmp/maya-upstream-v45-gates.sh orderedHIPmap/serve/tools thenfreshpairedS MTP/reuse/spec1API. Engine6a2f/b398unchanged; remote review6862700. PreserveallTP2/MTP/NextN/RDMA/cancel/lifecycle/wake/media/budget/tuningandownedREADME/AGENTS/VERSION/pin. Finalreport/headCI/PR/mainCI requiredbefore46. Protectedcanonical777userwork/daemon/YAMLunchanged. No44batchactive.

45ALLTARGETEDPASS2813EXIT0:HIPmap/602serve11skip/679tools33skip,freshpairedMTP/reuse/spec1ordinary64budget/PIDs/peerPASS. All5binarypathsb398;620runtimefilesexact44. Noenginebuild/math/wirechange. FunctionalDS4tailfixexplicitlyDEFERRED75OPEN, WMMA68OPEN; do notclaim45crashfixactive. Report/validationJSONpendingonlyexactheadCI/PR76merge/postmainCI before46. Bothengines/serverstopped; protectedcanonical777userwork/daemon/YAMLpreserved.

45PR76merged2df9e26be0783f7a84ae6c4e8dbc6e8d6d5fe933 afterfinal6f1e78c CI41346480733/41346487052PASS. Post-mainCI41346856373pending; #74open andno46yet. Runtime620files/b398exact44; functional75OPEN/WMMA68OPEN. Bothengines/serverstopped, remoterereview6f1e78c, protectedcanonical777userwork/daemon/YAMLpreserved. Original5a52backupverifiedbothhosts (remote maya/maya-build/maya-tmp allsame); YAML SHA256c5392d8c696f5ff8869510057427675b18f8a3cb482d0a136012495151297a20.

### 29.27 v1.1.45 semantic checkpoint ACCEPTED; v1.1.46 next

PR76 **2df9e26be0783f7a84ae6c4e8dbc6e8d6d5fe933**, final6f1e78c CI **41346480733/41346487052 PASS**, post-main **41346856373 PASS**. #74closed as completedscopedreview/ancestryintegration. **FunctionalDS4tailfixDEFERRED#75OPEN**, notclaimedactive. AutomaticNVIDIApathdisabledHIP; strictdrain/copyflush/lentguarddependenciesexcludedearlier, no partialport. DistinctWMMA68OPEN/failing. Report docs/upstream/v1.1.45.md/validation.json acceptedwithupstream_functional_change_applied=false.

620runtimefilesexact44, engine6a2f/b398all5paths, savedBOTHhosts /tmp/maya-upstream-v45-baseline/maya (same43/44). HIPmap/602serve11skip/679tools33skip+freshS MTP/reuse/spec1ordinarydraft56/47/0,64budget,PIDs/peerPASS;2813EXIT0. Noactivebatches; engines/serverstopped. ReusepriorGPU/strict/M-L/state/perfonlyforidenticalpaths. Remotereview6f1e78c;protectedcanonical777userwork/backups/daemon175392/YAMLpreserved. Original5a52baselineverifiedbothhostsforfulltaskrestoration.

Next46 **3589d6f12a3f77e813b184b51a586f8fc3de2185** (5commits/8files), ownissue/branchfromupdatedmain. Semanticreview/affectedgates/finalheadCI/postmainCI. Continue through46, thenrestorecoherentproductionruntimewithoutoverwritingprotectedsource/YAML, retainvalidatedreviewbuilds andfinalhandoff. NumericVERSION1.1.42, futuretp-vtags,no release. Keep33/42/46/52/54/60/11/16/68/75openunlessactuallyresolved.

### 29.28 Active v1.1.46 final tag (#77/#78/#79/#80)

Source5b13c83 merges3589d6f: malformedparameter/contentpartsfixes;78safechunkbounds/aliases(default0),80textparts count32Kcap. Windows-onlytrimexcluded. 79OPENforDeepSeeklocalcapmeasurements; noDSweightsinconfiguredmodelstore,no5%claim/defaultenable.75tail/68WMMAremainOPEN; no lending/stream/vision/multifragmentsreintroduced. GLM/core/transport/tuning/ownedfilesexact45, binaryb398/GPUimage958…unchanged. HIPDS4hostbinary51749e860763aea34847b183de95a933 built;DS4GPUimage92096f8e…exactbaseline.605serve11skip/679tools33skip/HIPmap,99focused,18hostactualpolicyPASS;270848boundaryproperties. Beforefailures/rawincorrectparserfixtureassumptionretained.

**ACTIVE22871** /tmp/maya-upstream-v46-live.sh output/tmp/maya-upstream-v46/: MTP7+simple state/large/reuse/stream/guards/real18Kcancel, thenfreshD2textparts20Kprefix+ordinary/MTPfollowups. No concurrentGPU/build/deploy. CPUpreflightOpenAItextpartsflattenstr, so directunitisolates80/liveAPIprovesformatcompatibility+policy, notindependent80A/B. Afterlivesuccess auditrounds/limits/cache,finalevidence/headCI/PR/mainCI,close77/78/80onlyaftermainPASS.79/75/68and33/42/46/52/54/60/11/16stayopen.

Remote review5b13c83; protectedcanonical777userwork/daemon/YAMLuntouched. OwnreviewSconfigsnow.exe review/build-hip/maya onBOTHhosts (M/Lvalidationusedexplicittransientconfigs); enablesproductionoriginal5a52restoreafterwhole46taskwithoutmixingreviewserverwitholdcanonicalengine. BothreviewDS4binaries51749; GLM5paths b398 matchedbeforelive. Finalproductionrestore/handoffs/index stillpending.

### 29.29 v1.1.46 runtime complete; final CI/merge/main pending

22871EXIT0.605serve11skip/679tools33skip/HIPmap,99focused,18host,HIPDS4buildPASS;DS4GPUimage92096…unchanged,newhostbinary51749e86; GLMb398/GPU958…exactprior. MTP7+ordinarylarge/reuse/stream/guards/idle andtrue18Kprefillcancel/cache0retry/followup/peerPASS;132rounds/8asymretriesmatch. ForcedidleMTP88.92/67.99ms,ordinary42.87/41.14retained60open,nospeedclaim. FreshtextpartsAPI20021initial/cache0→20020cached/collapsed0,ordinary/MTP64followupPASS. Directunitisolates80;APIflattensalltextpartsstr,notindependent80A/B.

Alltargetedruntimecomplete, bothengines/serverstopped, noactivebatches/builds. FinalexactheadCI/PR81merge/postmainCI pending. Close78/80aftermainPASS;77coversfinalproductionrestore/handoffs/indextoo,closeonlywholeobjectivecomplete.79capmeasurement(default0)/75tail/68WMMAremainOPENwith33/42/46/52/54/60/11/16. Protectedcanonical777userwork/daemon/YAMLpreserved. Reviewconfigsown.exepathsvalidated; after46accepted saveb398+5174baselinesBOTHhosts, restoreoriginal5a52ONLYcanonicalGLMpaths+remote/tmpmaya-tp2whilekeepingreviewGLMb398/DS45174. Neveroverwriteprotectedremote source/YAML. Finalrestoration/hash/source/GHancestry/CI/handoffchecksstillneeded.

46PR81merged1139fa27d7731941b187de57093c3abe424a5e1f afterfinalb5e46b2 CI41350489382/41350494526PASS. Post-mainCI41351082604pending. All10requestedupstream37–46ancestorsofmain; legacyforktagv1.1.42=64719953312df98c9fdf53a10198c106cb195345 unchanged,VERSION1.1.42,norelease/tagcreated. AftermainPASSmark46accepted/close78/80;77stillcoversrestore/finalhandoff/index. Protectedpre-restoresnapshot /tmp/maya-upstream-v46/protected-before-restore.json records777HEAD/userdiffsha/config/backups/YAML/daemon; compareexactafterrestoringonlycanonicalGLMpathsfrom5a52original. Reviewb398/DS45174kept,ownreviewconfigs.exe pathsalreadyvalidated. NoactiveGPU/tests.

### 29.30 Full requested integration and production restoration complete

Allrequestedupstream37–46areancestorsofmain. Final46PR81 **1139fa27d7731941b187de57093c3abe424a5e1f**, finalb5e46b2 CI **41350489382/41350494526 PASS**, post-main **41351082604 PASS**.46accepted;78/80closed. Detailedcompletion **docs/upstream/COMPLETION-2026-11-02.md** andcompletion-2026-11-02/completion.json. All §29.29gatespass; noactiveintegrationGPU/build/testbatches.79capmeasurements(default0),75functional45tailport,68fullDS4WMMAfailure stayOPEN alongwith33/42/46/52/54/60/11/16. No newrelease/tag; VERSION1.1.42,legacyv42tag6471995unchanged,futuretp-v*.

**ProductionGLM original5a52cd27167523b6ea4b0f0bde93f503RESTORED** at localcanonicalbuild-hip/maya,remotecanonicalbuild-hip/maya,andremote/tmp/maya-tp2 usinghashguards+atomicreplacement. BOTHreviewGLMb398d4a0711430edd6a28b95264c7293 andDS451749e860763aea34847b183de95a933 retained. Accepted46backupsBOTHhosts /tmp/maya-upstream-v46-baseline/maya andmaya-ds4-accepted. Localmaya-ds4inthatbaselinefolderisolderpre46controlcd177,NOTaccepted46hostbuild. Bothreviewsourceworktreescanfollowfinalmain docswithoutrebuildingunchangedCXXcode(GLM6a2f,DS4host5b13).

Protectedremote777cc92HEAD/userdiffs/stageddiffs/modelconfigs/backups/YAML/daemon175392exactlymatchbefore/afterrestoration; rawsundercompletion-2026-11-02/. YAMLsha c5392d8c696f5ff8869510057427675b18f8a3cb482d0a136012495151297a20,6entries/Lcontext524288unchanged. No canonicalremote sourcepull/reset/rebuild. Engines/testserverstopped,llama-swapreadytoloadproductiononrequest. **PermanentreadyreviewconfigisSonly**, .exe ownreviewbinaryonbothhosts; M/Lpastvalidationusedexplicittransientconfigs. FutureM/Lreviewlaunchmustselectsameexplicitmodel/pack/configonbothrankswith.exe review/build-hip/maya,notrestoredcanonicalexe. EarlierS/M/Lconfigurationwordingcorrected; metadataalwaysshowedSonly.

Final closeout completed: documentation checkpoint `dede882111be994b3e913463c8fa6b211f34633d`, main CI41352750402 PASS; #77/#36 closed. No further integration GPU tests/tag merges are required without a new request. Preserve all review/control/failed raw artifacts and original backups. Current operating instructions and the later user clarification are in §30.

## 30. Current restart state — documentation audit (2026-11-02, #82)

The requested v1.1.37–v1.1.46 integration and production restoration are complete. Final source merge
PR81/1139fa2 and documentation checkpoint dede882 passed CI41351082604 and CI41352750402 respectively.
#77/#36/#78/#80 are closed. No release/tag was created; numeric VERSION remains1.1.42, legacy release
v1.1.42 is immutable, and future fork releases use `tp-vMAJOR.MINOR.PATCH`.

**Process ownership:** the user wants to launch processes themselves and explicitly limited cleanup
to processes the agent started. All agent validation processes ended. Do not infer current user
process state from historical snapshots or inspect unrelated services. Future validation must track
its own PIDs/process groups and stop the owned server before its engine. No broad pkill/fuser cleanup.
Legacy `acceptance.sh`/`bench_mtp.py` cleanup is a separate open follow-up #83; these harnesses must
not run against user processes. Earlier permission to use the pair does not override this rule.

**Runtime split:** canonical GLM binaries on both hosts and rank0 `/tmp/maya-tp2` were restored to
production MD5 `5a52cd27167523b6ea4b0f0bde93f503`. Both review trees retain GLM
`b398d4a0711430edd6a28b95264c7293` and DS4 `51749e860763aea34847b183de95a933`.
The protected remote canonical checkout was777cc92 with user edits; no pull/reset/rebuild there.
Only the S review config is permanent. For M/L review launches use explicit temporary configs and
matching model/pack selection: rank0 config.exe and rank1 ENGINE/checkout path must both select the
review build. See [the runbook](docs/TP2-RUN.md#development-pair-production-and-review-builds) and
[completion evidence](docs/upstream/COMPLETION-2026-11-02.md). No runtime deployment occurred in #82.

**Documentation:** README now distinguishes unreleased main from the published release; MODELS
correctly describes opt-in TP2 MTP; CONTRIBUTING/AGENTS use the fork's HIP scope and issue workflow;
SETTINGS explains numeric thinking budgets separately from context size; the runbook uses corrected
MTP measurements. Original design/plan documents and inherited platform references are labeled as
historical/upstream material. [The documentation index](docs/README.md) identifies current guides.

**Open work:** #33 was found closed despite its keep-open body and later unresolved-work comments;
reopened during #82. Its cold/long-context 9.6tok/s case and ~3008-row cap remain unresolved; retain
the measured S32–37% improvement without claiming a general fix. #68 full DS4 WMMA failures, #75
deferred tail-drain port, #79 default-off chunk-cap measurements, #60 latency variation and
#42/#46/#52/#54/#11/#16/#10 remain open. No new GPU/performance claim follows from this docs audit.

## 31. Current restart state — upstream v1.1.49 (#85)

The user explicitly requested **one merge from the latest upstream tag**, rather than repeating the
previous per-tag sequence. #85 integrates v1.1.47–v1.1.49 directly from
`26aacb6bcea9333e4f408a68ee60c4c1c316bd68`, fork base d7626ac. The isolated worktree is
`/home/ugo/software/project-maya-upstream-v49`, branch `merge/upstream-v1.1.49`.
[Report](docs/upstream/v1.1.49.md) and raw evidence record the cumulative decisions; exact-head,
merge and main CI provenance is recorded on #85. The earlier v37–46 integration/#77 and docs
audit/#82 are complete; no old closeout task should be resumed.

Accepted: legacy/new real-engine names, Maya identity, monitor/TUI fixes, DeepSeek XS-v2/shared-table
installer fixes, soft thinking closure and read-order interleaving. Retain model-management opt-in,
all TP2 safeguards, numeric VERSION1.1.42, future `tp-v*` release namespace. Exclude unrelated
NVIDIA/Qwen allocator, local multi-GPU chunk default, Windows helper and conflict-reintroduced
Metal functions. README/AGENTS and catalog/settings were merged semantically.

**Validation:** ordered HIP-map/610 server (11 skips)/686 tools (35 skips) PASS; 118 focused tests,
42 bundled-wheel Textual/dashboard tests and 336010 actual DS4 request-policy checks PASS.
The changed DS4 entry point compiles/links using accepted read-only HIP dependencies into the
isolated `build-host-v49/maya-ds4`, MD5 `64f5092b583ea33815069fbbf28057ea`.
GPU image SHA25692096f8e… is identical to accepted DS451749e86. Across444 protected engine/header/
build/launcher/tuning files only DS4main changes; GLM/TP2/MTP/math/transport/tuning/launchers stay
exact to accepted main. New tests exercise both CLI names with the TP encoder guard and preserve
GLM wire keys and budget continuation. The initial focused-command class-name typo is retained
alongside the corrected passing run. No fresh GPU/S-M-L/performance claim.

**Runtime unchanged:** no production deployment or remote source update. Canonical GLM5a52,
retained review GLMb398/DS451749, protected remote777cc92 user edits and configurations remain
as recorded in the prior restoration snapshot. Do not inspect unrelated services or stop user
processes. Only test-owned fake-engine children were started and cleaned up. For future work,
record exact PIDs/process groups; no name-/port-wide cleanup. The user launches their processes.

**Open work:** #33/#42/#46/#52/#54/#60/#68/#75/#79/#83/#11/#16/#10 remain open. The new DeepSeek
soft-close fix does not resolve GLM prefill, full DS4 WMMA, tail-drain or cap-measurement issues.
No release or tag was created. This checkpoint's acceptance and final CI outcome live in #85.

## 32. Fresh v1.1.49 live acceptance complete (#87 / #88)

The user required the same live validation used after earlier merges; preliminary §31 CPU/host checks
alone were insufficient. Fresh GPU gates now pass on both gfx1151/RDMA hosts. Source main997bba7
contains the cumulative v47–49 merge; server fix **e1682cb** and durable evidence are in **PR89**.
Exact final-head/merge/main CI and issue closure provenance are recorded on #87/#85/PR89. If #87 is
still open after an interruption, finish its GitHub closeout; do not rerun completed GPU stages.

**No active validation engines/builds.** API38059, CLI30667, strict89906, prefill32674 and reverse10081
all EXIT0. Final audit52 local/26 remote saved ownership records stopped/alive=[]. Ordered CPU
HIPmap/613server11skip/686tools35skip PASS; final focused peer-loss3tests PASS. Detailed report
[docs/upstream/v1.1.49.md](docs/upstream/v1.1.49.md), [live evidence](docs/upstream/v1.1.49/live/README.md)
and machine-readable [summary](docs/upstream/v1.1.49/live/summary.json).

- S/M/L ordinary+MTP7 large/reuse/stream/forced-idle, lifecycle/media/context rejection, stable PIDs,
  true18K partial-prefill disconnect/cache0 retry/follow-up PASS. Long/short decode ratios0.30–1.43,
  unchanged bound4. MTP matching rounds S146/M127/L148, each8 asymmetric retries.
- First intentional peer-loss request degrades health/down, no rank0 respawn; restart BOTH ranks
  restores healthy MTP,8/8 recovery per model. S/M first hit queued poisoned errors; L first hits the
  existing mirror-send guard. No spontaneous peer-closed error in healthy-path suites.
- All6 paired CLI modes32/32; S exact acceptedv42 trajectories. L WARM0 workaround remains (#42),
  within unchanged600s bound. Strict S technical/code256 + M/L64: all32 warm/measured records raw0.
- Both prefill orders candidate→baseline and baseline→candidate, sameb398/normalized configs/tuning,
  normalidle/MTP2, fresh1K/4K/8Kx3/arm: all18 response/usage pairs equal/cache0. First full cycle of
  every arm retained/excluded from medians;4warm samples/length/server. Pooled throughput deltas
  −3.64/−1.66/−1.95%; no speedup claim. Extra MTPbias rejection/spec1ordinary, Responses/Anthropic
  output limits and MTP follow-up PASS.

#88 fixes a gap in original #6: a queued-collective drain can fail before mirror-send TP2_DOWN.
Both merged997bba7 and premerge serverd7626 reproduce the stale-health first poisoned error with
sameb398. The server now marks explicit fatal TP errors sticky-down immediately, retains cause
and poison, and refuses later GEN before an engine write. Never reset poison/retry collectives or
restart one rank. CPU regressions fail before/pass after. Original failed/control logs, runner-only
TIME_WAIT preflight failure and initial overly narrow artifact-audit expectation are all retained.
The latter incorrectly required poisoned text for L too; it now recognizes both legitimate fatal
paths. No live assertion/time bound/raw-pick tolerance was relaxed.

All live files were isolated under `/tmp/maya-v49-live-87` on both hosts; API18087/bootstrap18587,
usb4_rdma5. Archives997bba7 SHA44bb2781… and e1682cb SHAa91d3652…; both use matching retained GLM
**b398d4a0711430edd6a28b95264c7293**. GLM/transport/MTP/tuning/launchers remain source-identical to
premerge; four tuning tables and expert profile match. Pack/tokenizer/native-map proofs retained.
Owned supervisor checks PID/start ticks/unique owner marker and descendants; foreign/stale identities
are refused. Only recorded test processes were stopped, owned server before engine. Legacy broad
cleanup in acceptance.sh/bench_mtp.py remains open #83; temporary owned adaptations do not fix it.

**Production unchanged:** canonical GLM **5a52cd27167523b6ea4b0f0bde93f503** both hosts and rank0
/tmp/maya-tp2; review GLMb398/DS451749 retained. Protected remote canonical777cc92 user edits/configs
were not pulled/reset/rebuilt. No unrelated service inspection/control, production deployment, tag
or release. The user granted full host freedom during this run; no further availability question is
needed for its closeout. The user launches production processes themselves.

**Limits remain open:** cold/first-request latency, including candidate first4K23.5–27.0s,
M recovery22token prefill99.7s and L WARM0 CLI MTP32token decode326.6s, is retained under #33/#60.
No causal regression/speedup claim follows from these small timing samples. No fresh full-model
DS4 GPU acceptance: #68 WMMA failures, #75 deferred tail-drain and #79 default-off chunk-cap
measurements remain open. Also retain #42/#46/#52/#54/#83/#11/#16/#10. Future fork tags use tp-v*;
VERSION remains numeric1.1.42 and legacy v1.1.42 immutable. No more runtime validation is required
for this v1.1.49 batch; use linked GitHub records for final repository closeout status.

## 33. Prefill promotion and user acceptance (#90 / #91 / #93 / #33)

The user tested Maya-S, Maya-M and Maya-L through llama-swap and
reported prefill and generation working across all models. On the prompt “write an eBPF tutorial
to explain how to use eBPF”, their observed prefill / generation ranges and average generation were:

| Model | Prefill tok/s | Generation tok/s | Prompt average generation tok/s |
| --- | ---: | ---: | ---: |
| Maya-S | 131–308 | 29–46 | 30 |
| Maya-M | 187–299 | 31–45 | 32 |
| Maya-L | 194–310 | 25–33 | 28 |

The user considers Maya-L higher quality with similar prefill and requested promotion and closure.
Canonical `build-hip/maya` on BOTH hosts is the tested candidate **cbd9e99a6a49e9107b0db3190a40766a**;
primary hipBLASLt table SHA256 **084286e9620f01443f7d2b66b6312fbdcb64e436b518e170d7aafa73ce742aa5**
matched on both. Canonical rank0 `serve/`, both rank1 launchers, tuning tables and S/L configs were
aligned with tested source8d3; original files are backed up on BOTH hosts in
`/home/ugo/software/project-maya-before-prefill-90-promotion`. Rank0 llama-swap config restored
byte-for-byte to `~/.config/llama-swap/config.yaml.before-prefill-90`; the user restarts services.
Remote canonical Git checkout retains its other user changes; do not reset it. No process was stopped.

Changes: HIP APU complete pool retains experts if complete layout fits above headroom/reserve and
respects existing caps; TP2 scratch uses existing 3008-row receive bound; three DSA hipBLASLt entries
are calibrated for gfx1151/100401. `MAYA_GLM_COMPLETE_POOL=0` remains the capped-pool control.
PR #94 merged as `6f85d30`; commits 3da1347 and 8d3ce94 contain the engine changes, and both PR
checks passed. Issues #33/#90/#91 closed as completed; #93 closed as not planned at user request.

**Limits:** strict S technical historical-reference test reported 12/256 raw picks different, and
L balanced comparison found one differing cold output in the old baseline; full candidate API/CLI
regression was not repeated. User accepted their real-workload result; do not claim bit-exactness or
that these observations were fixed. Earlier raw logs are in `/tmp/maya-prefill-90`; #93 documents
the baseline variation. The 3008-row limit and intermittent cold/startup latency are not eliminated.
Checkpoint tag `tp-v1.1.43` uses numeric VERSION `1.1.43`; no GitHub release was requested.

## 34. DeepSeek setup HIP failure (#98)

User attempted `./maya.sh --setup --backend hip --gpu 0 --model deepseek-maya-xs ...`; build
failed before model download/packing. On this split ROCm install `/opt/rocm` lacks `.info/version`
while the resolved compiler's `/opt/rocm/core` has it. `HIP_PATH=/opt/rocm` made clang omit its HIP
runtime wrapper, so device math calls failed; `HIP_PATH=/opt/rocm/core` compiled the same source.

PR #99 merged as `21c9944` and #98 closed after the user successfully started DeepSeek. The installer fix selects the versioned compiler subroot;
focused split/normal tests and HIP mapping pass. Clean installer-driven gfx1151 `maya-ds4` build
completed in `/tmp/maya-ds4-build-98`, log `/tmp/maya-ds4-build-98.log`. The clean binary MD5
`0001d60d186d879b971eee254c03671a` is installed at local `build-hip/maya-ds4`; verified its
shared libraries and the real installer reuse check. `build-hip/MAYA-BUILD.json` records the actual
engine source fingerprint `957159b3d6717a89` and both build targets; its previous contents are in
`build-hip/MAYA-BUILD.before-issue-98.json`. Engine sources match the promoted GLM build's source
and `build-hip/maya` remains `cbd9e99a6a49e9107b0db3190a40766a`. No model was downloaded or started
at that checkpoint; the user subsequently set up DeepSeek Maya-XS and Maya-S (§§35–37).

Ordered CPU checks passed: HIP mapping, 613 server tests (11 skipped), 687 tools tests (3 skipped);
logs `/tmp/maya-98-{hip-map,serve,tools}.txt`. The installed DeepSeek binary passed the real build-reuse
check. A successful engine build is not full-model DeepSeek validation; #68 remains open.

## 35. DeepSeek performance and calibration (#100 / #101)

At the measurement checkpoint, the user ran DeepSeek Maya-XS on local port 8000. Process state may
have changed; do not stop or restart user-owned processes. Read-only `/health` and `/metrics` showed
262144 context, one GPU, DSpark
and prompt lookup enabled. Last completed request generated 853 tokens at 24.76 tok/s; 399/531
drafts accepted, only 6/12 from lookup, and reported VRAM hit 100% with CPU expert share 0.0%.
Startup logged 15730/15744 experts in VRAM, 14 on CPU lane, ~0.5 GiB free before prompt.
The 35-token prompt's 37.8 tok/s is not comparable to long-prompt prefill.

Setup calibration's no-DSpark arm reported 19.5 tok/s; its DSpark arm failed with a 600-second
watchdog and did not produce a valid final comparison. Three such earlier watchdog reports were
observed; #101 tracks them separately. Do not disable/extend watchdog without investigation.

User compared to upstream [#663](https://github.com/mw00/project-maya/issues/663): 43.2 tok/s
was a mean of three greedy 256-token answers (thinking off), 453 prefill at 8192 tokens; that was
Mix-128G on a Windows 192 GB Ryzen AI Max+ PRO 495 / 8065S. This user's 128 GB Max+ 395 / 8060S
Linux uses Maya-XS. Those figures do not establish a like-for-like comparison or a single cause.
Issue #100 tracks performance; #101 tracks calibration hangs. User authorizes reading anywhere but
issues only on uv-genai/maya-tp. User explicitly authorized benchmark requests to the running server. API benchmark session6443
EXIT0: the same greedy/thinking-off Python, prose and HTML questions yielded 39.3/28.5/47.9 tok/s
(mean 38.6 vs upstream 43.2). Fresh 2052 and 8196-token prompts, both cache0, yielded 203.5 and
298.1 tok/s (upstream 2048/8192: 278/453). Single samples; API and upstream direct engine differ,
as do model quant, OS and hardware; no causal attribution. Server was healthy, idle, DSpark enabled
afterward. Durable report and raw timing data: `docs/benchmarks/deepseek-maya-xs-2026-11-02/`.
#100 remains open for investigation, #101 for the 600-second calibration hang. Only open issues
on uv-genai/maya-tp; the user permits read-only access anywhere.

User requested one final coding prompt (thread-safe Python LRU cache and tests). With DSpark on,
greedy/thinking off, 38 uncached prompt tokens took 1438.8 ms; 512 output tokens took 12333.8 ms
(41.5 tok/s), 408/436 drafts accepted. The output hit the token limit and was incomplete. The user
server stayed healthy and idle. Timing data is in `docs/benchmarks/deepseek-maya-xs-2026-11-02/coding-last-summary.json`.

## 36. DeepSeek Maya-S slow decode (#102, 2026-11-02)

The user started a single-host DeepSeek V4.1 Flash **Maya-S** server on localhost port 8000.
At that checkpoint `maya-ds4-hip.json` selected the Maya-S pack, 262144 context and `--dspark on`;
§37 records the later config correction. Health and `/metrics` showed DSpark enabled. One completed
request generated 507 tokens in 125.140 s (4.05 tok/s), with 262 of 332 DSpark drafts accepted.
At startup 9,779 of 15,744 experts occupied GPU slots; 5,965 experts (~65.8 GB) were on the CPU
lane. A later active-request metric reported approximately 370 MB/token of process storage reads;
this counter measures all process reads and does not prove every byte was an expert read. At the
same time `/proc/meminfo` reported under 1 GB available. Unified-memory pressure and storage reads
are strong bottleneck candidates, not yet a proven complete breakdown. The server was not changed.
Issue #102 remains open for investigation; do not assume the smaller Maya-XS results in §35 apply
without qualification to this larger quant. No new validation requests were sent during this diagnosis.

## 37. DeepSeek XS llama-swap selection (#103, 2026-11-02)

The shared installer config had been overwritten for Maya-S, so llama-swap's `maya-deepseek-xs`
entry actually started Maya-S. Saved separate `maya-ds4-maya-xs-hip.json` and
`maya-ds4-maya-s-hip.json` configs and restored `maya-ds4-hip.json` to XS. Once llama-swap had no
child processes, updated its XS entry to reference the XS-specific config so a future installer run
cannot silently redirect that entry. YAML backup: `~/.config/llama-swap/config.yaml.before-deepseek-xs-103`.
Both configs' pack paths, tokenizer directories and source manifests were checked; XS sets DSpark on.
The user subsequently reported that everything was working well, but no separate fresh XS startup
or throughput measurement after the config correction was recorded. The user launches models.

## 38. Remote GLM startup repair (#104, 2026-11-03)

The user reported llama-swap GLM startup failure on strix-halo-1. Its `~/maya-tp2-swap.log`
showed `AttributeError: module 'maya_names' has no attribute 'export_env'` in `child_env`.
The promoted remote `serve/server.py` expected newer helpers than the remote
`tools/maya_names.py`, which was missing `both_names`, `lookup` and `export_env`; the earlier
selective GLM server deployment appears to have omitted that dependency. Updated only that remote
helper to the matching local version after confirming the difference was exactly those functions.
Backup is `/home/ugo/software/project-maya/tools/maya_names.py.before-104` on strix-halo-1.
Remote child environment initialization passed for the S and L configs, alias precedence passed,
and `serve.test_tp2_peer_loss` passed (3 tests). No user process was stopped. After the repair, S and M
reached ready, and Maya-L subsequently reached ready and completed requests through llama-swap.
The user reported that everything was working well. Issue #104 remains open in GitHub.

## 39. Maya-L startup exceeds timeout (#105, 2026-11-03)

Following #104, S and M reached ready but L repeatedly timed out at 600 seconds. Latest rank0
`maya-maya-l-hip.log` showed expert pool warming at 80% after 534 seconds; rank1 completed its
76.4 GB pool in 252 seconds. Both GLM binaries still have user-accepted MD5
`cbd9e99a6a49e9107b0db3190a40766a`. While remote llama-swap had no child process, changed its
`healthCheckTimeout` from 600 to 1800 seconds; backup on strix-halo-1 is
`~/.config/llama-swap/config.yaml.before-maya-l-timeout-105`. This mitigates the timeout only;
why rank0 loading is slow remains unestablished. Maya-L later reached ready and completed requests.
No user process was stopped by the agent. Issue #105 stays open.

Live #105 diagnosis: the user restarted Maya-L at 00:48 on Nov 3. Rank0 NVMe was 97–98% busy
with ~38,600 reads/s averaging 4 KiB (~158 MB/s); engine `/proc/<pid>/io` showed ~146 MB/s
physical reads. An idle 1 GiB direct sequential read of the same shard completed in under a second,
so the warmup's strided mmap access matters. A prelaunch fincore sample showed only ~29 GB of
156 GB of GGUFs resident. Earlier logs show warmups of 198–204 s but also older 394–561 s runs;
cache state is a plausible contributor but the reason for its change is not established.

Subsequent update: Maya-L reached ready and completed requests through llama-swap, and the user
reports everything is working well. The 1800-second timeout is a workaround; the cold-loading
slowdown remains unresolved. The user requested merging the DeepSeek investigation branch into
main. It was fast-forwarded and pushed as `76e0279`; post-merge tests are awaiting the user's report
(#106). Do not stop user-owned processes.

## 40. DeepSeek two-host design (#108, 2026-11-03)

The user requested analysis and design only, with implementation planning deferred. See
[DeepSeek TP2 and DSpark analysis/design](docs/DEEPSEEK-TP2-ANALYSIS-DESIGN.md). It compares
uneven tensor slicing, whole-expert ownership and layer splitting, describes distributed DSpark
state and failure contracts, and identifies measurement and correctness gates. DeepSeek's existing
`--tp` is same-host grouping; two-host support is not implemented. Maya-S's 2304-wide expert FFN
cannot be split evenly into two 256-aligned bands. No binaries or user processes were changed.

## 41. DeepSeek two-host implementation plan (#109, 2026-11-03)

The user requested an implementation plan. See
[DeepSeek TP2/DSpark implementation plan](docs/DEEPSEEK-TP2-IMPLEMENTATION-PLAN.md), based on
[the analysis/design](docs/DEEPSEEK-TP2-ANALYSIS-DESIGN.md). Seven independently tracked phases are
#112, #115, #111, #110, #113, #114 and #116 in plan order. The proposed uneven tensor-slicing
approach remains subject to P1 feasibility. The user subsequently authorized implementation;
no binaries, model processes or production configs were changed by the planning work.

## 42. DeepSeek TP2 implementation in progress (#110–#116, 2026-11-03)

**Current state:** experimental two-host target inference, chunked prefill,
DSpark and prompt lookup have been implemented and exercised on XS and S.
They are not yet approved for production: numerical/quality review and
fully warm/peak-budget validation remain open, and the user must review and
test the candidate before any promotion. Working
branch `feat/deepseek-tp2` (see `git log` for the latest checkpoint); no production binary
or config has been promoted. The retained `be3fd7a` review engine is
`/tmp/ds4-tp2-be3-engine`, with matching worker copy
`/tmp/ds4-tp2-coldwarm-engine` on strix-halo-1; their SHA-256 is
`bf1adc9fcf81f5478f5774b1fb961e3c950d0816fd28fa055128c70c548236f4`.
The previous `fed8567` smoke engine had SHA-256
`195a12a4fc390d3f7a43f1548d7618adc8f6661582818d06018ffb9fbe3fa9ae`.
The earlier fully tested sampled-lookup build had SHA-256
`d13f478d0702cebb35a40c99cdf0918787fa4079fa8de02cf4166a12663e203e`.
The earlier `a34d616` long-test engine at `/tmp/maya-ds4-tp2-p1-build/maya-ds4`
and its worker copy `/tmp/ds4-tp2-114-current-engine` have SHA-256
`f5d2ed9cf6714b38bc8320eb5e0d11658a67b7e60d0a6f57a30cdb74f050bff7`.
The earlier benchmark build had SHA-256
`31b2e21d529f5de3016c7ce58a3bab704f052ab5d22beea62b64829680a81fb1`.

**Active owned process:** none. The exact-head gfx1151 build and ordered HIP
mapping, server (618 tests, 11 skipped) and tools (707 tests, 3 skipped)
checks passed for `52cf519`. The XS exact-head API smoke passed DSpark on/off
and sampled lookup; both ranks exited. Review engine SHA-256
`a0dcc1cb91e9e52fa365291d68eef8203af714d8222e77e49d72741a4331ddc2`:
`/tmp/maya-ds4-tp2-113-point-build/maya-ds4`, worker copy
`/tmp/ds4-tp2-verified-engine` on strix-halo-1. Paths are temporary; verify the
hash before use. The test-only split-replay build log is
`/tmp/ds4-tp2-110-split-replay-build.log`.
It independently checks pack/partition/build identities, then the test
communicator uses its documented synthetic wire identity. It has a 3 GiB
per-host memory guard. The preceding strict test failed at 257 because the
unsupported direct whole feed and supported 256+1 split had max logit delta
1.059103, although their top token matched; its logs are saved as
`docs/validation/deepseek-tp2-control-2026-11-03/sweep-xs-whole257-failure-rank*.txt`.
The XS rerun recorded that diagnostic without demanding exact unsupported-call
logits and passed rank and window/chunk top-token agreement at all six lengths.
At 512 tokens both paths selected 9857 and whole/split logits matched exactly.
S also passed the six top-token checks; at 512 both paths selected 271 and
whole/split logits matched. Its unsupported 257-token direct whole call differed
from the supported split by max 3.5900075 in logits while keeping the same top.
The XS and S 545-token formatted prompt checks passed rank agreement,
window/chunk top-token agreement and exact whole/split logits; all ranks exited.
XS and S also passed those checks on the 3,204-token fixture and all ranks exited.
A test-only supported 256+1 split replay was added to `ds4_tp_text_rdma_test.cpp`.
Both XS and S passed with rank and replay max logit deltas 0; all ranks exited.
Three identical requests per XS/S single-host and TP2 configuration
produced identical answers within each configuration. Requests 2/3 on both
TP2 models had zero new rank-0 Engram row reads and zero recorded physical
read_bytes on both ranks, with prompt reuse 0. Single-host still read data
while Engram misses were zero and prompt reuse was 27 tokens. This is warmed
request working-set evidence, with startup and peak-budget scope separate.
All owned processes exited. Evidence is in
`docs/benchmarks/deepseek-tp2-review-2026-11-03/request-working-set.md`.
An extended XS/S DSpark rollback sweep crossed position 4096 from a
4,095-token prefix. Both models passed all 16 acceptance counts with exact
subsequent logits, five matching draft token IDs and confidence differences
within 0.0001. All ranks exited. HIP mapping, 618 server tests (11 skips) and
707 tools tests (3 skips) passed after the test-only prefix-bound change.
Evidence is in `docs/validation/deepseek-tp2-control-2026-11-03/rollback-4k.md`.
The isolated attention arithmetic investigation traced the first layer-0
query projection difference to per-32 Q8 versus FP16 GEMM arithmetic. An
independent CPU reference found the chunk projection closer to the dequantized
weights on the captured vector. Matching all attention projections reintroduced
EOS on the 512-token synthetic fixture, so those substitutions remain confined
to the diagnostic worktree. Both ranks exited. Evidence is in
`docs/validation/deepseek-tp2-control-2026-11-03/attention-arithmetic.md`.
GPU busy and rank-0 Engram lookup deltas were sampled once for XS/S on
single-host and TP2. Logical Engram reads, process-wide physical I/O and
device-wide busy are distinct; placement and cache differed, and the samples
are not a general speedup claim. All owned processes exited. Evidence:
`docs/benchmarks/deepseek-tp2-review-2026-11-03/utilization-engram.md`.
The review guide now includes a versioned-binary promotion and rollback
procedure. Its staging shell snippet passed isolated install, repeat,
conflicting-file and symlink-rejection checks. No deployment was performed.
A four-task coding comparison on XS and S single-host and TP2 passed all
23 fixed functional and edge cases on every path, using sandboxed generated
code; implementations differed in 2/4 XS and 1/4 S tasks. All processes
exited. This small sample does not establish general equivalence. Evidence:
`docs/validation/deepseek-tp2-control-2026-11-03/coding-comparison.md`.
An additional 20-case short objective cross-path check passed 19/20 strict
format checks on each XS path, and 19/20 single-host versus 20/20 TP on S;
the only strict misses gave the correct bitwise result with extra text.
All owned processes exited. Raw outputs and limitations are in
`docs/validation/deepseek-tp2-control-2026-11-03/cross-path-extended.md`.
XS and S each passed eight simple cross-path
sanity cases on one host and two hosts; 5/8 XS and 7/8 S answers were
text-identical, with remaining differences in JSON formatting or verifier
wording. All owned processes exited. The single-host runs used 8 GiB reserve,
zero read-ahead and a 3 GiB memory guard. The later current-build sweep no
longer reproduces the earlier synthetic 512-token chunk/window top-token
discrepancy; substantial full-logit differences remain. Evidence:
`docs/validation/deepseek-tp2-control-2026-11-03/cross-path-sanity.md`.
The cold/second-start comparisons used only
manifest-verified model source pages evicted via per-file `POSIX_FADV_DONTNEED`,
with sampled `mincore`; no system-wide cache drop was used.
The XS and S paired cold/warmer comparisons passed and all pairs exited:
startup was 46.17/46.67 s for XS and 91.30/67.20 s for S. Answers matched
within each pair, but sampled cache was only partially resident before the
second launch and both runs had substantial physical reads. Raw results are
`docs/benchmarks/deepseek-tp2-review-2026-11-03/{xs,s}-tp-coldwarm.json`.
The guarded single-host XS and S runs also passed, with startup of 25.03/23.03 s
and 27.05/27.03 s respectively and matching answers within each model. Before
the second launch 0/448 XS and 17/512 S sampled pages were resident, so they
cannot be called fully warm-cache measurements. Single-host S request-time
process reads were 48.83/45.78 GiB versus roughly 0.18–0.19 GiB per rank for
the paired requests; reserve, expert placement and memory pressure differed.
All owned processes exited. Raw results are
`docs/benchmarks/deepseek-tp2-review-2026-11-03/{xs,s}-single-coldwarm.json`.

The `fed8567` XS paired smoke passed
with 0 vs 88/100 DSpark drafts accepted/offered and identical 128-token greedy
answers, plus 163/164 sampled lookup drafts accepted/offered; both owned ranks
exited. Ordered host CI for `fed8567` passed: HIP mapping, 618 server tests
(11 skipped) and 707 tools tests (3 skipped); logs use
`/tmp/ds4-tp2-final-{hip,serve,tools}.log`.
The `69b63c7` XS smoke also passed with matching greedy answers, 88/100 DSpark
drafts and 165/166 sampled lookup drafts accepted; both ranks exited. Its
retained lead `/tmp/ds4-tp2-69-engine` and worker `/tmp/ds4-tp2-exact-engine`
have SHA-256 `cac60c7939efb8fb89b878d07450e2619ef12bcd1ad017efb50e2324db6b597e`.
Ordered host CI for `69b63c7` also passed: HIP mapping, 618 server tests
(11 skipped) and 707 tools tests (3 skipped), with logs under
`/tmp/ds4-tp2-latest-{hip,serve,tools}.log`.

The larger S sampled test passed all three
modes (10,000 plain vs 500 speculative): sampled-drafter p=0.6686, point with
penalties p=0.9782, and lookup after stale q p=0.9369. The rebuilt XS and S
pairs each passed all 16 forced rollback positions with matching next logits
and five matching DSpark drafts at each position. All owned test ranks exited.

**Completed API/recovery check:** the XS test passed Responses, Anthropic,
DSpark request off/on, lookup alone, output limits, streaming, media
rejection, sticky Anthropic peer loss and a fresh two-rank restart. Both owned
ranks exited. The 1,024-token eBPF example hit its limit; it is a partial
quality sample. Evidence: `docs/validation/deepseek-tp2-control-2026-11-03/api-recovery-xs.md`.

**Completed long-context check:** the preceding owned Maya-S pair and sampler
have exited. With checkpoint `a34d616` and configured context 524,288, a
32,772-token prompt generated 32 tokens and accepted 15/18 DSpark drafts; a
524,004-token prompt generated 32 and accepted 12/13 drafts. Results and
sampled telemetry are `docs/benchmarks/deepseek-tp2-review-2026-11-03/s-tp-long-*`.
The ordered host checks and an isolated engine build ran concurrently with
portions of this test, so do not treat its timing as a clean performance
measurement: HIP mapping and 617 server tests
(11 skipped) passed, followed by 707 tools tests (3 skipped). Logs use
`/tmp/ds4-tp2-116-ci-{hip,serve,tools}.log`. The same-process Maya-S single-host DSpark parity
test and guarded single-host
DSpark-on/off controls have exited. Parity logs
use `/tmp/ds4-tp2-113-s-single-spec-parity*`; control PID/log/config and
engine log use
`/tmp/ds4-tp2-116-s-safe-single-{on,off}-*`. On completed an 8K prefill in
809 seconds; off ran the short decode controls only. The tests used 8192 MiB
reserve, `MAYA_DS4_WARM_CAP_GB=0`, separate `/tmp` tuning stores, and a
3 GiB available-memory guard. The original uncapped attempt was stopped during
startup at ~127 MB available; memory recovered. Single-host read-ahead pressure
is tracked as #117. Separate on/off launches produced different S answers;
same-process fresh-prompt checks matched all 64 greedy tokens on both 19- and
545-token prompts, accepting 38/47 and 30/35 drafts. This narrows but does not
prove the cause of the separate-launch differences.

**Sampled-lookup follow-up:** an audit found that lookup-only sampled verification
could reuse stale sampled-drafter q after an unverified draft if the sampling
parameters and counter recur. The implementation explicitly marks lookup
proposals as point distributions and adds a paired regression that deliberately
leaves sampled q cached. The completed long run was on checkpoint `a34d616` and
does not include this later change. Main, HIP model and regression syntax checks
passed. The earlier isolated gfx1151 build (log `/tmp/ds4-tp2-113-point-build.log`)
had engine SHA-256
`d13f478d0702cebb35a40c99cdf0918787fa4079fa8de02cf4166a12663e203e`, paired
test SHA-256 `cc1736ab4be89d97a8897f7aadb0ece8be02c250ac04872712fd94c04bae2814`.
The new-build XS 50-trial test passed drawn, penalty and stale-q lookup modes;
the lookup comparison had p=0.5444, 15/50 accepted, and a deliberately stale-q
control changed 23/32 verdicts. The S test also passed: lookup p=0.3873,
15/50 accepted, stale-q control differed 19/32. Both pairs exited. Results
and limitations are in `docs/validation/deepseek-tp2-control-2026-11-03/sampled-lookup-proposal.md`.
Never overwrite an active rank executable. The prepared
`/tmp/ds4-tp2-114-api-recovery.py` and
`/tmp/ds4-tp2-113-sample-500-driver.py` use the new build. The latter accepts
trial count (50–1000) and model (`s` or `xs`). The prepared
`/tmp/ds4-tp2-113-rollback-point-driver.py` accepts a model and repeats all 16
rollback positions. All use isolated files, recorded PIDs and a 3 GiB guard.
Details and progress are tracked in #113.

**API follow-up (#114):** the implementation forwards per-request
`strata_tune.spec=0` to DeepSeek and reports `spec=1` as enabled for that
family, preserving GLM's existing 1=off convention. DeepSeek lookup remains a
separate source, so the monitor does not report all speculation off while
lookup remains enabled. Sixty-one focused sampling-key/dashboard tests passed
(`/tmp/ds4-tp2-114-spec-tests-v2.log`), as did the full 618-test server suite
(11 skipped; `/tmp/ds4-tp2-114-server-suite.log`). The completed API test
verified per-request DSpark off/on counts, monitor status and greedy equality
on the rebuilt pair; the long-context test had loaded earlier Python code.

**Completed:** XS and S TP2 with DSpark on/off produced identical outputs on
nine fixed prompts per model; same-binary S decode was 21.8–30.4 tok/s on
versus 17.6–19.9 off, XS 24.6–35.7 versus 17.1–23.4. GLM TP2/MTP HTTP
cancellation and peer-loss regression passed. Both XS and S passed all 16
forced rollback positions across the 128-token boundary. The Lt-store
read-only safeguard passed real GPU and paired-model tests. Raw results and
qualifications: `docs/benchmarks/deepseek-tp2-review-2026-11-03/README.md`
and `docs/validation/deepseek-tp2-control-2026-11-03/`.

**Final validation:** consult #116 for ordered host CI and paired GPU smoke on
the latest commit. Single-host and TP2 output
differences remain documented; do not claim exact cross-path equivalence or
eliminated disk I/O. #115 (partition and rank loading) and #111 (RDMA/control
lifecycle) are closed after their gates passed. Check GitHub for the current
states of #110, #112–#114, #116 and #117. Existing user processes and the
remote production checkout remain protected.

The entries below are a historical work log, not an ordered list of currently
running processes. Earlier statements about missing implementation, active
tests, or incomplete copies may be superseded by the current state above.

Working branch `feat/deepseek-tp2`; design and plan committed as `231a48b`. Both hosts were free of
model processes when P1 began; their llama-swap daemons remain user-owned. Pack inventory and an
isolated two-host RDMA smoke/benchmark are in `docs/benchmarks/deepseek-tp2-p1-2026-11-03/`.
Both ranks passed the transport test; P1 capacity, single-host reference and architecture gates
remain open. DeepSeek TP2 engine execution is not implemented yet. P2/P3 foundational helpers and
tests are in `2b6d544`–`e84734e` (#115/#111): whole-block ranges, expert byte spans, identity,
bounded control codec/channel and the shared transport's control-data-versus-EOF fix. The exact
bootstrap poison bug was reproduced with the old transport and fixed in real paired RDMA tests;
real peer loss still retains the sticky original cause. See
`docs/validation/deepseek-tp2-control-2026-11-03/`. These changes are not a substitute for full
GLM model regression or DeepSeek distributed execution and DSpark validation.
`ecbe2e6` adds a tested logical worker control/verify state machine; `91e6c54` adds a checked
host expert-slice packer. `279a04d` adds a real-pack byte-reconstruction test, which passed for
XS experts on layers 0 and 1 on strix-halo-1. None of these establish GPU computation parity.

An owned rsync started to copy Maya-XS, shared Engram, DSpark and vision into the previously absent
`/home/ugo/software/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/` on strix-halo-1. Local shell
PID 155996, rsync PID 156005 and SSH PID 156006 were recorded at launch; unified exec session 64870,
log `/tmp/ds4-p1-copy-112.log`. Recheck PIDs/start times before acting. The copy was initially
~110 MB/s, about 43 minutes; it is incomplete until rsync exits successfully and the remote pack is
checked. The copy subsequently completed with exit 0; rsync's size/mtime dry-run was clean and
remote `--pack-info` found all seven indexed files, but full cryptographic content comparison has
not yet been done.
An isolated transport build is at `/tmp/maya-ds4-tp2-p1-build`; its temporary test ranks have exited.
The reference API harness is `tools/ds4/bench_tp2_reference.py` (`0570a8e`); its mock checks passed,
but it has not yet been run against a model. `tools/ds4/estimate_tp2_experts.py` (`408c0b5`) estimates
logical routed expert tensor bytes for alternating 5+4 blocks: Maya-S ~84.510 GB/rank, XS
~54.814/54.751 GB. These exclude other allocations and do not prove fit or correct kernel layout.
An isolated clean `maya-ds4` build passed in `/tmp/maya-ds4-tp2-p1-build`, log
`/tmp/maya-ds4-tp2-p1-engine-build.log`. Its MD5 is `40b27560ae45924879fed5c5ef2808a7`; a matching
review copy is on strix-halo-1 at `/tmp/maya-ds4-tp2-p1-review-112`, not in the production tree.
The ordered HIP mapping, 613 server tests (11 skipped) and 687 tools tests (3 skipped) passed;
logs are `/tmp/ds4-p1-{hip,serve,tools}-112.log`. Do not run model performance tests while the
model transfer is active. After it completed, an isolated XS server (PIDs 176926/176928) completed
nine prompt samples and was stopped. Results are in `xs-single-reference.json` in the benchmark
directory. An isolated Maya-S reference server was then started as PID 177990, with engine PID
177992 and process group 177990, on local port 18701; config `/tmp/ds4-p1-s-112.json`, server log
`/tmp/ds4-p1-s-server-112.log`. Those owned test processes were stopped server first after the run.
User-owned llama-swap daemons remain untouched.
Maya-S reference harness was exec session 29729, log `/tmp/ds4-p1-s-bench-112.log`; it completed
nine samples and the owned S server/engine were subsequently stopped. Its JSON is in the same
benchmark directory. An isolated
remote source/build for the uncommitted `ds4_model.cu` two-way partition integration is at
`/tmp/ds4-tp2-review-115-{src,build}` on strix-halo-1; build session 64518 and log
`/tmp/ds4-tp2-review-115-build.log`. Its input `ds4_model.cu` SHA256 is
`8d05ce349b8a444242d8e428d2c87b0d6bd2470ba1f004a89d7815965bcca034`. Do not claim this change
compiled or passed GPU tests until the build and validation complete.
Update: that isolated candidate compiled, and 64-token single-host XS DSpark output matched the
reference (`f76114c`); the two-member path itself was not exercised. The owned Maya-S 262K reference
completed and was stopped (`6abd8c8`). The owned 512K Maya-S one-token load check and single-host XS
candidate `--sampler-test 50` also completed successfully. The former reported 9832/15744 experts in
GPU slots and 5912 on the CPU lane; the latter is a small statistical smoke test. These were isolated
single-host checks, not DeepSeek TP2 model validation. Evidence is in the P1 benchmark directory and
`docs/validation/deepseek-tp2-control-2026-11-03/`.

Owned full SHA-256 comparisons of the seven XS indexed source files completed successfully on both
hosts; all seven digests matched. The results are in the P1 benchmark directory's
`xs-source-sha256.txt`. This establishes copy equality at measurement time, not upstream provenance.

P5 refactor `b383b0e` separates target-window execution from DSpark commit/rollback. Isolated
gfx1151 build and single-host greedy plus small sampled checks passed. Both the XS and Maya-S
prefill harnesses completed and their owned servers and engines exited. Maya-S fresh-prompt
samples are in `s-prefill-reference.json`: 2K 92.0/125.6/140.7 and 8K 265.1/286.0/258.9
tokens/s, all with zero cached prompt tokens.

The P5 worker-only verify/commit API and internal rollback-buffer reservation have been tested
against the real XS pack on one gfx1151 host, using matched window paths. The three-token
zero/partial/full acceptance cases matched subsequent target logits and DSpark drafts, and
missing buffers and invalid verdicts were rejected. Initial mixed chunk/window comparisons
differed and are not evidence of a rollback failure; see
`docs/validation/deepseek-tp2-control-2026-11-03/worker-verify-single-host.md`. These checks do
not validate long contexts or two-host inference. Review sources/build remain isolated under
`/tmp/ds4-tp2-review-115-{src,build}` on strix-halo-1. No two-host DeepSeek model execution
exists yet; user processes remain untouched.

P2 now checks a joint two-rank geometry for expert blocks, attention heads and output groups.
The host test and real XS/S expert reconstructions passed with 32 heads and four output groups
per rank. A separate HIP uploader now transfers a rank's sliced expert directly into a compact
device layout; real XS/S expert-byte checks and synthetic quantized arithmetic passed on gfx1151
(`docs/validation/deepseek-tp2-control-2026-11-03/expert-upload.md`). A bounded, fail-closed GPU
expert pool was also checked with sampled XS/S slices. Neither component is connected to the model
loader; the checks do not establish full rank allocation or distributed inference.

A pack-backed dense loader also validates and uploads rank-local input or output matrix slices,
or a whole rank-owned matrix. XS Q6_K, S Q8_0 and BF16 byte checks and invalid-metadata checks
passed on gfx1151; the existing XS DSpark regression passed. It is not connected to the model
engine. See `docs/validation/deepseek-tp2-control-2026-11-03/dense-loader.md`.

The Maya-S directory was absent on strix-halo-1. An owned rsync copied it into that new
directory using `--ignore-existing --partial-dir=.rsync-partial`; it exited successfully after
transferring 339,486,892,688 bytes (log `/tmp/ds4-tp2-115-s-copy.log`). SHA-256 verification of
all six shards, the shared DSpark and vision files and pack metadata matched on both hosts
(16 files; `/tmp/ds4-tp2-115-s-sha-local.txt` and `/tmp/ds4-tp2-115-s-sha-remote.txt`).
The copied pack passed the isolated engine's `--pack-info`, which reported three DSpark blocks,
supported tensor types and all indexed files present; log `/tmp/ds4-tp2-115-s-packinfo.log` on
strix-halo-1. The checksum and rsync processes exited. User-owned processes remain untouched.

The pack-backed rank-expert loader passed 385-expert XS/S tests for both rank assignments, a
single-host DSpark regression, and full XS expert-pool loads on both hosts (#115). Full XS rank
allocations were 54,814,310,400 and 54,751,395,840 GPU bytes, with 4,249,600 pinned staging
bytes each; both owned test processes exited. See
`docs/validation/deepseek-tp2-control-2026-11-03/rank-expert-loader.md`. This covers routed
expert slices only; the loader is not connected to inference and does not establish full-rank fit.
The S copy and content verification have completed; paired S checks remain.

Both Maya-S routed-expert rank pools loaded sequentially on the source host (strix-halo-2): each
held all 15,360 expert slices in 84,509,982,720 GPU bytes, with 7,014,400 bytes of pinned staging.
Both owned test processes exited. This excludes dense weights, KV, Engram and DSpark; it does not
prove full-model fit or paired inference. See the rank-expert loader validation report and logs.

P4 has a fail-closed model-operation adapter for feed, step, verify, verdict, reset and shutdown.
It requires window-only prefill, validates the model's position and drains device work at operation
boundaries. Fake-model sequencing/failure tests and an isolated real-XS model dispatch passed;
the short DSpark and no-drafter verify parity checks also passed after this change. This adapter
does not yet perform model partitioning or distributed reductions. A paired real-model control
smoke is next. Standard HIP mapping, 613 server tests (11 skipped) and 687 tools tests (3 skipped)
also passed in this work session.

A real XS expert was also run as rank-local FFN slices on both hosts and its partials summed over
RDMA. Layer 0's 1280/1024 and layer 1's 1024/1280 slices matched each other exactly after reduction
and the full single-GPU expert within 1.2e-8. See
`docs/validation/deepseek-tp2-control-2026-11-03/rdma-expert-xs.md`. This is one routed expert,
not an end-to-end TP model, and uses synthetic test handshake identities.

Two individual XS Q6_K attention projections also passed two-host tests: gathered `wq_b` output
matched a full GEMV exactly, while summed `wo_b` partials differed by at most 1.2e-7. See
`docs/validation/deepseek-tp2-control-2026-11-03/rdma-dense-xs.md`. These are synthetic activations
through individual matrices, not full attention or an integrated model.

An owned replicated-XS paired control/RDMA smoke on port 18722 completed and both owned ranks
exited. The adaptive-lane attempt produced a 5.6557 maximum logit difference and the hosts chose
different CPU/stream splits. With identical fixed CPU execution, a rerun passed: all 129,280
subsequent logits matched exactly after a partial speculative verdict. See
`docs/validation/deepseek-tp2-control-2026-11-03/replicated-model-smoke.md`. These use full
replicated models and synthetic test identities, not TP model inference or production negotiation.

The experimental DeepSeek model loader now installs complete rank-local attention,
shared-expert and routed-expert slices, but deliberately rejects inference and restore
until the distributed execution path is connected. Both XS ranks loaded on the two
hosts with 4.22 GB dense weights and 54.81/54.75 GB expert shards; both S ranks loaded
sequentially on the source host with 5.22 GB dense weights and 84.51 GB expert shards,
leaving about 35 GiB free. Guard checks, the existing single-host XS DSpark regression,
and the HIP mapping check passed. This is load-only, not TP inference. Details:
`docs/validation/deepseek-tp2-control-2026-11-03/rank-model-loading.md`. The S rsync
and content verification have completed; paired S tests remain.

The first experimental XS model-pair execution on bootstrap port 18730 failed parity.
Both ranks agreed exactly on routes and logits, but maximum logit differences from a
single-host reference were 14.5–18.7, with a different top token at the fourth position.
Both owned ranks exited. Logs are `/tmp/ds4-tp2-110-tp-xs-rank0.log` on strix-halo-1
and `/tmp/ds4-tp2-110-tp-xs-rank1.log` locally. The experimental execution code is
uncommitted and restricted to the review test executable; trace the first divergent
layer before enabling anything. The second owned pair exited with the same failure.
Its reference and rank-0 traces are `/tmp/ds4-tp2-110-reference-layer.bin` and
`/tmp/ds4-tp2-110-rank0-layer.bin` on strix-halo-1. Attention residuals match within
1.8e-7 through layer 2; the next attention sees a 0.00665 maximum difference, so
inspect the preceding expert block and its routes. The matching pair on port 18732
also exited. Its route trace shows the first material residual difference occurs
on prompt row 2 after layer 2, with the same selected experts and near-identical
weights as the reference. Route order changes in layer 3 and some choices diverge
later. Both rank processes on port 18733 exited. Their layer-2 intermediates show
the routed expert hidden and reduced partials match within 2.4e-7 and 6e-8. The
shared Q6_K projection's per-32 Q8 input crosses a boundary at row 2 index 4921:
the reference ratio is 3.4999928 and the pair's is 3.5000079, rounding 3 versus 4.
The shared output then differs by up to 0.00036; later routing and logits diverge.
An isolated reference override confirmed the cause: using the pair's layer-2 FFN
input reduced row 2's residual difference after layer 2 to 1.79e-7, and it stayed
below 3e-7 through layer 4 before another boundary diverged. The override process
exited; data and log are under `/tmp/ds4-tp2-110-reference-override*` remotely.
This synthetic-token exact-logit comparison remains failed. The pair implementation
is test-only and no end-to-end gate has passed; compare actual text prompts next.
The ordered HIP map, 613 server tests (11 skipped), 687 tools tests (3 skipped),
and focused TP peer-loss regression passed. The test-only eBPF chat comparison
completed and both owned ranks exited. Both ranks' logits agreed within 1e-5;
63/64 teacher-forced top IDs matched the single-host reference. The pair and
reference generated 64-token introductions with a 55-token common prefix and
different coherent continuations. Details and limitations are in
`docs/validation/deepseek-tp2-control-2026-11-03/model-pair-numerics.md`.
The Maya-S copy passed SHA-256 and pack checks on both hosts. Sharded XS and S
typed worker feed/step/reset/shutdown tests passed: XS reproduced the previous
64-token pair sequence; S produced a coherent 64-token eBPF introduction.
All four owned ranks exited. Logs: `/tmp/ds4-tp2-control-{xs,s}-rank0.log` on
strix-halo-1 and the corresponding rank1 logs locally. The numerical execution
remains test-only; speculative rollback and DSpark gates are next.

P5 (#113): the sharded XS and S verify/commit tests passed acceptance counts 0–3
over a four-token window, with identical next logits to target-only continuations;
all owned ranks exited. Logs are `/tmp/ds4-tp2-113-rollback-{xs,s}-rank0.log` on
strix-halo-1 and corresponding rank1 logs locally. With a resident DSpark drafter
on rank 0 only, the XS and S pairs passed all four cases: next logits matched
exactly, and five next drafts and confidences matched the target-only path
(1e-4 confidence tolerance). All owned ranks exited. Logs are
`/tmp/ds4-tp2-113-dspark-{xs,s}-rank0.log` remotely and corresponding rank1 logs
locally. Test execution and identities are not production; actual acceptance
decisions are next.

Real greedy DSpark acceptance passed on XS and S: XS offered 106 drafts and
accepted 41 across 22 rounds; S offered 113 and accepted 41 across 23 rounds.
Each emitted the same 64 tokens as its target-only pair, and all owned ranks
exited. Logs: `/tmp/ds4-tp2-113-dspark-real-{xs,s}-rank0.log` remotely and
corresponding rank1 logs locally. These are isolated tests with synthetic
identity, not serving; sampled acceptance and the remaining gates are next.

The first sampled DSpark check was inconclusive because its chosen context had
only one meaningful chi-square group; both ranks exited. A revised XS smoke
selected a non-degenerate context using plain draws and passed: 1,000 target
versus 50 speculative first-token draws in sampled-q mode yielded p=0.8221;
point-q with penalties yielded p=0.4542. The first S point-q context observed
no accepts; a stronger test selected a context using independent target draws,
then required acceptance. On XS sampled-q and point-q-with-penalties yielded
p=0.8221 and p=0.7247 (11 and 71 accepted drafts); on S p=0.7755 and p=0.1919
(2 and 36 accepted). Each used 1,000 target and 50 speculative samples per
mode, and three fixed seed/counter replays matched. All owned ranks exited.
These are statistical smoke tests, not proofs of distributional equivalence.

P6 (#114): an isolated test exposed `control recv: Resource temporarily
unavailable` when the peer closed because `recv` returned zero with stale errno.
A narrow fix reports the peer-closed bootstrap cause on zero. Two-host control
EOF with sticky Worker cause, collective peer loss with sticky poison, queued
control coexistence and typed worker dispatch all passed; host socket-state and
`serve.test_tp2_peer_loss` passed too. All owned test processes exited. See
`docs/validation/deepseek-tp2-control-2026-11-03/control-eof.md`. Full DeepSeek
API pair-down behavior remains to be integrated and tested.

P3 (#111): `tools/ds4/tp2_identity.py` is a stdlib pack identity
helper under test. It hashes pack metadata and each indexed source, produces a
path-independent shared digest and checks local file identity before reuse.
Eleven synthetic unit tests pass. Independent full XS hashes produced the same
identity on both hosts (`0c4933e6b70f53f7eff163a4053f8970fce470cacc4065983ebe28bd507c57af`),
each cached check passed, and a deliberately modified source signature was
rejected. Independent full S hashes matched on both hosts
(`e8d41763b86d0ed5430202521f42c1089205d3bf14afc2e0308ca82e1d481c71`),
and cached checks passed. Both hash processes exited. See
`docs/validation/deepseek-tp2-control-2026-11-03/pack-identity.md`; local
manifests are in `/tmp/ds4-tp2-111-{xs,s}-local.manifest` and remote manifests
at the corresponding `-remote.manifest` paths. The cache assumes a trusted
filesystem and manifest. The helper is not yet wired to rank startup;
user-facing serving must not use synthetic test identities.

P3 controller integration is in progress: its host channel tests pass. A
test-only paired XS GPU run of ordinary feed, decode, reset and shutdown
matched the earlier 64-token pair sequence and explicitly rejected unsupported
checkpoint requests; both ranks exited. Logs are
`/tmp/ds4-tp2-111-controller-xs-rank0b.log` and remote
`/tmp/ds4-tp2-111-controller-xs-rank1b.log`. A previous pair was stopped by its
recorded PIDs because its 10.77 bootstrap address did not connect; the passing
retry used 192.168.1.92 for control and usb4_rdma5 for data.

Test-only DSpark model-wrapper validation on XS accepted 41 of 106 drafts and
matched the target-only pair's 64-token sequence. Both ranks exited; logs are
`/tmp/ds4-tp2-113-controller-xs-rank0.log` and remote
`/tmp/ds4-tp2-113-controller-xs-rank1.log`. The S run accepted 41 of 113
drafts and matched its target-only 64-token sequence; both ranks exited. Logs
are `/tmp/ds4-tp2-113-controller-s-rank0.log` and remote
`/tmp/ds4-tp2-113-controller-s-rank1.log`. The wrappers are not enabled for
production. The preflight helper now computes model, partition and binary
digests and returned matching S values independently on the two hosts. Real
startup and handshake integration remain open.

P3 actual-identity handshake test (uncommitted): both hosts prepared and
exchanged matching S content/partition/binary fingerprints, then rejected an
XS/S model mismatch and a separate executable mismatch on both sides before
model collectives. A full XS DSpark model run used the real preflight values,
accepted 41 of 106 drafts, matched its target-only 64-token sequence and shut
down both ranks. Logs `/tmp/ds4-tp2-111-real-dspark-xs-rank0.log` and remote
`/tmp/ds4-tp2-111-real-dspark-xs-rank1.log`. All owned test processes exited.
The test launcher binds the checked pack to the test arguments but is not a
production launcher; production startup is still guarded and unwired.

P3 review-binary linkage: the DeepSeek TP transport macro is shared between
the test and isolated `maya-ds4` targets. The first build failed because CMake de-duplicated
two `-include` switches. A grouped option corrected the generated C++ flags;
the isolated `maya-ds4` and companion test targets built, log
`/tmp/ds4-tp2-111-main-build2.log`. Single-host XS DSpark regression matched
the earlier 64-token SHA-256
`381baeb7bfea390e05fa6488507a4c43dcd5b845ffc556799aed32b44df67c89`
and accepted 33 of 37 drafts. The owned process exited; output and log are
`/tmp/ds4-tp2-111-single-host.tokens` and `/tmp/ds4-tp2-111-single-host.log`.

P6 (#114): the server retains the first `TP2_DOWN` or
poisoned-communicator cause and returns it on later requests before any new
GEN. The first multi-request test exposed a trailing newline; after trimming
the cause, all four `serve.test_tp2_peer_loss` tests pass, including an earlier
pair-down cause followed by a different error. See
`docs/validation/deepseek-tp2-control-2026-11-03/server-first-cause.md`.
An XS 262,144-context review pair is running sampled lookup, one-token
logprobs, and the P1 reference/prefill measurement harnesses. Local driver
session 72257, server PID/log `/tmp/ds4-tp2-116-xs-bench-server.{pid,log}`,
engine log `/tmp/ds4-tp2-116-xs-bench-engine.log`, remote worker PID/log
`/tmp/ds4-tp2-116-xs-bench-rank1.{pid,log}`. The run uses a generated
config and stops only its recorded processes.
Sampled XS lookup at temperature 0.2 kept lookup drafts for all three seeds;
the one-token logprobs request passed. The benchmark correctly stopped before
recording samples because the generated config had no model label. Both
owned processes exited. An optional explicit display label was added to the
config generator and server, without fabricating installer metadata; rerun
measurements with a fresh output file and verify the actual pack identity.
The fresh rerun is in local session 76046 with recorded server and worker
PIDs/logs under `/tmp/ds4-tp2-116-xs-bench-v2-*`; it uses the explicit
`DeepSeek V4.1 · Maya-XS` display label and checked pack manifest.
A read-only telemetry sampler is attached as local session 92077 and exits
when the owned server exits; its output is
`/tmp/ds4-tp2-116-xs-bench-v2-telemetry.jsonl`.
That run and sampler exited successfully. XS TP2 8K prefill was
201.0/199.9/203.5 tok/s versus earlier single-host 376.6/388.8/390.0;
short-prompt answers had different hashes in all nine comparisons. Raw data
is in `docs/benchmarks/deepseek-tp2-review-2026-11-03/`. The older baseline
used vision on; this is not an exact same-configuration comparison.
The full-window XS rollback test is running in local session 38214, remote
worker PID/log `/tmp/ds4-tp2-113-rollback-full-xs-rank1.{pid,log}`, local log
`/tmp/ds4-tp2-113-rollback-full-xs-rank0.log`. It uses 256-row prefill,
a 127-token prefix and forced acceptance counts 0–15.
The XS run passed every count with exact subsequent logits and matching five
DSpark draft IDs/confidences; both ranks exited. The same full-window test on
Maya-S runs as local session 9137, with remote owned worker PID/log
`/tmp/ds4-tp2-113-rollback-full-s-rank1.{pid,log}` and local log
`/tmp/ds4-tp2-113-rollback-full-s-rank0.log`.
The S sweep passed all 16 acceptance counts with exact next logits and
matching subsequent DSpark drafts; both ranks exited. Evidence is in
`docs/validation/deepseek-tp2-control-2026-11-03/sharded-rollback.md`.
The S 262,144-context sampled-lookup and three-run reference/prefill benchmark
is running under local session 36758; server PID/log and remote owned worker
PID/log use `/tmp/ds4-tp2-116-s-bench-*`. The benchmark saves answer text for
quality review and stops only its own server and worker.
Its read-only telemetry sampler is local session 52529 and exits with the
server; output `/tmp/ds4-tp2-116-s-bench-telemetry.jsonl`.
The S benchmark and sampler completed successfully. TP2 decode was 24.2–30.0
tok/s across nine prompts versus 2.6–7.7 in the earlier storage-pressured
single-host run; TP2 8K prefill was 187.7–190.9 versus 258.9–286.0 tok/s.
All matching answer hashes differed, and the saved 256-token answers were
truncated. Evidence and qualifications are in
`docs/benchmarks/deepseek-tp2-review-2026-11-03/README.md`.
During validation, the DeepSeek HIP Lt tuner may append to the pack's
`lt-picks.txt` despite `--profile-readonly`; the S files on both hosts have
mtime 2026-11-03 16:42 and size 2251 bytes. Preserve their contents rather
than guessing at a rollback. For future isolated review runs, copy any
existing picks to a new `/tmp` file per host and set `MAYA_DS4_LT_STORE` to
that file before launching; this isolates further tuning writes. The active
S benchmark was already launched using the pack defaults.
An unbuilt change makes DeepSeek TP2's Lt store read-only while still
reading existing picks; single-host DeepSeek and MiniMax retain their current
behavior. HIP mapping passes; a fresh gfx1151 engine build and actual
file-integrity check are required after the running benchmark.
The isolated rebuild is running under local session 85260, log
`/tmp/ds4-tp2-114-lt-store-build.log`; the earlier benchmark executable was
preserved as `/tmp/ds4-tp2-116-benchmark-engine`.
The engine and verification-test targets built. The same multi-target make
invocation initially reported no rule for the newly added Lt test after CMake
regenerated; building `ds4_lt_store_test` separately succeeded (log
`/tmp/ds4-tp2-114-lt-store-test-build.log`). Its GPU run awaits the GLM test.
An isolated GLM TP2/MTP HTTP cancellation and worker-loss regression is
running concurrently with that CPU build. Driver session 98949, server PID/log
`/tmp/ds4-tp2-114-glm-http-server.{pid,log}`, engine log
`/tmp/ds4-tp2-114-glm-http-engine.log`, and remote owned worker PID/log
`/tmp/ds4-tp2-114-glm-http-rank1.{pid,log}`. Its driver checks the recorded
worker executable before signaling that PID.
The GLM HTTP regression passed: real MTP acceptance, cancellation during
prefill and a successful follow-up, then owned-worker loss during a separate
prefill with the original poisoned-communicator cause retained in health and
later requests. Both owned processes exited. Evidence is in
`docs/validation/deepseek-tp2-control-2026-11-03/glm-regression.md`.
The standalone gfx1151 HIP Lt test passed both read-only and writable stores
with real GEMM candidates (log `/tmp/ds4-tp2-114-lt-store-test.log`); the
two-rank model-level file-integrity check remains next.
The two-rank XS check is running under local session 70451, remote owned
worker PID/log `/tmp/ds4-tp2-114-lt-readonly-rank1.{pid,log}` and local log
`/tmp/ds4-tp2-114-lt-readonly-rank0.log`. Each rank has a newly created empty
`/tmp/ds4-tp2-114-lt-rank{0,1}-picks.txt` store and verbose Lt logging; verify
that real tuning happened and neither store changed when it exits.
That check passed: each rank tuned 11 real hipBLASLt shapes while its empty
store remained zero bytes, and all 16 rollback/DSpark checks passed again.
Both owned ranks exited. The standalone test also confirmed writable mode
still appends. The rebuilt engine SHA-256 is
`31b2e21d529f5de3016c7ce58a3bab704f052ab5d22beea62b64829680a81fb1`.
The rebuilt engine is now running an S TP2 benchmark with DSpark and lookup
off at configured context 262144, using the same prompt harness. Local driver
session 31639, server and remote owned worker PID/logs under
`/tmp/ds4-tp2-116-s-off-*`. This also exercises the rebuilt launcher through
HTTP; the benchmark will stop its own processes.
Its read-only telemetry sampler is local session 81602 and exits with the
server; output `/tmp/ds4-tp2-116-s-off-telemetry.jsonl`.
The S speculation-off benchmark completed: 17.6–19.9 tok/s decode and
190.5–192.7 tok/s 8K prefill; both ranks and sampler exited. A matching
DSpark-on/lookup-off S run on the same binary is active under local driver
session 97608, with server/worker PID/logs under `/tmp/ds4-tp2-116-s-on-*`.
Its read-only telemetry sampler is local session 29015; it exits with the
server and writes `/tmp/ds4-tp2-116-s-on-telemetry.jsonl`.
Server and remote worker start-time ticks are recorded in the matching
`*-server.start` and `*-rank1.start` files.
The same-binary S comparison completed: lookup off in both cases, all nine
answers matched, DSpark on decode 21.8–30.4 versus off 17.6–19.9 tok/s, and
8K prefill about 191–193 tok/s in both. All six prefill prompt hashes matched.
Both pairs and samplers exited; raw evidence and limitations are in the
review benchmark report.
The equivalent XS TP2 speculation-off run is active under local driver
session 45477, with server and remote owned worker PID/logs under
`/tmp/ds4-tp2-116-xs-off-*`. Read-only telemetry sampler session 34861 exits
with the server. A same-binary DSpark-on/lookup-off run follows.
Checkpoint `a34d616` records the current experimental integration and staged
validation; #111 was subsequently closed. Do not treat
the checkpoint as final validation or a production promotion.
The XS speculation-off run completed (17.1–23.4 tok/s decode, 200.0–204.0
tok/s 8K prefill); both owned ranks and sampler exited. Its same-binary
DSpark-on/lookup-off counterpart is active in local driver session 78161,
with server/worker PID/logs under `/tmp/ds4-tp2-116-xs-on-*`; read-only
telemetry sampler session 26070 exits with that server.
The XS DSpark-on run completed: all nine answers matched off, decode was
24.6–35.7 versus 17.1–23.4 tok/s, and 8K prefill was about 199–204 tok/s in
both modes. All six prefill prompt hashes matched. Both pairs and samplers
exited; raw results are in the review benchmark report.
An isolated same-build XS single-host run with DSpark on and lookup off is
active under local session 49294; server PID/log and output use
`/tmp/ds4-tp2-116-xs-single-on-*`. Its Lt picks were copied to a separate
`/tmp` store so single-host tuning cannot append to the model pack. The
controlled settings disable flow and vision and use 256-row prefill; they
are not a claim about the best single-host settings.
Read-only telemetry sampler session 50360 follows that owned server and exits
with it, writing `/tmp/ds4-tp2-116-xs-single-on-telemetry.jsonl`.
The XS single-host DSpark-on run completed: one sample per prompt at
38.0/28.9/45.6 tok/s decode and 174.8/181.7 tok/s 2K/8K prefill. All three
answers differed from the TP2 run; their openings remain coherent. Its server
and sampler exited. The single-host DSpark-off counterpart is active under
local session 54726, with PID/logs under `/tmp/ds4-tp2-116-xs-single-off-*`
and a separate temporary Lt store.
Read-only sampler session 66813 follows that server and exits with it.
The XS single-host off run completed: all three answers matched its on run,
with 20.3 tok/s on each prompt off versus 38.0/28.9/45.6 on; 8K prefill was
182.0 off versus 181.7 on. Its owned server and sampler exited. An isolated
same-build S single-host DSpark-on run is active under local session 48370,
with PID/logs under `/tmp/ds4-tp2-116-s-single-on-*` and a separate temporary
Lt store. Read-only sampler session 80549 exits with the server.
The S single-host load was stopped via its recorded server PID while starting:
`MemAvailable` fell to about 127 MB and swap was in use. The server, its
engine and sampler exited; available memory subsequently recovered to about
120 GiB. No benchmark samples were produced. Do not repeat this exact
configuration; find and validate a safe memory/readahead setting first.
The retry is active under local session 31888 with an isolated config,
`MAYA_DS4_WARM_CAP_GB=0`, 8192 MiB reserve and a 3 GiB MemAvailable guard
that targets only its own recorded server and engine. PID/logs use
`/tmp/ds4-tp2-116-s-safe-single-on-*`. Its safety-constrained settings differ
from the TP2 runs and must be disclosed in any comparison.
Read-only sampler session 85845 follows the retry and exits with its server.
The guarded retry completed its three decode requests and the 2K prefill;
available memory remains about 6 GiB. Its flow-disabled 2K prefill took about
194 seconds, so the 8K request can take many minutes. This constrained run
is not representative of optimized/default single-host prefill.
The independent single-host unified-memory/read-ahead problem is tracked in
[#117](https://github.com/uv-genai/maya-tp/issues/117); its proposed cause
remains a hypothesis, and no production single-host fix has been made.
The guarded S DSpark-on run finished: decode 15.7/8.9/10.9 tok/s, 2K/8K
prefill 10.6/10.1 tok/s under its flow-disabled, zero-warm-up settings; the
server and sampler exited and available memory recovered. The short
DSpark-off control is active as local driver session 40466 and sampler 65489,
with PID/logs under `/tmp/ds4-tp2-116-s-safe-single-off-*`. Prefill is not
repeated in this control.
The off control completed and exited. All three on/off answers differed
across separate processes; expert placement differed because the off process
did not load the drafter. A same-process comparison is needed before drawing
conclusions about speculative decoding. The same-process check subsequently
passed both 19- and 545-token prompts, matching 64 greedy tokens each with
38/47 and 30/35 DSpark drafts accepted; its engine exited. Evidence is in
`/tmp/ds4-tp2-113-s-single-spec-parity-*`.
Ordered HIP/server/tools CI passed: HIP mappings, 614 server tests (11 skipped)
and 698 tools tests (3 skipped). Logs use prefix `/tmp/ds4-tp2-114-ci-`.
A real XS model-loss test stopped its owned worker
during generation; rank 0 preserved `verbs peer closed the bootstrap
connection` and rejected subsequent step/feed with the same cause. Both ranks
and their monitor exited. Logs are `/tmp/ds4-tp2-114-model-loss-rank0.log`
and the corresponding remote rank1 log.
Other DeepSeek failure timings and GLM GPU regression remain open.

Controller integration: attach/shutdown methods have been renamed
for eventual engine use, a one-shot failure callback and explicit post-shutdown
guard have been added, and the real loss test now checks callback count/cause.
The isolated test and engine build passed, log
`/tmp/ds4-tp2-111-api-build.log`. Paired XS GPU validation matched the earlier
64 tokens, cleanly shut down, rejected a post-shutdown step and invoked no
failure callback. Both ranks exited; logs use prefix
`/tmp/ds4-tp2-111-callback-normal-`. The live loss callback test stopped
the owned worker during generation. The callback was invoked exactly once
with the original cause; the rank, worker and monitor all exited. Logs use
prefix `/tmp/ds4-tp2-114-callback-`.
The experimental `tools/ds4/tp2_rank.py` prepares pack/partition/
binary identity and rank-specific arguments, with 14 focused launcher/identity
tests passing. Experimental rank startup, worker dispatch and serving hooks
passed an isolated XS protocol test: two requests each matched the 64-token
target-only pair and accepted 30/37 drafts, then both ranks shut down cleanly.
Logs use prefix `/tmp/ds4-tp2-111-serve-`; all owned test processes exited.
Do not use these for user serving until all remaining gates are complete.

P6 idle liveness: `Communicator::CheckPeer` polls without consuming
control data and poisons on peer close. Host socket-state and controller tests,
real two-host idle peer loss, and an RDMA/control coexistence test with a queued
frame all passed. The experimental server has a one-shot idle watcher;
its isolated build passed, log `/tmp/ds4-tp2-111-startup-build2.log`, and
its idle HTTP failure check is described below.

The first owned XS HTTP test completed a 64-token API request and shut down
cleanly, but its idle-loss check did not run because the driver looked for the
remote PID file locally. The corrected retry completed 64 tokens (30/36 drafts
accepted), stopped its recorded idle worker and observed degraded/down health
without another request. A later request was rejected with the original
peer-closed cause. The server, engine and worker exited. Logs use prefix
`/tmp/ds4-tp2-114-http-`.

P4 (#110) has uncommitted review-only chunk execution changes: deterministic
bounded feed, route checks and attention/MoE reductions. The isolated test
binary built and host worker/row-bound tests pass. The first paired XS test
failed before logits because `forward_chunks` still had a load-only guard;
both ranks exited. Logs use prefix `/tmp/ds4-tp2-110-chunk-xs-`. The guard was
narrowed to the explicitly enabled, bound review path and the paired test
passed: both ranks' full logits agreed within 1e-5 over 64 rows, their greedy
continuation matched the earlier window pair, and 63/64 reference top IDs
matched single-host (max logit difference 4.64711, RMS 0.404073). Both ranks
exited; logs prefix `/tmp/ds4-tp2-110-chunk-xs-v2-`. It used synthetic
handshake identities. See
`docs/validation/deepseek-tp2-control-2026-11-03/chunk-review.md`. This path
is not enabled by the launcher. A controller test with actual preflight
identities matched the 64-token window pair, deferred a cancellation callback
during the chunk, reset/replayed and acknowledged shutdown. Both ranks exited;
logs prefix `/tmp/ds4-tp2-110-control-xs-`. HTTP cancellation remains untested.
A DSpark-after-chunk XS test accepted 41/106 drafts and matched the target-only
64-token pair; both ranks shut down. Logs prefix `/tmp/ds4-tp2-110-dspark-xs-`.
The paired XS sweep at 17, 128, 129 and 256 tokens passed: the ranks' final-row
logits matched exactly and chunk/window top IDs matched at each length; max
absolute chunk/window logit differences were 2.5441, 7.7584, 3.6343 and
3.2556. The subsequent 64-token reference test passed. Both ranks exited;
logs prefix `/tmp/ds4-tp2-110-sweep-xs-`. This does not prove long-context or
quality equivalence.
An extended two-feed XS sweep passed through 257 tokens (rank logits matched,
chunk/window top ID 1192, max logit difference 5.7510), but **failed at 512**:
the ranks agreed with each other while chunk prefill chose EOS ID 1 and window
prefill chose ID 9857, max logit difference 7.4473. Both ranks exited. Logs
prefix `/tmp/ds4-tp2-110-multi-xs-`. Chunking stays disabled in the launcher;
trace the first divergence and do not weaken the test to claim parity.
An instrumented 512-token XS pair reproduced the failure and showed that a
whole two-chunk feed and two separate feeds were identical at every traced
layer and in all final logits (top ID 1). Both ranks exited. Logs prefix
`/tmp/ds4-tp2-110-trace-xs-`; position 511 residuals and analysis are in
`/tmp/ds4-tp2-110-trace-pos511.bin` and
`/tmp/ds4-tp2-110-trace-analysis.txt`. The trace did not include matching
attention-stage records for the chunk path; instrumentation is being extended
to locate the first window/chunk divergence. The owned one-host diagnostic
exited successfully and found the existing single-host window and chunk paths
also differ on this repeated-special-token fixture: window top 19851, chunk
EOS 1, max logit difference 8.5022. Log
`/tmp/ds4-tp2-110-single-compare.log`. This is not a correctness result for a
realistic long prompt. Rerun the matching-attention pair and test real text.
The matching-stage rerun has now exited: the window/chunk difference is
already 0.01365 maximum after layer 0 attention and grows to 5.116 after
layer 39; the two chunk-call shapes remain identical at every traced stage.
Analysis: `/tmp/ds4-tp2-110-attention-analysis.txt`. An owned single-host
545-token real eBPF prompt comparison exited successfully: one-host window
and chunk selected top token 5, max logit delta 3.6831. Log
`/tmp/ds4-tp2-110-real-single.log`. The paired comparison used PIDs
in `/tmp/ds4-tp2-110-real-rank0.pid` and the same path on strix-halo-1; logs
use `/tmp/ds4-tp2-110-real-rank{0,1}.log` on their respective hosts. It used
synthetic test identities, not the serving launcher. Both ranks exited cleanly
and matched all final logits exactly; chunk and window selected token 5 (`#`),
max window/chunk logit delta 4.3801. The single-host repeat exited; same-path
single/pair comparisons gave window max/RMS 3.9863/0.8676 (top-10 overlap
10/10), chunk 3.9327/0.7978 (8/10). Full logits and analysis use
`/tmp/ds4-tp2-110-real-{single,pair}.logits` and
`/tmp/ds4-tp2-110-real-logits-analysis.txt`. The P4 gate remains open.
An XS DSpark rollback test at position 127 with independently checked
identities passed forced acceptance 0..3 across the 128-token boundary: each
next full logit vector was identical to target-only, and all five drafts and
confidences matched. Both ranks shut down; logs use
`/tmp/ds4-tp2-113-boundary-rank{0,1}.log` on their respective hosts. This is
supplemental evidence, not a P5 gate pass; the harness and report are in
`4ae9925` (#113). Layer 0 has no Engram input.
Test-only tracing now records residuals before and after Engram as well as
after attention; it compiled and linked. The paired 512-token trace has exited
with the same retained gate failure. At layer 0 the window and chunk inputs
and post-Engram values were identical; their post-attention maximum difference
was 0.0136465. Analysis: `/tmp/ds4-tp2-110-inputs-analysis.txt`. The owned
single-host 545-token chunk-reference continuation completed 64 tokens;
`/tmp/ds4-tp2-110-real-reference.bin` contains its tokens and logits. The
paired continuation comparison exited cleanly: rank logits agreed within
1e-5, 59/64 teacher-forced top IDs matched the single-host reference (max
delta 7.9736, RMS 0.4767), and independently generated greedy outputs shared
12 initial tokens before diverging into two coherent tutorial openings. Logs:
`/tmp/ds4-tp2-110-cont-rank{0,1}.log` on their respective hosts. This is one
prompt, not a quality or exact-equivalence guarantee. A further test-only
layer-0 attention-input/projection probe compiled and linked in
`/tmp/maya-ds4-tp2-p1-build` (log
`/tmp/ds4-tp2-110-attention-probe-build.log`). The paired 512-token run
completed with the retained synthetic gate failure. The normalized input
matched exactly, while low-rank query and key/value projections differed by at
most 0.0005835 and 0.0005542; analysis is
`/tmp/ds4-tp2-110-math-analysis.txt`. A review-only option to use the window
projection math for the chunk built (log
`/tmp/ds4-tp2-110-matched-projection-build.log`). Its paired test exited 0:
layer-0 query and KV projections became bit-identical and at 512 tokens both
chunk and window selected 9857; the ranks' logits were identical, although
chunk/window max logit difference remained 6.8618. This is an experiment, not
a production fix. Logs: `/tmp/ds4-tp2-110-projection-rank{0,1}.log` on their
respective hosts; trace `/tmp/ds4-tp2-110-trace-projection.bin`. Extend the
sweep and test realistic prompts, DSpark, Maya-S and performance. The expanded
XS matched-projection sweep passed top IDs at 17, 128, 129, 256, 257 and 512;
rank logits matched exactly. Its short 64-row follow-up matched 63 single-host
top IDs. Both ranks exited; logs are
`/tmp/ds4-tp2-110-match-sweep-rank{0,1}.log` on their respective hosts.
The text harness now includes per-mode timing, and the probe's disabled path
avoids an environment lookup in single-host execution. Those changes are being
linked in the isolated test binary. The baseline 545-token timing pair exited
cleanly: lead-rank window 9.1835 s, chunk 2.9061 s, one sample including the
harness's synchronization and final logit exchange. Logs:
`/tmp/ds4-tp2-110-timing-base-rank{0,1}.log` on their respective hosts. The
same fixture with matched projections also exited cleanly: window 6.7692 s,
chunk 3.2084 s, pair logits identical and chunk/window both top 5. Logs:
`/tmp/ds4-tp2-110-timing-match-rank{0,1}.log`. These are single samples; the
window timings vary substantially, so they do not establish a reliable
performance difference. The matched mode still needs long-continuation,
DSpark, S, cancellation, failure and context tests.
Its 545-token paired continuation test exited cleanly: rank logits matched,
61/64 teacher-forced top IDs matched single-host (max/RMS 4.9847/0.4413), and
greedy continuations shared 21 initial tokens and then diverged coherently.
Logs: `/tmp/ds4-tp2-110-match-cont-rank{0,1}.log`. A DSpark pair with real
identities failed against its generated target-only sequence at output index
19, after six rounds and 13 accepted drafts; the worker observed peer close
after the lead test exited. Both ranks exited. Logs:
`/tmp/ds4-tp2-113-long-rank{0,1}.log`. A rebuilt diagnostic is running to
compare a fixed probe's full logits before and after resetting and replaying
the committed history on mismatch. It reproduced the mismatch (actual 2572,
expected 24353); after reset and replay, the next token was again 2572, and a
fixed probe's full logits matched the speculative state exactly (max/RMS 0).
Both ranks exited, with synchronized shutdown for the diagnostic. Logs:
`/tmp/ds4-tp2-113-replay-rank{0,1}.log`. This rules out a committed-state
difference for that probe only; the two target-only execution setups still
differ. The DSpark-loaded pair without any drafts generated 64 tokens and
exited cleanly: it shared 19 tokens with the old direct-feed reference and
chose the same 2572 at position 19 as the speculative run. Code inspection
found the likely test mismatch: a whole 545-token feed is balanced into
182+182+181 rows by `plan_chunks`, whereas the controller sends 256+256+33.
An independent controller pair without a loaded drafter generated exactly the
same 64 tokens as the drafter-loaded, no-draft pair. Both differ from the
direct-feed reference at position 19. The feed shapes differ, and the direct
harness also ran a teacher-forced phase before greedy generation, so the
specific cause is isolated below. The corrected-reference DSpark run passed:
it generated the same 64 target-only tokens, accepting 41/111 drafts over 23
rounds, and both ranks shut down. Logs:
`/tmp/ds4-tp2-113-corrected-rank{0,1}.log`; reference:
`/tmp/ds4-tp2-113-control-rank0.tokens`. Retain the original failed comparison
and numerical difference in the evidence. A test-only greedy-only whole/split
run exited successfully: whole vs split max logit delta 4.2774 (top token 5
in both); the whole-feed continuation matched the prior direct-feed result and
differed from the controlled split at token 19. Logs:
`/tmp/ds4-tp2-110-shape-rank{0,1}.log`. A review-only change makes TP chunk
planning use fixed-size boundaries like controller feeds; it built (log
`/tmp/ds4-tp2-110-fixed-chunks-build.log`). The 545-token test passed with
identical full logits for whole and split feeds and identical 64-token greedy
continuations; both ranks exited. Logs:
`/tmp/ds4-tp2-110-fixed-rank{0,1}.log`. Test DSpark, both quantizations and
the remaining gates before enabling it. The changed-binary XS DSpark test
passed against the controlled 545-token reference (64 matching tokens,
41/111 accepted); both ranks shut down. Logs
`/tmp/ds4-tp2-113-fixed-rank{0,1}.log` on their respective hosts. A Maya-S
545-token whole/split test passed: whole/split and pair full-logit deltas were
0, and 64 tokens were generated. Each rank loaded 84,509,982,720 expert
bytes; logs `/tmp/ds4-tp2-110-s-shape-rank{0,1}.log`. The real-identity S
controller test ran against that continuation, with PIDs/logs
`/tmp/ds4-tp2-110-s-control-rank{0,1}.{pid,log}` on their respective hosts.
That real-identity controller test passed all 64 tokens, deferred cancellation,
reset/replayed and shut down on both ranks. The real-identity S DSpark test
against the same continuation also passed: all 64 tokens matched, 40/116
drafts accepted over 24 rounds, and both ranks shut down. Logs
`/tmp/ds4-tp2-113-s-fixed-rank{0,1}.log` on their respective hosts. A
real-identity S controller test at configured context 524288 passed all 64
tokens and coordinated shutdown, matching its 32K-configured reference. Logs:
`/tmp/ds4-tp2-112-s-512k-rank{0,1}.log`. The corresponding DSpark test also
passed the same 64 tokens, accepting 40/116 drafts, and shut both ranks down;
logs `/tmp/ds4-tp2-112-s-512k-dspark-rank{0,1}.log`. During loading, after
the lead had loaded, system `free` reported 100 GiB used and 23 GiB available
on the lead; the worker was still loading, with 92 GiB used and 31 GiB
available. These are host snapshots, not isolated process or peak figures.
The prompt is only 545 tokens, so this checks allocation/startup, not a full
512K-token context.

The ordered HIP mapping/server/tools checkpoint passed: HIP mapping, 614
server tests (11 skips), and 701 tools tests (3 skips).
Logs `/tmp/ds4-tp2-checkpoint-{hip,serve,tools}.log`. An XS 3,204-token
real-format prompt passed: rank and whole/split full-logit deltas 0, window and
chunk top token 5, and 64 generated tokens; both ranks exited. Logs
`/tmp/ds4-tp2-110-3204-rank{0,1}.log`. This was a correctness run while CPU
tests were active, not a performance measurement. Fixture:
`tests/fixtures/ds4-ebpf-3204.tokens`. A real-identity XS DSpark comparison
against its continuation passed all 64 tokens, accepted 38/124 drafts over
25 rounds and shut down both ranks. Logs
`/tmp/ds4-tp2-113-3204-rank{0,1}.log`. The isolated GLM `maya` target built
and was copied to `/tmp/ds4-tp2-glm-review` on strix-halo-1; hashes match
`467170ceb00479d390ed6b12e11a8c15804fcc4abc7e0564c1be7f8bbc1c7a0c`.
An ownership-safe GLM TP2 regression passed: both ranks generated the same
32 tokens with MTP off and on, and the two modes matched each other. MTP
accepted 20/21 drafts. All owned ranks exited; report
`docs/validation/deepseek-tp2-control-2026-11-03/glm-regression.md`.
This short CLI test does not cover serve, cancellation or peer loss; avoid
the destructive repository harnesses for remaining tests.

Checkpoint `3f3c983` records the gated DeepSeek chunk work, evidence, and
GLM CLI regression. The launcher remains window-only. A test-only marker and
prefill-loss assertion built (log `/tmp/ds4-tp2-114-prefill-loss-build.log`).
The paired injection test passed: after the monitor saw a 256-row reduction
marker, it terminated only the recorded worker after verifying its isolated
executable. The lead retained `verbs peer closed the bootstrap connection`,
reported the failure once, and rejected subsequent feed and step with the same
cause. Both test and monitor exited. Logs
`/tmp/ds4-tp2-114-prefill-rank{0,1}.log` and
`/tmp/ds4-tp2-114-prefill-monitor.log`. The marker and assertion remain
uncommitted test-only changes; other failure points still need testing.
The prefill loss test and report were committed as `3df544a`. A new test-only
speculative failure harness is running: its own worker stops itself on the
first verify command, then an owned remote monitor verifies the executable and
kills that stopped PID. Lead/worker logs and PIDs use
`/tmp/ds4-tp2-114-verify-rank{0,1}.{pid,log}`; monitor PID/log on strix-halo-1
uses `/tmp/ds4-tp2-114-verify-monitor.{pid,log}`. The verify injection passed:
the first peer-close cause was retained and later feed/step calls rejected;
the lead and monitor exited. The verdict-boundary injection also passed with
the same sticky cause and rejection of later operations; logs
`/tmp/ds4-tp2-114-verdict-rank{0,1}.log` and remote monitor
`/tmp/ds4-tp2-114-verdict-monitor.log`. The rollback injection also passed:
rank 1 stopped after queuing ring restoration and before compressor restoration,
then the owned monitor verified and killed it. The lead retained the first
peer-close cause and rejected later feed/step calls. Logs
`/tmp/ds4-tp2-114-rollback-rank{0,1}.log` and remote monitor
`/tmp/ds4-tp2-114-rollback-monitor.log`. The stop hooks are now guarded by
`STRATA_DS4_TP2_TESTING`; test and isolated engine builds passed (log
`/tmp/ds4-tp2-114-test-guard-build.log`), and their strings are absent from
the regular `maya-ds4` binary. The guarded rollback injection passed again:
first peer-close cause retained, later operations rejected, all owned processes
exited. Logs `/tmp/ds4-tp2-114-guard-rollback-rank{0,1}.log` and remote
monitor `/tmp/ds4-tp2-114-guard-rollback-monitor.log`. HTTP cancellation and
remaining operational gates stay open.

Checkpoint `4b58729` records the guarded verify, verdict and rollback fault
tests. Chunk-prefill integration has begun: the internal option is renamed
`tp2_chunk_enabled`, the validated window-equivalent input projections are
always used in that mode, and temporary numerical tracing was removed. The
isolated test build passed (log `/tmp/ds4-tp2-110-deterministic-build.log`).
The XS 545-token whole/split check passed without a projection test knob:
full-logit delta 0 and the 64-token continuation matched its controlled
reference. Logs `/tmp/ds4-tp2-110-deterministic-rank{0,1}.log`. The same-mode
real-identity DSpark check passed 64 tokens with 41/111 drafts accepted; logs
`/tmp/ds4-tp2-113-deterministic-rank{0,1}.log`. Integration is in progress:
the launcher now accepts explicit 0 or 256 rows, binds the requested size,
and rejects conflicting flow/YOCO settings; its focused Python tests passed.
Engine preflight validates the requested size, includes it in peer feature
identity, and bounds the worker control path. Host preflight/identity tests
and the isolated `maya-ds4` build passed; CLI rejected a mismatched launcher
and engine chunk setting before loading. An isolated HTTP chunk/cancellation
test is running in session 50921, with driver PID/log
`/tmp/ds4-tp2-114-chunk-http-driver.{pid,log}`, server PID/log
`/tmp/ds4-tp2-114-chunk-http-server.{pid,log}`, engine log
`/tmp/ds4-tp2-114-chunk-http-engine.log`, and remote worker PID/log
`/tmp/ds4-tp2-114-chunk-http-rank1.{pid,log}`. The test passed: a 545-token
request generated 64 tokens (25/29 DSpark drafts accepted), a 3,204-token
streaming request was cancelled during prefill, and a follow-up completed with
the pair healthy. The driver stopped its server before its verified worker;
all owned processes exited. This is one smoke test, not final performance or
all API/failure gates. Do not deploy these changes.

The isolated serving integration now advertises `tp_prefill_chunk` in engine
INFO and `/health.tp.prefill_chunk`, and corrects the misleading rank
"load-only" startup message. Focused launcher, peer-loss/health, preflight,
identity and HIP mapping tests passed. Test and engine rebuilds passed (log
`/tmp/ds4-tp2-110-health-build.log`). A real paired mismatch test is running:
rank 0 requests 256 rows, rank 1 requests window-only. Both rejected before
inference with `context or requested features mismatch` and exited; logs
`/tmp/ds4-tp2-110-mismatch-rank{0,1}.log`. The rebuilt-engine HTTP
health/cancellation test passed with `/health.tp.prefill_chunk=256`, the
545-token request, cancellation during the 3,204-token prefill, and a healthy
follow-up. The driver stopped its server and verified worker; all owned
processes exited. Logs `/tmp/ds4-tp2-114-health-http-*`. The same HTTP check
also passed on Maya-S: 545-token request, 29/39 DSpark drafts accepted,
3,204-token prefill cancellation, and a healthy follow-up; all owned processes
exited. Logs `/tmp/ds4-tp2-114-s-http-*`. Experimental prompt lookup is now
being wired: launcher choices are 0 or 3, only rank 0 owns lookup, its value
is bound in peer identity, and engine INFO reports actual lookup to metrics.
Focused launcher/dashboard/preflight/identity tests passed. The engine/test
build passed (log `/tmp/ds4-tp2-113-lookup-build.log`), and a mismatched local
lookup preflight was rejected before loading. A real-identity lookup-only HTTP
pair passed: lookup-off/on responses matched exactly for one repetitive XS
prompt, and two internal lookup-on engine continuations accepted 90/92 and
81/81 drafts. Health reported 256 and metrics reported Lookup without DSpark.
All owned processes exited. Evidence:
`docs/validation/deepseek-tp2-control-2026-11-03/lookup-http.md`. Lookup with
DSpark, sampling, S, failures and broader quality remain open; the option is
experimental and defaults off. An XS HTTP test with lookup and DSpark enabled
also passed: lookup-off/on responses matched, and one internal lookup-on
continuation kept 82/92 lookup drafts and 15/15 DSpark drafts. Metrics
reported both modes, and all owned processes exited. Logs use
`/tmp/ds4-tp2-113-combined-http-*`. The equivalent Maya-S combined test is
running with driver/server/worker PIDs and logs under
`/tmp/ds4-tp2-113-s-lookup-http-*`. The Maya-S combined test passed as well:
lookup-off/on responses matched; one internal continuation kept 82/92 lookup
drafts and 15/15 DSpark drafts. Sampling and failure tests remain open.
An XS real-identity startup with mismatched lookup settings (3 on rank 0,
0 on rank 1) was rejected by both ranks before inference; both exited. Logs
`/tmp/ds4-tp2-113-lookup-mismatch-rank{0,1}.log`.
An XS sampled DSpark test with chunked prefill and checked identities is
running under local session 23767, with logs
`/tmp/ds4-tp2-113-sample-chunk-xs-rank{0,1}.log` and remote owned worker PID
`/tmp/ds4-tp2-113-sample-chunk-xs-rank1.pid`.
That run passed: 1,000 plain vs 50 speculative draws after a 19-token chunked
prefill gave p=0.612 for sampled proposals (35/250 accepted) and p=0.527 for
point proposals with penalties (8/250 accepted); seed replay also matched.
Both owned ranks exited. The equivalent Maya-S run is active under local
session 96004 and remote PID `/tmp/ds4-tp2-113-sample-chunk-s-rank1.pid`,
with logs `/tmp/ds4-tp2-113-sample-chunk-s-rank{0,1}.log`.
The Maya-S run passed the same smoke: sampled-q p=0.0129 (26/250 accepted)
and point-q with penalties p=0.365 (15/250 accepted), with seed replay.
These small samples cannot rule out subtle distribution errors. Both owned
ranks exited.
`tools/ds4/tp2_config.py` now prints a dedicated rank-0 API config using
explicit pack, manifest, executable and feature choices. Its focused tests
passed; a generated config will be exercised in the next real HTTP loss test.
The preliminary review procedure is in `docs/DEEPSEEK-TP2-RUN.md`. Server
health now exposes the first sticky TP failure as `tp.error`; the peer-loss
unit suite passed. These remain uncommitted experimental changes.
An isolated XS HTTP active peer-loss test is running with the generated
config, local driver session 44866, server PID/log
`/tmp/ds4-tp2-114-active-loss-server.{pid,log}` and remote owned worker
`/tmp/ds4-tp2-114-active-loss-rank1.{pid,log}`. Its driver verifies the
worker executable before targeting its recorded PID during prefill.
It killed the owned worker at 256/3,204 prefill tokens. Health went degraded
with the original peer-close cause and a later request was rejected. The
first response carried the raw cause without pair-down context, so server
error handling was tightened and a fifth peer-loss unit test passed. The
repeat runs in local session 45718, with server and worker PID/logs under
`/tmp/ds4-tp2-114-active-loss-v2-*`; the first pair has exited.
The repeat passed: the first HTTP error, degraded health and later error all
retained the peer-close cause with pair-down context. Its server and worker
exited. Evidence is in
`docs/validation/deepseek-tp2-control-2026-11-03/server-first-cause.md`.


## 43. DeepSeek TP2 launchers ready for user testing (#118, 2026-11-03)

The user requested the GLM-style rank scripts and a port-based llama-swap
wrapper, and will perform inference testing themselves. The new entry points
are `tools/ds4/rank0.sh`, `tools/ds4/rank1.sh` and
`tools/llama-swap/tp2-ds4.sh`, backed by `tools/ds4/tp2_launch.py`.
Per-model launcher configs select the pack, checked local manifest and engine.
Common context and feature choices are forwarded to both ranks. The pair
supervisor checks peer identities before model loading and records its process
ownership. `stop PORT` and llama-swap's `stop-pid ${PID}` request shutdown
through the owned control socket. API shutdown precedes cleanup of verified
child engines and the remote worker. Failed ranks require a fresh paired run.

An isolated runtime is staged on both hosts at
`/home/ugo/software/project-maya-ds4-tp2-review`. It has matching reviewed
engine SHA-256 `a0dcc1cb91e9e52fa365291d68eef8203af714d8222e77e49d72741a4331ddc2`,
per-host XS/S manifests and `maya-ds4-tp2-maya-{xs,s}.json` configs.
Rank 0 is strix-halo-1, rank 1 is strix-halo-2, bootstrap port 18525,
RDMA device usb4_rdma5, context 262144, DSpark on, prefill chunk 256,
lookup off and expert budget 92160 MiB. The runtime reuses each host's
existing virtualenv. The selected context matches the existing single-host
DeepSeek configs. User examples and a ready YAML fragment are staged in
`llama-swap-deepseek-tp2.yaml` under that runtime.

Fourteen process/config tests pass in both staged runtimes, covering duplicate
starts, mismatched identities, child exec transitions and PID reuse,
server/worker failure, restart, cleanup ordering and unrelated-process
preservation. Both hosts' pack, partition and executable preflights agreed.
Ordered repository checks passed HIP mapping, 618 server tests (11 skips)
and 721 tools tests (3 skips). Logs are recorded in #118 and
`/tmp/ds4-tp2-launch-ci-{hip,serve,tools}.log`.

The user can run on strix-halo-1 with
`/home/ugo/software/project-maya-ds4-tp2-review/tools/llama-swap/tp2-ds4.sh run 18080 --model Maya-XS`,
or `--model Maya-S`. Manual rank scripts are available in that runtime's
`tools/ds4/`. The llama-swap stop command uses `stop-pid ${PID}` because its
stop-command macro is the supervisor PID. Runtime logs are under
`~/.cache/maya/deepseek-tp2/PORT/run-*`, with the `latest` file identifying
the selected run directory. The operations guide documents standalone and
managed launches, config fields and overrides.

**Active owned model processes** are none. Validation for this workstream
uses fake processes and preflight checks. The user performs the next actual
model launch and inference tests.
