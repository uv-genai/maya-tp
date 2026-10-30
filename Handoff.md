# Handoff — Maya TP2 (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-30 (session 5, #15 MTP deployment + llama-swap launchers) · v1.1.37 · branch `main` on `uv-genai/maya-tp` · engine fix **`5b547ce`**; prior WIP preserved in `74a47d8`.
> **Both GPUs are free** (no engines or servers running). #15 deployment/launcher verification complete; see §14. Both deployed rank-0 paths and local binary match MD5 **`75a180099201dbc0abdf29808a536022`**. All #13 exit checks pass; MTP ships opt-in (§11).

## 1. What we are building

Two-way **tensor parallel (TP2)** for **GLM-5.3-Flash** (measured on **Maya-S** IQ2_XXS; Maya-M/L expected same geometry — code is pack-derived via #12) across two twin hosts:

| Host | Role | IP |
| --- | --- | --- |
| `strix-halo-1.local` | rank 0 (API server) | 192.168.1.22 |
| `strix-halo-2.local` | rank 1 (worker, **this machine**) | 192.168.1.92 |

- **Platform (hard rule):** HIP/ROCm **gfx1151 (Strix Halo) Linux only**. No CUDA/NVIDIA work.
- **Interconnect (hard rule):** **RDMA only** over Thunderbolt `usb4_rdma*` (gufo's path). No TCP data path (bootstrap TCP: QP handshake + serve instruction mirror + MTP accept wire only).
- **Reference implementation:** gufo at **`/home/ugo/projects/gufo`**. See `docs/models/qwen3.8-flash-next/TP2.md`, `src/cli/serve/tp_executor.cpp` + `tp_control.hpp`. MTP pattern: rank 0 `PlanDecode` → wire drafts → rank 1 `DecodeStepPlanned(forced_drafts)`.

## 2. Rules (must follow)

1. **Track all work through GitHub issues on `uv-genai/maya-tp`.** Comment while progressing. Link commits (`refs #N` / `closes #N`).
2. **Push only to `origin` (`uv-genai/maya-tp`). Never push to `upstream` (`mw00/project-maya`).**
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. Perf claims need **controlled** measurements. Use `--spec-follow` for near-tie counts.
5. **NEVER use nested SSH.** You are ON strix-halo-2. SSH only to strix-halo-1.
6. **Python uses `.venv/bin/python`.** Shell uses `jq`.
7. **Kill old processes before starting new ones.** `pkill -9 -x maya` — **do not** `pkill -f server.py` (matches the ssh shell and kills the session).
8. **Binaries are per-host.** After every build: `scp build-hip/maya strix-halo-1.local:.../maya` **and** `strix-halo-1.local:/tmp/maya-tp2` (used by `run_bench.sh`). Verify `md5sum` matches both sides before a test.
9. **Kill rank 0's server.py too before redeploying** — a live server respawns its engine (issue #27) and holds the binary open (`scp` fails "text file busy"); the respawned engine also inherits the OLD environment (no new env vars). Kill the server first to prevent respawn: `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'` (the `[s]` bracket stops the remote shell matching itself).

## 3. Git / GitHub

| | |
| --- | --- |
| `origin` | `https://github.com/uv-genai/maya-tp.git` |
| `upstream` | `https://github.com/mw00/project-maya.git` (read/merge only) |
| `main` | tracks `origin/main` |
| Latest release | **v1.1.37** |

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
| 14 | Controlled MTP on/off benchmark | complete; warm median off 38.255 / on 52.235 ms/token (§13) |
| 15 | MTP deployment + llama-swap launcher verification | complete; runtime flags and both-rank propagation (§14) |
| 16 | GLM vision on HIP + RDMA TP2 embedding synchronization | open, deferred by user (§15) |

## 5. What works (Maya-S pack)

- **RDMA pair:** `usb4_rdma5`, RoCE v2, `gid_index=1`, `mtu=4096`, write striping.
- **Full TP2 slicing:** mixer (KDA 64→32 heads, DSA head-slice), experts (2048→1024), shared expert, dense FFN (12288→6144), attn_output col-slice. Two ARs per layer.
- **Bench** (`tools/tp2/run_bench.sh`): both ranks `--tokens`; pair parity **100%**.
- **Serve** (`tools/tp2/rank1.sh` then `rank0.sh`): rank 0 mirrors `TpMirrorHdr` before every collective; rank 1 worker executes them. `curl /v1/chat/completions` works; multi-turn reuse works (simple loop).
- **Peer health:** `/health` → `tp.peer: up|down`, `status: degraded` after first failed mirror send. Sticky `TP2_DOWN` until both ranks restart.
- **TP2 + MTP:** opt-in `MAYA_TP_MTP=1` (leave `MAYA_GLM_NO_MTP` unset). Three-turn serve with restore/reuse, thinking-budget force, and cancel/follow-up verified (§11). Rank 0 owns inputs, acceptance and continuation; tensors stay on RDMA.
- **Acceptance:** `tools/tp2/acceptance.sh`.
- **Decode under TP2:** simple loop is the default; MTP is opt-in and greedy (see #13).

## 6. Performance (Maya-S IQ2_XXS)

Latest warm resident-pair test (#14, 256 identical output tokens, four measured runs per mode):
**38.255 ms/token without MTP vs 52.235 with MTP**, zero measured disk I/O. See §13 and the durable report.

Earlier 32-token measurements (2026-10-30):

| Config | ms/tok | notes |
| --- | --- | --- |
| single-host simple loop (warm) | **55.76** | pre-#13 baseline |
| single-host + MTP | **41.86** | 13 rounds, 2.46 tok/round (adaptive n) |
| TP2 + AR simple loop (spec-follow) | **41.1** | ~43 free greedy |
| **TP2 + AR + MTP** | **44.18** | 17 rounds, 1.88 tok/round (fixed n=2); drafts 9.3 ms + verify 73–77 ms + commit 0.1 ms per round |

- **MTP is not yet a win under TP2** (44.2 vs 41–43 simple). Verify is well-batched already; the lever is the **overlapped AR** (gufo `StartPartial`/`FinishPartial`; `MAYA_TP_OVERLAP` is NOT implemented — no such knob).
- Near-ties vs single-host MTP text (`--spec-follow`): **6/32** (simple-loop baseline 4/32 — normal AR reduction-order near-ties).
- Link: **6.04 µs** WRITE RTT; **37.4 Gb/s at 1 MiB**; **~9 Gb/s** at 10–48 KiB AR rows. Blocking AR ~58 µs. See §8.0.

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
6. **Near-ties:** use `--spec-follow`; free-generation agreement is not a quality gate.
7. **GCC 16 + HIP `<format>`:** `-include format` (already in CMake for TP targets).
8. **`ibv_create_cq_ex` / `IBV_SEND_INLINE` unsupported** on Thunderbolt.
9. **`RecvControl` must loop on EAGAIN.**
10. **`pkill -f server.py` over SSH kills the shell** — use `pkill -9 -x maya` for engines and the `[s]erver.py` bracket form for the server (§2.9).
11. **SETTINGS:** only implemented `MAYA_TP_*` knobs are real (see `docs/SETTINGS.md`). `MAYA_TP_OVERLAP` is NOT one.
12. **Per-host binaries + scripts** — always scp after build (§2.8/2.9).
13. **Gufo MTP pattern:** rank 0 plans and wires inputs/decisions; Maya uses fixed draft width (default 2), canonical input tokens (op 13), and accepted tokens/continuation (op 12).
14. **Idle TP worker holds ~45 GB VRAM with zero kernels** — normal (persistent worker waiting on the mirror socket); `pkill -9 -x maya` frees it.
15. **rank 0 logs are split:** engine stderr → repo `maya-maya-s-v2-iq2_xxs-hip.log` (**append mode** — multiple runs; segment by the last `glm prefill: CUDA0 chunks` line); server narration → `/tmp/tp2-r0.log`. rank 1 has everything in `/tmp/tp2-r1.log`.
16. **Engine restart during a request wipes TP INFO** — `/health` can briefly report `world_size: 1`; wait for `INFO tp_world=2` before judging. **Poisoned verbs communicator is sticky** — restart both ranks after any `size mismatch`/`timed out`.

## 9. How to run

### Benchmark (pair, controlled)
```sh
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P \
  --tokens $TOKS --max-new 32 --greedy --max-context 4096
bash tools/tp2/run_bench.sh   # both ranks, parity check
```
TP2+MTP variant: same but `MAYA_GLM_NO_MTP` **unset**, and **`MAYA_TP_MTP=1` on both CLI ranks**; rank 0 via `ssh strix-halo-1.local "/tmp/maya-tp2 --glm-pack $P --tokens $TOKS --max-new 32 --greedy --spec-follow-save /tmp/r0.txt --max-context 4096 --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5"`, rank 1 local with `--tp-rank 1 --tp-bootstrap-host strix-halo-1.local`.

### Serve (rank 1 first, no `/dev/tcp` probe)
```sh
bash tools/tp2/rank1.sh        # local; add env before it for the worker engine
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya || exit; \
  setsid nohup bash tools/tp2/rank0.sh 18080 >/tmp/tp2-r0.log 2>&1 </dev/null &'
curl http://192.168.1.22:18080/health   # expect tp.peer=up
```
Kill: `pkill -9 -x maya` locally; `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'`.

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
