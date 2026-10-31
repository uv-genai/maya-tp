# Handoff — Maya-TP (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-31 · **v1.1.41 published** at `a6d15a7` (CI41191735148 passed); #32 closed (§24).
> **v1.1.40** = `d9248e8`, **v1.1.39** = `20ff03a`. #29/#30: **Maya-S + Maya-M + Maya-L TP2/MTP validated** (§21, §23).
> Current binary **`5a52cd27167523b6ea4b0f0bde93f503`** on all three paths (TP2_RDMA=ON rebuild).
> llama-swap daemon/config preserved. v1.1.38 stays at c322b3e.

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
9. **Kill rank 0's server.py too before redeploying** — a live server respawns its engine and holds the binary open (`scp` fails "text file busy"); the respawned engine also inherits the OLD environment (no new env vars). Kill the server first to prevent respawn: `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'` (the `[s]` bracket stops the remote shell matching itself).

## 3. Git / GitHub

| | |
| --- | --- |
| `origin` | `https://github.com/uv-genai/maya-tp.git` |
| `upstream` | `https://github.com/mw00/project-maya.git` (read/merge only) |
| `main` | tracks `origin/main` |
| Release version | **v1.1.41** published from `a6d15a7` ([#32](https://github.com/uv-genai/maya-tp/issues/32)); previous v1.1.40 = `d9248e8`, v1.1.39 = `20ff03a`, v1.1.38 = `c322b3e` |

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
| 29 | Maya-M TP2 + model-selectable launchers | open; launchers parameterized; **Maya-M packed both hosts**; simple bench 32/32 @42.6–42.8 ms/tok; **full Maya-S MTP suite passed on Maya-M** (§21) |
| 30 | Maya-L TP2 + MTP validation | **complete** — suite PASS, report `docs/benchmarks/tp2-maya-l-mtp-2026-10-31`; docs updated |
| 31 | Release v1.1.40 | published from `d9248e8`; tag/CI/publication verified; closed (§22) |
| 32 | Release v1.1.41 | published from `a6d15a7`; tag/CI/publication verified; closed (§24) |
| 18 | MTP rank-local scratch retry diverges collective shapes | fixed and validated (§16) |
| 19 | Large prefill exceeds 48 MiB RDMA window | fixed and validated (§16) |
| 20 | Large-prompt cache double-slice | fixed and validated (§16) |
| 21 | MTP window attention head width / HIP scratch capacity | fixed and validated (§16) |

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

**Current corrected engine**, framed chat, warm resident 256-token common references, all raw picks agree:

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
10. **`pkill -f server.py` over SSH kills the shell** — use `pkill -9 -x maya` for engines and the `[s]erver.py` bracket form for the server (§2.9).
11. **SETTINGS:** only implemented `MAYA_TP_*` knobs are real (see `docs/SETTINGS.md`). `MAYA_TP_OVERLAP` is NOT one.
12. **Per-host binaries + scripts** — always scp after build (§2.8/2.9).
13. **Gufo MTP pattern:** rank 0 plans and wires inputs/decisions; Maya uses fixed draft width (default 2), canonical input tokens (op 13), and accepted tokens/continuation (op 12).
14. **Idle TP worker holds ~45 GB VRAM with zero kernels** — normal (persistent worker waiting on the mirror socket); `pkill -9 -x maya` frees it.
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

**llama-swap rank0 YAML (user-requested):** six entries — `maya-glm5-{s,m,l}-tp2` (`--no-mtp`) and
`maya-glm5-{s,m,l}-tp2-mtp` (`--mtp --drafts 2`), each with explicit `--model`. Backups beside the file
(`…pre-maya-l`). Watch-config lists all six. Do not edit YAML unless asked.

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
