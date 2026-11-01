# Handoff — Maya-TP (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated:2026-11-02 · Sequential integration (§29.21):37–42 accepted;43candidate #65 live validation active.
> Active batch8965; engine6a2fbe1/b398 all5paths matched. CPU/build/affectedEngram gates pass; known pre-existing DS4 WMMA failure68 remains open.
> Future fork release tags: **`tp-vMAJOR.MINOR.PATCH`** (#40); legacy published tags remain immutable.
> **v1.1.42 published** at `6471995` (CI41203815519 passed); release #35 (§28).
> Previous: **v1.1.41** at `a6d15a7` (CI41191735148 passed); #32 closed (§24).
> **v1.1.40** = `d9248e8`, **v1.1.39** = `20ff03a`. **Maya-S + Maya-M + Maya-L TP2/MTP validated** (§21, §23).
> Open perf: **#33** TP2 prefill — sliced GEMM calibration + rank1 config inheritance improve Maya-S
> median prompt throughput **32–37%** at 1K/4K/8K (§27). Binary **`5a52cd27167523b6ea4b0f0bde93f503`**
> retained as the pre-integration baseline (§29). Prefill commit `9bc0ec9`, PR #34 merged as `b63f8af`; local **main**
> tracks origin/main. Rank0 has user work (see §27); preserve it.
> Historical release-check runtime below is superseded by §29's authorized test window.
> Llama-swap daemon + 6 maya YAML entries preserved (Maya-L `--context 524288`).

## 1. What we are building

Two-way **tensor parallel (TP2)** for **GLM-5.3-Flash** (**Maya-S, Maya-M and Maya-L** all passed the full
TP2+MTP suite; geometry is pack-derived via #12) across two twin hosts:

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
| 33 | TP2 prefill throughput | **open** — sliced GEMM tuning + rank1 config inheritance improve S 32–37% (§27); row cap/short-request overhead remain |
| 35 | Release v1.1.42 | **published** — `6471995`, CI41203815519 PASS, annotated tag/public latest release verified (§28) |
| 42 | TP2 CLI readiness after expert warmup | **open** — existing Maya-L warmup skew needs a peer-ready handshake; WARM=0 cold CLI is very slow; tracking only during #37 |
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
