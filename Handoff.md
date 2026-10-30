# Handoff — Maya TP2 (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-30 (session 2, #13 deep dive) · v1.1.37 · branch `main` on `uv-genai/maya-tp` · HEAD `a4c10d5` + **uncommitted WIP (see §11 — commit it first thing)**.
> Both hosts' GPUs are **free** (all engines/servers killed). Rank binaries on disk: `26babb5e…` (accept wire, no traces); local `build-hip/maya` is `e668184c…` (+wire traces), **not yet scp'd to strix-halo-1** (last scp failed, text-file-busy).

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
9. **Kill rank 0's server.py too before redeploying** — a live server respawns its engine (issue #27) and holds the binary open (`scp` fails "text file busy"); the respawned engine also inherits the OLD environment (no new env vars). Safe form over ssh: `ssh strix-halo-1.local 'pkill -9 -x maya; pkill -9 -f "[s]erver.py --engine"'` (the `[s]` bracket stops the remote shell matching itself).

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
| 13 | MTP under TP2 | **in progress — (a) SLICE window fault FIXED (uncommitted); (b) slow verify RESOLVED; (c) accept wire implemented but turn-2 desyncs (root-cause with wire traces). MTP gate is currently OPEN (`!tp2` removed) for diagnosis — re-gate before any merge until (c) is fixed.** |

## 5. What works (Maya-S pack)

- **RDMA pair:** `usb4_rdma5`, RoCE v2, `gid_index=1`, `mtu=4096`, write striping.
- **Full TP2 slicing:** mixer (KDA 64→32 heads, DSA head-slice), experts (2048→1024), shared expert, dense FFN (12288→6144), attn_output col-slice. Two ARs per layer.
- **Bench** (`tools/tp2/run_bench.sh`): both ranks `--tokens`; pair parity **100%**.
- **Serve** (`tools/tp2/rank1.sh` then `rank0.sh`): rank 0 mirrors `TpMirrorHdr` before every collective; rank 1 worker executes them. `curl /v1/chat/completions` works; multi-turn reuse works (simple loop).
- **Peer health:** `/health` → `tp.peer: up|down`, `status: degraded` after first failed mirror send. Sticky `TP2_DOWN` until both ranks restart.
- **TP2 + MTP (CLI, `MAYA_GLM_NO_MTP` unset):** completes on both ranks, **pair parity 100% (32/32)**, round streams identical. Serve turn 1 works (~17–19 tok/s e2e).
- **Acceptance:** `tools/tp2/acceptance.sh`.
- **Decode under TP2:** simple loop is the default-shipped path; MTP currently ON only for diagnosis (see #13).

## 6. Performance (Maya-S IQ2_XXS, 32-tok controlled runs, 2026-10-30)

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

**MTP mirror:** `kTpMirrorMtp` + `TpMtpSp` (sampler, eos, think budget, penalty history); rank 1 runs `decode_mtp` with matching emit. `force_next` mirrored (`kTpMirrorForce`). **NEW `kTpMirrorMtpRound` (op 12):** per verify round, rank 0 sends `{n=keep, tokens[]}` after its accept loop; rank 1 receives before its accept loop and commits the wired tokens (glm_mtp.cu `decode_mtp`, wire lambdas in generate.cpp).

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
13. **Gufo MTP pattern:** rank 0 `PlanDecode` → wire draft count → rank 1 `DecodeStepPlanned(forced_drafts)`; one `Forward(verify chain)` per step. Maya analogue: fixed `n=2` + accept wire (op 12).
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
TP2+MTP variant: same but `MAYA_GLM_NO_MTP` **unset**; rank 0 via `ssh strix-halo-1.local "/tmp/maya-tp2 --glm-pack $P --tokens $TOKS --max-new 32 --greedy --spec-follow-save /tmp/r0.txt --max-context 4096 --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5"`, rank 1 local with `--tp-rank 1 --tp-bootstrap-host strix-halo-1.local`.

### Serve (rank 1 first, no `/dev/tcp` probe)
```sh
bash tools/tp2/rank1.sh        # local; add env before it for the worker engine
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya && \
  setsid nohup bash tools/tp2/rank0.sh 18080 >/tmp/tp2-r0.log 2>&1 </dev/null &'
curl http://192.168.1.22:18080/health   # expect tp.peer=up
```
Kill: `pkill -9 -x maya` locally; `ssh strix-halo-1.local 'pkill -9 -x maya; pkill -9 -f "[s]erver.py --engine"'`.

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
| 12 | **MTP accept wire: `n=keep`, payload = accepted token ids (rank 0 → rank 1, one per verify round, inside the op-10 decode)** |

Sticky pair-down prints `TP2_DOWN` on stdout; server sets `status=degraded` / `tp.peer=down`.

## 11. #13 current state (MTP under TP2) — resume here

### 11.1 Done this session (all in the uncommitted WIP — commit first)

1. **SLICE window fault FIXED (was blocker a).** `fast_moe_window` computed the shared-expert width as full `FFs = g.n_ff_exp * g.n_shared` while consumers run against **sliced** shared weights (gate/up row-sliced, down col-sliced at load, glm_model.cu post-load pass). The shared gate/up MvJobs, the shared-down MvJob, and `moe_gate_up`'s shared section all read past the sliced weights → the `moe_gate_up_kernel` HSA fault, slice-only, both WIN_MULTI styles. The token path was fine because it uses `ff_compute() * n_shared`. **Fix:** `FFs = ff_compute_for(il) * g.n_shared` (glm_window.cu:530). Buffers were never the problem — all hq/sh buffers are allocated full-width (`mt_sh_hq_s` is only a stride — verified).
2. **Slow verify (blocker b) resolved:** verify is now **73–77 ms/round** under slice+AR (the 2.8 s/window was a NOSLICE-era measurement, before the `57c30ce`/`64ba3c6` window width fixes).
3. **Accept wire implemented** (gufo's decide-and-wire pattern): `decode_mtp` gained `tp_round_send`/`tp_round_recv` params (glm_model.hpp, glm_mtp.cu); rank 0 sends `{keep, tokens}` per round (op 12) right after its accept loop, unconditionally; rank 1 receives before its accept loop and commits the wired tokens. Worker emit's rank1-only think-budget `force_next` removed (rank 0 never forces under TP2; wire is authoritative). Wire traces under `STRATA_GLM_TP_AR_TRACE` (`wire-send`/`wire-recv wait`/`wire-got`).
4. **MTP gate OPENED** in `run_request` (the `!tp2` on the MTP branch removed) — **diagnosis only; re-gate until 11.3 is fixed.**
5. Trace instrumentation added: prefill AR sites (glm_prefill.cu mixer/ffn), rows AR sites (glm_rows.cu ×4).

Verified on the pair (slice + MTP, gate open): 16/32-token CLI runs complete, **pair parity 100%**, `n=2 nd=2 T=3` streams identical; serve turn 1 ~17–19 tok/s; near-ties 6/32 vs single-host MTP text.

### 11.2 Measured (2026-10-30, this session)

- TP2+MTP CLI: 44.18 ms/tok (drafts 9.3 + verify 73–77 + commit 0.1 ms/round; 1.88 tok/round at fixed n=2). Single-host MTP 41.86 ms/tok (adaptive n, 2.46 tok/round). TP2 simple 41.1–43. → MTP not yet a win; overlap is the lever.
- Serve turn 1: works. Multi-turn (turn 2 with restore) = the current bug (11.3).

### 11.3 CURRENT BLOCKER — turn-2 desync with the accept wire (root-cause next)

Repro: fresh pair → turn-1 chat (64 tok) OK → turn-2 chat (restore + prefill 25 + decode) stalls after ~22 tokens:
- rank1 blocks in **round-13 `wire-recv`** (its log ends: `n=2 p0=69 produced=21`, `nd=2 T=3 ok=1`, then `TP2 worker mtp: TP2 pair down (mtp accept wire)`).
- rank0 advances **one round ahead** (its round-14 window AR gets no peer AR) → `verbs all-reduce timed out` (~150 s) → poison; server cancels; rank0's cleanup ops land in rank1's pending recv → recv rejects a non-op-12 header → mirror loop misaligned (`unknown op 154842` — a template token from a payload) → pair dead.
- So rank0 completed round 13 (and sent its wire message — its ph[2] shows fast sends for all its rounds) yet rank1's round-13 recv got nothing. **A one-round send/recv skew exists.** Both ranks report `nd=2 T=3` on every visible round, so the skew mechanism is still unknown.

**Next diagnostic step (do this first):** deploy the traced local binary (`e668184c…`) to **both** hosts (`build-hip/maya` **and** `/tmp/maya-tp2`, after killing engines+server per §2.9), run the 2-turn serve test with `STRATA_GLM_TP_AR_TRACE=1` on both ranks, then diff the `wire-send` (rank0 hip.log) vs `wire-recv wait`/`wire-got` (rank1) sequences — the first round where a `wire-send` has no matching `wire-got` localizes the skew. Check hypotheses in order:
1. **A round where one rank skips the wire** — e.g. `T==1` plain-token path taken on one rank only (mtp_choose must return constant 2 under TP — verify), or an early `break` between the window and the wire on one side (window error paths before the recv/send point).
2. **Off-by-one from an extra/missing socket message** — anything else sent on the bootstrap socket during decode (compare against the ops the worker logged). The op-10 header/TpMtpSp/penalty-history reads are byte-exact only if `h.next_token` (hist len) matches what rank0 sent.
3. **Turn-2 specifics:** restore→prefill→decode. Confirm both ranks' first post-restore round starts at the same `p0` and that no extra mirror op interleaves with the first wire message.

**Also to fix (after the skew): cancel robustness.** When rank0 aborts mid-round (window error / cancel), rank1 can sit in `wire-recv` until a cleanup op arrives, and the recv-fn consuming a non-op-12 header misaligns the op stream. Mitigations to consider: (a) `RecvControl` timeout in the wire recv (fail fast → both ranks restart, the accepted recovery for poisoned pairs), (b) a flush/abort round message from rank0 when its decode exits with an incomplete round, (c) on op mismatch, worker treats the pair as down and stops cleanly instead of reading on.

### 11.4 Carried-over notes (unchanged from previous session)

- `glm_rows.cu` still has the **full-width `FFs`** pattern (lines ~87/672/759) — same latent OOB as the window bug; rows never runs under TP2 today (`multi` gated `!tp2`). Fix with `ff_compute_for(il) * g.n_shared` when re-enabling multi (#11).
- NOSLICE+AR doubles expert compute — for isolation use `NOSLICE=1`+`NOAR=1`.
- `W.whq` stays full `n_ff_exp` (window runs unsliced `mtp_il_`); `W.whq` full-width fix (`64ba3c6`) is in the verified set.
- Window TP checklist (durable): residual ARs after mixer and FFN; expert launches use `ff_compute_for(il)`/`ff_compute_dense()`; head-dependent MvJobs use `head_compute_for(il)` widths; window buffers serving `mtp_il_` stay full-width.

### 11.5 Exit criteria for #13

1. Two-turn (and three-turn) serve chat works with MTP under TP2 (restore/reuse + wire).
2. Cancel mid-request + kill/recover leave the pair restartable (no sticky misalignment beyond the accepted "restart both ranks").
3. CLI pair parity 100%; near-tie rate vs single-host in line with simple-loop (4–6/32).
4. Decide the gate: keep MTP opt-in under TP2 vs default-on, given 44.2 vs 41–43 ms/tok (probably ship gated/opt-in until the overlapped AR lands).
5. Run `tools/tp2/acceptance.sh` + the CI trio (`tools/check_hip_compat.py`, serve tests, tools tests), commit (`refs #13`), update the issue and this file.

## 12. Key commits

`a4c10d5` Handoff (pre-#13 restart) · `bd4607d`/`af0c398` AR traces · `63d3966` emit/greedy/n=2 sync · `57c30ce` window/rows TP dims · `64ba3c6` whq/n=0 drafts · `aec2829` peer health · `805ee0b` serve mirroring · `0b9f5c4` v1.1.37 · earlier TP2 slice/transport.
**Uncommitted WIP (this session):** window FFs slice fix + prefill/rows AR traces + MTP accept wire (op 12) + gate opened + wire traces — see §11.1. Changed files: `include/strata/core/glm_model.hpp`, `src/core/glm_mtp.cu`, `src/core/glm_prefill.cu`, `src/core/glm_rows.cu`, `src/core/glm_window.cu`, `src/program/generate.cpp`.
