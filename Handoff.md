# Handoff — Maya TP2 (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-30 · v1.1.37 · branch `main` on `uv-genai/maya-tp` · HEAD after `bd4607d`

## 1. What we are building

Two-way **tensor parallel (TP2)** for **GLM-5.3-Flash** (measured on **Maya-S** IQ2_XXS; Maya-M/L expected same geometry — code is pack-derived via #12) across two twin hosts:

| Host | Role | IP |
| --- | --- | --- |
| `strix-halo-1.local` | rank 0 (API server) | 192.168.1.22 |
| `strix-halo-2.local` | rank 1 (worker, **this machine**) | 192.168.1.92 |

- **Platform (hard rule):** HIP/ROCm **gfx1151 (Strix Halo) Linux only**. No CUDA/NVIDIA work.
- **Interconnect (hard rule):** **RDMA only** over Thunderbolt `usb4_rdma*` (gufo's path). No TCP data path (bootstrap TCP: QP handshake + serve instruction mirror only).
- **Reference implementation:** gufo at **`/home/ugo/projects/gufo`**. See `docs/models/qwen3.8-flash-next/TP2.md`, `src/cli/serve/tp_executor.cpp` + `tp_control.hpp`.

## 2. Rules (must follow)

1. **Track all work through GitHub issues on `uv-genai/maya-tp`.** Comment while progressing. Link commits (`refs #N` / `closes #N`).
2. **Push only to `origin` (`uv-genai/maya-tp`). Never push to `upstream` (`mw00/project-maya`).**
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. Perf claims need **controlled** measurements. Use `--spec-follow` for near-tie counts.
5. **NEVER use nested SSH.** You are ON strix-halo-2. SSH only to strix-halo-1.
6. **Python uses `.venv/bin/python`.** Shell uses `jq`.
7. **Kill old processes before starting new ones.** `pkill -9 -x maya` — **do not** `pkill -f server.py` (matches the ssh command line and kills the shell).
8. **Binaries are per-host.** After every build: `scp build-hip/maya strix-halo-1.local:.../maya` (and `mv` to final path). Verify `md5sum` matches both sides before a test.

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
| 2 | P1 partition + pack audit | closed |
| 3 | QueuePartial / link metrics | closed |
| 4 | P4 greedy parity | closed — PASS |
| 5 | P5 fast path + prefill | closed — +32% |
| 6 | P6 serve + launch | closed — mirroring, peer `/health`, kill/recover |
| 7 | P7 calibration / release | closed — CI + tables + v1.1.37 |
| 8 | P2 RDMA transport | closed |
| 9 | MTP / spec decode TP | closed by **gating** (not a TP-aware window); see #13 |
| 10 | Gather bit-exact variant | open (optional design) |
| 11 | Multi-seq + snapshot reuse | **partial — snapshot/reuse + rows ARs done; multi gated (`seq_count>1` prompt `mv_rdna`)** |
| 12 | Pack-derived TP geometry | **closed — `peek_pack_geometry` + post-load check** |
| 13 | MTP under TP2 | **partial — cycle sync proven; slow verify + SLICE `moe_gate_up`; gated** |

## 5. What works (Maya-S pack)

- **RDMA pair:** `usb4_rdma5`, RoCE v2, `gid_index=1`, `mtu=4096`, write striping.
- **Full TP2 slicing:** mixer (KDA 64→32 heads, DSA head-slice), experts (2048→1024), shared expert, dense FFN (12288→6144), attn_output col-slice. Two ARs per layer.
- **Bench** (`tools/tp2/run_bench.sh`): both ranks `--tokens`; pair parity **100%**.
- **Serve** (`tools/tp2/rank1.sh` then `rank0.sh`): rank 0 mirrors `TpMirrorHdr` before every collective; rank 1 worker executes them. `curl /v1/chat/completions` works.
- **Peer health:** `/health` → `tp.peer: up|down`, `status: degraded` after first failed mirror send. Sticky `TP2_DOWN` until both ranks restart.
- **Multi-turn reuse:** snapshot/restore mirrored (`kTpMirrorSnapshot`/`kTpMirrorRestore`); turn 2 shows `cached_tokens>0`.
- **Acceptance:** `tools/tp2/acceptance.sh`.
- **Decode under TP2:** **simple loop only** (MTP/DFlash/spec forced off). `MAYA_GLM_SEQS` ignored under TP2 (#11).

## 6. Performance (controlled, no-MTP, 32 tok)

| Config | ms/tok |
| --- | --- |
| single-host (warm) | **55.76** (56.81 on 10-29) |
| TP2 + AR (free greedy) | ~43 |
| TP2 + AR (spec-follow) | **41.1** |

Pair parity 32/32. Spec-follow: **4/32** near-ties vs single-host continuation.

Link: **6.04 µs** WRITE RTT; **37.4 Gb/s at 1 MiB**; **~9 Gb/s** at 10–48 KiB AR rows. Blocking AR ~58 µs. See §8.0.

## 7. Architecture

```
include/strata/core/tp_partition.hpp   TpPartition + tp_slice_expert
include/strata/core/tp_context.hpp     TpContext (create from pack widths — #12)
include/strata/net/{communicator,verbs}.hpp + src/net/{verbs.cpp,tp_kernels.cu}
src/core/{tp_context,glm_model,glm_window,glm_rows,glm_mtp}.cu  slice + AR hooks
src/program/generate.cpp               TpMirrorHdr, tp_send, worker, serve gates, MTP wire
serve/server.py                        TP2_DOWN → /health peer down
tools/tp2/{rank0.sh,rank1.sh,run_bench.sh,acceptance.sh}
tools/calibrate_glm_tp.py              microbench + e2e tables
build-tp2/tp_comm_rdma_test            --bench --hidden N (needs STRATA_ENABLE_TP2_RDMA=ON)
docs/TP2-RUN.md                        ops runbook
```

**Call mirroring:** send immediately **before** each collective `prefill`/`forward` (not inside `forward()`). Rank 1: `--tp-rank 1` without `--tokens`.

**MTP mirror (in tree, gated off):** `kTpMirrorMtp` + `TpMtpSp` (sampler, eos, think budget, penalty history); rank 1 runs `decode_mtp` with matching emit. `force_next` mirrored (`kTpMirrorForce`).

## 8. Critical gotchas

0. **6 µs / 37 Gb/s ≠ AR cost.** 6 µs = 2 B WRITE RTT; 37 Gb/s = 1 MiB plateau. Blocking AR ~58 µs; overlapped residual = gufo’s 6 µs form.
1. **Do not probe bootstrap with `/dev/tcp`** — steals rank 0’s `accept()`.
2. **Start rank 1 first** (or within rank 0’s 120 s accept window).
3. **Shard stride:** GGUF expert stride is full `gu_row * n_ff_exp`; `Ly.gu_bytes` is sliced.
4. **`NOSLICE=1` + AR doubles the FFN** (both ranks full compute then AR). Isolation = `NOSLICE=1` + `NOAR=1`.
5. **Drain:** `hipStreamSynchronize` before `CheckQueued`/`EndOperation`.
6. **Near-ties:** use `--spec-follow`.
7. **GCC 16 + HIP `<format>`:** `-include format` (already in CMake for TP targets).
8. **`ibv_create_cq_ex` / `IBV_SEND_INLINE` unsupported** on Thunderbolt.
9. **`RecvControl` must loop on EAGAIN.**
10. **`pkill -f server.py` over SSH kills the shell** — use `pkill -9 -x maya`.
11. **SETTINGS:** only implemented `MAYA_TP_*` knobs are real (see `docs/SETTINGS.md`).
12. **Per-host binaries** — always scp after build (gotcha 8 in §2).
13. **Gufo MTP pattern:** rank 0 `PlanDecode` → wire draft count → rank 1 `DecodeStepPlanned(forced_drafts)`; one `Forward(verify chain)` per step. Maya analogue: constant `n=2` under TP + identical `produced`/`p0` (emit must not double-increment `decode_mtp`’s counter).

## 9. How to run

### Benchmark
```sh
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P \
  --tokens $TOKS --max-new 32 --greedy --max-context 4096
bash tools/tp2/run_bench.sh
```

### Serve (rank 1 first, no `/dev/tcp` probe)
```sh
bash tools/tp2/rank1.sh
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya && \
  setsid nohup bash tools/tp2/rank0.sh 18080 >/tmp/tp2-r0.log 2>&1 </dev/null &'
curl http://192.168.1.22:18080/health   # expect tp.peer=up
```

Runbook: **`docs/TP2-RUN.md`**. Settings: **`docs/SETTINGS.md`**.

### Debug knobs (MTP / #13)
- `STRATA_GLM_TP_AR_TRACE=1` — log every window `all_reduce` site/size + `decode_mtp` round-end
- `STRATA_GLM_MTP_FIXED=<n>` / `STRATA_GLM_MTP_ADAPT=0` — fixed draft width
- Logs: rank1 `/tmp/tp2-r1.log`; rank0 engine `maya-maya-s-v2-iq2_xxs-hip.log` in the repo

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

Sticky pair-down prints `TP2_DOWN` on stdout; server sets `status=degraded` / `tp.peer=down`.

## 11. #13 current state (MTP under TP2) — resume here

**Done in tree (keep):**
- Window KDA/DSA/q_b use `head_compute_for` (token-path NH/DIw); mixer + dense ARs; `W.whq` full `n_ff_exp`; EXL3 `n_ff = ff_compute_for`
- Rows: mixer + dense/grouped FF ARs; KDA NH; q_b sliced n_out
- `decode_mtp`: emit no longer double-increments `produced`; greedy forced under TP+MTP; `mtp_choose` returns constant width under TP; skip `hx_pos→n=0` when TP
- Mirror protocol: `kTpMirrorMtp`, `kTpMirrorForce`, `TpMtpSp`
- Optional AR/round traces (`STRATA_GLM_TP_AR_TRACE=1`)

**Measured (NOSLICE + MTP, emit-fixed):**
- Both ranks: `n=2 nd=2 T=3`, `produced`/`p0`/`keep` **matched** through `p0≈31`
- Window AR sequences identical (45 × 12288 floats/window)
- **Not a deadlock:** rank0 `verify≈22.6s/8 rounds` (~2.8s/window) → **0.1 tok/s** → server **cancel ~183s**
- NOSLICE+AR **doubles** expert compute (both ranks full then AR) — for isolation use `NOSLICE=1`+`NOAR=1`

**Still blocked (MTP gated `!tp2` in `run_request`):**
1. **SLICE** — `moe_gate_up_kernel` HSA fault in window (NOSLICE does not fault). Token-path `moe_gate_up` is fine. Next: compare window `md.plan` / `W.xq` / `Ly.blob` vs token path under slice.
2. **Slow verify** (~2.8s/window) — profile AR wait vs window compute (add timestamps to `STRATA_GLM_TP_AR_TRACE`).

**Gufo pattern to finish against:** `PlanDecode` → wire `drafts` → `DecodeStepPlanned(forced_drafts)`; one `Forward(verify chain)` per step (`tp_executor.cpp`, `engine.hpp` `forced_drafts`).

## 12. Key commits

`bd4607d`/`af0c398` AR traces · `63d3966` emit/greedy/n=2 sync · `57c30ce` window/rows TP dims · `64ba3c6` whq/n=0 drafts · `aec2829` peer health · `805ee0b` serve mirroring · `0b9f5c4` v1.1.37 · earlier TP2 slice/transport.
