# Handoff — Maya TP2 (two-host tensor parallel)

> **Live document.** Update this whenever state changes so a fresh session can restart from here.
> Last updated: 2026-10-30 · v1.1.37 · branch `main` on `uv-genai/maya-tp`

## 1. What we are building

Two-way **tensor parallel (TP2)** for **GLM-5.3-Flash** (measured on **Maya-S** IQ2_XXS; Maya-M/L expected same geometry — **#12**) across two twin hosts:

| Host | Role | IP |
| --- | --- | --- |
| `strix-halo-1.local` | rank 0 (API server) | 192.168.1.22 |
| `strix-halo-2.local` | rank 1 (worker, **this machine**) | 192.168.1.92 |

- **Platform (hard rule):** HIP/ROCm **gfx1151 (Strix Halo) Linux only**. No CUDA/NVIDIA work.
- **Interconnect (hard rule):** **RDMA only** over Thunderbolt `usb4_rdma*` (gufo's path). No TCP data path (bootstrap TCP: QP handshake + serve instruction mirror only).
- **Reference implementation:** gufo at **`/home/ugo/projects/gufo`**. See `docs/models/qwen3.8-flash-next/TP2.md`, `src/cli/serve/tp_executor.cpp`.

## 2. Rules (must follow)

1. **Track all work through GitHub issues on `uv-genai/maya-tp`.** Comment while making progress. Link commits (`refs #N` / `closes #N`).
2. **Push only to `origin` (`uv-genai/maya-tp`). Never push to `upstream`.**
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. Perf claims need **controlled** measurements. Use `--spec-follow` for near-tie counts.
5. **NEVER use nested SSH.** You are ON strix-halo-2. SSH only to strix-halo-1.
6. **Python uses `.venv/bin/python`.** Shell uses `jq`.
7. **Kill old processes before starting new ones.** `pkill -9 -x maya` — **do not** `pkill -f server.py` (matches the ssh command line and kills the shell).

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
| 9 | MTP / spec decode TP | closed — fixed (MTP still **off** under two-host TP; see #13) |
| 10 | Gather bit-exact variant | open (optional design) |
| 11 | Multi-seq + snapshot reuse | open |
| 12 | Pack-derived TP geometry | open |
| 13 | MTP under TP2 | open |

## 5. What works (Maya-S pack)

- **RDMA pair:** `usb4_rdma5`, RoCE v2, `gid_index=1`, `mtu=4096`, write striping.
- **Full TP2 slicing:** mixer (KDA 64→32 heads, DSA head-slice), experts (2048→1024), shared expert, dense FFN (12288→6144), attn_output col-slice. Two ARs per layer.
- **Bench** (`tools/tp2/run_bench.sh`): both ranks `--tokens`; pair parity **100%**.
- **Serve** (`tools/tp2/rank1.sh` then `rank0.sh`): rank 0 mirrors `TpMirrorHdr` (reset / prefill / forward) before every collective call; rank 1 worker executes them. `curl /v1/chat/completions` works.
- **Peer health:** `/health` → `tp.peer: up|down`, `status: degraded` after first failed mirror send. Sticky `TP2_DOWN` until both ranks restart.
- **Acceptance:** `tools/tp2/acceptance.sh` (pair up, chat, kill rank1, recover).
- **Decode under TP2:** **simple loop only** (MTP/DFlash/spec forced off — #13). Conversation reuse and `MAYA_GLM_SEQS` off (#11).

## 6. Performance (controlled, no-MTP, 32 tok)

| Config | ms/tok |
| --- | --- |
| single-host (warm) | **55.76** (56.81 on 10-29) |
| TP2 + AR (free greedy) | ~43 |
| TP2 + AR (spec-follow) | **41.1** |

Pair parity 32/32. Spec-follow: **4/32** positions where TP2’s own pick differs from single-host’s continuation (float reduction order).

AR ≈ 5 ms/tok (90 × ~55 µs host-bound). Link: **6.04 µs** WRITE RTT, **37.4 Gb/s at 1 MiB**; at 10–48 KiB AR rows **~9 Gb/s**. See Handoff §8.0 and `docs/TP2-RUN.md`.

## 7. Architecture

```
include/strata/core/tp_partition.hpp   TpPartition + tp_slice_expert
include/strata/core/tp_context.hpp     TpContext (create hardcodes 2048/12288 until #12)
include/strata/net/{communicator,verbs}.hpp + src/net/{verbs.cpp,tp_kernels.cu}
src/core/{tp_context,glm_model,...}.cu slice + AR hooks
src/program/generate.cpp               TpMirrorHdr, tp_send, worker loop, serve gates
serve/server.py                        TP2_DOWN → /health peer down
tools/tp2/{rank0.sh,rank1.sh,run_bench.sh,acceptance.sh,calibrate via tools/}
docs/TP2-RUN.md                        ops runbook
```

**Call mirroring:** send immediately **before** each collective `prefill`/`forward` (not inside `forward()` — `prep_prompt` blocks first). Rank 1: `--tp-rank 1` without `--tokens` (any `--serve`).

## 8. Critical gotchas

0. **6 µs / 37 Gb/s ≠ AR cost.** 6 µs = 2 B WRITE RTT; 37 Gb/s = 1 MiB plateau. Blocking AR ~58 µs; overlapped residual ~gufo’s 6 µs form (`QueuePartial`).
1. **Do not probe bootstrap with `/dev/tcp`** — steals rank 0’s `accept()`.
2. **Start rank 1 first** (or within rank 0’s 120 s accept window). Engine retries connect (`MAYA_TP_BOOTSTRAP_TIMEOUT`).
3. **Shard stride:** GGUF expert stride is full `gu_row * n_ff_exp`; `Ly.gu_bytes` is sliced.
4. **Shared expert summed** across ranks. `MAYA_TP_SH_BOTH=1`.
5. **Drain:** `hipStreamSynchronize` before `CheckQueued`/`EndOperation`.
6. **Near-ties:** `dot(half0)+dot(half1) != dot(full)`. Use `--spec-follow`.
7. **GCC 16 + HIP `<format>`:** `-include format`.
8. **`ibv_create_cq_ex` / `IBV_SEND_INLINE` unsupported** on Thunderbolt driver.
9. **`RecvControl` must loop on EAGAIN** (bootstrap socket non-blocking).
10. **`pkill -f server.py` over SSH kills the shell** — use `pkill -9 -x maya` and port `fuser` carefully.
11. **SETTINGS:** only implemented `MAYA_TP_*` knobs are real; `CONTROL_*`, `AR_FP16`, `PREFILL_ROWS`, `OVERLAP` are not implemented.

## 9. How to run

### Benchmark (both ranks get `--tokens`)
```sh
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack
TOKS=154822,25062,287,29905,371,25,7487,785,6722,315,9621,374
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P \
  --tokens $TOKS --max-new 32 --greedy --max-context 4096
bash tools/tp2/run_bench.sh
```

### Serve (rank 1 first, no `/dev/tcp` probe)
```sh
# rank 1 (this host)
bash tools/tp2/rank1.sh
# rank 0 (strix-halo-1)
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya && \
  setsid nohup bash tools/tp2/rank0.sh 18080 >/tmp/tp2-r0.log 2>&1 </dev/null &'
curl http://192.168.1.22:18080/health   # expect tp.peer=up
```

Full runbook: **`docs/TP2-RUN.md`**. Settings: **`docs/SETTINGS.md`** (GLM TP2 section).

## 10. Serve mirroring (how it works)

Rank 0 `tp_send` before each collective: `{op, n, next_token}` + `int32` tokens on bootstrap TCP. Ops: reset=0, prefill=1, forward=2. Sticky pair-down prints `TP2_DOWN` on stdout; `serve/server.py` sets `status=degraded` / `tp.peer=down`.

Gufo reference: `TpMirroredRunner`, `kPrefill`/`kAdvance`/`kDecode` in `/home/ugo/projects/gufo/src/cli/serve/`.

## 11. Key commits

`aec2829` peer health + pair-down · `10ffb91` P7 docs/calibrator · `9518a40` AR vs 6µs/37Gb/s · `805ee0b` serve mirroring · `0b9f5c4` v1.1.37 · earlier TP2 slice/transport on main.
