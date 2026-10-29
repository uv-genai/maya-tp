# TP2 Run Book — two-host tensor parallel for GLM-5.3-Flash

Operations guide for Maya TP2 on twin Strix Halo hosts over Thunderbolt RDMA.
Platform: **HIP/ROCm gfx1151 Linux only**. Interconnect: **RDMA only** (`usb4_rdma*`).

## Prerequisites

- Both hosts: gfx1151, ROCm, `thunderbolt_ibverbs` loaded (`usb4_rdma5` up; `ibv_devices` shows it)
- Same model pack on both hosts (`P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack`)
- Engine built on **both** hosts with TP2:
  ```sh
  cmake -S . -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_ENABLE_TP2_RDMA=ON
  cmake --build build-hip --target maya -j"$(nproc)"
  ```
- SSH from rank 1 to rank 0 (`strix-halo-1.local`)

**Roles:** rank 0 = API server (`strix-halo-1` / 192.168.1.22), rank 1 = worker (`strix-halo-2` / this host).

## What “6 µs / 37 Gb/s” means

Do **not** compare AR latency to the link microbench without reading gufo’s Thunderbolt doc:

| Metric | Benchmark | Meaning |
|---|---|---|
| ~6 µs | `ib_write_lat -s 2` | WRITE **round-trip** (2 B) — pure link |
| ~37 Gb/s | `ib_write_bw -s 1048576` | **1 MiB** sustained WRITE only |
| ~9 Gb/s | same at 10–48 KiB | Real AR row sizes (overhead-bound) |
| ~58 µs | `ExchangePartial` 48 KiB | Blocking AR (host staging + driver) |
| ~6 µs residual | `QueuePartial` Stage/Wait/Add | Overlapped AR (what the engine uses) |

Link microbench: `tools/tp2` / `build-tp2/tp_comm_rdma_test --bench --hidden 12288`.

## Quick start (benchmark, both ranks with `--tokens`)

```sh
# Single-host reference
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P \
  --tokens 154822,25062,287,29905,371,25,7487,785,6722,315,9621,374 \
  --max-new 32 --greedy --max-context 4096

# TP2 pair (both get the same --tokens; no control-channel mirror)
bash tools/tp2/run_bench.sh
```

## Production launch (serve mode)

**Start rank 1 first**, then rank 0. Never probe port 18515 with `/dev/tcp` — that completes a
TCP connection and rank 0’s single `accept()` takes it as the peer (handshake fails EAGAIN/EISDIR).
The engine already retries connect for `MAYA_TP_BOOTSTRAP_TIMEOUT` (default 120 s).

```sh
# Rank 1 worker (strix-halo-2) — no --serve, no --tokens
bash tools/tp2/rank1.sh

# Rank 0 API server (strix-halo-1) — via SSH; server.py spawns the engine with --serve
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya && nohup bash tools/tp2/rank0.sh 18080 \
  > /tmp/tp2-rank0.log 2>&1 &'

# Smoke test
curl -s http://192.168.1.22:18080/health
curl -s http://192.168.1.22:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"maya","messages":[{"role":"user","content":"Say hello."}],"max_tokens":8}'
```

Rank 1 log should show `TP2 rank 1 worker — mirroring…` then `reset` / `prefill N` / `forward 1` per request.

## Config (`maya-*.json` `tp` block)

```json
{
  "tp": {
    "world_size": 2,
    "rank": 0,
    "bootstrap_port": 18515,
    "rdma_device": "usb4_rdma5"
  },
  "args": ["--glm-pack", "/path/to/pack", "--max-context", "4096"]
}
```

Rank 1 adds `"bootstrap_host": "strix-halo-1.local"` and `"rank": 1`.
See `docs/SETTINGS.md` for the full `MAYA_TP_*` list (only implemented knobs are listed).

## Isolation flags

| Flag | Effect |
|---|---|
| `MAYA_TP_NOSLICE=1` | Full weights on both ranks |
| `MAYA_TP_NOAR=1` | Skip all-reduce (wrong answers; isolates slice vs AR) |
| `MAYA_TP_SH_BOTH=1` | Shared expert on both ranks |
| `MAYA_GLM_NO_MTP=1` | Disable MTP (controlled benchmarks) |
| `GLM_CB_DIR=<dir>` | Dump `ffn_out-<il>.f32` / `l_out-<il>.f32` for A/B |

## Health check

```sh
curl http://192.168.1.22:18080/health
```

`/health` reports `tp.world_size` / `tp.rank` from the engine INFO line. **It does not check that
rank 1 is alive** — if rank 1 dies mid-request, rank 0 hangs on the all-reduce or fails the request
while `/health` still says `ok` (issue #6).

## Failure playbook

| Symptom | Cause | Fix |
|---|---|---|
| `bootstrap accept timed out` | Rank 1 never connected | Start rank 1; check `usb4_rdma*` on both hosts |
| `bootstrap receive: Resource temporarily unavailable` / `Is a directory` | Something stole accept (`/dev/tcp` probe) | Remove port probes; restart both |
| `bootstrap connect: Operation now in progress` | Rank 0 not listening yet | Wait; engine retries 120 s (`MAYA_TP_BOOTSTRAP_TIMEOUT`) |
| `TP2 mirror send: Broken pipe` | Rank 1 died | Restart both; check rank 1 log |
| Request hangs after “reading the prompt” | Rank 1 not in worker mode / AR mismatch | Kill both; rank 1 without `--tokens` must log “mirroring” |
| `ibv_post_send failed` | RDMA link down | `ip link show`, reload `thunderbolt_ibverbs` |
| Pair parity < 100% | Slicing bug | `greedy_parity.py`; try `MAYA_TP_NOSLICE=1` |

## Performance reference (measured)

Controlled bench (no-MTP, 32 tokens, same prompt, warm engine), 2026-10-29/30:

| Config | ms/tok | vs single |
|---|---|---|
| single-host | **55.76** (56.81 on 10-29) | — |
| TP2 full slice, no AR | 36.57 | +35.6% |
| TP2 full slice + AR (free greedy) | ~43 | +23% |
| TP2 full slice + AR (spec-follow) | **41.1** | **+26%** |

**Quality (2026-10-30):**
- **Pair parity 100%** (rank0 vs rank1 free generation, 32/32).
- **Spec-follow vs single-host continuation:** output stream 32/32 forced; **4 of 32** positions where TP2’s own greedy pick differs (near-ties / float reduction order) — same class as the earlier 31/32.

AR cost ≈ 5 ms/tok (90 exchanges × ~55 µs). Link is fine (6 µs RTT, 37 Gb/s at 1 MiB); blocking AR is host/driver-bound.

End-to-end + microbench: `.venv/bin/python tools/calibrate_glm_tp.py`.

**CI:** `check_hip_compat.py` + `serve` (464) + `tools` (614) green. Single-host with TP flags off is unchanged (mirroring and reuse gates are `tp_world>1` / serve-only).

## Monitoring

- `STRATA_GLM_PROF=1`: per-stage GPU timing (`tp_ar_mixer`, …)
- `STRATA_GLM_TIMING=1`: ms/token summary
- `--spec-follow-save PATH`: save generated tokens for parity
- `tools/tp2/greedy_parity.py A B --min-agree 1.0`

## Calibration / microbench

```sh
# rebuild the AR microbench (needs STRATA_ENABLE_TP2_RDMA=ON in that build dir)
cmake -S . -B build-tp2 -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_ENABLE_TP2_RDMA=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-tp2 --target tp_comm_rdma_test -j"$(nproc)"

# both hosts; rank 0 first
ssh strix-halo-1.local '/tmp/tp_comm_rdma_test --rank 0 --bootstrap-port 18515 \
  --rdma-device usb4_rdma5 --exchanges 20 --bench --hidden 12288' &
./build-tp2/tp_comm_rdma_test --rank 1 --bootstrap-host strix-halo-1.local \
  --bootstrap-port 18515 --rdma-device usb4_rdma5 --exchanges 20 --bench --hidden 12288
```

`--hidden 12288` = Maya decode AR (48 KiB/row). `--hidden 2560` = gufo Flash-Next geometry.
