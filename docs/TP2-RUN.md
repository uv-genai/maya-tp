# TP2 Run Book — two-host tensor parallel for GLM-5.3-Flash

Operations guide for running Maya TP2 on twin Strix Halo hosts over Thunderbolt RDMA.

## Prerequisites

- Both hosts: gfx1151, ROCm, `thunderbolt_ibverbs` loaded (`usb4_rdma5` up)
- Same model pack on both hosts (`P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack`)
- Engine built: `cmake -S . -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_ENABLE_TP2_RDMA=ON`
- SSH access from rank 1 to rank 0 (`strix-halo-1.local`)

## Quick start (benchmark)

```sh
# Single-host reference
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P \
  --tokens 154822,25062,287,29905,371,25,7487,785,6722,315,9621,374 \
  --max-new 32 --greedy --max-context 4096

# TP2 pair (rank 0 first, rank 1 within 30 s)
bash tools/tp2/run_bench.sh
```

## Production launch

```sh
# Rank 0 (API server) on strix-halo-1
tools/tp2/maya-rank0.sh --config maya-glm5.json

# Rank 1 (worker) on strix-halo-2
tools/tp2/maya-rank1.sh --config maya-glm5.json --host strix-halo-1.local
```

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

Rank 1 adds `"bootstrap_host": "strix-halo-1.local"` and sets `"rank": 1`.

## Isolation flags

| Flag | Effect |
|---|---|
| `MAYA_TP_NOSLICE=1` | Full weights on both ranks (isolation) |
| `MAYA_TP_NOAR=1` | Skip all-reduce (slice vs AR bugs) |
| `MAYA_TP_SH_BOTH=1` | Shared expert on both ranks (AR doubles it) |
| `MAYA_GLM_NO_MTP=1` | Disable MTP (controlled benchmarks) |
| `GLM_CB_DIR=<dir>` | Dump `ffn_out-<il>.f32` / `l_out-<il>.f32` for A/B |

## Health check

```sh
curl http://127.0.0.1:8080/health
```

## Failure playbook

| Symptom | Cause | Fix |
|---|---|---|
| `bootstrap accept timed out` | Rank 1 not connected in 30 s | Start rank 1 sooner, check SSH |
| `bootstrap connect: Operation now in progress` | Rank 0 not ready | Wait for rank 0's expert warm-up (~20 s) |
| `ibv_post_send failed` | RDMA link down | `ip link show usb4_rdma5`, reload `thunderbolt_ibverbs` |
| `verbs exchange timed out` | Peer died | Restart both ranks |
| `HSA_STATUS_ERROR_MEMORY_FAULT` | Sliced weight OOB | Check `MAYA_TP_NOSLICE=1` works; report bug |
| Pair parity < 100% | Slicing bug | Run `greedy_parity.py`, compare with `MAYA_TP_NOSLICE=1` |

## Performance reference (2026-10-29)

| Config | ms/tok | vs single |
|---|---|---|
| single-host | 56.81 | — |
| TP2 full slice + AR | 38.52 | **+32.0%** |
| TP2 full slice, no AR | 36.57 | +35.6% |

Pair parity 100% (32/32). Spec-follow 31/32 (1 near-tie).

## Monitoring

- `STRATA_GLM_PROF=1`: per-stage GPU timing (kda_proj, moe_gate_up, tp_ar_mixer, …)
- `STRATA_GLM_TIMING=1`: ms/token summary
- `--spec-follow-save PATH`: save generated tokens for parity comparison
