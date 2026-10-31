# Maya-L TP2 + MTP validation — 2026-10-31

Same suite as Maya-S (#22) and Maya-M (#29), for **issue #30**.

## Environment

| | |
| --- | --- |
| Hosts | strix-halo-1 (rank0) / strix-halo-2 (rank1), gfx1151, `usb4_rdma5` |
| Pack | `/home/ugo/software/Maya-data/models/Maya-L/pack` (identical MD5s both hosts) |
| Binary | `5a52cd27167523b6ea4b0f0bde93f503` (all three paths) |
| Context | 32768 (API), 4096 (CLI / bench) |
| MTP | `MAYA_TP_MTP=1`, fixed/cap **7** (serve/CLI) or **2** (`bench_mtp.py`) |
| Launch | `--model Maya-L` / `MAYA_MODEL=Maya-L` |

Engines stopped after the run.

## Results

| Test | Result | Artifact |
| --- | --- | --- |
| `audit_split.py --world 2` | **all splits block-safe** (after adding Q3_K/Q4_K/Q5_K/IQ4_NL/BF16 to the tool) | `audit-split.txt` |
| `run_bench.sh --model Maya-L` | pair parity **32/32**, **64.34 / 64.42 ms/token** (r1/r0) | `run-bench.txt` |
| `mtp_acceptance.py` | **PASS** (reuse, cancel, limits 1/2/3, peer up) | `mtp-acceptance.txt` |
| `regression_collectives --mode mtp` | **PASS** (990/7290/14550, cache 7289, stream) | `mtp-matrix.json` |
| `regression_collectives --mode simple` | **PASS** | `simple-matrix.json` |
| CLI parity simple | **32/32** | `cli-parity.txt` |
| CLI parity 7-draft MTP | **32/32** (`mtp=1`) — with `STRATA_GLM_WARM=0` both ranks | `cli-parity.txt` |
| `acceptance.sh` MTP kill/recover | **8 passed, 0 failed** | `acceptance-mtp.txt` |
| `bench_mtp.py --model Maya-L --drafts 2` | **PASS** (raw pick diffs 0) | `bench-mtp-results.json` |

### Notes on Maya-L warm skew

Rank0’s expert warm on Maya-L was much slower than rank1’s (up to ~8 min vs ~40 s cold). For **paired CLI** decode,
rank1 finished warming first and the first all-reduce timed out (`verbs all-reduce timed out`), poisoning the
communicator (Handoff: restart both ranks). Workaround used for CLI MTP: **`STRATA_GLM_WARM=0` on both ranks**
(skip the pre-warm; fill on demand) so both enter decode together. Serve mode does not have this issue (rank1
waits for mirrors). `bench_mtp.py` READY wait raised to **900 s** so a cold rank0 can finish warming.

### Controlled MTP vs simple (`bench_mtp`, 256-token common reference, ABBA, warmups excluded)

| Mode | median ms/tok | median tok/s |
| --- | ---: | ---: |
| simple | **42.81** | **23.36** |
| MTP (2 drafts) | **32.62** | **30.65** |

MTP throughput **+31.24%**, latency **−23.80%**. Drafts 158/193 accepted. One MTP warm measure was a 47.7 ms
outlier; medians exclude warmups. Not a universal gain claim.

## Outstanding tooling fixes in this session

- `tools/tp2/audit_split.py`: GGML table missing Q3_K/Q4_K/Q5_K/IQ4_NL/BF16 (Maya-L down quants).
- `tools/tp2/bench_mtp.py`: READY/DONE timeouts 240/180 → 900/600.
- Installer `maya.py`: `-DSTRATA_ENABLE_TP2_RDMA=ON` (`62ee2e7`).
