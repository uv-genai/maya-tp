# TP2 Run Book — two-host tensor parallel for GLM-5.3-Flash

Operations guide for Maya TP2 on twin Strix Halo hosts over Thunderbolt RDMA.

Both ranks use the selected config's KV format and streaming/resident settings.
The worker forwards `--kv`, `--kv-streaming` and `--kv-resident` in config order,
including the GLM Maya-XS installer's INT8 KV selection.
Platform: **HIP/ROCm gfx1151 Linux only**. Interconnect: **RDMA only** (`usb4_rdma*`).
For DeepSeek TP2 and DSpark, use its family-specific launchers in the
[DeepSeek TP2 operations guide](DEEPSEEK-TP2-RUN.md).

The source checkpoint through upstream v1.1.53 is `de184d9`, with final main CI
41834772034 passing. Per-tag validation used isolated GLM Maya-S and DeepSeek
XS/S builds. Accepted runtime paths are recorded below, and the full source
checkpoint is in [Handoff §47](../Handoff.md#47-sequential-upstream-v1150v1153-integration).

## Prerequisites

- Both hosts: gfx1151, ROCm, `thunderbolt_ibverbs` loaded (`usb4_rdma5` up; `ibv_devices` shows it)
- Same model pack on both hosts (default `P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack`;
  any pack works — see **Choosing a model** below)
- Engine built on **both** hosts with TP2:
  ```sh
  cmake -S . -B build-hip -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_ENABLE_TP2_RDMA=ON
  cmake --build build-hip --target maya -j"$(nproc)"
  ```
- SSH from rank 1 to rank 0 (`strix-halo-1.local`)

**Roles:** rank 0 = API server (`strix-halo-1` / 192.168.1.22), rank 1 = worker (`strix-halo-2` / this host).

## Development pair: production and review builds

The user accepted the prefill candidate after Maya-S, Maya-M and Maya-L llama-swap tests. Canonical
`build-hip/maya` on both hosts is MD5 `cbd9e99a6a49e9107b0db3190a40766a`, with matching calibrated
tuning. The user validated this selection through llama-swap. The
previous binaries, configs, rank1 launchers and tuning are backed up on both hosts in
`/home/ugo/software/project-maya-before-prefill-90-promotion`; the rank0 backup also contains its
previous server files. The rank0 llama-swap backup is `~/.config/llama-swap/config.yaml.before-prefill-90`. The remote canonical checkout
still contains user edits; do not reset it. See [Handoff §33](../Handoff.md#33-prefill-promotion-and-user-acceptance-90--91--93--33)
for performance and validation limits.

## Process ownership and validation

The user starts their own processes. Coding agents may clean up only their recorded validation
PIDs/process groups, stopping their server before its engine. A prior permission to use the hosts
does not authorize future cleanup of user processes or inspection/control of unrelated services.

`acceptance.sh` and `bench_mtp.py` currently use name-wide process cleanup when starting a pair.
They require explicitly reserved, free hosts; an agent must not run them against user processes.
Owned-process cleanup is tracked in [#83](https://github.com/uv-genai/maya-tp/issues/83).
The API probes below use an already-running pair; the benchmark/recovery harnesses start and stop
engines and must be scheduled accordingly.

## Choosing a model

Launchers do **not** hardcode Maya-S. Pick a model by **model id** or **pack path**, with a CLI flag or an
environment variable. The shared resolver is `tools/tp2/model_env.sh` (sourced by the rank scripts,
`run_*`, and the llama-swap wrapper).

### Selection inputs

| CLI | Environment | Meaning |
|---|---|---|
| `--model NAME` | `MAYA_MODEL` | `$MAYA_MODELS_DIR/NAME/pack` |
| `--pack PATH` | `MAYA_PACK` | explicit pack directory (wins over `--model`) |
| `--config PATH` | `MAYA_CONFIG` | base `maya-*.json` (exe, env, tokenizer, context defaults) |
| — | `MAYA_MODELS_DIR` | model root (default `/home/ugo/software/Maya-data/models`) |

**Precedence (pack):** `--pack` / `MAYA_PACK` → `--model` / `MAYA_MODEL` → default
`Maya-S-v2-IQ2_XXS`. **Precedence (serve config):** `--config` / `MAYA_CONFIG` →
`maya-<model>-hip.json` beside the repo root if that file exists → `maya-maya-s-v2-iq2_xxs-hip.json`.

When only the model/pack changes and no `--config` is given, serve launchers start from that base JSON and
rewrite `--glm-pack` (and `tokenizer` when set) to the selected pack. **Both hosts must have the same pack
tree** (index, dense weights, tokenizer, and the GGUF shards beside `pack/`).

Both serve rank scripts apply the base config's `env` to their engines, including prefill
sub-batches and hipBLASLt tuning. Config entries override inherited values, as on the server. Both aliases are exported; `MAYA_X` wins over `STRATA_X` when both are in the config.
Keep those entries consistent on both hosts so a slower worker does not hold up every collective.
The gfx1151 hipBLASLt 100401 table includes measured TP2 projection shapes. A controlled Maya-S
comparison improved 1K/4K/8K prompt throughput by 32–37%; see the
[prefill report](benchmarks/tp2-prefill-2026-10-31/RESULTS.md). Reproduce against an idle pair with
`.venv/bin/python tools/tp2/bench_prefill.py --output /tmp/prefill-results.json`.

### Per-script reference

| Script | Model flags | Notes |
|---|---|---|
| `tools/tp2/rank0.sh [PORT]` | `--model` `--pack` `--config` | API server on rank 0; env works too |
| `tools/tp2/rank1.sh` | `--model` `--pack` `--config` | worker on rank 1; start first |
| `tools/llama-swap/tp2.sh run PORT` | `--model` `--pack` `--config` (+ `--mtp` `--drafts` `--context`) | forwards selection to both ranks over SSH |
| `tools/tp2/run_bench.sh` | `--model` `--pack` | paired `--tokens` bench + parity (scps rank0’s save file) |
| `tools/tp2/run_prof.sh` / `run_noar.sh` | `--model` `--pack` | AR profile / no-AR isolation |
| `tools/tp2/acceptance.sh` | env only: `MAYA_MODEL` / `MAYA_PACK` / `MAYA_CONFIG` | forwards env to both rank launches |
| `tools/tp2/bench_mtp.py` | `--model` / `--pack` (or env) | MTP vs simple harness; `--output` required |
| `tools/calibrate_glm_tp.py` | `--pack` | end-to-end + microbench |

All of the above also accept the matching environment variables, which is how `acceptance.sh` and
the llama-swap wrapper propagate a choice to the remote rank.

### Examples

```sh
# --- pack selection (defaults to Maya-S-v2-IQ2_XXS) ---
bash tools/tp2/run_bench.sh                         # default Maya-S
bash tools/tp2/run_bench.sh --model Maya-M          # $MAYA_MODELS_DIR/Maya-M/pack
bash tools/tp2/run_bench.sh --pack /data/packs/mymodel
MAYA_MODEL=Maya-M bash tools/tp2/run_bench.sh

# --- serve one model on both ranks (rank 1 first) ---
bash tools/tp2/rank1.sh --model Maya-M
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya || exit; \
  setsid nohup bash tools/tp2/rank0.sh 18080 --model Maya-M \
    >/tmp/tp2-r0.log 2>&1 </dev/null &'
curl -s http://192.168.1.22:18080/health            # tp.peer: up

# --- serve with an explicit pack + a dedicated base config ---
bash tools/tp2/rank1.sh --config ./maya-maya-m-hip.json --pack /home/ugo/software/Maya-data/models/Maya-M/pack
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya || exit; \
  setsid nohup bash tools/tp2/rank0.sh 18080 --config ./maya-maya-m-hip.json \
    >/tmp/tp2-r0.log 2>&1 </dev/null &'

# --- llama-swap wrapper (run on rank 0 only; it SSHes to rank 1) ---
tools/llama-swap/tp2.sh run 18080 --model Maya-M --mtp --drafts 2 --context 32768
tools/llama-swap/tp2.sh stop

# --- acceptance / MTP harness ---
MAYA_MODEL=Maya-M MAYA_TP_MTP=1 bash tools/tp2/acceptance.sh
.venv/bin/python tools/tp2/bench_mtp.py --output /tmp/mtp-maya-m --model Maya-M --tokens 256 --runs 4
```

### If `pack/` is missing

The engine needs `<model>/pack/` (index, dense, tokenizer, `native_experts.txt`), not only GGUF shards.
Pack each host from shard 1 (few minutes; writes ~0.9 GiB beside the shards):

```sh
export STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py"
.venv/bin/python tools/iq_pack.py \
  --gguf /home/ugo/software/Maya-data/models/Maya-M/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf \
  --out  /home/ugo/software/Maya-data/models/Maya-M/pack \
  --compat-bf16
# repeat on the other host (or run the installer’s pack step / ./maya.sh --repack)
```

Before a live pair, confirm identical pack hashes on both hosts and `audit_split.py`:

```sh
.venv/bin/python tools/tp2/audit_split.py --world 2 \
  /home/ugo/software/Maya-data/models/Maya-M/pack    # expect: all splits block-safe
```

### Measured on this pair (Maya-M)

Controlled `run_bench.sh --model Maya-M` (2026-10-31, binary `5dbc9356…`, simple decode, 32 tokens):
**pair parity 32/32**, decode **42.57 / 42.83 ms/token** (rank0 / rank1), prefill 11 tokens ≈ 470 ms.

**Maya-S, Maya-M and Maya-L** passed these suites on earlier builds (S/M: 7-draft API matrices,
`mtp_acceptance`, CLI 32/32 both modes, kill/recover 8/8, `bench_mtp`). The promoted prefill build
was accepted in user llama-swap trials; its complete API/CLI regression was not repeated. Reports:
[Maya-M](benchmarks/tp2-maya-m-mtp-2026-10-31/RESULTS.md),
[Maya-L](benchmarks/tp2-maya-l-mtp-2026-10-31/RESULTS.md).
Maya-S raw MTP data remains in [tp2-mtp-review](benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md).
On Maya-L, rank0’s cold expert warm can exceed rank1’s by minutes — for **paired CLI** decode set
`STRATA_GLM_WARM=0` on both ranks (or wait out the warm); serve mode is unaffected.

## Changing the context or active model

The dashboard shows the running TP2 context as read-only. Its context API and active-model restart
controls reject changes that would restart only rank 0. A pair needs matching context/model selection
and a coordinated restart of both ranks (#53).

Stop both ranks, then relaunch with the same `CONTEXT` and model/pack/config selection on both rank
scripts, or use the rank-0 pair wrapper's `--context` / `--model` options above. The guard does not
change ordinary single-host reloads. The transient TP2 config's empty Models catalog is tracked
separately in #54; it does not affect API inference.

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
# Resolve the pack you want (or export MAYA_MODEL / MAYA_PACK)
P=/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack   # default; or Maya-M/pack

# Single-host reference
MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 ./build-hip/maya --glm-pack $P \
  --tokens 154822,25062,287,29905,371,25,7487,785,6722,315,9621,374 \
  --max-new 32 --greedy --max-context 4096

# TP2 pair (both get the same --tokens; no control-channel mirror)
bash tools/tp2/run_bench.sh                        # default Maya-S
bash tools/tp2/run_bench.sh --model Maya-M         # any pack under MAYA_MODELS_DIR
bash tools/tp2/run_bench.sh --pack "$P"
```

`run_bench.sh` starts rank 0 on `strix-halo-1.local` via SSH (binary `/tmp/maya-tp2`) and rank 1
locally, then **scps rank 0’s `--spec-follow-save` file** back before `greedy_parity.py`
(compare on the rank that wrote the file — a stale local `r0.txt` produces a false FAIL).

## Production launch (serve mode)

**Start rank 1 first**, then rank 0. Never probe port 18515 with `/dev/tcp` — that completes a
TCP connection and rank 0’s single `accept()` takes it as the peer (handshake fails EAGAIN/EISDIR).
The engine already retries connect for `MAYA_TP_BOOTSTRAP_TIMEOUT` (default 120 s).

```sh
# Rank 1 worker (strix-halo-2) — no --serve, no --tokens
bash tools/tp2/rank1.sh            # optional: --model Maya-M | --pack PATH | --config PATH

# Rank 0 API server (strix-halo-1) — via SSH; server.py spawns the engine with --serve
ssh strix-halo-1.local 'cd /home/ugo/software/project-maya || exit; setsid nohup bash tools/tp2/rank0.sh 18080 \
  > /tmp/tp2-rank0.log 2>&1 </dev/null &'
# Pass the same model flags on rank 0 as on rank 1, e.g. --model Maya-M on both.

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
Partition widths (`n_ff_exp`, `n_ff_dense`) are **read from the pack’s GGUF** before bootstrap (issue #12); the engine refuses a pack whose widths do not split under TP. See `docs/SETTINGS.md` for `MAYA_TP_*`.

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

`tp.world_size` / `tp.rank` come from the engine INFO line. **`tp.peer`** is `up` while the pair is healthy.
The first fatal transport error or pair-down notice sets it to `down` and sets
`status` to **`degraded`**. The first cause stays in `tp.error`. Restart **both**
ranks to recover (`rank1.sh` then `rank0.sh`).

Acceptance: `tools/tp2/acceptance.sh` (pair up, chat, kill rank1 → degraded, restart → recover).
Select the model with env vars (the script does not take `--model`):

```sh
MAYA_MODEL=Maya-M bash tools/tp2/acceptance.sh
MAYA_PACK=/path/to/pack MAYA_TP_MTP=1 bash tools/tp2/acceptance.sh
```

### Opt-in MTP

The simple decode loop remains the default. Set `MAYA_TP_MTP=1` on rank 0's server to use MTP;
leave `MAYA_GLM_NO_MTP` unset. For paired CLI requests, set `MAYA_TP_MTP=1` on both ranks.
MTP uses greedy verification with a fixed draft width (default 2); rank 0 sends anchor/draft token IDs,
accepted tokens, and the continue/stop decision over the bootstrap control channel. Tensors stay on RDMA.
Thinking-budget forces and client cancellation use this same round protocol. Both hosts need matching binaries.
For a numeric reasoning cap independent of context length, see
[Limiting GLM thinking](SETTINGS.md#limiting-glm-thinking-for-a-coding-agent).

The [corrected resident-pair review](benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md) measured
25.82 tok/s ordinary versus 28.60 with one draft on the technical prompt (+10.8%), and 25.86 versus
33.53 with two drafts on code (+29.6%). Seven drafts slowed the technical workload. The older
[MTP comparison](benchmarks/tp2-mtp-2026-10-30/RESULTS.md) predates numerical fixes and is historical.
Re-run with `.venv/bin/python tools/tp2/bench_mtp.py --output /tmp/mtp-comparison` on reserved free hosts
(see [process ownership](#process-ownership-and-validation)).
Add `--model Maya-M` / `--pack PATH` (or `MAYA_MODEL` / `MAYA_PACK`) to target another pack; keep
`--output` unique per run.

Validate MTP correctness with:

```sh
# Against a running MTP-enabled pair: 3 turns with reuse, cancellation, and small output limits
.venv/bin/python tools/tp2/mtp_acceptance.py
# Starts its own pair and checks peer death/restart with MTP enabled
MAYA_TP_MTP=1 MAYA_MODEL=Maya-M bash tools/tp2/acceptance.sh
```

A malformed or interrupted MTP frame stops the worker; rank 0 marks the pair down on decode failure.
Restart both ranks after a protocol or RDMA failure.

### llama-swap on strix-halo-1

The installed `~/.config/llama-swap/config.yaml` (on **strix-halo-1**) uses
`tools/llama-swap/tp2.sh run ${PORT} …` and the matching `stop` command. The wrapper starts
`tools/tp2/rank1.sh` over SSH on strix-halo-2, then runs `tools/tp2/rank0.sh` locally in the foreground.
`tools/tp2/tp2-swap.sh` delegates to this same wrapper.

**Rank0 must have model-aware launchers** (`tools/llama-swap/tp2.sh`, `tools/tp2/model_env.sh`,
`rank0.sh`, `rank1.sh`). The protected canonical checkout is `777cc92`; model-aware launcher files were
deployed separately. Do not pull/reset it or copy files as routine validation setup. Use isolated
review trees when testing newer launchers; a production deployment is a separate requested action.

The user-configured entries preserved at the integration checkpoint were as follows; this is a
configuration snapshot, not an instruction to inspect or manage the user's running service:

```yaml
models:
  maya-glm5-s-tp2:
    cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT} --model Maya-S-v2-IQ2_XXS --no-mtp
    cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
    proxy: http://127.0.0.1:${PORT}
  maya-glm5-s-tp2-mtp:
    cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT} --model Maya-S-v2-IQ2_XXS --mtp --drafts 2
    cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
    proxy: http://127.0.0.1:${PORT}
  maya-glm5-m-tp2:
    cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT} --model Maya-M --no-mtp
    cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
    proxy: http://127.0.0.1:${PORT}
  maya-glm5-m-tp2-mtp:
    cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT} --model Maya-M --mtp --drafts 2
    cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
    proxy: http://127.0.0.1:${PORT}
  maya-glm5-l-tp2:
    cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT} --model Maya-L --no-mtp
    cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
    proxy: http://127.0.0.1:${PORT}
  maya-glm5-l-tp2-mtp:
    cmd: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh run ${PORT} --model Maya-L --mtp --drafts 2
    cmdStop: /home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
    proxy: http://127.0.0.1:${PORT}
```

Both packs must exist on both hosts (`…/Maya-S-v2-IQ2_XXS/pack`, `…/Maya-M/pack`). `--watch-config`
reloads on edit. Manual examples **on strix-halo-1**:

```sh
/home/ugo/software/project-maya/tools/llama-swap/tp2.sh run 18080 --model Maya-M --mtp --drafts 2
/home/ugo/software/project-maya/tools/llama-swap/tp2.sh stop
```

`--no-mtp` uses the simple loop; no flag preserves the default off (or `MAYA_TP_MTP` from the environment).
`--drafts N` accepts 1–8, defaults to 2, and forwards both `MAYA_GLM_MTP_FIXED=N` and
`MAYA_GLM_MTP_DRAFT=N` to both hosts. `--context N` also sets the same context on both ranks;
otherwise the context comes from the base Maya JSON config. `--model NAME` / `--pack PATH` / `--config PATH`
select the model pack (forwarded to both ranks). One sequence is used while #11 remains gated.

For direct engine/manual rank launches, put these environment assignments on **both** ranks before the command:

```sh
MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=4 MAYA_GLM_MTP_DRAFT=4 <engine-or-rank-script> ...
```

There is no separate MTP build switch: the HIP `maya` target includes it. The `--mtp` and `--drafts` options
above belong to the llama-swap wrapper; direct engine invocations use the environment variables.
Unset `MAYA_GLM_NO_MTP` / `MAYA_GLM_NO_SPEC` (and their `STRATA_` aliases) when launching manually.
MTP verification currently uses greedy sampling under TP2.

## Failure playbook

| Symptom | Cause | Fix |
|---|---|---|
| `bootstrap accept timed out` | Rank 1 never connected | Start rank 1; check `usb4_rdma*` on both hosts |
| `bootstrap receive: Resource temporarily unavailable` / `Is a directory` | Something stole accept (`/dev/tcp` probe) | Remove port probes; restart both |
| `bootstrap connect: Operation now in progress` | Rank 0 not listening yet | Wait; engine retries 120 s (`MAYA_TP_BOOTSTRAP_TIMEOUT`) |
| `TP2 mirror send: Broken pipe` | Rank 1 died | Restart **both**; `/health` → `status=degraded`, `tp.peer=down` |
| Request: `TP2 pair down` | Peer already lost | Restart both ranks (engine sticky until restart) |
| `TP2 pair down: verbs communicator is poisoned…` | A fatal collective failure was detected before a mirror send | Health reports degraded/down on this first failure; preserve its cause and restart both ranks. Never reuse or clear the poisoned communicator (#88). |
| Request hangs after “reading the prompt” | Rank 1 not in worker mode / AR mismatch | Kill both; rank 1 without `--tokens` must log “mirroring” |
| `ibv_post_send failed` | RDMA link down | `ip link show`, reload `thunderbolt_ibverbs` |
| Pair parity < 100% | Slicing bug | `greedy_parity.py`; try `MAYA_TP_NOSLICE=1` |

## Historical performance reference (before numerical fixes)

The early figures below describe the 2026-10-29/30 implementation. Use the
[corrected review](benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md) and
[current README measurements](../README.md#measured-performance) for performance comparisons.

Controlled bench (no-MTP, 32 tokens, same prompt, warm engine), 2026-10-29/30 — **Maya-S**:

| Config | ms/tok | vs single |
|---|---|---|
| single-host | **55.76** (56.81 on 10-29) | — |
| TP2 full slice, no AR | 36.57 | +35.6% |
| TP2 full slice + AR (free greedy) | ~43 | +23% |
| TP2 full slice + AR (spec-follow) | **41.1** | **+26%** |

**Quality (Maya-S, 2026-10-30):**
- **Pair parity 100%** (rank0 vs rank1 free generation, 32/32).
- **Spec-follow vs single-host continuation:** output stream 32/32 forced; **4 of 32** positions where TP2’s own greedy pick differs (near-ties / float reduction order) — same class as the earlier 31/32.

**Maya-M** (`run_bench.sh --model Maya-M`, 2026-10-31): pair parity **32/32**, decode **42.57 / 42.83 ms/token**
(r0/r1). Not an MTP or serve-API claim; single-host Maya-M on the same prompt streams experts from disk and is
not a parity oracle.

AR cost ≈ 5 ms/tok (90 exchanges × ~55 µs). Link is fine (6 µs RTT, 37 Gb/s at 1 MiB); blocking AR is host/driver-bound.

End-to-end + microbench: `.venv/bin/python tools/calibrate_glm_tp.py --pack …`.

**Latest integration CI:** HIP mapping, 605 server tests (11 skips) and 679 tools tests (33 skips)
passed at the v1.1.46 checkpoint. See the [completion report](upstream/COMPLETION-2026-11-02.md) for
exact-head/main CI and the separate unresolved DeepSeek GPU limitations.

## Monitoring

- `STRATA_GLM_PROF=1`: per-stage GPU timing (`tp_ar_mixer`, …)
- `STRATA_GLM_TIMING=1`: ms/token summary
- `--spec-follow-save PATH`: save generated tokens for parity
- `tools/tp2/greedy_parity.py A B --min-agree 1.0`

## Calibration / microbench

```sh
# rebuild the AR microbench (needs STRATA_ENABLE_TP2_RDMA=ON in that build dir)
cmake -S . -B build-tp2 -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_ENABLE_TP2_RDMA=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-tp2 --target tp_comm_rdma_test -j"$(nproc)"

# both hosts; rank 0 first
ssh strix-halo-1.local '/tmp/tp_comm_rdma_test --rank 0 --bootstrap-port 18515 \
  --rdma-device usb4_rdma5 --exchanges 20 --bench --hidden 12288' &
./build-tp2/tp_comm_rdma_test --rank 1 --bootstrap-host strix-halo-1.local \
  --bootstrap-port 18515 --rdma-device usb4_rdma5 --exchanges 20 --bench --hidden 12288
```

`--hidden 12288` = Maya decode AR (48 KiB/row). `--hidden 2560` = gufo Flash-Next geometry.

### Cancelling a prompt and changing the pair

A TP2 prompt cancellation completes the current mirrored prefill call on both ranks before resetting them. The client can disconnect immediately; the server drains that call before admitting another request. This can take as long as the current call, and avoids leaving one rank inside abandoned collectives. Cancelled-checkpoint restoration remains single-host-only, even with `MAYA_GLM_KEEP_CANCELLED=1`. Live regression: `tools/tp2/prefill_cancel.py` (both simple and MTP).

Model load/unload, idle unload and automatic dead-engine restart are guarded under TP2, like context/current-model changes. Stop and relaunch both ranks with matching model/context options through the pair launcher. The single-host lifecycle APIs retain their normal behavior.

## Unsupported local-only operations

TP2 refuses images/videos before encoding or lending the expert cache. Direct `GENI`, `VLEND`, `VRECLAIM` and `LOGP` commands are rejected before state changes because no paired operation exists. A vision config is ignored for a TP2 launch, keeping text inference available without starting an encoder. RDMA embedding synchronization remains deferred in #16.

To stress idle wake-up (#64), configure BOTH ranks with `MAYA_GLM_SERVICE_IDLE_MS=10`, `MAYA_GLM_SERVICE_DEEP_S=1`, `MAYA_GLM_SERVICE_DEEP_MS=3000`, then run `tools/tp2/regression_collectives.py --mode mtp --idle-seconds 3 --max-long-decode-ratio 4 --output /tmp/idle-wake.json`. The optional bound rejects repeated deep-sleep delays in decode after the long prefill; prefill time is excluded. Restore normal settings afterward. The regression keeps the ordinary correctness/reuse/stream/peer checks.

## Continuation boundary probes

On a matching4096-context pair with MTP enabled, `tools/tp2/continuation_boundaries.py --eos-id ID --output PATH` verifies unsupported MTP-bias rejection, supported simple-loop EOS continuation at the context boundary, and normal MTP follow-up. Use the EOS id from the selected pack. Its `--forced-follow` mode requires the explicit EOS follow-token diagnostic on both ranks and tests MTP/simple continuation branches; forced-token results are not numerical-fidelity evidence.

TP2 MTP currently enforces greedy verification and rejects nonempty `logit_bias` (#69). For ordinary decoding with bias, use `strata_tune: {"spec": 1}`. No MTP sampling algorithm was changed.

For long-loop cache validation on a fresh matching32768-context pair, run `tools/tp2/long_loop_reuse.py` with `--expect-preserved` at the default collapse limit32000, or `--expect-collapsed` with config limit0. Synthetic tool history is input data; no tool executes. Keep separate output paths for both arms.
