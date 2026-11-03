# DeepSeek TP2 review guide

DeepSeek two-host execution is **experimental and under validation** on this
branch. The established GLM production setup is separate; this guide does not
replace it. The example starts processes explicitly and writes a dedicated
server config. It does not change llama-swap or any existing config.

The implementation runs `maya-ds4` on two gfx1151 HIP hosts, with one GPU per
host and `usb4_rdma*` for tensor traffic. Each host needs the same DeepSeek GGUF
model and pack, a matching TP2-enabled executable, sufficient memory, and a
local identity manifest. The manifest hashes the pack, Engram, tokenizer and
indexed source files; its first creation can read hundreds of gigabytes. Use
separate manifests on the two hosts, because they also record local file
metadata. If a source changes, regenerate the affected manifest and confirm
that both hosts still have matching model contents.

Build matching review executables with `STRATA_ENABLE_HIP=ON`,
`STRATA_ENABLE_CUDA=OFF`, `STRATA_PREFILL_MMQ=ON`,
`STRATA_ENABLE_TP2_RDMA=ON` and `CMAKE_HIP_ARCHITECTURES=gfx1151`; build the
`maya-ds4` target. Do not overwrite the current production executable for a
review run. The same executable digest is required on both ranks.

On **each host**, choose its local pack, executable and a new manifest path:

```sh
TP2_PACK=/absolute/path/to/DeepSeek-model/maya-ds4
TP2_ENGINE=/absolute/path/to/review/maya-ds4
TP2_MANIFEST=/absolute/path/to/deepseek-tp2-manifest.json
.venv/bin/python tools/ds4/tp2_identity.py create --pack "$TP2_PACK" --manifest "$TP2_MANIFEST"
```

DeepSeek TP2 ranks read existing HIP Lt picks but do not append new picks to
the tuning file. Earlier review builds and single-host DeepSeek can append
to the pack's `lt-picks.txt` even with `--profile-readonly`. To isolate those
builds, give each host a **new** local tuning file:

```sh
TP2_LT_STORE=/tmp/new-deepseek-tp2-lt-picks.txt
if [ -f "$TP2_PACK/lt-picks.txt" ]; then cp -n "$TP2_PACK/lt-picks.txt" "$TP2_LT_STORE"; fi
export MAYA_DS4_LT_STORE="$TP2_LT_STORE"
```

The two hosts may use different local paths. When running the API server,
export this variable in the shell that starts it so its engine inherits it.

Choose rank 0's reachable control address as `TP2_HOST`, a free bootstrap `TP2_BOOTSTRAP_PORT`,
the local RDMA device as `TP2_RDMA_DEVICE`, the same `TP2_CONTEXT` and per-rank expert budget
in MiB as `TP2_BUDGET_MIB`. Both ranks must select the same model contents, executable,
context, DSpark, prefill and lookup options. The two local pack, manifest,
executable and RDMA device names may differ. The example explicitly enables
256-row experimental chunked prefill and DSpark; prompt lookup defaults off.
For example, set these on each host using the values for your pair:

```sh
TP2_HOST=rank0.example.local
TP2_BOOTSTRAP_PORT=18700
TP2_RDMA_DEVICE=usb4_rdma5
TP2_CONTEXT=32768
TP2_BUDGET_MIB=92160
TP2_API_PORT=18080  # rank 0 only
```

On rank 1, start the worker and keep its terminal and log:

```sh
.venv/bin/python tools/ds4/tp2_rank.py --rank 1 \
  --pack "$TP2_PACK" --manifest "$TP2_MANIFEST" --engine "$TP2_ENGINE" \
  --host "$TP2_HOST" --port "$TP2_BOOTSTRAP_PORT" --rdma-device "$TP2_RDMA_DEVICE" \
  --context "$TP2_CONTEXT" --expert-budget-mib "$TP2_BUDGET_MIB" \
  --prefill-chunk 256 --dspark on --suffix-draft 0
```

On rank 0, choose a **new** config path and log path, generate a config, then
start the API server. Set `TP2_API_PORT` independently of the bootstrap port.

```sh
TP2_CONFIG=/absolute/path/to/new-deepseek-tp2-config.json
TP2_ENGINE_LOG=/absolute/path/to/new-deepseek-tp2-engine.log
.venv/bin/python tools/ds4/tp2_config.py --engine-log "$TP2_ENGINE_LOG" \
  --pack "$TP2_PACK" --manifest "$TP2_MANIFEST" --engine "$TP2_ENGINE" \
  --host "$TP2_HOST" --port "$TP2_BOOTSTRAP_PORT" --rdma-device "$TP2_RDMA_DEVICE" \
  --context "$TP2_CONTEXT" --expert-budget-mib "$TP2_BUDGET_MIB" \
  --prefill-chunk 256 --dspark on --suffix-draft 0 > "$TP2_CONFIG"
.venv/bin/python -m serve.server --engine maya --config "$TP2_CONFIG" --port "$TP2_API_PORT"
```

Optionally add `--model-label 'DeepSeek V4.1 · Maya-XS'` (or the correct label
for the selected pack) when generating the config to show that name in health
and metrics. This label is descriptive; the checked manifest binds the actual
pack contents.

The generated config refers to `tp2_rank.py`, which rechecks the local manifest
and hashes the executable each time it launches. `/health` reports the peer
and negotiated prefill chunk. A fatal pair error makes health degraded and
retains its first cause in `tp.error`; stop the server and worker, then start
**both** afresh, worker first. Never attempt to reuse a poisoned communicator.

`--prefill-chunk 0` selects the window-only path; `256` selects the bounded
chunk path. `--dspark off` disables the drafter on both ranks. Experimental
prompt lookup is opt-in with `--suffix-draft 3` on both ranks; the launcher
keeps lookup ownership on rank 0. Rank choices must match even though the
worker does not run the drafter or lookup itself. Vision, parallel requests,
and performance at a filled 512K context are not validated for this path.
The resident expert split does not imply that Engram and all model I/O have
been eliminated. See the [validation evidence](validation/deepseek-tp2-control-2026-11-03/)
and [implementation plan](DEEPSEEK-TP2-IMPLEMENTATION-PLAN.md) for current
gates and limitations.
