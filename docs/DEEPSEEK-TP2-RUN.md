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

## Promotion and rollback

The candidate remains experimental while the numerical review in #110 and the
remaining validation in #116 are open. The user should review and test it
before deciding whether to promote it. The steps below are a procedure for that
decision; they do not change the active setup by themselves.

Before a change, record the exact commands and environment used to start both
previous ranks, including their binary paths and hashes, local pack and manifest
paths, context, RDMA device, bootstrap port and feature settings. Preserve the
previous server config at a separate path and keep the previous executables and
packs available on both hosts. Check that the two previous ranks' selections
match. For a single-host DeepSeek setup, preserve its original server config,
executable and launch command instead. GLM has separate executables and configs.

After the user accepts the candidate, install a **versioned copy** on each host,
using a directory dedicated to locally managed releases. This example refuses to
reuse a filename containing different bytes. Set `TP2_ENGINE` independently to
the reviewed local executable on each host, and choose the same release directory
convention; the directory itself may be different on the two hosts.

```sh
TP2_RELEASE_DIR=/absolute/path/to/deepseek-releases
stage_tp2_release() {
  TP2_SHA=$(sha256sum -- "$TP2_ENGINE") || return 1
  TP2_SHA=${TP2_SHA%% *}
  mkdir -p -- "$TP2_RELEASE_DIR" || return 1
  TP2_VERSIONED_ENGINE="$TP2_RELEASE_DIR/maya-ds4-$TP2_SHA"
  if test -L "$TP2_VERSIONED_ENGINE"; then
    printf '%s\n' 'Release path is a symlink; stop.' >&2
    return 1
  fi
  if test -e "$TP2_VERSIONED_ENGINE"; then
    TP2_EXISTING_SHA=$(sha256sum -- "$TP2_VERSIONED_ENGINE") || return 1
    test "${TP2_EXISTING_SHA%% *}" = "$TP2_SHA" || {
      printf '%s\n' 'Existing release file has the wrong hash; stop.' >&2
      return 1
    }
  else
    install -m 0755 -- "$TP2_ENGINE" "$TP2_VERSIONED_ENGINE" || return 1
  fi
  TP2_INSTALLED_SHA=$(sha256sum -- "$TP2_VERSIONED_ENGINE") || return 1
  test "${TP2_INSTALLED_SHA%% *}" = "$TP2_SHA" || {
    printf '%s\n' 'Installed release hash does not match; stop.' >&2
    return 1
  }
  printf 'Release SHA-256: %s\nBinary: %s\n' "$TP2_SHA" "$TP2_VERSIONED_ENGINE"
}
stage_tp2_release
```

Compare the printed SHA-256 on both hosts with the reviewed build's hash. Set
`TP2_ENGINE` to the versioned path on **each** host, and generate a **new**
server config using the earlier instructions, retaining the previous config
unchanged. Record the new worker command and config path with the release hash.
Both ranks must use the same pack contents, context and feature selections;
local paths and RDMA device names can differ. The user then stops their old
server and worker in their usual controlled manner and starts the new worker,
followed by the server, as shown above. Verify `/health` reports the peer up,
the expected chunk setting and the selected model, then send a test request.

To roll back, the user stops the new server first and then its worker. Start the
previous configuration using the recorded commands, original binary paths and
matching rank settings; for a prior single-host setup, start its original
server command and config. Verify health and a test request. Do not mix a new
rank with an old rank or reuse a poisoned pair. Keep the versioned candidate and
its config separate until the user has finished comparing or troubleshooting.

## Rank scripts and llama-swap

`tools/ds4/rank0.sh [PORT]` starts the HTTP API and rank-0 engine. Start
`tools/ds4/rank1.sh` on the worker host first. Both scripts select a launcher
JSON with `--config PATH` or `--model NAME`. For example, `--model Maya-XS`
selects `maya-ds4-tp2-maya-xs.json` beside the repository root. With neither
option, `MAYA_DS4_TP2_CONFIG` supplies the path, falling back to
`maya-ds4-tp2.json`. Keep one local checked manifest per model on each host.

The example config gives the schema. Paths to pack, manifest and engine are
resolved relative to that config file. Common model, context, bootstrap and
feature settings apply to both ranks. Optional `rank0` and `rank1` objects
can override local paths, `rdma_device`, `gpu`, `cpu_threads`,
`expert_budget_mib`, `reserve_mib` and `env`. Use absolute rank-1 paths when
the peer has a different directory layout. `peer_root` selects the peer's
runtime checkout and `peer_python` optionally selects its Python executable.
[Example configuration](../tools/ds4/tp2-launch.example.json).

```sh
# On rank 1
./tools/ds4/rank1.sh --model Maya-XS
# On rank 0
./tools/ds4/rank0.sh 18080 --model Maya-XS
```

For an SSH-managed pair, run the single wrapper on the API host. It checks
both ranks' pack, partition and executable identities before starting model
loading, then starts the worker and API supervisor. It returns an error if
another pair owns its API or bootstrap port. A worker failure ends the pair.
A new launch starts both ranks afresh.

```sh
./tools/llama-swap/tp2-ds4.sh run 18080 --model Maya-XS
./tools/llama-swap/tp2-ds4.sh stop 18080
```

Overrides `--context N`, `--dspark on|off`, `--prefill-chunk 0|256` and
`--suffix-draft 0|3` apply to both ranks. `--peer HOST` overrides the SSH
host. The foreground supervisor handles SIGTERM, SIGINT and SIGHUP. Shutdown
ends the API supervisor, cleans up its recorded child engines, then closes
the remote worker's control pipe. The remote supervisor also stops its worker
when that pipe closes. Cleanup checks recorded process start times and
executable paths. Runtime logs and ownership records are stored under
`~/.cache/maya/deepseek-tp2/PORT/run-*`, with a `latest` file giving the current
run directory. `--state-dir PATH` selects another runtime-state root and must
be passed to the corresponding stop command.

For llama-swap, `${PORT}` belongs in `cmd` and `${PID}` identifies the
foreground supervisor in `cmdStop`. The PID stop action verifies its recorded
identity and requests shutdown through that pair's control socket. These macros
are documented by llama-swap. [llama-swap command documentation](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/writing-cmd.md).

```yaml
healthCheckTimeout: 1800
models:
  deepseek-maya-xs-tp2:
    cmd: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh run ${PORT} --model Maya-XS
    cmdStop: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh stop-pid ${PID}
    checkEndpoint: /health
  deepseek-maya-s-tp2:
    cmd: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh run ${PORT} --model Maya-S
    cmdStop: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh stop-pid ${PID}
    checkEndpoint: /health
```

Use the two entries with llama-swap's normal one-model-at-a-time behavior.
The review configs use bootstrap port 18525, leaving the established GLM
bootstrap selection separate. The user controls loading and inference testing.
