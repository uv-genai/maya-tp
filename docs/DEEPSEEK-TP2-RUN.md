# DeepSeek TP2 and DSpark guide

DeepSeek Maya-XS and Maya-S have paired gfx1151 validation and successful user
trials. The user reports hours of working DeepSeek and GLM inference, including
DeepSWE benchmarks, and accepted integration into `main`. Numerical and
peak-capacity follow-ups are tracked in #110/#112. DeepSeek uses `maya-ds4`
and its own launcher configs. The manual examples below create a dedicated
server config; the final section covers the SSH-managed llama-swap wrapper.

PR #122 merged the implementation into `main` at `75e18e8`. Feature-head and
post-merge CI passed. [Merged pull request](https://github.com/uv-genai/maya-tp/pull/122).

The source checkpoint through upstream v1.1.53 is `de184d9`, with final main CI
41834772034 passing. Per-tag validation used isolated GLM Maya-S and DeepSeek
XS/S builds. Accepted runtime paths are recorded below, and the full source
checkpoint is in [Handoff §47](../Handoff.md#47-sequential-upstream-v1150v1153-integration).

## Accepted runtime on the user's pair

Start the existing `maya-deepseek-xs-mtp` or `maya-deepseek-s-mtp` llama-swap
entry on strix-halo-1.local. The wrapper starts rank 1 on strix-halo-2.local.
Both hosts use the runtime at
`/home/ugo/software/project-maya-ds4-tp2-review` and the versioned executable
`build-hip/queued-a950e7e9350b/maya-ds4` within it. Its SHA-256 is
`a950e7e9350be4fa49a7063c67ef901ebbc1bd2a9b583d07250eb016bcf72102`.
The directory name reflects its original staging role. This is the build
accepted in the user's extended trials.

The per-model launcher files are `maya-ds4-tp2-maya-xs.json` and
`maya-ds4-tp2-maya-s.json`. They select distinct packs and local manifests,
context 262144, DSpark on, prefill chunk 256, lookup off, bootstrap port
18525 and RDMA device usb4_rdma5. The per-rank expert budget is 92160 MiB
with reserve 1024 MiB and 14 CPU threads. Run one model pair at a time.
Use `main` for source development and this section for the accepted runtime
selection. The following manual steps also support setting up a new pair.

A fatal GPU-state `END` notice marks the pair down and retains the first failure
cause. Recovery uses the pair launcher to restart both ranks together. The
updated Models/Settings context controls retain the paired restart guard.

## Prepare a new pair

The implementation runs `maya-ds4` on two gfx1151 HIP hosts, with one GPU per
host and `usb4_rdma*` for tensor traffic. Each host needs the same DeepSeek GGUF
model and pack, a matching TP2-enabled executable, sufficient memory, and a
local identity manifest. The manifest hashes the pack, Engram, tokenizer and
indexed source files; its first creation can read hundreds of gigabytes. Use
separate manifests on the two hosts, because they also record local file
metadata. If a source changes, regenerate the affected manifest and confirm
that both hosts still have matching model contents.

New manifests record the stable filesystem ID alongside local file metadata.
On Btrfs it includes the subvolume identity, so a device-number change after
remount or reboot can pass when the filesystem ID, inode, size and nanosecond
modification/change times still match. Hashing still rejects files that change
during the read, and startup rehashes the pack metadata.

Older manifests without a filesystem ID retain their strict device-number
check. If they report `identity input changed` after a remount, create a new
manifest at a fresh path on each affected host using the command below. Compare
its printed model digest with the previous manifest and the peer before updating
the launcher's `manifest` path. This performs a full content hash once. The
identity helper on both hosts must support the new local metadata fields.

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
256-row bounded chunked prefill and DSpark. Prompt lookup defaults off.
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
worker does not run the drafter or lookup itself. Vision and performance at a filled 512K context are not validated for this
path. The accepted deployed build serves one sequence; new opt-in batching
requires the current matching source/engine described below.
The resident expert split does not imply that Engram and all model I/O have
been eliminated. See the [validation evidence](validation/deepseek-tp2-control-2026-11-03/)
for the measured scope. The [implementation plan](DEEPSEEK-TP2-IMPLEMENTATION-PLAN.md)
records the original phase gates.

## Promotion and rollback

The queued build above has user acceptance, and operational validation #116 is
closed. Numerical/capacity follow-ups are tracked in #110/#112. Use the
procedure below when selecting a future versioned build or restoring a previous
pair configuration. The accepted executable and config paths are recorded above.

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
`--suffix-draft 0|3` and `--parallel 1..4` apply to both ranks. `--peer HOST` overrides the SSH
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
  maya-deepseek-xs-mtp:
    cmd: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh run ${PORT} --model Maya-XS
    cmdStop: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh stop-pid ${PID}
    checkEndpoint: /health
  maya-deepseek-s-mtp:
    cmd: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh run ${PORT} --model Maya-S
    cmdStop: /absolute/path/to/project-maya/tools/llama-swap/tp2-ds4.sh stop-pid ${PID}
    checkEndpoint: /health
```

Use the two entries with llama-swap's normal one-model-at-a-time behavior.
The accepted configs use bootstrap port 18525, leaving the established GLM
bootstrap selection separate. The user controls loading and inference testing.

The accepted 2026-11-04 configs select the queued-collective build on both
hosts. XS measured 23–24% higher generation throughput than the same single-host
build on two prompts at 262144 context. The user reports Maya-S eBPF generation
slightly above 30 tok/s with prefill similar to XS and a larger relative speedup.
[Measurements and validation scope](benchmarks/deepseek-tp2-queued-2026-11-04/README.md).


## Opt-in continuous batching

Current source adds checked 2–4-slot TP2 serving for XS/S, including real DSpark
proposals and per-request commits. Keep `parallel` at its default 1 for the
accepted single-sequence flow. The versioned accepted executable documented
above predates this option; select a matching newly built `maya-ds4` on both
hosts and matching launcher source before opting in. Model manifests remain
valid: slot/context/speculation choices have a separate execution identity.

Launch from the API host only, using a dedicated port/config while comparing:

```sh
./tools/llama-swap/tp2-ds4.sh run 18080 --model Maya-XS --parallel 4 \
  --engine /absolute/path/to/new/maya-ds4 \
  --peer-root /absolute/path/to/matching/peer/source
./tools/llama-swap/tp2-ds4.sh stop 18080
```

`--engine` requires that explicit path on both hosts. For different local
executable paths, put them in the launcher's `rank0`/`rank1` objects and omit
that override. `rank0.sh`, `rank1.sh` and `tp2_config.py` also accept
`--parallel`; use the same count on both. A shared JSON `"parallel": 4`
provides the persistent equivalent. Preserve the existing pack/context,
manifest, bootstrap and RDMA selections. An old peer launcher fails preflight
rather than silently changing batch semantics.

Batch mode requires lookup off (`--suffix-draft 0`). DSpark drafting uses
argmax point proposals on rank zero in fitting groups, so all four eligible
clients can draft even when their combined drafter block exceeds scratch.
Target windows remain bounded to 16 rows. Temperature/top-p/top-k/min-p,
per-request seed, bias and penalties apply to target selection. The optional
`DS4_DSPARK_SAMPLE=1` proposal-distribution mode is rejected for batching;
the default single-sequence implementation retains it. A request can disable
DSpark with `strata_tune: {"spec": 0}`. `--dspark off` disables it globally.

Admission, cancellation, EOS/output cuts and slot reuse occur between complete
mirrored transactions. Each speculative verdict covers every member before
reuse, and leaves the final emitted token pending. Prompt pieces stay bounded
to one negotiated chunk even when only one request is active. Exact in-place
continuations reuse committed tokens; file slots and checkpoint restoration
remain outside TP2 support. Four full 262144-token XS/S slots passed the D1
capacity gate with complete target-expert residency. Longer filled contexts
and operational user trials remain separate checks.

Use `tools/ds4/validate_batch.py` for owned isolated model/lifecycle validation;
it records PIDs/start identities and stops only its own pair. Shared HTTP
measurements use `tools/bench_concurrency.py`; DeepSeek `--spec 0` disables
DSpark, whereas GLM uses 1 to disable proposals. Raw throughput, numerical
checks and rollout limits are recorded in the D5 validation report when that
checkpoint completes. Production configs and executables are not changed by
source integration.
