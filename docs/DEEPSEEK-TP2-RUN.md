# DeepSeek TP2 and DSpark guide

DeepSeek Maya-XS and Maya-S have paired gfx1151 validation and successful user
trials. The user reports hours of working DeepSeek and GLM inference, including
DeepSWE benchmarks, and accepted integration into `main`. Numerical and
peak-capacity follow-ups are tracked in #110/#112. DeepSeek uses `maya-ds4`
and its own launcher configs. The manual examples below create a dedicated
server config; the final section covers the SSH-managed llama-swap wrapper.

PR #122 merged the implementation into `main` at `75e18e8`. Feature-head and
post-merge CI passed. [Merged pull request](https://github.com/uv-genai/maya-tp/pull/122).

The source includes sequentially validated upstream v1.1.54–v1.1.58 through
[PR #168](https://github.com/uv-genai/maya-tp/pull/168), merged as `b7458c4`.
[#169](https://github.com/uv-genai/maya-tp/issues/169) installs the validated
engines at existing executable paths, without changing launchers/configs.
The current user-accepted installation is recorded below and in Handoff §55.

<a id="accepted-runtime-on-the-users-pair"></a>

## Current runtime on the user's pair

Start the existing `maya-deepseek-xs-mtp` or `maya-deepseek-s-mtp` llama-swap
entry on strix-halo-1.local. The wrapper starts rank 1 on strix-halo-2.local.
Both hosts use the runtime at
`/home/ugo/software/project-maya-ds4-tp2-review` and the versioned executable
`build-hip/queued-a950e7e9350b/maya-ds4` within it. At the user's request this
same file path now contains the validated v58 engine, SHA256
`17f42fa16d52dd59c2843138a23a2a2dbcbecc40826d623272e2456f82a11076`.
The directory name is historical: it no longer names the contents' hash. No
launcher/config path was changed. The user confirmed the installed builds are
working fine on 2026-11-11; this installation is accepted (#169).
An already-loaded process retains the previous executable until the user
unloads/reloads it. Original files are backed up on each host under
`/home/ugo/software/maya-binary-backups/issue-169-20261111-v58/`.
[Deployment evidence](deployments/v58-existing-paths-2026-11-11/README.md) records
exact paths, backups and unchanged launcher/config fingerprints. Model manifests
were unchanged; the launcher hashes the executable afresh on each start.

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
path. The preserved deployed configuration selects one sequence. The installed v58
engine also supports opt-in batching, which requires matching current server/
launcher source and an explicit parallel setting as described below.
The resident expert split does not imply that Engram and all model I/O have
been eliminated. See the [validation evidence](validation/deepseek-tp2-control-2026-11-03/)
for the measured scope. The [implementation plan](DEEPSEEK-TP2-IMPLEMENTATION-PLAN.md)
records the original phase gates.

## Promotion and rollback

The initial queued engine was accepted in extended user trials; the v58 engine
now installed at the same path is also user-tested and accepted (#169). Numerical/
capacity follow-ups remain in #110/#112. Preserve the existing commands,
executable paths, model/config selections and launcher files during binary-only
updates.

For a future user-authorized installation, record each selected engine's path,
hash, owner/mode and current pair settings, then copy and verify the original
file into a separate backup directory on each host. Stage the validated new
binary beside the existing target, verify its hash and shared libraries, and
atomically rename it over that target on both hosts. Compare installed hashes.
Do not create new configured engine paths, rewrite launcher JSON, or redirect
llama-swap entries. A binary-only update needs no model repack or manifest
rehash: executable identity is freshly checked at launch.

The user controls model unload/reload. An existing process can keep running its
old executable inode after atomic replacement; the next model start loads the
new file. Do not stop user processes or manage services as implicit installation
steps. After the user's restart, check health and a normal request.

To roll back on user instruction, atomically restore the matching backup binary
contents at those same paths on both hosts and verify the hashes. Preserve the
launchers/configs. The user reloads both ranks together; never mix a new rank
with an old rank or reuse a poisoned pair. The [v58 deployment record](deployments/v58-existing-paths-2026-11-11/README.md)
contains exact current paths and backups.

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
proposals and per-request commits. The newly installed v58 binary contains this
support. Existing model entries/configs and the protected runtime's launcher/
server source were preserved by the binary-only installation; the default is
still one conversation. Binary replacement does not opt those entries into a
new mode or upgrade their Python/launcher code.

Opting into batching separately requires matching current server/launcher code
on both hosts and explicit `--parallel 2`–`4` (or the launcher's `parallel`
setting). Preserve existing executable paths, packs, manifests, contexts and
RDMA selections. Do not rewire existing launchers or configs as an automatic
installation step. An older peer launcher fails preflight instead of silently
changing semantics. Source-level setup and isolated validation are recorded in
the [D5 report](validation/tp2-batching-d5-2026-11-05.md); the current stable-path
installation is recorded in #169. Slot/context/speculation choices have a
separate execution identity; replacing the engine does not invalidate model
manifests.

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
capacity gate with complete target-expert residency. The user subsequently tested and accepted the integrated new mode; #11/#140
are closed. Longer filled contexts remain outside the recorded performance scope.

Use `tools/ds4/validate_batch.py` for owned isolated model/lifecycle validation;
it records PIDs/start identities and stops only its own pair. Shared HTTP
measurements use `tools/bench_concurrency.py`; DeepSeek `--spec 0` disables
DSpark, whereas GLM uses 1 to disable proposals. Raw throughput, numerical
checks and rollout limits are in the [D5 validation report](validation/tp2-batching-d5-2026-11-05.md)
and [matched XS/S measurements](benchmarks/deepseek-tp2-batching-2026-11-05/README.md). Source integration alone does not authorize production changes. The explicit
subsequent binary-only installation in #169 preserves launcher/config files.
