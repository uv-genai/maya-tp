# v58 engines installed at existing paths

[#169](https://github.com/uv-genai/maya-tp/issues/169) records the user's explicit
request to replace binary contents while keeping executable paths, launcher
scripts and configs unchanged. Installation is complete, and on 2026-11-11
the user confirmed that the installed builds are working fine. The installation
is user-tested and accepted (#169). No user process was stopped or model
started. An already-loaded model retains the previous executable until the
user unloads/reloads it.

The installed engines are the frozen Release gfx1151 HIP/MMQ/RDMA builds from
source `2485cb609da0c436e1b7d9012d31a136656dd013`, validated during the upstream
v54–v58 integration and merged through [PR #168](https://github.com/uv-genai/maya-tp/pull/168)
at `b7458c4`. Later publication commits changed documentation/evidence only.
The fork's numeric VERSION remains 1.1.44; v58 names the integration checkpoint.
Per-tag Flash GPU and final source/PR/main CI passed before installation.

## Existing file paths

| Host | Existing executable replaced | SHA256 |
| --- | --- | --- |
| Both | `/home/ugo/software/project-maya/build-hip/maya` | `dd37dbd84cabcfce75c4072a8fdac95b08a065ae47481c8bba77fbb9ab07f965` |
| Both | `/home/ugo/software/project-maya-ds4-tp2-review/build-hip/queued-a950e7e9350b/maya-ds4` | `17f42fa16d52dd59c2843138a23a2a2dbcbecc40826d623272e2456f82a11076` |
| Host2 | `/home/ugo/software/project-maya/build-hip/maya-ds4` | same DeepSeek SHA256 above |

Host1 had no canonical `build-hip/maya-ds4`; no new path was created there.
The DeepSeek TP2 directory's `queued-a950…` name is historical and no longer
identifies the binary contents. Its configured path was deliberately retained.
Existing XS/S model packs, local manifests, contexts, DSpark/MTP controls,
calibration/tuning and llama-swap entries were preserved. The model identity
manifests do not include executable contents; the launcher hashes the engine
freshly on each start. No model rehash, repack or download was needed.

## Verification and backups

Both hosts' replacements were staged, checked for the expected content hash
and resolvable libraries, then renamed atomically over the existing files.
Ownership and executable modes were preserved. Installed hashes, `ldd` and
`--help` exit 0 were verified. No lifecycle/API/model validation was repeated:
the exact engines already passed those gates in the
[v58 report](../../upstream/v1.1.58.md).

All 134 host2 and 109 host1 launcher/config fingerprints match their pre-install
values, including the relevant llama-swap config where present. The protected
peer checkout was not pulled/reset or rewritten. The upstream merge did not
change executable-selection wrappers/paths; its DS4 AI_RUN identity helper
change does not select a different engine file.

Original files and exact plans/results are preserved on **each host** in:

```text
/home/ugo/software/maya-binary-backups/issue-169-20261111-v58/
```

| Backup file | Original target |
| --- | --- |
| `0-maya.before` | canonical GLM, both hosts |
| `1-maya-ds4.before` | canonical single-host DeepSeek, host2 only |
| `2-maya-ds4.before` | existing DeepSeek TP2 executable, both hosts |

[Host1 plan](host1-plan.json), [verification](host1-verified.json),
[host2 plan](host2-plan.json), [verification](host2-verified.json) record full
hashes, paths and unchanged fingerprints. [install.py](install.py) is the exact
archived deployment/rollback helper, with its fixed paths/run ID; its `prepare`
phase refuses an existing backup directory. Rollback is a separate user-directed
action that restores backed-up binary contents at the same paths. Never stop
user processes or modify launchers as an implicit part of rollback.
