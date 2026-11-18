# v1.1.65 engines installed at existing paths

[#182](https://github.com/uv-genai/maya-tp/issues/182) records the user's request on 2026-11-18 ("Yes, deploy"),
made after PR #180 merged the upstream v1.1.59–v1.1.65 integration into `main`. The request was to install the
validated v1.1.65 engines at the same executable paths and to sync the strix-halo-2 checkout. The procedure is
#169's, unchanged. Installation is complete. The user's first reload is the end-to-end test, and **that test is
still pending**.

The installed files are the frozen Release gfx1151 HIP/MMQ/RDMA builds from source
`736e1cd7616c61b4de53cf2c0d6777a285fb4cfd`. Those exact bytes passed the v1.1.65 checkpoint
([report](../../upstream/v1.1.65.md)):

- local gates;
- the nine pre-bootstrap refusals;
- six Flash TP2 cases with 0 differences from v1.1.64, including MTP/DSpark;
- injected peer loss, with the first cause kept;
- a matched GLM repeat.

The source was merged through PR #180 (`08cbbfc4`). The numeric `VERSION` stays `1.1.45`.

## Existing file paths

| Host | Existing executable replaced | New SHA256 | Previous (v58, #169) |
| --- | --- | --- | --- |
| Both | `/home/ugo/software/project-maya/build-hip/maya` | `5556b6fe13932d220bf412a0aced76dab68bf669d8ca8469cc351cf484e891b3` | `dd37dbd8…` |
| Both | `/home/ugo/software/project-maya-ds4-tp2-review/build-hip/queued-a950e7e9350b/maya-ds4` | `66e1973d56fddbf3389d4588d3496819598c0410029921b184a5c144663d5cf8` | `17f42fa1…` |
| strix-halo-2 | `/home/ugo/software/project-maya/build-hip/maya-ds4` | same DeepSeek SHA256 | `17f42fa1…` |

strix-halo-1 has no canonical `build-hip/maya-ds4`, and none was created. The `queued-a950…` directory name is
historical. No model was loaded on either host during the replacement, so no process held the old files. No user
process was stopped and no model was started.

## Verification and backups

**Install on each host.** The replacements were first staged next to their targets, then checked:

- the expected content hash;
- that every library resolves;
- that ownership and mode match the old file.

They were then renamed atomically over the existing files. Verification found:

- installed hashes match;
- backups are intact;
- `ldd` passes and `--help` exits 0.

All 134 strix-halo-2 and 109 strix-halo-1 launcher/config fingerprints match their pre-install values, including
the llama-swap configs.

**Backups.** The originals are kept on **each host** in:

```text
/home/ugo/software/maya-binary-backups/issue-182-20261118-v65/
```

| Backup file | Original target |
| --- | --- |
| `0-maya.before` | canonical GLM, both hosts (v58 `dd37dbd8…`) |
| `1-maya-ds4.before` | canonical single-host DeepSeek, strix-halo-2 only (v58 `17f42fa1…`) |
| `2-maya-ds4.before` | existing DeepSeek TP2 executable, both hosts (v58 `17f42fa1…`) |

[install.py](install.py) is #169's archived helper with only these changed: the run ID, the source folder
(`engines/v1.1.65`), the expected hashes and the source commit. Its phases are `prepare`, `commit`, `verify` and
`rollback`, and `prepare` refuses an existing backup folder. Rollback is a separate action, taken only when the
user directs it. It restores the backed-up v58 contents at the same paths, and it never stops user processes or
changes launchers. [Host1 plan](host1-plan.json), [verification](host1-verified.json),
[host2 plan](host2-plan.json) and [verification](host2-verified.json) record full hashes, paths and fingerprints.

## strix-halo-2 checkout

The canonical checkout `/home/ugo/software/project-maya` on strix-halo-2 was clean. It was fast-forwarded from
`d6b53f8` (tp-v1.1.45) to `main` `8a2dd0d7`, and it imports `serve.server` with its venv. Its fingerprint set went
from 134 to 135 files ([before](host2-checkout-selectors-before.json), [after](host2-checkout-selectors-after.json)):

- **Added:** `tools/ds4/download-2.25bpw.json`.
- **Changed:** `tools/ds4/pack.py`, `tools/ds4/slot_keep_test.py`, `tools/ds4/validate_batch.py` and
  `tools/tp2/validate_batch.py`.

No launch script, model config or llama-swap config changed. The GLM tuning tables keep the legacy
`STRATA_HIPBLASLT_TUNING_V1` header and differ only in comment lines.

strix-halo-1's checkout (`777cc92e` plus 75 protected uncommitted user edits) was not touched. It runs rank 0 and
the API server.

## What was and was not validated

**Validated:** the v1.1.65 engines with the v1.1.65 server code, on both ranks. strix-halo-2's own checkout now
matches that combination.

**Not run:** production's rank-0 server from strix-halo-1's edited checkout, with these engines. That server
already runs the v1.1.58 engines. It has no `END`-line handling, so v1.1.65's changed DeepSeek `END` text does not
affect it. Older `STRATA_*` settings it passes reach the engines through their alias layer.

The user's first reload of the usual llama-swap entries is the end-to-end test. A model that was already loaded
would keep its old executable until it is reloaded; none was loaded during this install.
