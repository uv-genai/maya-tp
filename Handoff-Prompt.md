# Handoff-Prompt — Maya-TP restart

Read `/home/ugo/software/project-maya/Handoff.md` in full, especially §§18,21,23,27–29 and latest §29.21, then `AGENTS.md`.

## Active43 update

42 remains accepted. **43 is now in progress: #65/#66/#67/#69, branch merge/upstream-v1.1.43.** Engine source6a2fbe1, MD5b398d4a0711430edd6a28b95264c7293, all5paths matched. Remote review6a2fbe1. CPU602serve11skip/676tools33skip/HIPmap/Node,3HIP/17host/affectedEngram0fail pass. Small-context normal/forced probes46313 exited0. #66 context-note bounds, #67 collapse config aliases and #69 unsupported MTP-bias rejection are fixed but open until acceptance.

**ACTIVE GPU batch8965** `/tmp/maya-upstream-v43-batch.sh`, outputs `/tmp/maya-upstream-v43/`: S state/cancel, long-loop prefix/cache A/B, CLI/recovery,16strictframed Sraw records and prefillAB vs saved42. No concurrentGPU/build/deploy. Inspect outputs before resuming; do not rerun completed gates. All43 final evidence/headCI/PR/mainCI still pending; no44.

**#68 remains OPEN:** full DS4 GPU suite fails packed-KV WMMA stress. Independentlybuilt accepted42 control reproduces both failures:3alternating KVruns/arm counts[1,2,1] vs[1,1,2]. WMMAfunction source unchanged. Engram-only decoder checks pass; full suite is not passing. Never relax/omit the recorded exception. Control source/build `/home/ugo/software/project-maya-v42-ds4-control`; raw /tmp/maya-upstream-v43/wmma-ab. GLM's GPUimage is identical42, so42M/L/math evidence is reusable only where execution is unchanged; server paths need liveS/CPU coverage.

Initial live bias fixture failed to force EOS because pre-existing TP MTP ignored bias; raw failure kept. New guard rejects before state changes; spec1 ordinary bias still works. Forced-follow variant tests EOS branches separately and must not count as raw-pick fidelity. See docs/upstream/v1.1.43.md.

## Objective and checkpoint

Integrate upstream **v1.1.37–v1.1.46 one tag at a time**. Review semantically, validate each checkpoint, and require **post-merge main CI before starting the next tag**. Continue through46. Preserve TP2/MTP, measured tuning and fork-owned README/AGENTS/docs. Push only to `origin` (`uv-genai/maya-tp`); official tags are `upstream/v*`.

**37–42 are accepted.** Latest42: #61 /PR63, merge **b1867c8bfa0e7576754208abb9f2ff66303af1b3**. Final head **aad4e031b174fb931e081c2ef15d4986359a725b**, CI **41334558152/41334563528 PASS**; post-merge **main41334938602 PASS**. #61/#62/#64 closed. Evidence: `docs/upstream/v1.1.42.md` and its `validation.json`. Earlier accepted checkpoints have their own reports;41main41327238265 passed.

**43 is active under #65 / draft PR70**, upstream acdab34bd58eec957715392a1d822777c3c579c3. The active update above is the current state; §§29.19–29.20 in Handoff.md are historical. Next after acceptance/mainCI:44=5d20b0e,45=6ecbd3c,46=3589d6f, each with its own issue, semantic review and validation before proceeding.

Accepted42 engine837a679/bc4e90a607b87fe609398fbdba0fb98c is saved on BOTH hosts at `/tmp/maya-upstream-v42-baseline/maya`. Candidate43 engine6a2fbe1/b398d4a0711430edd6a28b95264c7293 is deployed for validation. Protected remote canonical source777cc92 is unchanged.

Future fork release tags use **`tp-vMAJOR.MINOR.PATCH`**; keep VERSION numeric and legacy tags immutable. Integration does not authorize a release. Published legacy v1.1.42 at6471995 remains unchanged; older updaters need one clean manual pull for prefix support.

## What42 verified

Ordered HIP mapping,584server tests/11skips,662tools/33skips and actual Node monitor pass. All3gfx1151HIP engines build;16host checks,196fused cases,MMQ/attention and396strictIQ cases pass. Read-only ELF audit proves the GLM GPU image equals accepted41 (SHA95821268…). Initial Windows fixture errors and pre-fix latency data are retained, with their corrections documented.

Simple/MTP7 matrices cover990/7290/14550 fresh tokens,7289-token reuse, streaming, real lifecycle/media rejection without PID/context/peer changes, and idle follow-up. Both modes pass true18K partial-prefill disconnect/cache0retry/follow-up.132round tuples and8asymmetric retries match. CLI32/32 on both ranks/modes matches41; peer recovery8/8. **S16+M8+L8 framed records all have zero ordinary AND MTP raw-pick differences.** M/L API/reuse/cancel/limits/fresh1K/4K/8K pass. L final warm731/219s stayed within the unchanged900s bound; MTP35.16ms/token is within its historical range.

#62 rejects unmirrored `GENI`, `VLEND`, `VRECLAIM`, `LOGP` before local state changes and media before encoding. TP startup skips encoder measurement/start and uses the last repeated world flag, matching the engine/launcher. This protects the boundary; RDMA vision #16 remains deferred.

#64 wakes BOTH service lanes at mirrored calls and nested-MTP boundaries. The exaggerated1s/3000ms idle probe caught32-token decode23.683s after a14.55K prefill; fixed code takes2.2415s. The optional4x long/short latency gate rejects actual pre-fix data and accepts fixed data (12.26x→1.14x). Normal-default gfx1151 idle A/B reduces engine voluntary switches422/428→11/10per second and CPU use, with successful MTP follow-up/peer checks. Short CPU samples are not power measurements.

Controlled S prefill vs41: all9responses/usage match, all cache0; warm throughput−0.18/+0.30/+1.57% at990/4062/8158. No material regression or new speedup claim. #33 remains open for the exact cold9.6tok/s case,~3008-row cap and unimplemented `MAYA_TP_PREFILL_ROWS`. #60 remains open for transient MTP/host-wait variability; snapshots identify file-page waits in one cold L load, not its general root cause.

## Safeguards for later merges

- HIP/ROCm **gfx1151 Linux only**; **RDMA tensors only**. TCP is for bootstrap/control/token metadata. No new backend/NVIDIA work. No subagents authorized.
- #59/32650e8: rank0's progress callback must keep `return tp2 || !read_stopped();`. Complete the ENTIRE mirrored prefill call before STOP/reset. Single-host early cancellation stays; TP2 mirrors reset. KEEP_CANCELLED remains off/single-host-only. Live regression: `tools/tp2/prefill_cancel.py`.
- #56/#53: reject rank0-only load/unload/idle/dead-engine/context/current-model restarts before mutation. Relaunch BOTH ranks with the pair launcher. Preserve #47 failed-start liveness/reaping/INFO/routes.
- #57: internal loop-guard continuations share the TOTAL client output budget. Keep all-limit answer reserves, bounded soft closure, final STAT and cancel draining.
- Preserve canonical MTP inputs/accept/continuation/abort, global retry/chunk agreement, replicated NextN/full shared widths/trunk ARs, primary100401 tuning and config inheritance. Route rank/fast and SNAP_GEN remain opt-in/off; snapshot mirror precedes local save.
- #46 streamed fetch remains deferred ENTIRELY: reader/header/kernels unchanged, even `MAYA_GLM_STREAM_FETCH=1` does nothing. Do not reimport fragments. Keep DS4 prefix/tail/link-ring opt-in and unsupported platform/same-host multi-GPU additions excluded.

Keep #33/#42/#46/#52/#54/#60/#11/#16 open. #42's L CLI readiness workaround is WARM=0; no peer-ready handshake was added. #54 transient config still hides the Models catalog.

## Hosts, tools and protected state

User authorization persists: **“Hosts are free; you may stop the pair for validation.”** This is strix-halo-2 rank1. SSH ONLY to strix-halo-1.local rank0; no nested SSH or llama-swap wrapper over SSH.

Remote canonical `/home/ugo/software/project-maya` stays777cc92 with user maya.py changes,2L-config backups and #33 rank1/tuning deployments. **No pull/reset/rebuild there.** Preserve daemon175392 and six-entry YAML (Lcontext524288). Use `/home/ugo/software/project-maya-upstream-review` on both hosts. At the FULL task end leave a coherent production pair; original canonical binary restoration is available while retaining final validated review builds. Do not overwrite user source/YAML to deploy.

Stop rank0 server BEFORE its engine:

```sh
ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'
pkill -9 -x maya
```

Never broad `pkill -f server.py` or `/dev/tcp` bootstrap probes. Start rank1 first, then rank0 through a fully redirected direct SSH command. Poisoned communication requires both ranks restarted. After each build deploy/hash-check local review/canonical, remote review/canonical and rank0 `/tmp/maya-tp2`.

Backups on BOTH hosts: original5a52 in `/tmp/maya-upstream-baseline-2026-11-01/`; accepted38cff75b48,39c3132ee4,40=8facc62c,41a617 and42bc4 in corresponding `/tmp/maya-upstream-vNN-baseline/maya`. Rejected initial42d8 is separately retained in `/tmp/maya-upstream-v42-before-idle-fix/maya`.

Use `.venv/bin/python`. Review `STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py`; pin3cf03257f219afbe7334045ff7c6a06ac68c627d stays unchanged. Ordered gates: HIP map→server discovery with `-t .`→tools. Isolated build: gfx1151,HIP ON,CUDA OFF,MMQ ON,TP2_RDMA ON,compiler `/opt/rocm/lib/llvm/bin/clang++`,GGML_DIR points to canonical pinned checkout; -j4.

Reusable helpers: `/tmp/maya-upstream-live.py`, `/tmp/maya-upstream-cli.py`. Both ranks use the same model/pack/config through `model_env.sh`. MTP needs MAYA_TP_MTP=1 plus matching MAYA_GLM_MTP_FIXED/DRAFT; simple uses MAYA_GLM_NO_MTP=1. Common S256 references are in docs/benchmarks/tp2-mtp-review-2026-10-30; M/L64 in docs/upstream/v1.1.39/{Maya-M,Maya-L}/reference.txt. Require strict ordinary AND MTP raw-pick0, investigate differences and keep failed/outlier data. Do not relax the900s startup bound for convenience.

#52: GGML archive metadata can inherit parent git HEAD and cause harmless host relinks. Inspect GPU sections READ-ONLY via readelf offsets and Path.read_bytes; NEVER objcopy in place.
