# Handoff-Prompt — Maya-TP restart

Read `/home/ugo/software/project-maya/Handoff.md` in full, especially §§18,21,23,27–29 and **latest §29.17**, then `AGENTS.md`.

## Task and current checkpoint

Integrate upstream **v1.1.37–v1.1.46 one at a time**. Validate each checkpoint and its post-merge main CI before starting the next. Preserve TP2/MTP; merge owned README/AGENTS/docs semantically. The audit is #36 /`docs/upstream/INTEGRATION-2026-11-01.md`; official tags are `upstream/v*`.

Accepted37–41:

- v1.1.37: #37 /PR38 /7ea9d71, full S/M/L gates; report `docs/upstream/v1.1.37.md`.
- v1.1.38: #43 /PR44 /ad96053. Its post-main failure was fixed separately by #47 /PR48 /f479b91; fixed main41303961934 passed. Preserve failed-start liveness; do not retry away that historical failure.
- v1.1.39: #45 /PR49 /d2708b2; main41311902227 passed; report `docs/upstream/v1.1.39.md`.
- v1.1.40: #50 /PR51 /0a10f5c; main41316911427 passed. Preserve #53's paired context/current-model lifecycle guard.

- v1.1.41: #55 /PR58 /0fea27a, final39523ae CI41326842619/41326839560 and **main41327238265 PASS**. #55/#56/#57/#59 closed. Full report docs/upstream/v1.1.41.md and validation.json.

**Active42 #61 /PR63 /#62 /#64**: code/runtime fixes validated, final model/idle/evidence/CI gates pending. Review branch merge/upstream-v1.1.42, engine source **837a679**, MD5 **bc4e90a607b87fe609398fbdba0fb98c** all5paths matched. Python policy456b4a5; remote review456, canonical source777cc92/user work protected. No42acceptance/no43; continue through46 sequentially after each checkpoint/mainCI.

**ACTIVE batch19655** /tmp/maya-upstream-v42-batch.sh, output /tmp/maya-upstream-v42/: S force-deep-idle simple/MTP7 matrices, real startup/media/lifecycle guards, trueprefillcancelboth,CLI32/32matching41,recovery8/8,16strictframedSrawpick0,PREFILL A/B andMsmoke+8rawpick0 ALL PASS. LAPI/cancel/reuse/limits/fresh1K/4K/8K PASS; L64framedbenchmark is currently warming.900s startup boundunchanged. Do not start concurrentGPU/build/deploy or rerun finished gates. Batch86028 was terminated for64 (notcurrent).

**Required AFTER19655 completes with exit0**: run `.venv/bin/python /tmp/maya-upstream-v42-idle-ab.py > /tmp/maya-upstream-v42/idle-ab.txt 2>&1` in review. This normal-default S comparison (candidatebc4 vs saved41a617) waits65squiet+10sCPUticks/voluntarywake sample onbothranks, thenMTPfollowup/peerhealth. Prepared/pycompiled, NOT RUN YET. It validates default idle polling benefit on gfx1151. Do not omit it or runconcurrently. Then collectL/idledata, finalreview/evidence/headCI/PRmerge/mainCI, close61/62/64onlyaftermainPASS, accepted42baselinebothhosts, then43.

#64 force-idle1s/3000ms exposedinitiald59/d8 long14.55K MTP32decode23.683s. Fixed837 wakesBOTHlanes atmirrored modelcalls/nestedMTP; sameprobe2.2415s, prefill41.463→41.399s; long/short12.26x→1.14x. Durable optional4x gate rejectsactualpre-fixobservations/passesfixed. Initiallogs/binary preserved /tmp/maya-upstream-v42/idle-before-fix and /tmp/maya-upstream-v42-before-idle-fix/maya BOTHhosts. GPUimageidenticalaccepted41/d8/fixed. No defaultspeedclaimfromexaggeratedsettings.

132matchedMTProunds/8asymmetricretries, bothmodespartialprefillcancel/cache0retry/followup PASS. Firstaudit249/132mixedappendedrank0startups; segmentedlastheader(line784), preservesrawlogs. #62 all4GENI/VLEND/VRECLAIM/LOGP refusals followedby strictfidelity; mediaguard/startupbogusencoder/PIDs/context/peerpreserved. Python456last-worldflagstartupguard22focusedtestsPASS. #16 remainsdeferred.

OrderedHIPmap/584serve11skip/662tools33skip/Node,3HIPengines/16host/196fused/MMQ/attention/396strictIQ PASS. InitialWindowsmockfixturefailureCI41329879097retained/fixedba4c34e; runtime837CI41330559259/Python456CI41331029193/779CI41331428973+41331432952PASS. LatestdocsheadCIstillrequired.

S prefillvs41 all9messages/usageequal/cache0, warmthroughput−0.18/+0.30/+1.57% at990/4062/8158; no materialregression/newgainclaim. #33 staysopenexactcold9.6case/~3008rowcap. M64ordinary40.62/MTP31.90ms,8raw0; coldsmokeprefill11.32/39.66/24.97sretained. LAPIstartupR0=186/R1=73s; rawbenchmarkstartupstillrunning. #60 remainsopenvariability; requirestrictLrawpicks, inspecttimingsbeforeacceptance.

Accepted41 baselinea617 remains BOTHhosts /tmp/maya-upstream-v41-baseline/maya. Completed41batches15734/61183/96027 need no rerun. No release/tag authorized by integration; futureforktags tp-v.

## Validated41 behavior

Ordered HIP map,580server tests/11skips,654tools/31skips passed. Three gfx1151 HIP engines build;15host checks,396strict IQ cases,196fused cases, MMQ/attention, CPU exact rows with PDEP0/1 and compact1024 widths, EXL3 tiers and actual Node monitor checks passed.

Simple/MTP7 matrices at990/7290/14550 tokens, large reuse/streaming, thinking/output limits and generation cancellation passed. True partial-prefill disconnect/cache0 retry/follow-up passed in BOTH modes. Lifecycle rejections preserved PIDs/context/peer. CLI32/32 in both modes matches accepted39/40; peer recovery8/8. Default135round records and8asymmetric retries match; snapshot mode103rounds and8asymmetric plus1natural both-rank clash also match. All64candidate framed records have zero raw-pick differences: Sdefault16+routeopt16,M8,L8,Lrepeat8,snapshot-own8. Saved40 compatibility8also passed.

Keep these safeguards in later merges:

- **#59 /32650e8:** rank0 must finish the entire mirrored prefill call before honoring STOP/reset. Initial candidate13a failed with poisoned collectives because rank1 continued the abandoned call. Single-host early cancellation remains. The durable live regression is `tools/tp2/prefill_cancel.py`; pre-fix logs are archived. The host-only fix preserves the GPU image.
- **#56:** reject load/unload/idle/dead-engine single-rank restarts before mutation. Preserve #53's context/current-model guards; relaunch both ranks with the pair launcher.
- **#57:** internal loop-guard continuations share the TOTAL client output budget. Pre-fix10>9/134>133 failures are retained; both paths and adjacent boundaries pass after the fix.
- Route rank/fast and SNAP_GEN remain opt-in/off. KEEP_CANCELLED remains off and single-host-only; TP2 always resets both ranks. Preserve canonical MTP inputs/accept/continuation/abort wire, global retry/chunk agreement, replicated NextN/full shared widths, trunk ARs and measured primary100401 tuning/config inheritance.
- **#46:** streamed fetch is deferred entirely. Buffered reader/header/fetch kernels remain; even `MAYA_GLM_STREAM_FETCH=1` does nothing. Do not reimport fragments. DS4 prefix/tail/link-ring settings remain opt-in. Unsupported platforms, new same-host peer/helper/drafter placement and unmeasured lazy/fair-copy additions were excluded.

Prefill vs40: all9messages/usage equal; warm throughput+0.57/−0.97/−1.84% at1K/4K/8K. No material regression or new gain claim. **#33 stays open** for cold/long-context/request11 at9.6tok/s and the~3008-row cap/unimplemented `MAYA_TP_PREFILL_ROWS`. Preserve #34's worker config inheritance and18calibration rows, which improve S32–37%.

**#60 stays OPEN:** initial L64 MTP76.23/77.20ms/token; controlled accepted40 gave71.02then35.34; repeated candidate gave37.06/37.65, within historical34.82–37.65. Acceptance36/52 and raw picks agree. No persistent candidate regression reproduced; no aggregate speedup claim. Startup rank0/rank1 seconds: initial649/249,40=414/33,repeated617/159, within the unchanged900s bound. The integration was accepted after final/main CI; transient drafting/host-wait variability remains unresolved. Raw arms and phase times are committed.

## Hosts and operations

**HIP/ROCm gfx1151 Linux only. RDMA tensors only; TCP carries bootstrap/control/token metadata.** No new backend/NVIDIA work. No subagents unless explicitly authorized.

User authorization persists: **“Hosts are free; you may stop the pair for validation.”** Do not ask again. This machine is **strix-halo-2 rank1**; SSH only to **strix-halo-1.local rank0**. Never nested SSH or invoke the llama-swap wrapper over SSH.

Remote canonical `/home/ugo/software/project-maya` stays777cc92 with user `maya.py` edits, two L-config backups and #33 launcher/tuning deployments. **No pull/reset/rebuild there.** Preserve daemon175392 and its six-entry YAML. Use the isolated `/home/ugo/software/project-maya-upstream-review` worktrees on both hosts; remote review runtime source isd59ea06.

Stop rank0 server BEFORE its engine:

```sh
ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'
pkill -9 -x maya
```

Never broad `pkill -f server.py` or `/dev/tcp` bootstrap probes. Start rank1 first, then rank0 via a fully redirected direct SSH command. Restart BOTH after poisoned communication. After every build deploy/hash-check local canonical/review, remote canonical/review and rank0 `/tmp/maya-tp2`.

Backups on BOTH hosts:

- Original5a52cd27167523b6ea4b0f0bde93f503: `/tmp/maya-upstream-baseline-2026-11-01/`.
- Accepted38cff75b48: `/tmp/maya-upstream-v38-baseline/maya`.
- Accepted39c3132ee4: `/tmp/maya-upstream-v39-baseline/maya`.
- Accepted40=8facc62c: `/tmp/maya-upstream-v40-baseline/maya`.
- Accepted41a617: `/tmp/maya-upstream-v41-baseline/maya`; extra candidate backup `/tmp/maya-upstream-v41-candidate/maya`.

The accepted41 baseline has been saved before42 builds. At the end of the FULL task leave a coherent production source/binary pair; restoring original canonical binaries is available while final validated builds remain in the review worktrees. Never overwrite remote user source/YAML to deploy.

## Tools and remaining work

Use `.venv/bin/python`. In review set `STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py`. The llama.cpp pin is3cf03257f219afbe7334045ff7c6a06ac68c627d. Ordered gates: HIP map→server discovery with `-t .`→tools. Isolated build: gfx1151,HIP ON,CUDA OFF,MMQ ON,TP2_RDMA ON; tests as needed; compiler `/opt/rocm/lib/llvm/bin/clang++`, `STRATA_GGML_DIR` points to canonical pinned llama.cpp.

Helpers: `/tmp/maya-upstream-live.py`, `/tmp/maya-upstream-cli.py`. Completed41 batch: `/tmp/maya-upstream-v41-batch.sh`; output `/tmp/maya-upstream-v41/`. Use the same model/pack/config on both ranks through `tools/tp2/model_env.sh`. MTP uses `MAYA_TP_MTP=1` and BOTH `MAYA_GLM_MTP_FIXED=N`/`MAYA_GLM_MTP_DRAFT=N`; simple uses `MAYA_GLM_NO_MTP=1`. L CLI needs `STRATA_GLM_WARM=0` (#42, no readiness-handshake fix).

Common references are reusable only with identical pack/prompt/token count and zero ordinary AND MTP raw-pick differences. S256: `docs/benchmarks/tp2-mtp-review-2026-10-30/`; M/L64: `docs/upstream/v1.1.39/{Maya-M,Maya-L}/reference.txt`. Investigate differences; never weaken the gate or use forced IDs alone. Snapshot mode has its own ordinary reference and completed fidelity evidence in41.

#52: archive GGML metadata inherits parent git ID and can cause harmless host relinks. Inspect GPU sections READ-ONLY via readelf offsets and Path.read_bytes; NEVER objcopy in place. #54: transient rank0 config still hides the Models catalog. #11multi/#16vision stay gated; upstream HIP vision does not implement TP2 RDMA embedding synchronization.

41 is accepted and55/56/57/59 are closed; keep33/42/46/52/54/60/11/16 open. Active42 issue61 is open; begin its semantic review only from the accepted main. Require42 final/main CI before43 and continue sequentially through46.

Future fork release tags: **`tp-vMAJOR.MINOR.PATCH`**, with numeric VERSION. Legacy published `v*` tags are immutable; upstream tags stay `upstream/v*`. Publishedv1.1.42=64719953312df98c9fdf53a10198c106cb195345(public/latest,0assets). Integration alone does not authorize a release. Legacy updaters throughv42 need one manual clean-main pull for prefix support; mention it in the next release notes.
