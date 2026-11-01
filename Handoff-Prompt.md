# Handoff-Prompt — Maya-TP restart

Read `/home/ugo/software/project-maya/Handoff.md` in full, especially §§18,21,23,27–29 and **latest §29.15**, then AGENTS.md. Follow issue tracking and all platform/host safeguards.

## Active task

Integrate upstream **v1.1.37–v1.1.46 ONE AT A TIME**, validating each and post-merge main CI before the next. Preserve TP2/MTP. README/AGENTS and fork documentation are owned: semantic merging only. Audit #36 /docs/upstream/INTEGRATION-2026-11-01.md; tags fetched as `upstream/v*`.

Accepted37–40:

-37: #37/PR38/7ea9d71, report docs/upstream/v1.1.37.md; full S/M/L gates.
-38: #43/PR44/ad96053. Post-main failure was fixed separately #47/PR48/f479b91; fixed main41303961934 PASS. Preserve failed-start liveness, do not retry away that historical failure.
-39: #45/PR49/d2708b2; main41311902227 PASS, report docs/upstream/v1.1.39.md.
-40: #50/PR51/0a10f5c; main41316911427 PASS, report docs/upstream/v1.1.40.md. #53 paired context/current-model restart guard retained.

**Active41: #55/PR58, upstream171077787036ccc2784250ca644760f654e8d4a3. Runtime gates COMPLETE; finalCI/merge/mainCI pending.** Review branch merge/upstream-v1.1.41 head7ab98c8. Runtime32650e8, fixed engine **a6170c07503e139d73789184d9a93971** all5paths matched. Latest report docs/upstream/v1.1.41.md and validation.json. Final7ab98c8 CI41326427259/41326422499 running at handoff; reconcile latest mainhandoff (docs only), require finalheadCI before PRmerge, then mainCI before42. No42sourcework yet.

All test engines/server stopped, no active batch. Completed15734 (state/CLI/recovery/fidelity/A-B/M-L),61183(LtimingAB),96027(SNAPown/registration). Inspect actual status before starting anything; do not rerun completed gates without reason.

## v41 results and safeguards

580serve/11skips,654tools/31,HIPmap,3gfx1151HIPengines,15host,396strictIQcases,196fused,MMQ/attention,CPUexactPDEP0/1/compact1024,EXL3/Node PASS. Simple/MTP7 matrices990/7290/14550+reuse/stream, limits/generationcancel and TRUE partial-prefillcancel/cache0retry/followup bothmodes PASS. Lifecycle400preservesPIDs/context/peer. CLI32/32bothmodes matchesaccepted39/40; peerrecovery8/8. Default135matchingrounds/8asymretries; SNAP103matchingrounds/8asym plus1naturalboth-rank clash, allretryshapesmatch.64candidateframedrecords0rawdifferences(S32,M8,L8,Lrepeat8,SNAPown8); baseline40compatibility8also0.

- **#59 fixed32650e8:** initial candidate13a failed cancelled-prefill/retry. Rank0 abandoned a mirrored call while rank1 continuedcollectives. Finish the entire call before STOP/reset onTP2; keep single-host earlycancel. Durable regression tools/tp2/prefill_cancel.py. Never drop this guard in later state merges. Rejectedcandidate/logs preserved; a617GPUimage matches13a (read-only audit), not the same host code.
- **#56:** reject model load/unload/idle/dead-engine single-rank restarts beforemutation; preserve #53context/currentmodelguard. Relauchbothranks throughpairlauncher.
- **#57:** loop-guard internal continuations respect TOTAL max_tokens. Pre-fix10>9/134>133reproduced; bothpaths/adjacentboundariescovered.
- New routing rank/fast and SNAP_GEN are opt-in/off. KEEP_CANCELLED off/single-host-only; TP2alwayspairedreset. Preserve canonical MTP inputs/accept/continuation/abort wire, global retries/chunks, replicated NextN/fullshared widths, trunk ARs andprimary100401 tuning/configinheritance.
- **#46 complete streamed-fetch deferral:** buffered reader/header/fetch kernels remain, even explicitMAYA_GLM_STREAM_FETCH1doesnothing. Do not reimport partial streaming. DS4prefix/tail/link-ring defaults opt-in; newv41same-hostpeer/helper/placement/lazy/fair-copy andunsupportedplatformfeaturesexcluded. HIPpin unchanged.

PrefillA/Bvs40:9messages/usageequal, warmthroughput+0.57/-0.97/-1.84% at1K/4K/8K, no materialregression/newgainclaim. **#33 remains open** for cold/long-context/request11 at9.6tok/s and~3008rowcap/unimplementedMAYA_TP_PREFILL_ROWS. Existing#34configinheritance+18calibrationrows improveS32–37%; preserve.

**#60 remains OPEN:** initialL64MTP76.23/77.20ms/token, accepted40controlled71.02then35.34, repeatedcandidate37.06/37.65 withinhistorical34.82–37.65. Identicalacceptance36/52/rawpicks. No persistentcandidate regression reproduced; no aggregate speedupclaim. Initial/old/repeatedR0startup649/414/617s, R1249/33/159, withinexisting900s; alloutliersretained. IntegrationholdclearedsubjectfinalCI; underlyingtransientwaitvariabilityisnotfixed.

## Hosts and safe operations

**HIP/ROCm gfx1151 Linux ONLY; RDMA tensors ONLY**, TCPbootstrap/control/tokenmetadata. No new backends/NVIDIA work. No subagents unless explicitly authorized.

User authorized: **“Hosts are free; you may stop the pair for validation.”** Persists; do not ask again. We are **strix-halo-2 rank1**; SSH only **strix-halo-1.local rank0**, never nested or invoke llama-swap wrapper overSSH.

Remote canonical `/home/ugo/software/project-maya` stays777cc92 with usermaya.py edit,twoLconfigbackups,#33deployedrank1/tuning. **No pull/reset/rebuild there.** Preserve daemon175392 andYAML. Use isolated review worktrees `/home/ugo/software/project-maya-upstream-review` on BOTHhosts; remotereviewsource32650e8, source/usercanonicalprotected.

Stop rank0 server BEFOREengine: `ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'`; locally `pkill -9 -x maya`. Never broad pkill-fserver.py, never /dev/tcp bootstrap probe. Start rank1first, rank0directSSHfullyredirected. RestartBOTHafteranypoison. Deploy/hash-check bothrank0paths(build-hip/maya,/tmp/maya-tp2),reviewcopies/localcanonical afterbuild.

Backups BOTHhosts: original5a52cd27167523b6ea4b0f0bde93f503 `/tmp/maya-upstream-baseline-2026-11-01/`; accepted38cff75b48 `/tmp/maya-upstream-v38-baseline/maya`;39c3132ee4 `/tmp/maya-upstream-v39-baseline/maya`;40=8facc62c `/tmp/maya-upstream-v40-baseline/maya`; candidate41a617 `/tmp/maya-upstream-v41-candidate/maya`. Save accepted41baseline before42build. At fulltaskend leave coherent production source/binarypair; restoring originalcanonicalbinaries is available whilefinalvalidatedreviewbuild/source remains. Never overwrite remoteuser source/YAML.

## Tools / next steps

Use .venv/bin/python; in review setSTRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py. Pin3cf03257f219afbe7334045ff7c6a06ac68c627d. OrderedgatesHIPmap→serve(discover-t.)→tools. Buildisolatedgfx1151,HIPON,CUDAoff,MMQON,TP2_RDMAON,testsON asneeded; compiler/opt/rocm/lib/llvm/bin/clang++, STRATA_GGML_DIR canonicalpinnedllama.

Livehelpers /tmp/maya-upstream-live.py andcli.py; completed41batch /tmp/maya-upstream-v41-batch.sh,outputs/tmp/maya-upstream-v41. Same model/pack/configselection bothranks via tools/tp2/model_env.sh. MTPexplicitMAYA_TP_MTP1,FIXEDandDRAFTsameN;simpleNO_MTP1. LCLIneedsSTRATA_GLM_WARM0(#42), no readinesshandshakefixyet.

Strictcommonreferences canbe reused ONLY samepack/prompt/tokens: S256 docs/benchmarks/tp2-mtp-review-2026-10-30/{technical,code} files; M/L64 docs/upstream/v1.1.39/{Maya-M,Maya-L}/reference.txt. Bothordinary/MTPrawpicks mustbe0; investigate differences, never weaken gate/forcedIDs-only. Snapshotmodegetsownreference; framedproofin41directory.

#52archiveGGMLmetadata inheritsparentgitID, harmlesshostrelinks. InspectGPUimage READONLY viareadelfoffsets+Path.read_bytes; NEVER objcopyin-place. #54transientrank0config hidesModelscatalog remainsopen. #11multi/#16visiongated; HIPvisionupstreamaloneisnotTP2RDMAembeddingport. #33/#42/#46/#52/#54/#60/#11/#16stayopen. After41mainCIpasses, close55/56/57/59withfinalcomments,updatehandoff/report, then open nexttag42issue and startsemanticreview. Continue through46, notpartialcompletion.

Futureforkreleases use **tp-vMAJOR.MINOR.PATCH**, VERSIONnumeric. Legacyv*publishedtagsimmutable, upstreamfetchedasupstream/v*. Publishedv1.1.42=64719953312df98c9fdf53a10198c106cb195345(public/latest,0assets). No releaseauthorizedmerelybyintegration. Legacyupdater<=v42needsoneclean-mainmanualpullforprefixsupport; nextreleasenotesmustmentionit.
