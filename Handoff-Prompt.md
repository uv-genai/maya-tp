# Handoff-Prompt — Maya-TP restart

Read `/home/ugo/software/project-maya/Handoff.md` in full, especially18/21/23/27–29 and latest29.18, then AGENTS.md.

## Task and current state

Integrate upstream **v1.1.37–v1.1.46 one tag at a time**, semantic review, per-tag validation and **post-merge main CI before the next tag**. Continue through46, not a partial status response. Preserve TP2/MTP, owned README/AGENTS/docs, numeric fork VERSION1.1.42 and measured tuning. Push only origin uv-genai/maya-tp. Official refs upstream/v*. Future releases use **tp-vMAJOR.MINOR.PATCH**; legacy v* immutable. Integration does not authorize a release.

Accepted37–41 have full reports in docs/upstream/. PR38/44+48/49/51/58;41merge0fea27a, final39523aeCI41326842619/41326839560, main41327238265 PASS. Main9bbdcb5 currently includes documentation checkpoints only beyond41;42CODE NOT MERGED.

**Active42 #61 /PR63 /#62 /#64**. Review branch merge/upstream-v1.1.42; upstream cd48762 ancestor. ALL runtime gates PASS. Engine source837a679, MD5 **bc4e90a607b87fe609398fbdba0fb98c** all5paths matched. Python policy456b4a5; remote review456, protected canonical777cc92 unchanged. Report docs/upstream/v1.1.42.md +validation.json. **Only final headCI/PRmerge/post-mainCI remain.** Close61/62/64 only after mainPASS, update accepted report/handoffs/index, save42baselineBOTHhosts, then open43issue/branch from updatedmain. No43work before42acceptance/mainCI.

**No active GPU/server/test batches.** Fixed fullbatch19655 and idleAB26121 exited0. Do not rerun them without a new reason. Output /tmp/maya-upstream-v42/. Initialbatch86028 was terminated for64 and is not accepted; failedlogs/binary retained separately.

## Completed42 gates and fixes

Ordered HIPmap/584serve11skip/662tools33skip/Node PASS. All3gfx1151HIPengines/16host checks/196fused/MMQ/attention/396strictIQ PASS. Read-only GPUimage SHA95821268… equalsaccepted41/initial42/fixed42, so host-only fix reuses kernel gates. InitialWindowsmockfixturefailureCI41329879097 retained/fixedba4c34e. Runtime837CI41330559259/Python456CI41331029193/779CI41331428973+41331432952 PASS; finalheadCI still required.

Simple/MTP7 fresh990/7290/14550,7289reuse/stream, real lifecycle/media400/PID/context/peerpreservation/idlefollowup, true18Kpartialprefillcancel/cache0retry/followup PASS.132matchedMTProunds/8asymretries. Firstaudit249/132mixedappendedrank0startups; lastheader(line784) gives132/132; rawlogs retained. CLI32/32bothranks/modes matches41; kill/recover8/8. **32candidateframedrecordsraw0: S16+M8+L8**, ordinary AND MTP against established common references.

#62 refuses local-only GENI/VLEND/VRECLAIM/LOGP before file/cache/reset/collectives, media before rendering/encoding and skipsencoderstartup forTP config. Last repeatedworldflag controlsstartup, matchingengine/launcher;22focusedCPUchecks andlivebogusencoder/4directcommandrejections+strictfidelitypass. #16vision remainsdeferred.

#64 forcedidle1s/3000ms exposedinitiald59/d8 long14.55K MTP32decode23.683s.837wakes BOTH lanes atmirrored modelcalls/nestedMTP. Sameprobe2.2415s, prefill41.463→41.399s; long/short12.26x→1.14x. Durable4xgate rejectsactualpre-fix/passesfixed. Initial /tmp/maya-upstream-v42/idle-before-fix; binary /tmp/maya-upstream-v42-before-idle-fix/maya BOTHhosts. Do not use rejected d8.

DefaultidleA/B completed:65squiet+10ssamples, saved41→42 engineCPU R0/R1 .88/.78%→.20/.00%; voluntaryswitches422/428→10.94/10.00persec. BotharmsMTPfollowup/peerpass; rawcounters/hash/intervals committed. Zero means noCPUtick inshortsample, notpowermeasurement. This supports retainingtheidledefault ongfx1151.

S prefillvs41all9messages/usageequal/cache0, warmthroughput−.18/+.30/+1.57% at990/4062/8158; no materialregression/newgainclaim. #33openforexactcold9.6case/~3008rowcap/unimplementedMAYA_TP_PREFILL_ROWS. M64ordinary40.62/MTP31.90; L42.82/35.16ms, allraw0. LAPIstartup186/73s; finalframed731/219s within unchanged900. ColdM/Lsmokeoutliersretained. LoadsnapshotR0D/folio_wait_bit_common/read14.8GBover109s; identifiesfilepagewaitinthatloadonly. #60 remainsopen fortransientMTP/hostwaitvariability.

Retained42: idle+pairedwake/aliasparity/monotonicGLMchunk/applicableDS4-MM3guards. Deferred allvision/vendor/setup andsame-hostDS4partialread/multi-restore/lending/DMA. Owned README/AGENTS/VERSION/pin unchanged. #16/#11/#46staygated.

## Invariants to preserve in later tags

- HIP/ROCm **gfx1151 Linux only**; **RDMA tensors only**. TCP bootstrap/control/tokenmetadata permitted. No new backends/NVIDIA work. No subagents authorized.
- #59/32650e8: rank0 progress callback must return `tp2 || !read_stopped()` and complete the ENTIRE mirrored prefill call before STOP/reset. Singlehost earlycancel remains; TP2 alwaysmirrorsRESET, KEEP_CANCELLED off/singlehostonly. Durable tools/tp2/prefill_cancel.py.
- #56/#53: refuse rank0-only load/unload/idle/dead-engine/context/currentmodel restarts beforemutation. Pairlauncher relaunches BOTH ranks. #47failedstart marksended/reapsworker andpreservesINFO/routes.
- #57: loopguardinternalcontinuations share TOTAL client outputbudget. Preserveall-limitanswerreserves/boundedsoftgrace/finalSTAT/canceldrain.
- Canonical MTPinputs/accept/continuation/abort wire, globalretry/chunkagreement, replicatedNextN/fullsharedFF/trunkARs, measuredprimary100401tuning/configinheritance. Route rank/fast andSNAP_GEN opt-in/off. SNAPsnapshotmirrorbeforelocalSave.
- #46 streamedfetch is deferred ENTIRELY, including reader/header/kernels; MAYA_GLM_STREAM_FETCH=1doesnothing. Do not reimportfragments. DS4prefix/tail/linkring opt-in. Unsupportedplatform/localmultiGPU/helper/placement/lazy/faircopy additions excluded.

## Hosts and protected production state

User authorization persists: **“Hosts are free; you may stop the pair for validation.”** This is strix-halo-2 rank1; SSH ONLY strix-halo-1.local rank0, no nestedSSH or llama-swapwrapper viaSSH.

Remote canonical `/home/ugo/software/project-maya` stays777cc92 with user maya.py changes,2Lconfigbackups and #33 rank1/tuning deployments. **No pull/reset/rebuild there.** Daemon175392/six-entryYAML (Lcontext524288) preserved. Build onlyisolated `/home/ugo/software/project-maya-upstream-review` onbothhosts. At fulltaskend leave coherentproductionpair; originalcanonicalbinaryrestoration is available while retaining finalvalidatedreviewtrees. Neveroverwriteuserwork/YAML fordeployment.

Stop rank0 server BEFORE engine:

```sh
ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'
pkill -9 -x maya
```

Never broad pkill-fserver.py or /dev/tcpbootstrapprobes. Rank1first, thenrank0directnohupSSH withredirection. Poisonedcommunicator requiresbothrestart. Afterbuilddeploy/hash-check localreview/canonical,remotereview/canonical,rank0/tmp/maya-tp2.

BackupsBOTHhosts: original5a52 at/tmp/maya-upstream-baseline-2026-11-01/;38cff75b48 at/tmp/maya-upstream-v38-baseline/maya;39c3132ee4 atv39;40=8facc62c atv40;41a6170c07503e139d73789184d9a93971 at/tmp/maya-upstream-v41-baseline/maya. Also41candidatea617 andrejectedinitial42d8backup. Saveaccepted42bc4baselinebefore43.

## Tools and remaining sequence

Use .venv/bin/python. Review STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py. Pin3cf03257f219afbe7334045ff7c6a06ac68c627d unchanged. HIPmap→serve discover-t.→tools. Buildgfx1151,HIPON,CUDAOFF,MMQON,TP2_RDMAON,compiler/opt/rocm/lib/llvm/bin/clang++,GGMLDIRcanonicalpinnedcheckout. -j4.

Reusable /tmp/maya-upstream-live.py,/tmp/maya-upstream-cli.py. Bothranks same --model/--pack/--config throughmodel_env.sh. MTP needsMAYA_TP_MTP1+MAYA_GLM_MTP_FIXED/DRAFTsameN; simpleMAYA_GLM_NO_MTP1. LCLI WARM0 workaround#42 only. CommonS256referencesdocs/benchmarks/tp2-mtp-review-2026-10-30/; M/L64docs/upstream/v1.1.39/{Maya-M,Maya-L}/reference.txt; strictordinaryANDMTPrawpick0. Investigatedifferences, neverrelaxfidelity/forceIDs-only. Startup900s unchanged.

#52 GGMLarchivemetadata inheritsparentgitHEAD/hostrelinks: inspectELFsections READ-ONLY viareadelfoffsets+Path.read_bytes. NEVERobjcopyinplace. #54transientconfigstillhidesModelscatalog. Keep33/42/46/52/54/60/11/16open.

Finish42finalCI/PR/main, close61/62/64 withcomments, updateacceptedhandoffs/report/index andbaseline. Then43acdab34,44=5d20b0e,45=6ecbd3c,46=3589d6f, each ownissue/semanticreview/scopedvalidation/final+mainCI. No release/tag merelyforintegration. Legacyv1.1.42release6471995 remainsimmutable; olderupdatersneedonecleanmanualpull forprefixsupport.
