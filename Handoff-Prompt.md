# Handoff-Prompt — Maya-TP restart

Read AGENTS.md, then Handoff.md §33 for current state. Earlier sections retain historical evidence.

## Active prefill fix (#90, parent #33)

New user request: fix prefill. Branch perf/apu-prefill-90 from accepted4d1183e. Full host freedom
already authorized; no availability question. HIP-only prototype lets the Linux APU complete pool
keep experts+spares+prompt tail when all fit above RAM headroom/reserve, with old capped fallback
and user caps preserved. Current code excluded APU and needlessly evicted residents for scratch.
Existing MAYA_GLM_COMPLETE_POOL=0 is same-binary control. Host memory boundary tests PASS.
Isolated build /tmp/maya-prefill-90/build-hip (maya + glm_memory_test, -j4) PASS, logs build.txt and
build-final.txt. HIPmap/host memory tests PASS. No GPU model started. Use new API18090/bootstrap18590 and
exact-owned supervisor, matching binaries/explicit packs both ranks, oldb398 + new0/1 before/after.
Require measured fresh/warm gains, strict fidelity and MTP/reuse/cancel/peer-health. Preserve all
production/review/protected source states. Track #90, keep #33 remaining work honest. Details§33.
Prototype3da1347 pushed/CI41392437370 PASS; matching candidate48d67746 / oldb398 on both hosts. Pilot35741 EXIT0,
/tmp/maya-prefill-90/pilot.py → pilot.txt, S old0/new0/new1 six fresh1K/4K/8K/arm MTP2/context32768.
All6 registries stopped; all6 response/usage pairs equal. On stages0 experts/copies; warm throughput
essentially unchanged. Cold off1K26.87s/on4.53s retained, no statistical gain claim. Need balanced
M/L+largecontext eligibility and fidelity before accepting. Separate missing DSA GEMM shape
f16N8192K1536ldy8192 identified; #91 calibration. ACTIVE GPU tuner99452, stock tool64/256/1024,
16MiBworkspace unchanged accuracy gates; dsa-tuner.txt/dsa-tuning.txt. No models active. Inspect
tuner completion before any GPU stage. Prepared sweep.py, API/CLI/strict owned helpers under90.
LATEST: tuner99452/tuned-pilot79153 EXIT0,20 GPU parity cases pass both hosts with actualLt10/fallback0
each; all6 tuned pilot outputs/usage equal, warm variation small. Old L512K probe38122 failed new
cache>0-on-everyturn assumption at turn4; generation/peer healthy, both roles stopped, data retained.
#92 tracks cache/render/checkpoint investigation; record misses, retain unchanged three-turn cache gate.
User workload/timing async question pending; continue independently. HIP workspace now also sized
to SAME TP3008 limit (old startup reserved20–25K/6GiB), exact3008 if fit else old256-grid fallback.
Build24599/host memory+TP tests/HIPmap PASS, no GPU active. Need new candidate deployment/provenance
(keep48d prototype andb398), then balanced S/M/L beforeold/afternew, highcontext, cap fallback,
full API/CLI/strict fidelity, final evidence/docs/orderedCPU/headCI/PR/main. #90/#91 open,33/60/92 open.

## Fresh live validation complete (#87 / #88)

Read Handoff.md §32 and docs/upstream/v1.1.49.md/live evidence. Main997bba7 integrates upstream
v1.1.47–49; fix e1682cb/PR89 adds first-fatal-transport sticky-down health and a pre-GEN refusal.
Exact-final-head/merge/main CI and closure provenance are on #87/#85/PR89. If #87 is still open
after interruption, finish GitHub closeout only; all runtime gates have completed successfully.
No new GPU work/tag/release/deployment is required for this batch.

API38059/CLI30667/strict89906/prefill32674/reverse10081 EXIT0. All6 S/M/L ordinary/MTP7 modes pass
large/reuse/stream/forcedidle/guards/18Kcancel; MTP rounds146/127/148,8 asymmetric retries each.
Peer loss immediately down/degraded/no respawn and dual-rank recovery8/8 per model. S/M exercise
queued poison, L existing mirror-send path. Healthy suites have no spontaneous peer-closed error.
CLI32/32 all6 modes, S exact acceptedv42, L WARM0 retained. Strict S256technical/code+M/L64:
all32 warm/measured records raw-pickdiff0. Both prefill orders all18 message/usage pairs equal/cache0;
pooled warm throughput−3.64/−1.66/−1.95% at1K/4K/8K, no speedupclaim. Extra bias/spec1/Responses/
Anthropic limits/MTP follow-up PASS. CPU HIPmap/613serve11skip/686tools35skip + finalfocused3 PASS.
Final52 local/26 remote saved ownership records all stopped/alive=[]. Raw failures/outliers retained.

Both premerge d7626ac and merged997bba7 reproduce the first poisoned-error stale-health gap in #6;
#88 preserves original cause/poison and blocks subsequent GEN. Never clear poison, retry collectives
or restart one rank. No GLM C++/transport/MTP/tuning/launcher change. Live source archives/configs/logs
under /tmp/maya-v49-live-87 on both hosts, API18087/bootstrap18587/usb4_rdma5, matchingb398 binaries.
Durable proofs: docs/upstream/v1.1.49/live/{summary.json,manifest.json,README.md}.

No active engines/builds. Production5a52/reviewb398+DS451749 split and protected remote777cc92 user
source/configs remain unchanged. User granted full host freedom for this run; no availability question
is needed for its closeout. Only owned exact PID/starttick/marker cleanup was used, server first.
Legacy broad harness cleanup #83 remains open. No unrelated service inspection/control.

Cold/first-request outliers remain #33/#60, L WARM0 #42. No full-model DeepSeek GPU acceptance;
#68/#75/#79 remain open along with all other previously unresolved issues. No tag/release;
VERSION1.1.42, futuretp-v*.

## Latest requested batch: direct v1.1.49 merge (#85)

The user requested one cumulative merge from latest upstream v1.1.49 (26aacb6), including v47/v48.
Worktree /home/ugo/software/project-maya-upstream-v49, branch merge/upstream-v1.1.49, fork base d7626ac.
Report docs/upstream/v1.1.49.md and its evidence directory; exact-head/merge/main CI and acceptance
are recorded in #85. This replaces the per-tag sequence only for this requested batch.

Keep model management opt-in and all TP2/MTP/cancel/budget/media/lifecycle safeguards. Both --engine
maya and legacy strata work; GLM wire keys remain unchanged. Accept DS4 XS-v2/shared-table/soft-close/
read-order and monitor/TUI fixes. Exclude NVIDIA/Qwen allocator, local multi-GPU chunk default,
Windows pipe helper and conflict-reintroduced Metal functions. VERSION1.1.42, future tp-v*, no release/tag.

Preliminary source/host gates before #87: HIP-map,610serve/11skip,686tools/35skip,118focused,42Textual/dashboard and336010 actual
DS4request-policy checks PASS. Changed DS4main compiles/links against read-only accepted HIP deps;
isolated build-host-v49/maya-ds4 MD5 64f5092b583ea33815069fbbf28057ea, same GPUimage92096f8e… as51749.
Only DS4main differs among444 protected engine/build/header/launcher/tuning files; no GLM/TP2/MTP
math/transport/tuning changes. Fresh GLM GPU results are above; no full DS4 GPU claim or runtime deployment.

The user launches their processes. Stop only your own recorded validation PIDs/process groups;
do not inspect/manage unrelated services or run broad-cleanup harnesses against user processes.
Production/review binary sets and protected remote source below are unchanged. Keep all previously
open issues, including #83, open unless independently resolved. Prior v37–46/#77 and docs/#82 work
are complete; do not resume their historical pending steps.

## Completed integration

Requested upstream v1.1.37–v1.1.46 integration and production restoration are complete. Tags were handled ONE AT A TIME with semantic review, affected validation, exact-final-head and post-merge main CI before proceeding. All10tags are ancestors of main. Final PR81/1139fa27d7731941b187de57093c3abe424a5e1f; finalb5e46b21e8dd855563dae54d3ddd9897bc9b723a CI41350489382/41350494526 andpost-main41351082604 PASS. #78/#80closed. Reports: docs/upstream/INTEGRATION-2026-11-01.md, v1.1.37…46.md, COMPLETION-2026-11-02.md andcompletion-2026-11-02/completion.json.

Final documentation checkpoint dede882111be994b3e913463c8fa6b211f34633d passed main CI41352750402. Issues #77/#36 are closed. Nothing remains from the integration task. No further GPU tests, tag integration, release or production deployment is required without a new request. The integration validation batches ended;46batch22871EXIT0. Documentation audit #82 corrected stale guides and process-ownership rules; #83 tracks legacy harness cleanup.

No release/tag was created. VERSION remains numeric1.1.42; legacyforkv1.1.42 is immutable64719953312df98c9fdf53a10198c106cb195345. Futureforktags use tp-vMAJOR.MINOR.PATCH; upstreamtags are namespaced upstream/v*. Push only origin uv-genai/maya-tp, never upstream.

## Current runtime and protected state

Production original GLM **5a52cd27167523b6ea4b0f0bde93f503** restored at localcanonical build-hip/maya, remotecanonical build-hip/maya andrank0/tmp/maya-tp2. BOTHreviewworktrees retain GLM **b398d4a0711430edd6a28b95264c7293** (enginecode6a2fbe1) andDS4 **51749e860763aea34847b183de95a933** (hostcode5b13c83). Reviewsource can follow finalmain docs without rebuilding unchanged engine code. Accepted46backups BOTHhosts: /tmp/maya-upstream-v46-baseline/maya andmaya-ds4-accepted. The LOCAL maya-ds4 in that folder is olderpre46controlcd177 forELFproof, notaccepted46hostbuild.

The permanent self-contained **Sreviewconfig** points at review/build-hip/maya on BOTHhosts. M/Lvalidation used explicit temporaryconfigs. For futureM/Lreview launches select the same model/pack/config on both ranks. Rank0 uses config.exe; rank1 uses ENGINE or its checkout's build-hip/maya, so run from the review trees or select both paths explicitly. A canonicalconfig still targets the restored productionbinary. Do not assume permanentM/Lreviewconfigs exist.

Protected remote canonical /home/ugo/software/project-maya was777cc92eeab582688c54ae78fcd214427453ccc4 at restoration, with user maya.py/tuning/rank1 edits andtwoLconfigbackups. NO pull/reset/rebuild there. Userdiffs/stageddiffs/configs/backups/YAML/daemon matched exactly before/afterrestoration; raw snapshots committed. These are historical preservation records, not live service status. Only agent-started validation processes were stopped; the user launches their processes themselves.

This is strix-halo-2 rank1. SSH ONLY strix-halo-1.local rank0; never nestedSSH. The user clarified: **stop only processes you started**. Record validation PIDs/process groups; stop the owned rank0 server before its owned engine to prevent respawn. Never use name-wide pkill or port-wide killing. Do not inspect or manage unrelated services, including llama-swap, without a task explicitly requesting it. Prior host-validation permission is not standing authorization to stop user processes. Legacy acceptance.sh/bench_mtp.py use broad cleanup (#83); do not run them against user processes.

Never use /dev/tcp bootstrap probes. Start rank1first, then rank0 directSSH fully redirected. A poisoned communicator requires restarting BOTH owned validation ranks. For new engine validation, match selected binaries on both hosts and explicitly select canonical-versus-review paths; the current split is intentional.

## Validation and limits

46: HIPmap,605server/11skip,679tools/33skip,99focusedPython,18hostchecks andgfx1151HIPDS4build PASS. Actual chunk policy passes aliases/overflow/exactarchitecture and270848plan properties. Read-only ELF: DS4GPUimage92096f8e… unchanged; GLMbyteidenticalb398/GPUimage95821268… unchanged. Reuse prior kernel/strict S/M/L evidence only for unchanged paths; no fresh46M/L/performance claim.

Fresh MTP7+ordinary large/reuse/stream/lifecycle/media/idle gates andtrue18Kpartial-prefill cancel/cache0retry/follow-up/peer PASS.132rank-matched rounds/8asymmetric retries. Text-partsAPIhistory20021/cache0→20020cached/collapsed0, ordinary/MTP64follow-ups PASS. DirectCPUregression isolates parts-count80; OpenAItext-onlyparts flatten tostr, soAPIformatprobe is notindependent80A/B. Raw beforefailures, initial mistakenparserfixtureexpectation andtimingoutliers retained. No general speedup claim.

43also retained final strictStechnical/code256 warmup+measured16records raw-pick0,CLI32/rank/mode exact42,recovery8/8. FreshprefillABvs42 all9messages/usageequal/cache0,warm+0.94/−0.36/−0.37%. Syntheticlongloop policy0vs32000 cache0→20020/prefill61.76→1.14s is fixture-specific, notuniversal throughput.42has freshM/L/API/raw64data andnormalidle A/B; reports document reused paths and limits.

## Explicit open work

- #75: upstream45functional DS4tail-drain fix DEFERRED, notactive. AutomaticNVIDIApath disabledHIP; copyflush/strictdrain/lentguard dependencies excluded earlier. No partialport or32K/64K/128KDS4stabilityclaim.
- #68: FULLDS4WMMA GPU suite FAILS on accepted42control and43.3alternatingKVruns/arm control[1,2,1],candidate[1,1,2]; no thresholdrelaxed. NewEngramchecks pass, unchangedDS4GPUimage doesnotfixWMMA.
- #79: optional MAYA_DS4_CHUNK_GFX11 default0, not4096. Bounds/aliases tested under78; configuredmodelstore has GLMS/M/L only, noforkDeepSeek5%speedup/default-enable measurement.
- Keep33/42/46/52/54/60/11/16open unless actually resolved.33cold9.6tok/s/~3008rowcap/missingMAYA_TP_PREFILL_ROWS remains;60latencyvariability unresolved.42LCLI WARM=0 workaround, nopeerreadyhandshake.54transientconfig hidesModelscatalog.46streamedfetch deferredENTIRELY evenenv1;16RDMAvision and11multiseq gated (latentfullsharedFFbuginmultirows).

## Critical safeguards

HIP/ROCm gfx1151Linux ONLY, RDMA tensors ONLY (~37Gb/s/~6µs). TCP bootstrap/control/tokenmetadata only. No newbackend/NVIDIA work or subagents authorized.

#59 rank0progress callback MUST keep `return tp2 || !read_stopped();`: finish ENTIRE mirroredprefillcall beforeSTOP/reset. Singlehostearlycancel unchanged;TP alwaysmirrorsRESET, KEEP_CANCELLED off/singlehostonly. Regression tools/tp2/prefill_cancel.py.

#56/#53 reject single-rank load/unload/idle/dead-engine/context/current-model restart beforemutation; relaunchBOTH. Preserve47failedstart ended/reap/INFO/routes. #57 internalcontinuations shareTOTAL clientbudget (`while n < max_new`), plus43context-room check; preserveall-limitanswerreserve/boundedsoftgrace/finalSTAT/canceldrain.

#64 wake BOTHservice lanes atmirroredcalls/nestedMTP andrank1inputpoke. #62 rejectTP GENI/VLEND/VRECLAIM/LOGP beforemutation,media beforeencoding,skipTPencoderstartup,lastrepeatedworldflagwins. #69 rejectnonemptybiasbeforestate/table/mirror underTPMTP;spec1ordinarybiasworks. #67 config/legacycollapsealiases preserved;80counts parts. #71 benchmarkseparatestdout/stderrreaders/queues/logs;nevermergefragmenteddiagnosticsintoDONE.

PreservecanonicalMTPinput/accept/continuation/abort,globalretry/chunkagreement,replicatedNextN/fullsharedwidths/trunkARs,primary100401tuning/configinheritance. Route rank/fast/SNAP_GEN opt-in/off; snapshotmirror precedeslocalSave. RawGGUF O_DIRECT/driveprobe stayopt-in/offHIP,packedTPsourceunchanged. Neverreimportdeferredstream/vision/multifragments.

Use .venv/bin/python; STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py; pin3cf03257f219afbe7334045ff7c6a06ac68c627d unchanged. OrderedHIPmap→serve(-t .)→tools. IsolatedHIPbuild:gfx1151,HIPON,CUDAOFF,MMQON,TP2_RDMAON,-j4,compiler/opt/rocm/lib/llvm/bin/clang++,GGML_DIR canonicalpinnedcheckout. #52 archiveparentgitHEAD can causeharmlesshostrelinks; NEVERobjcopyinplace. InspectELFread-onlywithreadelfoffsets/Path.read_bytes.

Backups BOTHhosts: original5a52 in/tmp/maya-upstream-baseline-2026-11-01/;accepted38cff,39c313,40=8facc,41a617,42bc4,43–45b398 in/tmp/maya-upstream-vNN-baseline/maya;46acceptedpaths describedabove. Preserveallrawfailures/controlworktree /home/ugo/software/project-maya-v42-ds4-control. Sharedhelpers /tmp/maya-upstream-live.py,/tmp/maya-upstream-cli.py; noactivebatches. Use model_env.sh --model/--pack/--config sameBOTHranks; MTP opt-in withmatchingFIXED/DRAFT,ordinaryNO_MTP=1. StrictS256refs docs/benchmarks/tp2-mtp-review-2026-10-30;M/L64refs docs/upstream/v1.1.39. RequireordinaryANDMTPraw-pick0;preserveoutliers,keep900sstartupbound.
