# Handoff-Prompt — Maya-TP restart

Read Handoff.md (especially §§18,21,23,27–29 and latest §29.29), then AGENTS.md.

## Objective and current checkpoint

Integrate upstream v1.1.37–v1.1.46 ONE TAG AT A TIME. Semantic review and relevant validation, exact-final-head CI and post-merge main CI must pass before starting the next tag. Continue through46. Preserve TP2/MTP and fork-owned README/AGENTS/docs; push only origin uv-genai/maya-tp. No release is authorized by integration. Future fork release tags are tp-vMAJOR.MINOR.PATCH; VERSION stays numeric and published legacy tags immutable. Upstream tags are upstream/v*.

**37–45 semantic checkpoints accepted.** Latest45 PR76/2df9e26, final6f1e78c CI41346480733/41346487052,post-main41346856373PASS.74closed; functionalDS4tailfixDEFERRED75OPEN.44/43acceptedreports/handoffsretainfullprovenance.

**46 runtime COMPLETE (#77/#78/#80, draftPR81, merge/upstream-v1.1.46).** Upstream3589d6f,code5b13c83.605serve11skip/679tools33skip/HIPmap,99focused,18host,HIPDS4buildPASS. Newcapdefault0(gfx1151only,bounds/aliases78);79measurementOPEN,no5%claim. Contentparts/malformedparameterfixes+80capcount. Windows-onlytrimexcluded. BothGPUimagesunchanged;GLMb398,DS4host51749e86.

**No active batches/builds.22871EXIT0:** MTP7+ordinarylarge/reuse/stream/guards/idle,real18Kprefillcancel/cache0retry/followup/peerPASS;132rounds/8asymretriesmatch. FreshtextpartsAPI20021initial/cache0→20020cached/collapsed0,ordinary/MTP64followupsPASS. Directunitisolates80;OpenAItextonlypartsflattenstr,soAPIprobeisformat/prefixendtoend,notindependent80A/B. Initialbadparserfixtureexpectationandallbeforefailuresretained. Donotreruncompletedgateswithoutnewreason.

**Remaining:** finalexactheadCI,PR81merge,post-mainCI. Thenmark46accepted/saveGLMb398+DS45174baselinesBOTHhosts,close78/80; finish77wholeobjective: restoreoriginal5a52toONLYlocalcanonicalbuild-hip/maya,remotecanonicalbuild-hip/maya,andremote/tmp/maya-tp2. Keep BOTHreviewGLMb398/DS45174 andsource. ReviewS/M/Lconfigsnow.exe review/build-hip/maya(verifiedlive), so restoration keepsstackscoherent. Protectedremote777userwork/backups/daemon175392/YAMLmustremainunchanged. Thenfinalhandoffs/index/ancestry/hash/CIchecks,close77andindex36,andfinalansweronlywhencomplete. No newrelease;futuretagstp-v*.

GLMsource6a2fbe11aeae4fbc0242f0fa2d3cf2ba9d510316 MD5b398d4a0711430edd6a28b95264c7293,DS4hostcode5b13c83 MD551749e860763aea34847b183de95a933. Remote review5b13c83;canonical777unchanged;engine/serverstopped. Original5a52cd27167523b6ea4b0f0bde93f503 in/tmp/maya-upstream-baseline-2026-11-01/maya BOTHhosts;remote also maya-build/maya-tmp same5a52. YAML SHA256c5392d8c696f5ff8869510057427675b18f8a3cb482d0a136012495151297a20. Baselines38–45retained;45b398same43/44.79/75/68and33/42/46/52/54/60/11/16remainOPEN.

## What43 verified

Ordered HIP map,602server/11skip,677tools/33skip,Node pass;3gfx1151HIP engines,17host,affectedEngram parity pass. Full GLM GPU image SHA95821268… equals42. Reuse42M/L/math evidence only for unchanged executed paths; no fresh43M/L startup/perf claim. Broader DS4 stress fails #68, reproduced independently from accepted42 b1867c8:3alternatingKVruns/arm control[1,2,1], candidate[1,1,2], unchanged WMMA function. Control worktree /home/ugo/software/project-maya-v42-ds4-control. All failed raw runs retained.

Simple/MTP7 large/reuse/stream/lifecycle/media/idle and real18Kpartial-prefill disconnect/cache0retry/followup pass.133round tuples/8asymmetric retries match. CLI32/rank/mode exactly42, recovery8/8. Strict technical/code256warmup+measured16records ordinary ANDMTPraw-pick0. FreshprefillABvs42 all9messages/usageequal/cache0, warm+0.94/−0.36/−0.37% at990/4062/8158. Both live executable hashes verified in both arms. No material regression/new general gain claim.

Long-loop policy A/B JSON0vs32000 preserves20020cached instead of0; follow-up prefill61.76→1.14s for syntheticfixture only. Initial model content/usage equal after normalizing only random allocated tool IDs; raw data retained. Follow-up history/answers intentionally differ. #33 remains open for original cold9.6tok/s/~3008rowcap/missing MAYA_TP_PREFILL_ROWS. #60 remains open for latency variability;43short exaggerated-idle MTP491.19ms/token outlier retained.

#66 stops length before oversized internal GEN when a server note exhausts context; retain total client budget. #67 JSON whitelist/aliases and legacy fallback. #69 rejects nonempty logit_bias before state/table/mirror under TP2 MTP; explicit spec1 ordinary bias works. Normal/forced4096context probes pass; forced EOS is branch coverage, not fidelity. #71 benchmark separates stdout/stderr readers/queues/logs: old diagnostic fragment spliced into DONE; interrupted run retained. Real pipe regression reproduces2errors before/pass after. Production server already separates streams.

## Safeguards for all later merges

- HIP/ROCm gfx1151 Linux ONLY; RDMA tensors ONLY (~37Gb/s,~6µs). TCP only bootstrap/control/token metadata. No new backends/NVIDIA work. No subagents authorized.
- #59 rank0 progress callback must keep `return tp2 || !read_stopped();`: finish ENTIRE mirrored prefill call before STOP/reset. Single-host earlycancel unchanged; TP always mirrorsRESET. KEEP_CANCELLED stays off/single-host-only. Regression tools/tp2/prefill_cancel.py.
- #56/#53 reject single-rank load/unload/idle/dead-engine/context/current-model restart before mutation. Relaunch BOTH ranks. Preserve #47 failed-start ended/reaping/INFO/routes.
- #57 internal continuations share TOTAL client output budget (`while n < max_new`), plus43 context-room bound. Preserve all-limit answer reserve, bounded soft grace, finalSTAT and cancel drain.
- #64 wake BOTH CPU service lanes at mirrored calls and nested-MTP boundaries, including rank1 worker input poke. #62 reject TP GENI/VLEND/VRECLAIM/LOGP before mutation, media before encoding, skip TP encoder startup; last repeated world flag wins.
- Preserve canonical MTP input/accept/continuation/abort, global retry/chunk agreement, replicatedNextN/fullsharedwidths/trunkARs, primary100401 tuning/config inheritance. Route rank/fast/SNAP_GEN opt-in/off; snapshot mirror precedes localSave.
- #46 streamed fetch deferred ENTIRELY: reader/header/kernels unchanged, even MAYA_GLM_STREAM_FETCH=1 does nothing. #11 multi and #16 HIP/RDMA vision remain deferred. Exclude unsupported platforms/same-host multi-GPU additions. Raw-GGUF O_DIRECT and new drive probe stay opt-in/off on HIP pending measurement; packed TP source unchanged.

Keep33/42/46/52/54/60/11/16/68/75/79 open unless a later change actually resolves them. #42 L CLI uses WARM=0; no ready handshake. #54 transient rank0 config hides Models catalog. #52 archive metadata may inherit parentgitHEAD and cause harmless host relinks; pin unchanged. Never objcopy in place: inspect ELF read-only with readelf offsets and Path.read_bytes.

## Hosts and tools

User authorization persists: “Hosts are free; you may stop the pair for validation.” This is strix-halo-2 rank1. SSH ONLY strix-halo-1.local rank0; no nested SSH or wrapper over SSH. Protected remote canonical /home/ugo/software/project-maya remains777cc92 with user maya.py/tuning/rank1 deployment edits and2Lconfigbackups. NO pull/reset/rebuild there. Preserve daemon175392 and six-entry YAML (Lcontext524288). Use isolated /home/ugo/software/project-maya-upstream-review on both hosts.

Stop rank0 server BEFORE engine:
```
ssh strix-halo-1.local 'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'
pkill -9 -x maya
```
Never broad pkill-f-server.py or /dev/tcp probes. Start rank1 first, then rank0 via fully redirected direct SSH. Restart BOTH after poisoned communication. After engine builds deploy/hash-check local review/canonical, remote review/canonical and rank0 /tmp/maya-tp2.

Backups BOTH hosts: original5a52cd27167523b6ea4b0f0bde93f503 in /tmp/maya-upstream-baseline-2026-11-01/; accepted38cff75b48,39c3132ee4,40=8facc62c,41a617,42bc4e90a6,43b398d4a0 in /tmp/maya-upstream-vNN-baseline/maya. Rejected42d8 and41pre-cancel13a separately preserved. At FULL task end leave coherent production runtime without overwriting protected user source/YAML; original canonical binary restoration is available, retaining final validated review builds.

Use .venv/bin/python; STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py; pin3cf03257f219afbe7334045ff7c6a06ac68c627d unchanged. Ordered HIPmap→serve discovery(-t .)→tools. Build gfx1151,HIP ON,CUDA OFF,MMQ ON,TP2_RDMA ON,-j4,compiler/opt/rocm/lib/llvm/bin/clang++,GGML_DIR canonical pinned checkout.

Reusable /tmp/maya-upstream-live.py and /tmp/maya-upstream-cli.py. Same model/pack/config selection BOTHranks via model_env.sh. MTP needs MAYA_TP_MTP=1 and matching MTP_FIXED/DRAFT; simple GLM_NO_MTP=1. Strict S256 refs docs/benchmarks/tp2-mtp-review-2026-10-30; M/L64 refs docs/upstream/v1.1.39/{Maya-M,Maya-L}/reference.txt. Require ordinary ANDMTPraw-pick0; preserve failures/outliers; do not relax900s startup bound. tools/tp2/bench_mtp.py now keeps separate rank0 protocol and stderr diagnostic logs (#71).
