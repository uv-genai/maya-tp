# Handoff-Prompt — Maya-TP restart

Read [AGENTS.md](AGENTS.md), then Handoff.md §47 for the sequential upstream v1.1.50–v1.1.53 integration. The source incorporates the applicable changes through v53, with isolated GLM TP2/MTP and DeepSeek TP2/DSpark checks per tag. PRs #126/#129/#131/#134 completed the batch at `de184d9`. Final exact-head push CI 41833758436, PR CI 41833925330 and merged-main CI 41834772034 passed. Issue #132 records completion. Reports and raw evidence are in `docs/upstream/`. Deferred v53 engine experiments are tracked in open #133. No owned model processes remain. Accepted user runtime paths are in §§33/44, and user acceptance/main merge history is in §§45–46. Preserve the protected remote checkout and use isolated builds for future validation. Handoff.md §48 records the post-v53 documentation audit and publication status in #135.

The latest DeepSeek startup repair is in Handoff.md §49 and issue #138.
Btrfs device renumbering invalidated the old identity caches. Both runtime
helpers now use stable filesystem/subvolume identity, and both hosts' XS/S
manifests were fully rehashed and refreshed. Paired preflight passes, and the
user confirmed Maya-S running through llama-swap, closing #138. Handoff.md §50
and #139 record the user's 1–4 client concurrency benchmark. All 20 requests
succeeded, with aggregate throughput around 28 tok/s and queueing latency
increasing with concurrency. Publication of the identity fix, benchmark and
related handoff updates on `main` is tracked in #138/#139. Those issues record
the publication commit and CI outcomes.
DeepSeek TP2 continuous batching with DSpark is tracked in open #140, with
the initial design and phased implementation plan pending. GLM multi-sequence
tracking remains in #11.

## Current GLM and DeepSeek state

The user accepted the DeepSeek TP2/DSpark work after hours of successful testing
across most DeepSeek and GLM models, including DeepSWE benchmarks. The work is
merged through PR #122 at `75e18e8`, and issues #116/#121 are closed. Use `main`
as the source development entry point. Feature-head and post-merge CI passed
HIP mapping, 618 server tests and 721 tools tests, with platform-dependent skips
recorded in Handoff.md §45. Target inference, chunked prefill, DSpark and prompt lookup
have been exercised on XS and S. Numerical/capacity follow-ups remain in #110/#112.
Read §43 and #118 for
the staged runtime, config choices and ownership-safe stop actions. Issue #119
installed `maya-deepseek-xs-mtp` and `maya-deepseek-s-mtp` in llama-swap on
strix-halo-1.local. The user launches from that host. Issue #120 replaces
blocking collectives with GLM's queued GPU/RDMA
sequence and GPU router comparison. XS generation measured 23–24% above the
same single-host build on two prompts. Both review configs select the measured
versioned binary. Focused correctness checks passed. The user reports Maya-S
eBPF generation slightly above 30 tok/s, prefill similar to XS and a larger
relative speedup. Issue #120 is complete, and extended user trials complete
operational acceptance in #116. Do not restart the broad
validation campaign; the user explicitly requested a few checks and then testing
through the existing entries. No owned validation processes remain.
Use Handoff.md §47 for the source checkpoint and §§33/44 for accepted runtime selection. §42, the implementation plan and
phase issues #110–#116 record the implementation history and separate follow-ups.
Review any recorded owned test PIDs before acting and preserve user-owned processes. The review
procedure and accepted deployment paths are in
[docs/DEEPSEEK-TP2-RUN.md](docs/DEEPSEEK-TP2-RUN.md) and Handoff.md §46.

For the remote GLM startup repair and Maya-L cold-load investigation, read Handoff.md §§38–39
and issues #104–#105. The user reports that everything is working; Maya-L has completed TP2
requests. The cold-loading slowdown remains under investigation.

Read Handoff.md §§35–37 and issues #100–#103. DeepSeek Maya-XS measurements and comparison
limitations are in `docs/benchmarks/deepseek-maya-xs-2026-11-02/`; the DSpark calibration watchdog
is tracked in #101. The larger DeepSeek Maya-S ran slowly with memory pressure and heavy storage
reads (#102). Separate XS and S configs prevent the llama-swap XS entry from silently selecting S
(#103). Read from any repository, but open issues only on uv-genai/maya-tp.

The earlier single-host DeepSeek investigation was merged at `76e0279`.
The later TP2/DSpark implementation and user acceptance are recorded by PR #122.
Treat model processes as user-owned and preserve their recorded ownership boundaries.

## Accepted GLM runtime and checkpoint

- PR [#94](https://github.com/uv-genai/maya-tp/pull/94) merged as `6f85d3036a1010b3a2f1b0cfca2105b3f28fdb94`. It includes HIP APU complete-pool sizing, TP2 startup scratch bounded by the existing receive window, and gfx1151/hipBLASLt 100401 DSA projection tuning. Issues #33, #90, #91 and #93 are closed at the user's request.
- The user tested Maya-S, Maya-M and Maya-L through llama-swap and reported working prefill and generation. Their eBPF tutorial prompt averaged 30, 32 and 28 generation tok/s respectively; [report and raw results](docs/benchmarks/tp2-prefill-2026-11-02/README.md).
- The promoted `build-hip/maya` on both hosts has MD5 `cbd9e99a6a49e9107b0db3190a40766a`. Both hosts have matching tuning and model configs. Rank0 llama-swap's health timeout was later raised to 1800 seconds as a cold-start workaround (#105). Previous files are backed up on both hosts in `/home/ugo/software/project-maya-before-prefill-90-promotion`.
- Acceptance is not a bit-exactness claim: a strict Maya-S historical reference differed on 12 of 256 raw picks, and one cold Maya-L baseline response differed from repeats. Full candidate API/CLI regression was not repeated. The existing 3008-row receive limit and cold/startup variability remain. Preserve the evidence; don't claim these were fixed.
- The checkpoint tag is `tp-v1.1.43`, with numeric VERSION `1.1.43`; no GitHub release was requested. No further GLM prefill investigation or deployment is requested. Push only to `origin`.

## Operating rules

The user owns their processes. For new validation, record exact owned PIDs and start times; stop only processes you started and stop an owned server before its engine. Do not use name-wide or port-wide cleanup or inspect unrelated services. The legacy `acceptance.sh` and `bench_mtp.py` cleanup is tracked in #83. Prior permission to use the hosts is not standing permission to stop user processes.

The usual workspace is on strix-halo-2 (rank 1). SSH directly to strix-halo-1.local (rank 0) from there, never nested SSH. GLM model selection uses `tools/tp2/model_env.sh`. DeepSeek uses its per-model `maya-ds4-tp2-*.json` launcher configs and `tools/llama-swap/tp2-ds4.sh`. Keep pack contents, executable digests, context and feature settings matched across both ranks. Never probe the bootstrap port with `/dev/tcp`. Restart both ranks after a poisoned communicator. Preserve #88's first-failure health guard, #59's completed mirrored prefill before cancellation, #57's total output budget, #64's dual-lane wake-up, #62's unmirrored-operation guards and #69's MTP bias guard.

Use `.venv/bin/python` without resolving the symlink, `STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py`, and the pinned llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Run the ordered HIP mapping, server and tools checks in AGENTS.md. The remote canonical checkout contains user edits; do not pull, reset or rebuild it as routine test setup. Use isolated review builds for future tests.

Outstanding issues include #10, #11, #16, #42, #46, #52, #54, #60, #68, #75, #79 and #83; consult GitHub for current status. Full DeepSeek WMMA failures (#68), tail draining (#75), and default-off gfx1151 chunk-cap measurements (#79) remain unresolved. Historical integration evidence and prior GPU validation are linked from [docs/README.md](docs/README.md).
