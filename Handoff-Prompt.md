# Handoff-Prompt — Maya-TP restart

Read [AGENTS.md](AGENTS.md), then [Handoff.md §33](Handoff.md#33-prefill-promotion-and-user-acceptance-90--91--93--33). Sections 10–32 of the handoff are dated history and may describe superseded processes, issues, binaries or pending steps.

## Current state

- PR [#94](https://github.com/uv-genai/maya-tp/pull/94) merged as `6f85d3036a1010b3a2f1b0cfca2105b3f28fdb94`. It includes HIP APU complete-pool sizing, TP2 startup scratch bounded by the existing receive window, and gfx1151/hipBLASLt 100401 DSA projection tuning. Issues #33, #90, #91 and #93 are closed at the user's request.
- The user tested Maya-S, Maya-M and Maya-L through llama-swap and reported working prefill and generation. Their eBPF tutorial prompt averaged 30, 32 and 28 generation tok/s respectively; [report and raw results](docs/benchmarks/tp2-prefill-2026-11-02/README.md).
- The promoted `build-hip/maya` on both hosts has MD5 `cbd9e99a6a49e9107b0db3190a40766a`. Both hosts have matching tuning and model configs. The original rank0 llama-swap configuration was restored; the user restarts and launches their own services. Previous files are backed up on both hosts in `/home/ugo/software/project-maya-before-prefill-90-promotion`.
- Acceptance is not a bit-exactness claim: a strict Maya-S historical reference differed on 12 of 256 raw picks, and one cold Maya-L baseline response differed from repeats. Full candidate API/CLI regression was not repeated. The existing 3008-row receive limit and cold/startup variability remain. Preserve the evidence; don't claim these were fixed.
- The checkpoint tag is `tp-v1.1.43`, with numeric VERSION `1.1.43`; no GitHub release was requested. No further prefill investigation or deployment is requested. Push only to `origin`.

## Operating rules

The user owns their processes. For new validation, record exact owned PIDs and start times; stop only processes you started and stop an owned server before its engine. Do not use name-wide or port-wide cleanup or inspect unrelated services. The legacy `acceptance.sh` and `bench_mtp.py` cleanup is tracked in #83. Prior permission to use the hosts is not standing permission to stop user processes.

The usual workspace is on strix-halo-2 (rank 1); SSH directly to strix-halo-1.local (rank 0) from there, never nested SSH. Use matching model/pack/config on both ranks via `tools/tp2/model_env.sh`. Never probe the bootstrap port with `/dev/tcp`. Restart both ranks after a poisoned communicator; never clear poison, retry a failed collective, or restart one rank alone. Preserve #88's first-failure health guard, #59's completed mirrored prefill before cancellation, #57's total output budget, #64's dual-lane wake-up, #62's unmirrored-operation guards and #69's MTP bias guard.

Use `.venv/bin/python` without resolving the symlink, `STRATA_GGUF_PY=/home/ugo/software/project-maya/third_party/llama.cpp/gguf-py`, and the pinned llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Run the ordered HIP mapping, server and tools checks in AGENTS.md. The remote canonical checkout contains user edits; do not pull, reset or rebuild it as routine test setup. Use isolated review builds for future tests.

Outstanding issues include #10, #11, #16, #42, #46, #52, #54, #60, #68, #75, #79 and #83; consult GitHub for current status. Full DeepSeek WMMA failures (#68), tail draining (#75), and default-off gfx1151 chunk-cap measurements (#79) remain unresolved. Historical integration evidence and prior GPU validation are linked from [docs/README.md](docs/README.md).
