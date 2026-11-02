# v1.1.49 live validation evidence (#87 / #88)

This directory preserves the live validation of merged source `997bba7` and the first-failure
peer-health fix `e1682cb`. The final result is described in the parent integration report.
`summary.json` records all passing gates and both prefill orders; `ownership-final.json` confirms
all registered test processes ended. Exact-final-head/merge/main CI is tracked in #87/PR89.
The scripts contain the original absolute paths for this specific two-host experiment; they
are evidence, not general-purpose launchers. Do not run them against an occupied pair.

## Provenance and ownership

- Rank0: `strix-halo-1.local`; rank1: `strix-halo-2`; HIP/ROCm gfx1151; `usb4_rdma5`.
- Isolated root on both hosts: `/tmp/maya-v49-live-87`; API18087 / bootstrap18587.
- Same retained GLM binary on both hosts: MD5 `b398d4a0711430edd6a28b95264c7293`.
  GLM C++/transport/math/tuning sources did not change in the cumulative merge or #88.
- Merged archive `997bba7`: SHA256
  `44bb2781fe6a62768adaaffb893dcc63153172816bdf21fbf78ee20e09e7a8ad`.
- Fixed archive `e1682cb`: SHA256
  `a91d36523ffaf5bcdcf2d99ff1d7d3307ca5e456a916bb92be8ce6759c3e31e3`.
- Premerge server control: `d7626ac`, with the same GLM binary and pack/config.
- `pack-proofs.json` records matching pack-index/native-map/tokenizer hashes and dense sizes.
  API configs derive the retained S base, so display labels still say Maya-S; the explicit
  `--glm-pack` and tokenizer paths select S/M/L on both ranks, as recorded in each config/spec.
- `owned.py` records exact PIDs/start ticks and a unique environment owner marker, checks recorded
  descendants, and stops the owned server before its engines. Foreign/stale identities are refused.
  `ownership-tests.json` and `stdio-selftest-result.json` preserve the supervisor checks.

Production binaries/configs, protected remote source and unrelated services were not changed.
Source archives, model data and binaries are intentionally excluded from this evidence directory.
`manifest.json` lists file sizes and hashes; raw terminal logs retain original formatting.

## Data map

- `api-batch-fixed.txt`, `api_batch_fixed.py`, `pair.py`: sequential S/M/L ordinary and MTP7 API
  matrix, forced idle, reuse/stream/cancel/output limits, lifecycle/media/context guards and
  intentional peer loss followed by restarting both ranks. Per-model directories contain configs,
  process identities, raw rank/server logs, responses, round audits and recovery results.
- `api-batch.txt`, `Maya-S-v2-IQ2_XXS-mtp`: initial healthy-path passes followed by the deliberate
  peer-kill assertion failure; retained rather than discarded.
- `peer-candidate-delay1` and `peer-control-rerun-delay1`: merged/premerge controls both reproduce
  first poisoned-error health incorrectly remaining up. `peer-control-delay1` is the initial
  runner-only TIME_WAIT preflight failure. SO_REUSEADDR/partial-start logging were corrected;
  no transport deadline or health assertion was relaxed.
- `peer-cpu-before.txt`: expected failing first-failure regressions. `peer-cpu-after.txt` and
  `peer-cpu-final-clean.txt`: passing regressions. Earlier final-test logs retain fixture resource
  warnings; the final fixture closes engine pipes and the HTTP error response explicitly.
- `summary-audit-before.txt`: the final artifact auditor initially over-required poisoned text on
  L, whose first failure reaches the mirror-send guard. The corrected audit recognizes both fatal
  paths; S/M exercise queued poison. No live assertion or deadline changed.
- `cpu-fixed`: ordered HIP mapping, full server and installer/tools suites.
- `cli-fixed`, `cli_parity.py`: paired 32-token ordinary/MTP7 CLI trajectories; Maya-S is also
  compared to the accepted v1.1.42 trajectories. Maya-L retains the existing WARM=0 workaround (#42).
- `strict-*`, `strict_batch.py`, `bench_mtp_owned.py`: canonical benchmark adapted only for
  owned-process cleanup, isolated paths/ports and binary checking. Separate stdout/stderr readers,
  ABBA order, time bounds and maximum raw-pick differences0 are preserved. S technical/code use
  256-token historical references; M/L use 64. Warmups are retained separately from measurements.
- `prefill-ab`, `prefill-ab-reverse`, `prefill_ab*.py`: fixed/premerge server comparisons in both
  orders, with the same engine/config,
  normal idle, MTP2 and fresh 1K/4K/8K synthetic prompts. Three full cycles per arm; first cycle
  retained/excluded from timing medians. Additional API bias/Responses/Anthropic probes follow the
  candidate timing arm of the first comparison. All18 paired messages/usage match/cache0; pooled
  warm throughput changes are −3.64/−1.66/−1.95% at1K/4K/8K. Repeated-word data does not establish
  answer quality or general throughput. `prefill-*` pair directories retain configs/rank logs.

Cold expert-loading and first-request latency outliers remain in the raw data. The traced forced-idle
API runs and CLI cold loads are correctness tests, not resident-performance measurements. Existing
prefill/latency issues #33/#60 and the separate full DeepSeek GPU issues remain outside this fix.
The temporary owned benchmark does not resolve the legacy harness cleanup work in #83.
