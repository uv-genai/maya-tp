# TP2 collective regression validation — 2026-10-30

> These fixes ship in v1.1.38. Version/release status below records the validation state before the release bump.

HIP/ROCm gfx1151, two Strix Halo APUs, `usb4_rdma5` RDMA. Maya-S-v2 IQ2_XXS,
context 32768, one conversation. Binary MD5 `d6cd72ab0376c8df1d58cd1c2a20441b`
matched local `build-hip/maya` and rank 0 `build-hip/maya` + `/tmp/maya-tp2`.
Version stays v1.1.37; release #17 remains deferred.

Engine commits: `aee442e` (#18), `9623e7e` (#19), `a732362` (#20), `2e44a8a` (#21).

## Fixes

- #18: agree over RDMA before retrying an MTP window. A scratch clash on either rank
  makes both replay with per-token MoE collectives.
- #19: bound prefill batches by the actual receive payload and agree on chunk/layout
  choices. Hidden 4096 yields 3008 aligned rows, at most 49283072 bytes, below
  50331584. The exact reported 118521856-byte / 7234-row boundary has a host regression.
  Dense lead FFN slices now also receive the all-reduce previously bypassed by `continue`.
- #20: RAM experts already contain compact rank bands; copy them through the ordered
  GPU copy stream rather than slicing them a second time.
- #21: MTP window MLA computes only the rank's attention heads while retaining full
  buffer strides. HIP partial scratch capacity also accounts for head/KV geometry.

## Live API results

Both MTP off and MTP on with **seven drafts** completed this matrix. Raw usage,
timings and synthetic response text are in [simple.json](simple.json) and [mtp.json](mtp.json).

| Case | Prompt tokens | Output tokens | Result |
| --- | ---: | ---: | --- |
| Seven-draft window | 990 | 64 | both pass |
| Beyond receive-window payload | 7290 | 32 | both pass |
| Large-conversation reuse | 7333 | 24 | both pass; 7289 cached |
| Multiple prefill chunks | 14550 | 32 | both pass |
| Streaming follow-up to large prompt | >14550 | up to 24 | both reach `[DONE]`; peer up |

MTP run forced **eight local scratch retries on rank 1 only** via
`MAYA_TP_TEST_WINDOW_RETRY=8`. All eight matching rank-0 records report
`local-clash=0 retry=1`; rank 1 reports `local-clash=1 retry=1` at the same
positions. All **150/150** MTP round-end `(p0, produced, keep)` records match
across ranks, including subsequent cancellation tests. No collective mismatch,
receive-window overflow, nonfinite routes or memory faults in these final runs.
Diagnostic retry injection stays off by default.

Seven-draft `mtp_acceptance.py` passes three turns with reuse, cancellation after
eight streamed text chunks plus a successful follow-up, and output limits 1/2/3.

## Other gates

- HIP name mapping: pass.
- Serve unittest: 464 tests, 11 skips, pass.
- Tools unittest: 621 tests, 31 skips, pass.
- Host `tp_collective_test`: pass, via unittest compiler harness and CMake target.
- `glm_attn_bench --parity-only` on gfx1151: pass. New 32->64-head scratch regression
  at 2051 selected cells passes for all/tail/chunk masking; shared-head outputs
  are bit-identical and finite.
- CLI rank parity: **32/32 tokens** in simple and seven-draft MTP modes.
- Peer kill/recovery: **8 passed, 0 failed**, with seven drafts on both ranks, context 32768.
  Used `acceptance.sh` orchestration with the remote draft environment explicitly set to 7;
  peer death fails chat cleanly and degrades health, restarting both restores chat.
- Final state: both test engines and the test API server stopped; llama-swap daemon still running.
  Rank-0 YAML and its simple/MTP-seven entries remain as configured by the user.

## Reproduction

Start matching ranks with `CONTEXT=32768`. For MTP set `MAYA_TP_MTP=1`,
`MAYA_GLM_MTP_FIXED=7` and `MAYA_GLM_MTP_DRAFT=7` on both; set the diagnostic
retry count on rank 1 alone. Use canonical rank scripts directly from their own
hosts (never nested SSH). For the simple run use `MAYA_GLM_NO_MTP=1` on both.

```sh
.venv/bin/python tools/tp2/regression_collectives.py --mode mtp --output /tmp/mtp.json
.venv/bin/python tools/tp2/mtp_acceptance.py
# Restart both ranks in simple mode before the next command.
.venv/bin/python tools/tp2/regression_collectives.py --mode simple --output /tmp/simple.json
```

These are correctness tests, not a controlled performance comparison. Forced
retries, seven drafts and varying context length affect the timings; the older
#14 benchmark predates these correctness fixes and needs a fresh run for current
performance claims. Dense prefill summation and corrected MTP attention can
change generated answers.
