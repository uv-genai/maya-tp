# Two-host replicated DeepSeek model control smoke

Issue #110, 2026-11-03. Both hosts ran isolated copies of the same gfx1151 review
test executable (`sha256 dbf97905a16dd60ef166ebf99fa91d6a933806bee2abf81201e92574110421f3`)
with their previously compared Maya-XS files. Each rank loaded a **complete model**;
rank 0 loaded DSpark and rank 1 loaded no drafter but reserved worker verification
buffers. This is a control and state test, not tensor-parallel model inference.

The test used the DeepSeek control codec and worker dispatcher over the existing
two-host communicator, executing feed, step, a three-row verification, a verdict
accepting one draft, step and shutdown. Each model then evaluated the same probe;
their 129,280 logits were exchanged using RDMA and compared. The controlled run
used window-only prefill, `DS4_STREAM=0`, 14 CPU threads, mmap experts and no GPU
expert slots on either rank. Both ranks passed with maximum absolute logit difference
zero. `replicated-fixed-rank0.txt` and `replicated-fixed-rank1.txt` contain their logs.

The first run used the engine's adaptive expert-lane calibration and failed the
comparison with a maximum absolute difference of 5.6557. Each host chose a different
CPU/stream split, so the experiment changed both model execution path and host;
the controlled rerun establishes matching state under one fixed path, not which
individual operation caused the first divergence. Preserve those failed-run logs
(`replicated-first-rank*.txt`): rank-local adaptive decisions and numerical
differences can change routes, and the distributed implementation must ensure both
ranks make compatible decisions and share authoritative router information.

The handshake uses a synthetic fixed identity in this smoke, not a production model
fingerprint. The test does not establish sharded allocation, reductions inside the
model, sampled speculative decoding, throughput, or long-context correctness.
Both owned processes exited; no user service or production executable was changed.
