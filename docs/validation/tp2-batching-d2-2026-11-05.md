# DeepSeek checked batch protocol — D2

2026-11-05, [#150](https://github.com/uv-genai/maya-tp/issues/150).
The [wire contract](../DEEPSEEK-TP2-BATCH-PROTOCOL.md) adds bounded negotiated
metadata and per-slot state without enabling production model batching yet.

Seven CPU suites pass with assertions active and warnings as errors: control,
identity, worker state, model worker, channel, batch codec/state and batch
channel. Codec/state also pass ASan/UBSan. Tests cover malformed/truncated
frames, stale identities, incomplete verdicts, canonical retained prefixes,
Feed versus metadata limits, acceptance/completion ordering and sticky errors.
CI runs these suites without ROCm using only an opaque stream-type declaration.
HIP engine/fixture builds and the HIP mapping gate pass.

The pack-free paired gfx1151 fixture passes ragged verification, whole-plan
commit, generation reuse and three integer-exact queued RDMA reductions
(10, 4 and 2 rows), exiting 0/0. Existing worker dispatch and intentional
control-connection loss also pass, exiting 0/0. Owned identities and logs are
under `/tmp/maya-batching-validation/d2-production-path` and
`d2-legacy-{worker,control-eof}`; all owned processes exited.
Frozen fixture SHA256:
`1443ea0da10c27843adce502bcff68c8f30de1b74bb81e29852a5154439ef48f`.

The first synthetic reduction used dormant WaitAdd and failed a row sum.
The production engines use WaitValue plus AddRowsBroadcast, which passes.
The unrelated helper defect remains open in #154; no transport arithmetic was
changed to hide it. This phase establishes control safety, not model-row
correctness or throughput. D3–D5 carry those gates.
