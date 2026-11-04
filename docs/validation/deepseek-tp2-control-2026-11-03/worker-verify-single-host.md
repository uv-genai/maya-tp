# DeepSeek worker verification: isolated model checks

Issue #113. On strix-halo-1, an isolated gfx1151 build and the copied, hash-checked
Maya-XS pack exercised the experimental worker verification entry points. The
model source SHA-256 was
`9497e2fc9fde069f6ab8377dbaefaa6f9fd6748ff116cca8f9aa2cd2e088f6c7`.

With both reference and verification using the window path (`prefill_chunk=0`),
the model-backed test compared three-token target windows with zero, one and two
accepted drafts. It reset and replayed the same prompt for each comparison. In the
worker-only buffer case with no local drafter, the next-token logits were exactly
equal. With DSpark enabled, the next-token logits and five draft token IDs matched;
the corresponding confidence values were within 1e-4. The test also checks that
invalid acceptance counts do not change the pending position. With no speculation
buffers at all, both worker and ordinary verification reject the request before
touching model state.

An earlier test mixed the model's selectable chunked prompt path and window path:
it produced differing target logits and drafts. The paths use different numerical
operations, so that mixed-path comparison does not establish a rollback failure.
The exact-path comparison above passes; this is not a claim of parity across
prefill paths.

This is a small single-host test with short windows, not a validation of long
contexts, sampled acceptance, all rejection positions, or two-host inference. The
worker API is not yet connected to a distributed model engine. Tests used isolated
review binaries; no production executable was replaced.
