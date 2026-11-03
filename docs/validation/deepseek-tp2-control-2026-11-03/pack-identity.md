# DeepSeek TP2 pack identity checks

Issue #111. The review helper `tools/ds4/tp2_identity.py` computes a
path-independent SHA-256 identity from the normalized pack configuration,
tensor index, tokenizer files and every indexed GGUF source. It checks indexed
tensor spans and records local file signatures for later checks. Both hosts
independently read the full packs and obtained the following identities:

| Pack | Identity |
| --- | --- |
| Maya-XS | `0c4933e6b70f53f7eff163a4053f8970fce470cacc4065983ebe28bd507c57af` |
| Maya-S | `e8d41763b86d0ed5430202521f42c1089205d3bf14afc2e0308ca82e1d481c71` |

The indexed sources include Engram, the DSpark drafter and vision. Cached
checks on each host succeeded; an intentionally altered cached source
signature was rejected. Nine synthetic tests also cover changed files and
metadata, redirects, invalid spans and atomic no-overwrite publication.

The helper also derives a partition fingerprint from pack geometry and hashes
the selected engine executable. On both hosts, preflight for the S pack and
the same review executable returned partition fingerprint
`b71b12627fd73832ac3ebace373bcd574b63f46f021104d711de5f9900304e94`
and executable SHA-256
`59b50149213fd42c3e2e3126f6b40f410846747ed93c9c4b249d5102f56a516f`.
Its geometry vectors match the C++ partition unit tests; this is still
preflight evidence, not a handshake using these values.

The cache rehashes metadata but uses file identity and timestamps to detect
subsequent source changes; it assumes a trusted local manifest and filesystem.
It is not protection against a malicious actor or a change between the final
check and the engine opening a source. The helper is not yet connected to
production startup or the rank handshake. It does not by itself validate the
rank partition, binary identity, or running model state.
