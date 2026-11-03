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

The cache rehashes metadata but uses file identity and timestamps to detect
subsequent source changes; it assumes a trusted local manifest and filesystem.
It is not protection against a malicious actor or a change between the final
check and the engine opening a source. The helper is not yet connected to
production startup or the rank handshake. It does not by itself validate the
rank partition, binary identity, or running model state.
