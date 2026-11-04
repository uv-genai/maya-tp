# Two-host DeepSeek XS expert arithmetic

Issue #110. Isolated gfx1151 test executable SHA-256
`883ee168508d61d0dba8a923d7b7a652bb8130b5a89f5e184f54a28fa343af2b`
loaded the real XS expert 0 separately on both hosts with the rank-local loader.
Each rank ran its gate, up, activation and down operations on its own FFN slice;
the output partials were exchanged and summed over RDMA. The resulting rank
outputs matched each other exactly. Rank 0 also ran the same real expert in full
on one GPU and compared the result with the distributed sum.

| Layer | Rank widths | Maximum absolute difference from full expert |
| --- | --- | ---: |
| 0 | 1280 / 1024 | 1.1175871e-8 |
| 1 | 1024 / 1280 | 7.4505806e-9 |

The two layers use different down quantization; the test uses a deterministic
synthetic input and one routed expert. Logs are in `rdma-expert-xs.txt`. The
handshake has synthetic test identities; there is no production pack negotiation,
router agreement, attention, shared expert, full layer, DSpark, or end-to-end
model execution in this test. Both owned processes exited.
