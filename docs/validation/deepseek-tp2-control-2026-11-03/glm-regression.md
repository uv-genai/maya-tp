# GLM TP2 and MTP regression after shared transport changes

Issues #110 and #114. An isolated gfx1151 HIP review build at source checkpoint
including the shared `CheckPeer` and bootstrap EOF changes was copied to both
hosts (SHA-256 `467170ceb00479d390ed6b12e11a8c15804fcc4abc7e0564c1be7f8bbc1c7a0c`).
No production binary or service was touched. Both ranks used the same existing
Maya-S-v2-IQ2_XXS pack, context 4096, USB4 RDMA, the 12-token fixture
`154822,25062,287,29905,371,25,7487,785,6722,315,9621,374`, and greedy
32-token CLI decoding. The two processes were individually launched and
recorded; no name-wide cleanup was used.

With MTP disabled, the two ranks' 32 token IDs matched exactly. With TP2 MTP
enabled and draft width 2, the two ranks again matched all 32 tokens; the
engine reported 20 of 21 proposed drafts accepted. MTP and non-MTP output
also matched on this prompt. All owned processes exited. This single short CLI
fixture is not a throughput comparison or a replacement for serve, cancellation,
peer-loss and long-context regression checks. Logs use
`/tmp/ds4-tp2-glm-{simple,mtp}-rank{0,1}.log` on their respective hosts.
