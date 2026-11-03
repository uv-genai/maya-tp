# DeepSeek TP2 control and RDMA coexistence regression

Issue #111, 2026-11-03. Isolated gfx1151 review builds, two hosts over `usb4_rdma5`, no model
processes and no production binary changes. `ds4_tp_channel_rdma_test` exchanges a test identity,
then rank 0 sends a control command and delays its RDMA exchange by 500 ms while rank 1 is waiting
for the collective. Both ranks complete the collective and rank 1 reads the queued control command.

* Before: detached source `6f1d64e`, with the same test and adapter added but the prior `verbs.cpp`;
  both ranks failed with `verbs peer closed the bootstrap connection` (rank 1 exit 1). Raw output:
  `before-rank0.txt`, `before-rank1.txt`.
* After: modified review build, both ranks passed (`after-rank*.txt`). The transport treats actual
  socket readability with data as pending control instead of EOF and suppresses repeated `POLLIN`
  while waiting; it still monitors errors and peer shutdown.
* Separate `peer-loss` run: rank 1 deliberately exits after identity exchange, rank 0 starts an RDMA
  operation, and asserts the original `peer closed the bootstrap connection` error plus sticky
  poison on the next operation. Both ranks passed (`peer-loss-rank*.txt`).
* Socketpair tests exercise non-consuming peek, suppression of a readable-data busy loop, clean
  half-close, and (where `POLLRDHUP` is available) half-close with unread data. The Python
  `serve.test_tp2_peer_loss` regression passed all three tests.
* The test's `worker` mode exercised a typed step, an actual scoped two-rank GPU partial exchange,
  transactional worker position, shutdown and acknowledgement over the real RDMA communicator.
  Both ranks passed (`worker-rank*.txt`). The callback is synthetic, not a DeepSeek model layer.

This validates the transport/control race and peer-loss behavior, not DeepSeek distributed inference
or the full GLM TP2/MTP model. The latter still needs isolated paired model validation before merge.
