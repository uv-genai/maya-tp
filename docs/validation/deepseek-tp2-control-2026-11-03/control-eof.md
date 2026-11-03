# TP2 control EOF reporting

Issue #114. A review test intentionally closed its rank-0 peer while rank 1
was receiving a control message. Previously `recv` returning zero could be
formatted with stale `errno` as `control recv: Resource temporarily unavailable`.
The receive path now reports `verbs peer closed the bootstrap connection` for
that EOF; other receive errors keep their system error.

On two gfx1151 hosts over `usb4_rdma5`, a new real-RDMA `control-eof` case
confirmed that the rank-1 Worker reports that cause and retains it on the next
call. The existing real-RDMA collective peer-loss test still reported its
sticky poison and original peer-closed cause; queued control during an RDMA
exchange and typed worker dispatch also passed. The host socket-state test and
`python -m unittest serve.test_tp2_peer_loss` passed. All owned processes exited.
The review test executable SHA-256 was
`23b7878668789b35e8a401aae9f873235750bf82c165418a04d4a802f74fd165`.

This validates the socket and worker failure boundaries; it is not a complete
DeepSeek API or production pair-down test.
