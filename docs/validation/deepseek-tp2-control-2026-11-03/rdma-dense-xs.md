# Two-host DeepSeek XS dense attention projections

Issue #110. On two gfx1151 hosts, an isolated test executable with SHA-256
`cb8e37fe73142873e152bd6f912c02c06b875c80e8f23e55eda5f2ec8bdd644f`
loaded real XS layer-0 Q6_K attention matrix slices using the rank-local dense
loader. Deterministic synthetic activations were used.

For `wq_b`, each rank computed half the output rows; the gathered result matched
the full matrix's one-GPU GEMV exactly. For `wo_b`, each rank computed half the
input columns and exchanged its partial output over RDMA. The reduced results on
the two ranks matched exactly and differed from the full-matrix GEMV by at most
1.1920929e-7. The rank logs are in `rdma-dense-xs.txt`.

The identity used here is synthetic. This checks two individual projections,
not attention scores or KV state, router synchronization, a full layer, DSpark,
or end-to-end model inference. Both owned processes exited.
