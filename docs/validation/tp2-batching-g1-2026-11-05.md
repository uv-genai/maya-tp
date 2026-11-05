# GLM batching G1 state validation

Checkpoint on 2026-11-05 for #144, following C1 at `2ba0803`.
The paired HIP build targets gfx1151 with APU RDMA on `usb4_rdma5`.
Rank 0 was strix-halo-2, rank 1 strix-halo-1, bootstrap port 19151.

## Scope and outcome

The historical multi-sequence prompt GPU fault did not reproduce with the
current source in these 32K-context cases. The serving scheduler remains gated
during G1. G2 exercises its actual row messages and admission path. Larger
contexts and the other GLM quants retain their later capacity gates.

The arena-layout guard now checks every KDA and DSA offset vector against the
initial allocation. The new manual `glm_tp2_sequences_test` runs the same prompt
through every slot, checks positions and logits, saves per-slot snapshots,
advances each slot and restores them in reverse order. Replay logits agree
within an absolute 1e-4, and prompt picks agree between ranks. Its explicit
checks execute in Release builds.

| Case | Result |
| --- | --- |
| Default one-sequence TP2/MTP, two 32-token responses | Passed. 37 of 47 proposals accepted in total. |
| Two allocated arenas with the legacy scheduler | Passed. Both 32-token outputs exactly match the one-sequence control. MTP remains disabled by the existing multi-arena loading gate. |
| Four 32768-token slots, 28-token prompt, bind/snapshot/replay | Passed on both ranks. Both processes exited 0. |

The extra arenas and all snapshots cost 2.06 GB beyond the primary 0.58 GB
arena. The routed-expert pool remained complete at 42.45 GB. Free device bytes
were 64,903,409,664 on rank 0 and 64,908,959,744 on rank 1. This is the measured
32K/4-slot capacity checkpoint for Maya-S. The report retains the raw token
controls, counters, memory values and binary hashes.
[Machine-readable results](tp2-batching-g1-2026-11-05.json).

## Reproduction

Build `glm_tp2_sequences_test` with HIP, RDMA and tests enabled. Supply the same
pack, token file, environment and requested context/slot count on each rank.
The checked token file contains whitespace-separated vocabulary IDs.

```sh
# Rank 0 on strix-halo-2
glm_tp2_sequences_test 0 192.168.1.92 19151 usb4_rdma5 \
  /home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack \
  32768 4 prompt-ids.txt rank0.json

# Rank 1 on strix-halo-1
glm_tp2_sequences_test 1 192.168.1.92 19151 usb4_rdma5 \
  /home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack \
  32768 4 prompt-ids.txt rank1.json
```

The validation reused the S config's prefill/tuning environment, with a 16 GB
host-memory reserve and 1024 MB device reserve. The fixture sets its requested
sequence count and uses the existing model allocation and snapshot APIs.

## Evidence and ownership

Logs and owned process identities are under
`/tmp/maya-batching-validation/g1-{control-p1-r2,baseline-p2,four-slots-r2}/`.
The control engine is frozen on both hosts under
`/home/ugo/software/project-maya-batching-validation/baseline-72d9260/`.
The paired fixture is under the sibling `g1/` directory.

Two preliminary harness attempts ended before model validation because of
remote session setup and a missing local token-file copy. The corrected
coordinator records child identities and cleans up those exact processes.
The legacy serving worker exits after observing the owned leader's shutdown
socket close. The four-slot fixture uses a final peer exchange and exits cleanly.

HIP mapping and both affected HIP targets built successfully. Python source
and behavior match C1's passing full suite. The phase issue records exact
commit CI and subsequent integration status.
