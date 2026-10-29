# TP2 run book — Maya GLM Maya-S on two Strix Halo hosts

Ops notes for the two-host tensor-parallel pair (design: [TP2-DESIGN.md](TP2-DESIGN.md),
plan: [TP2-PLAN.md](TP2-PLAN.md)). **HIP/ROCm gfx1151 Linux only. Interconnect: Thunderbolt RDMA
(`usb4_rdma*`), ~37 Gb/s / ~6 µs target.**

## Hosts

| Host | Role | IP |
| --- | --- | --- |
| `strix-halo-1.local` | rank 0 (API server) | 192.168.1.22 |
| `strix-halo-2.local` | rank 1 (worker) | 192.168.1.92 |

Same binary on both hosts (`sha256sum` must match). Start **rank 0 first**, rank 1 within 30 s.

## RDMA link

```sh
ibv_devices                      # usb4_rdma* (lowest GUID = pick, both hosts)
ibv_devinfo -d usb4_rdma5 | head # state: PORT_ACTIVE, link_layer: Ethernet (RoCE v2)
sudo cat /sys/kernel/debug/thunderbolt_ibverbs/summary | grep striping
#   native_write_striping: 1  →  good (one QP over both cables)
```

- Links should train at **20 Gb/s × 2 lanes** per cable (`/sys/bus/thunderbolt/devices/*/rx_speed`).
  `10.0 Gb/s × 2` is a 20 Gb/s cable or poor training — try re-plugging.
- RoCE GID needs an IPv4 on dummy netdev `tbv0` (10.77.0.1/24 vs .2).
- After a cable re-plug: wait ~15 s for the link; if `usb4_rdma*` names shift, reload
  `thunderbolt_ibverbs` (`sudo rmmod thunderbolt_ibverbs && sudo modprobe thunderbolt_ibverbs`).

## Exchange test (no model)

```sh
# both hosts, same --rdma-device (lowest GUID)
build-tp2/tp_comm_rdma_test --rank 0 --bootstrap-port 18515 --rdma-device usb4_rdma5 --bench
build-tp2/tp_comm_rdma_test --rank 1 --bootstrap-host strix-halo-1.local \
    --bootstrap-port 18515 --rdma-device usb4_rdma5 --bench
```

Expected: `rdma_ready … link=roce_v2 gid_index=1 mtu=4096`, `tp_comm_rdma_test OK`,
`allreduce rows=…` lines. Compare against [#3](https://github.com/uv-genai/maya-tp/issues/3).

## Failure playbook

| Symptom | Cause | Fix |
| --- | --- | --- |
| `identity or size mismatch` (same scope/op, different bytes) | ranks issued different collectives — mismatched binaries or a skip/branch desync | `sha256sum` both binaries; check draft/row counts are mirrored |
| `status 5` / `IBV_WC_MW_BIND_ERR` | stale Thunderbolt-RDMA state after a bad restart | reboot both hosts (or reload the module); **do not restart-loop** |
| `verbs operation scope ended with exchanges in flight` | `EndOperation` before the stream drained | `hipStreamSynchronize` then `CheckQueued` then `EndOperation` |
| `tp_pair_lost` | the other rank died / link dropped | expected on kill; restart both (rank 0 first) |
| `no usb4_rdma device found` | links re-negotiating, or `thunderbolt_net` stole the rings | wait 15 s; `sudo rmmod thunderbolt_net` + `modprobe thunderbolt_ibverbs` |
| handshake HTTP 500 naming a setting | ranks disagree on model/context/sessions | make flags identical |
| decode / prefill much slower than expected | cables trained at 10 Gb/s×2, or striping off | check `rx_speed`, `native_write_striping: 1` |

## Token parity

```sh
# single-host reference
MAYA_GLM_SPEC_FOLLOW_SAVE=/tmp/single.txt ./maya.sh --bench   # or a generate run
# TP2 pair
MAYA_GLM_SPEC_FOLLOW_SAVE=/tmp/tp2.txt   …
python tools/tp2/greedy_parity.py /tmp/single.txt /tmp/tp2.txt --min-agree 0.95
```

Near-tie flips (reduction order) are expected at a few percent; a flipped prefix or
empty stream is a bug (see collective invariant in [TP2-DESIGN.md](TP2-DESIGN.md)).

## Collective invariant

> Both ranks must issue the **same** collectives — order, count, bytes.

Any rank-divergent branch (draft acceptance, cache restore, skip) breaks the pair.
Single-host-only features stay gated off under TP2 (MTP/DFlash, KV streaming).
