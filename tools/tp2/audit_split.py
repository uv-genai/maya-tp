#!/usr/bin/env python3
"""tools/tp2/audit_split.py - check a Maya pack's tensors for two-host tensor parallel (world=2).

Every split must land on a quantization-block boundary.  For routed experts and the dense lead FFN that means
the intermediate width halves cleanly (n_ff 2048 -> 1024, n_ff_dense 12288 -> 6144) and the down projection's
column share is a whole number of blocks.  Gate/up split is by whole rows and is always safe.

    python tools/tp2/audit_split.py /path/to/pack [--world 2] [--embd 4096] [--ff 2048] [--ff-dense 12288]

Reads `pack/native_experts.txt` (expert geometry) and `pack/index.txt` (dense tensors).  No model weights are
opened.  Exit 0 when every planned split is block-safe (prints the plan); exit 1 and names the tensor otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ggml type id -> (name, block values, bytes per block) for the types Maya-S uses.
# Sizes from ggml-common.h static_asserts (IQ2_XXS 66, IQ3_XXS 98, IQ2_S 82, ...); block length
# is what matters for split alignment.
GGML = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
}


def share(whole: int, rank: int, world: int) -> tuple[int, int]:
    if whole % world:
        raise ValueError(f"{whole} does not split into {world}")
    n = whole // world
    return n * rank, n


def check_split(name: str, whole: int, block: int, world: int, axis: str) -> list[str]:
    """Return a list of problems (empty = green)."""
    bad = []
    if whole % world:
        bad.append(f"{name}: {axis} {whole} does not split into {world}")
        return bad
    n = whole // world
    if block and n % block:
        bad.append(f"{name}: {axis} share {n} is not a multiple of block {block}")
    return bad


def parse_index_row(parts: list[str]) -> tuple[str, int, int, int] | None:
    """index.txt: name <6 fields> ne0 ne1 type ...  -> (name, ne0, ne1, type)."""
    if len(parts) < 10:
        return None
    try:
        return parts[0], int(parts[7]), int(parts[8]), int(parts[9])
    except ValueError:
        return None


def audit_pack(pack: Path, world: int, embd: int, ff: int, ff_dense: int) -> int:
    problems: list[str] = []
    plans: list[str] = []

    native = pack / "native_experts.txt"
    if native.is_file():
        for line in native.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                problems.append(f"native_experts.txt: short line {line!r}")
                continue
            layer = int(parts[0])
            gu_type, d_type = int(parts[1]), int(parts[2])
            blob_bytes = int(parts[4])
            gu, dt = GGML.get(gu_type), GGML.get(d_type)
            if not gu or not dt:
                problems.append(f"layer {layer}: unknown ggml type gu={gu_type} d={d_type}")
                continue
            gu_row = embd // gu[1] * gu[2]
            d_row = ff // dt[1] * dt[2]
            expect = 2 * ff * gu_row + embd * d_row
            if expect != blob_bytes:
                problems.append(
                    f"layer {layer}: blob_bytes {blob_bytes} != computed {expect} "
                    f"(gu_row {gu_row} d_row {d_row})"
                )
            for axis, whole, block in (
                ("n_ff gate/up rows", ff, gu[1]),
                ("n_ff down cols", ff, dt[1]),
            ):
                problems.extend(check_split(f"layer {layer} expert {axis}", whole, block, world, axis))
            if ff % world == 0:
                begin, count = share(ff, 0, world)
                plans.append(
                    f"  expert L{layer:2d}  {gu[0]:8s}/{dt[0]:8s}  "
                    f"ff {ff}->{count}  gate/up rows [{begin},{begin + count})  "
                    f"down cols [{begin},{begin + count})  blob {blob_bytes}"
                )
    else:
        problems.append(f"missing {native}")

    index = pack / "index.txt"
    if index.is_file():
        for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            row = parse_index_row(line.split())
            if not row:
                continue
            name, ne0, ne1, ty = row
            g = GGML.get(ty)
            if not g:
                continue
            block = g[1]
            if any(
                name.endswith(s)
                for s in (
                    "ffn_gate.weight",
                    "ffn_up.weight",
                    "ffn_gate_shexp.weight",
                    "ffn_up_shexp.weight",
                )
            ):
                problems.extend(check_split(name, ne1, block, world, "out (n_ff)"))
                plans.append(f"  dense  {name}  {g[0]}  out {ne1}->{ne1 // world}")
            elif any(name.endswith(s) for s in ("ffn_down.weight", "ffn_down_shexp.weight")):
                problems.extend(check_split(name, ne0, block, world, "in (n_ff)"))
                plans.append(f"  dense  {name}  {g[0]}  in  {ne0}->{ne0 // world}")
            elif name.endswith(("attn_output.weight", "attn_v.weight", "attn_k.weight", "attn_q.weight")):
                if ne1 % world:
                    problems.append(f"{name}: out {ne1} does not split into {world}")
                else:
                    plans.append(f"  attn   {name}  {g[0]}  out {ne1}->{ne1 // world}")
    else:
        problems.append(f"missing {index}")

    print(f"pack {pack}  world={world}  embd={embd} n_ff={ff} n_ff_dense={ff_dense}")
    for line in plans:
        print(line)
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nall splits block-safe")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", type=Path, help="pack directory (contains native_experts.txt / index.txt)")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--embd", type=int, default=4096)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--ff-dense", type=int, default=12288, dest="ff_dense")
    a = ap.parse_args(argv)
    return audit_pack(a.pack, a.world, a.embd, a.ff, a.ff_dense)


if __name__ == "__main__":
    raise SystemExit(main())
