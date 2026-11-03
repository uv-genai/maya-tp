#!/usr/bin/env python3
"""Estimate rotated two-rank expert tensor bytes from a DeepSeek GGUF pack index.

This reports logical tensor bytes only, not actual allocation, caches, or a supported TP layout.
"""
import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")


def estimate(pack, block):
    config = {}
    for line in (pack / "ds4.txt").read_text().splitlines():
        key, sep, value = line.partition(" ")
        if sep:
            config[key] = value
    layers = int(config["num_hidden_layers"])
    experts = int(config["n_routed_experts"])
    width = int(config["moe_intermediate_size"])
    hidden = int(config["hidden_size"])
    if block <= 0 or width % block or width // block < 2:
        raise ValueError("expert width must contain at least two whole blocks")
    blocks = width // block
    rank = [0, 0]
    counts = [0] * layers
    totals = [0] * layers
    seen = set()
    with (pack / "tensors.txt").open() as f:
        for line in f:
            fields = line.split()
            if len(fields) < 8 or fields[0] != "t":
                continue
            match = PATTERN.fullmatch(fields[1])
            if not match:
                continue
            layer, expert, role = map(int, match.groups())
            key = (layer, expert, role)
            if layer >= layers or expert >= experts or key in seen:
                raise ValueError(f"invalid or duplicate expert tensor: {fields[1]}")
            seen.add(key)
            size = int(fields[5])
            shape = [int(x) for x in fields[6:]]
            expected = [hidden, width] if role == 2 else [width, hidden]
            if shape != expected or size % blocks:
                raise ValueError(f"unexpected tensor shape or proportional byte layout: {fields[1]}")
            # One extra whole block alternates by layer as in the local splitb(..., layer % 2).
            n0 = blocks // 2 + int(blocks % 2 != 0 and layer % 2 == 0)
            b0 = size // blocks * n0
            rank[0] += b0
            rank[1] += size - b0
            totals[layer] += size
            counts[layer] += 1
    if any(n != experts * 3 for n in counts):
        raise ValueError(f"incomplete trunk expert tensors per layer: {counts}")
    return {"pack": str(pack), "block": block, "width": width, "layers": layers,
            "experts_per_layer": experts, "trunk_total_bytes": sum(totals),
            "rank_expert_bytes": rank, "per_layer_bytes": totals,
            "note": "logical expert tensor bytes only; excludes replicated tensors, padding, cache and all runtime allocations"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--block", type=int, default=256, help="Assumed common block size; verify against each dtype")
    args = ap.parse_args()
    print(json.dumps(estimate(args.pack, args.block), indent=2))


if __name__ == "__main__":
    main()
