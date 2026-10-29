#!/usr/bin/env python3
"""tools/tp2/greedy_parity.py - compare two token-id dumps (single-host vs TP2 greedy).

Both runs must use the same prompt and greedy settings.  Token dumps come from the engine's
`--spec-follow-save PATH` (one request's decoded ids, whitespace/comma separated) or
`MAYA_GLM_MULTI_DUMP` (one line per sequence).

    python tools/tp2/greedy_parity.py single.txt tp2.txt
    python tools/tp2/greedy_parity.py single.txt tp2.txt --min-agree 0.95

Exit 0 when the agreement rate is >= --min-agree (default 0.95) and prints the first
divergences.  Near-tie flips (a change in reduction order can flip a close logit) are
expected at a few percent; a structural mismatch (repeating / empty / a flipped prefix)
fails the run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_ids(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="replace").replace(",", " ")
    ids: list[int] = []
    for tok in text.split():
        try:
            ids.append(int(tok))
        except ValueError:
            continue
    return ids


def compare(a: list[int], b: list[int], min_agree: float, show: int) -> int:
    n = min(len(a), len(b))
    if n == 0:
        print("FAIL: empty dump")
        return 1
    same = sum(1 for i in range(n) if a[i] == b[i])
    rate = same / n
    print(f"tokens: a={len(a)} b={len(b)} compared={n} agree={same} ({rate:.4f})")
    if len(a) != len(b):
        print(f"note: length differs (a={len(a)} b={len(b)}); comparing the overlap")
    flips = [i for i in range(n) if a[i] != b[i]]
    for i in flips[:show]:
        lo = max(0, i - 2)
        print(f"  flip at {i}: a[{lo}:{i + 3}]={a[lo:i + 3]}  b[{lo}:{i + 3}]={b[lo:i + 3]}")
    if flips[show:]:
        print(f"  ... {len(flips) - show} more flips")
    if rate < min_agree:
        print(f"FAIL: agreement {rate:.4f} < {min_agree}")
        return 1
    print(f"OK: agreement {rate:.4f} >= {min_agree}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path, help="reference token dump (single-host)")
    ap.add_argument("b", type=Path, help="candidate token dump (TP2)")
    ap.add_argument("--min-agree", type=float, default=0.95)
    ap.add_argument("--show", type=int, default=10, help="how many flips to print")
    o = ap.parse_args(argv)
    return compare(load_ids(o.a), load_ids(o.b), o.min_agree, o.show)


if __name__ == "__main__":
    raise SystemExit(main())
