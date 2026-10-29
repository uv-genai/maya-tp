#!/usr/bin/env python3
"""TP2 calibration: measure AR cost and recommend MAYA_TP_* settings.

Runs a short benchmark on both ranks and reports:
  - AR cost (ms/tok) at current settings
  - Compute-only cost (MAYA_TP_NOAR=1)
  - Recommended settings

Usage:
  python tools/calibrate_glm_tp.py --pack /path/to/pack [--tokens ...] [--max-new N]

Requires both hosts running the engine (rank 0 via SSH, rank 1 locally).
No GPU needed for the config tests; the benchmark needs both hosts.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK = "/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack"
DEFAULT_TOKENS = "154822,25062,287,29905,371,25,7487,785,6722,315,9621,374"
RANK0_HOST = "strix-halo-1.local"
ENGINE = str(ROOT / "build-hip" / "maya")


def run_rank(cmd: list, env: dict = None) -> str:
    """Run a command and capture stderr (where the timing output goes)."""
    e = dict(__import__("os").environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=300)
    return r.stderr + r.stdout


def parse_ms(text: str) -> float | None:
    """Extract ms/token from the engine's timing output."""
    m = re.search(r"(\d+) tokens, ([\d.]+) ms/token", text)
    return float(m.group(2)) if m else None


def bench(pack: str, tokens: str, max_new: int, no_ar: bool = False) -> float | None:
    """Run one benchmark and return ms/tok."""
    env = {"MAYA_GLM_NO_MTP": "1", "STRATA_GLM_TIMING": "1"}
    if no_ar:
        env["MAYA_TP_NOAR"] = "1"

    # rank 0 (remote)
    remote_cmd = (
        f"pkill -x maya-tp2 2>/dev/null; "
        f"MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 {'MAYA_TP_NOAR=1' if no_ar else ''} "
        f"/tmp/maya-tp2 --glm-pack '{pack}' --tokens '{tokens}' "
        f"--max-new {max_new} --greedy --max-context 4096 "
        f"--tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515 --tp-rdma-device usb4_rdma5"
    )
    subprocess.Popen(
        ["ssh", RANK0_HOST, remote_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(8)

    # rank 1 (local)
    local_cmd = [
        ENGINE, "--glm-pack", pack, "--tokens", tokens,
        "--max-new", str(max_new), "--greedy", "--max-context", "4096",
        "--tp-world-size", "2", "--tp-rank", "1",
        "--tp-bootstrap-host", RANK0_HOST, "--tp-bootstrap-port", "18515",
        "--tp-rdma-device", "usb4_rdma5",
    ]
    output = run_rank(local_cmd, env)
    return parse_ms(output)


def main() -> int:
    p = argparse.ArgumentParser(description="TP2 calibration")
    p.add_argument("--pack", default=DEFAULT_PACK)
    p.add_argument("--tokens", default=DEFAULT_TOKENS)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--host", default=RANK0_HOST)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    global RANK0_HOST
    RANK0_HOST = args.host

    print("=== TP2 calibration ===")
    print(f"  pack:     {args.pack}")
    print(f"  tokens:   {args.tokens}")
    print(f"  max-new:  {args.max_new}")
    print(f"  rank 0:   {RANK0_HOST}")
    print()

    # warmup
    for i in range(args.warmup):
        print(f"  warmup {i + 1}/{args.warmup}...")
        bench(args.pack, args.tokens, args.max_new)

    # measure with AR
    print("  measuring TP2 + AR...")
    ms_ar = bench(args.pack, args.tokens, args.max_new)
    if ms_ar is None:
        print("  ERROR: could not parse timing from TP2 + AR run")
        return 1
    print(f"    {ms_ar:.2f} ms/tok")

    # measure without AR
    print("  measuring TP2 no AR...")
    ms_noar = bench(args.pack, args.tokens, args.max_new, no_ar=True)
    if ms_noar is None:
        print("  ERROR: could not parse timing from TP2 no-AR run")
        return 1
    print(f"    {ms_noar:.2f} ms/tok")

    ar_cost = ms_ar - ms_noar
    ar_pct = ar_cost / ms_ar * 100 if ms_ar > 0 else 0

    print()
    print("=== results ===")
    print(f"  TP2 + AR:     {ms_ar:.2f} ms/tok")
    print(f"  TP2 no AR:    {ms_noar:.2f} ms/tok")
    print(f"  AR cost:      {ar_cost:.2f} ms/tok ({ar_pct:.1f}%)")
    print()
    print("=== recommended settings ===")
    print(f"  MAYA_TP_NOAR=0 (AR is {ar_pct:.0f}% of runtime)")
    if ar_pct > 15:
        print("  Consider: reduce AR count (combine mixer+FFN) or optimize transport")
    else:
        print("  AR cost is acceptable")

    return 0


if __name__ == "__main__":
    sys.exit(main())
