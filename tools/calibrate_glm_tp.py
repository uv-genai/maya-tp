#!/usr/bin/env python3
"""TP2 calibration: AR microbench + end-to-end ms/tok tables.

Runs on the twin Strix Halos (rank 1 here, rank 0 via SSH). Produces:
  - link / AR microbench (tp_comm_rdma_test --bench) if the binary exists
  - end-to-end: single-host, TP2 + AR, TP2 no-AR
  - pair token parity via --spec-follow-save when both ranks complete

Usage:
  .venv/bin/python tools/calibrate_glm_tp.py [--pack DIR] [--max-new 32] [--skip-e2e]

No GPU needed for --help; benches need both hosts and the pack.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK = "/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack"
DEFAULT_TOKENS = "154822,25062,287,29905,371,25,7487,785,6722,315,9621,374"
RANK0 = "strix-halo-1.local"
ENGINE = ROOT / "build-hip" / "maya"
COMM_TEST = ROOT / "build-tp2" / "tp_comm_rdma_test"
RDEV = "usb4_rdma5"


def sh(cmd: list[str], env: dict | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout)


def kill_pair() -> None:
    subprocess.run(["pkill", "-9", "-x", "maya"], capture_output=True)
    sh(["ssh", RANK0, "pkill -9 -x maya 2>/dev/null; true"])
    time.sleep(1)


def parse_ms(text: str) -> float | None:
    # "32 tokens, 38.50 ms/token"
    m = re.search(r"(\d+) tokens, ([\d.]+) ms/token", text)
    return float(m.group(2)) if m else None


def parse_tokens(text: str) -> list[str]:
    return re.findall(r"^T (\d+)$", text, re.M)


def e2e(pack: str, tokens: str, max_new: int, tp: bool, no_ar: bool, follow: Path | None) -> tuple[float | None, list[str]]:
    kill_pair()
    env = {"MAYA_GLM_NO_MTP": "1", "STRATA_GLM_TIMING": "1"}
    if no_ar:
        env["MAYA_TP_NOAR"] = "1"
    r1 = [str(ENGINE), "--glm-pack", pack, "--tokens", tokens, "--max-new", str(max_new),
          "--greedy", "--max-context", "4096"]
    if follow:
        r1 += ["--spec-follow-save", str(follow)]
    if tp:
        r1 += ["--tp-world-size", "2", "--tp-rank", "1",
               "--tp-bootstrap-host", RANK0, "--tp-bootstrap-port", "18515",
               "--tp-rdma-device", RDEV]
        remote = (
            f"pkill -9 -x maya 2>/dev/null; "
            f"MAYA_GLM_NO_MTP=1 STRATA_GLM_TIMING=1 "
            f"{'MAYA_TP_NOAR=1 ' if no_ar else ''}"
            f"{ENGINE} --glm-pack '{pack}' --tokens '{tokens}' --max-new {max_new} "
            f"--greedy --max-context 4096 --tp-world-size 2 --tp-rank 0 "
            f"--tp-bootstrap-port 18515 --tp-rdma-device {RDEV}"
        )
        if follow:
            remote += f" --spec-follow-save {follow}.r0"
        sh(["ssh", RANK0, f"nohup bash -c {remote!r} > /tmp/tp2-cal-r0.log 2>&1 & echo started"])
        time.sleep(3)
        proc = sh(r1, env, timeout=400)
        time.sleep(1)
        r0txt = sh(["ssh", RANK0, "cat /tmp/tp2-cal-r0.log 2>/dev/null || true"]).stdout
        return parse_ms(proc.stderr + proc.stdout), parse_tokens(proc.stderr + proc.stdout) + parse_tokens(r0txt)
    proc = sh(r1, env, timeout=400)
    return parse_ms(proc.stderr + proc.stdout), parse_tokens(proc.stderr + proc.stdout)


def microbench() -> None:
    bin_r0 = "/tmp/tp_comm_rdma_test"
    if COMM_TEST.is_file():
        sh(["scp", "-o", "BatchMode=yes", str(COMM_TEST), f"{RANK0}:{bin_r0}"])
        sh(["ssh", RANK0, f"chmod +x {bin_r0}"])
    else:
        print(f"  (no {COMM_TEST}; skip microbench)")
        return
    for hidden in (12288, 2560):
        print(f"\n=== microbench hidden={hidden} ===")
        r0 = sh(["ssh", RANK0,
                 f"{bin_r0} --rank 0 --bootstrap-port 18515 --rdma-device {RDEV} "
                 f"--exchanges 20 --bench --hidden {hidden}"])
        time.sleep(1)
        r1 = sh([str(COMM_TEST), "--rank", "1", "--bootstrap-host", RANK0,
                 "--bootstrap-port", "18515", "--rdma-device", RDEV,
                 "--exchanges", "20", "--bench", "--hidden", str(hidden)])
        for label, out in (("rank1", r1.stdout + r1.stderr), ("rank0", r0.stdout + r0.stderr)):
            for line in out.splitlines():
                if any(k in line for k in ("bench ", "allreduce ", "FAIL", "OK ")):
                    print(f"  {label}: {line}")


def main() -> int:
    p = argparse.ArgumentParser(description="TP2 calibration and tables")
    p.add_argument("--pack", default=DEFAULT_PACK)
    p.add_argument("--tokens", default=DEFAULT_TOKENS)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--host", default=RANK0)
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--skip-micro", action="store_true")
    args = p.parse_args()
    global RANK0
    RANK0 = args.host

    print("=== TP2 calibration ===")
    print(f"  pack:   {args.pack}")
    print(f"  tokens: {args.tokens}")
    print(f"  max-new:{args.max_new}")
    print(f"  rank 0: {RANK0}")

    if not args.skip_micro:
        microbench()

    if args.skip_e2e:
        return 0

    follow = Path("/tmp/tp2-cal-follow.txt")
    if follow.exists():
        follow.unlink()
    rows = []

    print("\n=== e2e single-host ===")
    ms, toks = e2e(args.pack, args.tokens, args.max_new, tp=False, no_ar=False, follow=follow)
    print(f"  {ms} ms/tok  tokens={len(toks)}")
    rows.append(("single-host", ms, None, toks))

    print("\n=== e2e TP2 + AR ===")
    ms_ar, toks_ar = e2e(args.pack, args.tokens, args.max_new, tp=True, no_ar=False, follow=None)
    print(f"  {ms_ar} ms/tok  tokens={len(toks_ar)}")
    rows.append(("TP2 + AR", ms_ar, None, toks_ar))

    print("\n=== e2e TP2 no AR ===")
    ms_no, toks_no = e2e(args.pack, args.tokens, args.max_new, tp=True, no_ar=True, follow=None)
    print(f"  {ms_no} ms/tok")
    rows.append(("TP2 no AR", ms_no, None, toks_no))

    kill_pair()
    print("\n=== table ===")
    base = rows[0][1]
    print("| Config | ms/tok | vs single |")
    print("|---|---|---|")
    for name, ms, _, _ in rows:
        if ms is None:
            print(f"| {name} | — | — |")
            continue
        vs = "—" if base is None or name == "single-host" else f"{100.0 * (ms - base) / base:+.1f}%"
        print(f"| {name} | {ms:.2f} | {vs} |")
    if base and ms_ar and ms_no:
        ar = ms_ar - ms_no
        print(f"\nAR cost: {ar:.2f} ms/tok ({100.0 * ar / ms_ar:.1f}% of TP2)")
    if toks and toks_ar:
        n = min(len(toks), len(toks_ar))
        same = sum(1 for a, b in zip(toks[:n], toks_ar[:n]) if a == b)
        print(f"Single vs TP2 first {n} tokens: {same}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
