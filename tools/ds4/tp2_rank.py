#!/usr/bin/env python3
"""Prepare an experimental DeepSeek TP2 rank from a checked local pack."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ds4.tp2_identity import IdentityError, MAX_META, prepare_identity


def bounded(low, high):
    def parse(value):
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("expected an integer") from exc
        if not low <= n <= high:
            raise argparse.ArgumentTypeError(f"expected an integer from {low} to {high}")
        return n
    return parse


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serve", action="store_true", help="serve rank 0 through stdin/stdout")
    p.add_argument("--rank", type=bounded(0, 1), required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--host", required=True, help="rank 0's control address")
    p.add_argument("--port", type=bounded(1, 65535), required=True)
    p.add_argument("--rdma-device", required=True)
    p.add_argument("--context", type=bounded(1, 1048576), required=True)
    p.add_argument("--expert-budget-mib", type=bounded(1, 1048576), required=True)
    p.add_argument("--reserve-mib", type=bounded(0, 1048576), default=1024)
    p.add_argument("--gpu", type=bounded(0, 255), default=0)
    p.add_argument("--cpu-threads", type=bounded(1, 1024), default=14)
    p.add_argument("--dspark", choices=("on", "off"), default="on")
    p.add_argument("--suffix-draft", type=int, choices=(0, 3), default=0,
                   help="experimental synchronized prompt lookup (0: off; 3: minimum match length)")
    p.add_argument("--prefill-chunk", type=int, choices=(0, 256), default=0,
                   help="experimental synchronized prefill rows (0: window-only; 256: chunked)")
    return p


def prepare_launch(opts, environ):
    if opts.serve != (opts.rank == 0):
        raise IdentityError("rank 0 requires --serve; rank 1 must not use it")
    if not opts.host.strip() or not opts.rdma_device.strip():
        raise IdentityError("control host and RDMA device must be nonempty")
    manifest = Path(opts.manifest)
    if manifest.stat().st_size > MAX_META:
        raise IdentityError("manifest is too large")
    data = manifest.read_bytes()
    if len(data) > MAX_META:
        raise IdentityError("manifest is too large")
    pack = str(Path(opts.pack).resolve(strict=True))
    engine = str(Path(opts.engine).resolve(strict=True))
    identity = prepare_identity(json.loads(data), pack, engine)
    env = dict(environ)
    if env.get("DS4_STREAM", "0") != "0":
        raise IdentityError("DeepSeek TP2 requires DS4_STREAM=0")
    env["DS4_STREAM"] = "0"
    if opts.prefill_chunk:
        for name in ("DS4_FLOW", "DS4_YOCO"):
            if env.get(name, "0") != "0":
                raise IdentityError(f"DeepSeek TP2 chunk prefill requires {name}=0")
            env[name] = "0"
        if "DS4_CHUNK" in env:
            raise IdentityError("DeepSeek TP2 chunk prefill does not support DS4_CHUNK")
    env.setdefault("MAYA_TP_BOOTSTRAP_TIMEOUT", "900")
    env["MAYA_DS4_TP2_PACK"] = pack
    env["MAYA_DS4_TP2_DSPARK"] = "1" if opts.dspark == "on" else "0"
    env["MAYA_DS4_TP2_CHUNK"] = str(opts.prefill_chunk)
    env["MAYA_DS4_TP2_LOOKUP"] = str(opts.suffix_draft)
    for kind in ("model", "partition", "build"):
        env["MAYA_DS4_TP2_" + kind.upper() + "_DIGEST"] = identity[kind + "_digest"]
    draft = opts.rank == 0 and opts.dspark == "on"
    args = [engine, "--pack", pack, "--gpus", str(opts.gpu), "--max-context", str(opts.context),
            "--reserve-mib", str(opts.reserve_mib), "--cpu-threads", str(opts.cpu_threads),
            "--tp", "1", "--threads", "off", "--prefill-chunk", str(opts.prefill_chunk), "--vision", "off",
            "--mmap-experts", "on", "--no-expert-cache", "--profile-readonly", "--vram-experts",
            "1" if draft else "0", "--suffix-draft", str(opts.suffix_draft if opts.rank == 0 else 0),
            "--dspark", "on" if draft else "none",
            "--tp2-rank", str(opts.rank), "--tp2-host", opts.host, "--tp2-port", str(opts.port),
            "--tp2-rdma-device", opts.rdma_device, "--tp2-expert-budget-mib", str(opts.expert_budget_mib)]
    args.append("--serve" if opts.rank == 0 else "--tp2-worker")
    return args, env


def main(argv=None):
    opts = parser().parse_args(argv)
    try:
        args, env = prepare_launch(opts, os.environ)
        os.execve(args[0], args, env)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"DeepSeek TP2 preflight: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
