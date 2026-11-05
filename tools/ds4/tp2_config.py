#!/usr/bin/env python3
"""Print a dedicated experimental DeepSeek TP2 rank-0 server config as JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ds4 import tp2_rank


def make_config(opts, engine_log, model_name, model_label=None):
    """Use absolute paths so launching the server from another directory is safe."""
    pack = str(Path(opts.pack).resolve())
    manifest = str(Path(opts.manifest).resolve())
    engine = str(Path(opts.engine).resolve())
    args = ["--serve", "--rank", "0", "--pack", pack, "--manifest", manifest,
            "--engine", engine, "--parallel", str(opts.parallel), "--host", opts.host, "--port", str(opts.port),
            "--rdma-device", opts.rdma_device, "--context", str(opts.context),
            "--expert-budget-mib", str(opts.expert_budget_mib), "--reserve-mib", str(opts.reserve_mib),
            "--gpu", str(opts.gpu), "--cpu-threads", str(opts.cpu_threads), "--dspark", opts.dspark,
            "--prefill-chunk", str(opts.prefill_chunk), "--suffix-draft", str(opts.suffix_draft)]
    config = {"exe": str(ROOT / "tools/ds4/tp2_rank.py"), "args": args, "cwd": str(ROOT),
              "tokenizer": str(Path(pack) / "tokenizer"), "backend": "hip",
              "model_name": model_name, "log": str(Path(engine_log).resolve())}
    if model_label:
        config["model_label"] = model_label
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    p.add_argument("--engine-log", help="dedicated engine log path")
    p.add_argument("--model-name", default="deepseek-v4.1-flash")
    p.add_argument("--model-label", help="optional display label for this exact selected pack")
    p.add_argument("--help", action="store_true")
    options, rank_args = p.parse_known_args(argv)
    if options.help:
        p.print_help()
        print("\nRank settings:")
        tp2_rank.parser().print_help()
        return 0
    if not options.engine_log:
        p.error("--engine-log is required")
    if any(a == "--serve" or a == "--rank" or a.startswith("--rank=") for a in rank_args):
        p.error("rank 0 and --serve are supplied automatically")
    opts = tp2_rank.parser().parse_args(["--serve", "--rank", "0", *rank_args])
    print(json.dumps(make_config(opts, options.engine_log, options.model_name, options.model_label), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
