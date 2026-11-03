#!/usr/bin/env python3
"""Measure fresh DeepSeek prefill samples with deterministic synthetic prompts."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_tokenizer import Tokenizer  # noqa: E402
if __package__:
    from .bench_tp2_reference import call  # noqa: E402
else:
    from bench_tp2_reference import call  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--dspark", required=True, choices=("on", "off"))
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--source-tokens", type=int, action="append", dest="lengths",
                    help="fresh prompt length; repeat for multiple lengths (default: 2048 and 8192)")
    ap.add_argument("--timeout", type=int, default=900, help="request timeout in seconds (default: 900)")
    ap.add_argument("--output-tokens", type=int, default=1, help="maximum generated tokens (default: 1)")
    args = ap.parse_args()
    if args.runs < 1 or args.runs > 20:
        ap.error("--runs must be between 1 and 20")
    lengths = args.lengths or [2048, 8192]
    if any(n < 1 or n > 1048576 for n in lengths):
        ap.error("--source-tokens must be between 1 and 1048576")
    if args.timeout < 1 or args.timeout > 86400:
        ap.error("--timeout must be between 1 and 86400 seconds")
    if args.output_tokens < 1 or args.output_tokens > 4096:
        ap.error("--output-tokens must be between 1 and 4096")
    stride = sum(lengths) + 1024 * len(lengths)
    tokenizer = Tokenizer.from_dir(args.tokenizer)
    # Unique indexed text avoids sending the same prefix in successive trials.
    lines = max(5000 * args.runs, (stride * args.runs + 7) // 8)
    corpus = "\n".join(f"Reference segment {i:06d}: evaluate the recorded measurements, explain the outcome, "
                       f"and check the assumptions for case {i * 17 + 3}." for i in range(lines))
    ids = tokenizer.encode(corpus)
    if len(ids) < args.runs * stride:
        raise RuntimeError("synthetic corpus tokenized to fewer tokens than expected")
    with args.output.open("x", encoding="utf-8") as out:
        result = {"url": args.url, "label": args.label, "dspark": args.dspark, "samples": []}
        def save():
            out.seek(0); json.dump(result, out, indent=2); out.write("\n"); out.truncate(); out.flush()
        save()
        for run in range(args.runs):
            offset = 0
            for source_tokens in lengths:
                prompt = tokenizer.decode(ids[run * stride + offset:run * stride + offset + source_tokens])
                offset += source_tokens + 1024
                state = call(args.url, "/metrics", timeout=5)
                engine, live = state.get("engine") or {}, state.get("live") or {}
                if engine.get("model_label") != args.label or engine.get("dspark") is not (args.dspark == "on"):
                    raise RuntimeError("model label or DSpark state does not match expected configuration")
                if live.get("active") or live.get("queued"):
                    raise RuntimeError("server is busy; no further request sent")
                payload = {"model": engine["model"], "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": args.output_tokens, "temperature": 0, "top_p": 1, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}}
                started = time.monotonic()
                response = call(args.url, "/v1/chat/completions", payload, timeout=args.timeout)
                wall = time.monotonic() - started
                after = call(args.url, "/metrics", timeout=5)
                result["samples"].append({"run": run + 1, "source_tokens": source_tokens,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "wall_s": wall, "usage": response.get("usage"),
                    "timings": response.get("timings"), "engine": engine,
                    "tiers": (after.get("tiers") or {}).get("now")})
                save()
                print(f"run {run + 1}, {source_tokens} source tokens", flush=True)


if __name__ == "__main__":
    main()
