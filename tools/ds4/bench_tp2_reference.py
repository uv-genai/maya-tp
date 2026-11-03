#!/usr/bin/env python3
"""Capture comparable DeepSeek API samples on an explicitly selected, idle server."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

CASES = (
    ("python", "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring."),
    ("prose", "Explain how photosynthesis works, step by step, for a high-school student."),
    ("html", "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript."),
)


def call(url, path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="Explicit API server URL")
    ap.add_argument("--label", required=True, help="Expected exact engine model_label")
    ap.add_argument("--dspark", required=True, choices=("on", "off"))
    ap.add_argument("--output", required=True, type=Path, help="New JSON file; never overwritten")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--save-answers", action="store_true", help="include generated text for qualitative comparison")
    a = ap.parse_args()
    if a.runs < 1 or a.tokens < 1:
        ap.error("--runs and --tokens must be positive")
    # Reserve the destination before making requests; do not clobber existing measurements.
    with a.output.open("x", encoding="utf-8") as out:
        result = {"url": a.url, "expected_label": a.label, "dspark": a.dspark,
                  "runs": a.runs, "max_tokens": a.tokens, "samples": []}
        def save():
            out.seek(0)
            json.dump(result, out, indent=2)
            out.write("\n")
            out.truncate()
            out.flush()
        save()
        for run in range(a.runs):
            for name, prompt in CASES:
                state = call(a.url, "/metrics", timeout=5)
                engine = state.get("engine") or {}
                live = state.get("live") or {}
                if engine.get("model_label") != a.label or engine.get("dspark") is not (a.dspark == "on"):
                    raise RuntimeError("model label or DSpark state does not match expected configuration")
                if live.get("active") or live.get("queued"):
                    raise RuntimeError("server is busy; no further request sent")
                payload = {"model": engine["model"], "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": a.tokens, "temperature": 0, "top_p": 1, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}}
                started = time.monotonic()
                response = call(a.url, "/v1/chat/completions", payload)
                wall = time.monotonic() - started
                message = response["choices"][0]["message"]
                if message.get("reasoning_content"):
                    raise RuntimeError("server returned thinking despite thinking-off request")
                answer = message.get("content") or ""
                after = call(a.url, "/metrics", timeout=5)
                result["samples"].append({"run": run + 1, "case": name,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    **({"answer": answer} if a.save_answers else {}),
                    "wall_s": wall, "usage": response.get("usage"), "timings": response.get("timings"),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "engine": engine, "tiers": (after.get("tiers") or {}).get("now")})
                save()
                print(f"run {run + 1} {name}: {wall:.2f}s", flush=True)


if __name__ == "__main__":
    main()
