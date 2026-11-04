#!/usr/bin/env python3
"""Measure concurrent streamed chat requests against an existing OpenAI API."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import statistics
import tempfile
import threading
import time
import urllib.error
import urllib.request


DEFAULT_PROMPT = (
    "Write a detailed eBPF tutorial explaining how to use eBPF on Linux. Include prerequisites, "
    "the compiler toolchain, an example program and its userspace loader, compilation commands, "
    "tracing, verification and troubleshooting. Explain each part with code examples. "
    "Aim for at least 1000 words."
)


def request(url, model, prompt, request_id, max_tokens, timeout, api_key=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content":
                      f"Request identifier {request_id}. Omit this identifier from your answer.\n\n{prompt}"}],
        "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    http_request = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                          json.dumps(body).encode(), headers=headers)
    result = {"request_id": request_id, "request": body, "error": None,
              "ttft_s": None, "usage": {}, "timings": {}, "finish_reason": None}
    content, reasoning = [], []
    started = time.perf_counter()
    try:
        completed = False
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    completed = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(json.dumps(chunk["error"]))
                result["response_model"] = chunk.get("model", result.get("response_model"))
                result["usage"] = chunk.get("usage") or result["usage"]
                result["timings"] = chunk.get("timings") or result["timings"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if result["ttft_s"] is None and (text or thought):
                        result["ttft_s"] = time.perf_counter() - started
                    content.append(text)
                    reasoning.append(thought)
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
        tokens = result["usage"].get("completion_tokens", 0)
        if not completed or not result["finish_reason"] or not tokens or result["ttft_s"] is None:
            raise RuntimeError("Incomplete stream or missing completion usage")
        if any(reasoning):
            raise RuntimeError("Reasoning tokens received with thinking disabled")
    except urllib.error.HTTPError as exc:
        result["error"] = f"HTTP {exc.code} {exc.read(4096).decode('utf-8', 'replace')}"
    except (OSError, ValueError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__} {exc}"
    result.update(elapsed_s=time.perf_counter() - started,
                  content="".join(content), reasoning_content="".join(reasoning))
    return result


def wave(options, prompt, run_id, clients, round_number):
    start = threading.Barrier(clients + 1)

    def client(index):
        start.wait()
        return request(options.url, options.model, prompt,
                       f"{run_id}-c{clients:02d}-r{round_number:02d}-i{index:02d}",
                       options.max_tokens, options.timeout, options.api_key)

    with ThreadPoolExecutor(max_workers=clients) as executor:
        futures = [executor.submit(client, index) for index in range(clients)]
        started = time.perf_counter()
        start.wait()
        results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
    return {"concurrency": clients, "round": round_number, "elapsed_s": elapsed, "requests": results}


def summarize(waves):
    summary = []
    for clients in sorted({item["concurrency"] for item in waves}):
        group = [item for item in waves if item["concurrency"] == clients]
        results = [result for item in group for result in item["requests"]]
        good = [result for result in results if result["error"] is None]
        decode = [result["timings"]["predicted_per_second"] for result in good
                  if result["timings"].get("predicted_per_second") is not None]
        tokens = sum(result["usage"]["completion_tokens"] for result in good)
        summary.append({
            "concurrency": clients, "requests": len(results), "errors": len(results) - len(good),
            "output_tokens": tokens,
            "aggregate_tok_s": tokens / sum(item["elapsed_s"] for item in group),
            "median_ttft_s": statistics.median(result["ttft_s"] for result in good) if good else None,
            "median_latency_s": statistics.median(result["elapsed_s"] for result in good) if good else None,
            "median_server_decode_tok_s": statistics.median(decode) if decode else None,
        })
    return summary


def save(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name, delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="API base including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--api-key", default=os.environ.get("MAYA_API_KEY"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if min(options.concurrency) < 1 or options.rounds < 1 or options.max_tokens < 1 or options.timeout <= 0:
        parser.error("Concurrency, rounds, token limit and timeout must be positive")
    prompt = options.prompt_file.read_text() if options.prompt_file else DEFAULT_PROMPT
    options.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"started_utc": datetime.now(timezone.utc).isoformat(), "url": options.url,
            "model": options.model, "prompt": prompt, "max_tokens": options.max_tokens,
            "rounds": options.rounds, "concurrency": options.concurrency,
            "timeout_s": options.timeout, "warmup": None, "waves": [], "summary": []}
    with options.output.open("x") as output:
        json.dump(data, output, indent=2)
    run_id = secrets.token_hex(6)
    print("Warmup", flush=True)
    data["warmup"] = wave(options, prompt, run_id, 1, 0)
    save(options.output, data)
    if any(result["error"] for result in data["warmup"]["requests"]):
        print("Warmup failed. Details saved to", options.output, flush=True)
        return 1
    for clients in options.concurrency:
        for round_number in range(1, options.rounds + 1):
            print(f"Running concurrency {clients}, round {round_number}", flush=True)
            measured = wave(options, prompt, run_id, clients, round_number)
            data["waves"].append(measured)
            data["summary"] = summarize(data["waves"])
            save(options.output, data)
            tokens = sum(result["usage"].get("completion_tokens", 0) for result in measured["requests"])
            errors = sum(result["error"] is not None for result in measured["requests"])
            print(f"Completed {tokens} tokens in {measured['elapsed_s']:.2f}s, "
                  f"{tokens / measured['elapsed_s']:.2f} tok/s, {errors} errors", flush=True)
            if errors:
                return 1
    print(json.dumps(data["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
