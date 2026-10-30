#!/usr/bin/env python3
"""Live TP2 MTP regression: thinking-budget force, snapshot reuse, and cancellation.

Start rank1.sh then MAYA_TP_MTP=1 rank0.sh. This uses the running pair;
acceptance.sh separately covers peer death and restarting both ranks.
"""
import argparse
import json
import time
import urllib.request


def post(base, messages, max_tokens, stream=False):
    body = json.dumps(dict(model="maya", messages=messages, max_tokens=max_tokens,
                           temperature=0, stream=stream)).encode()
    return urllib.request.urlopen(urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=120)


def chat(base, messages, max_tokens):
    started = time.monotonic()
    with post(base, messages, max_tokens) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    assert result["choices"] and result["usage"]["completion_tokens"] > 0, result
    print(f"chat: {time.monotonic() - started:.2f}s "
          f"tokens={result['usage']['completion_tokens']} "
          f"cached={result['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)} "
          f"drafts={result.get('timings', {}).get('draft_n', 0)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://192.168.1.22:18080")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    messages = []
    for i, prompt in enumerate(("Say hello.", "Now say goodbye.", "What is two plus two?")):
        messages.append(dict(role="user", content=prompt))
        result = chat(base, messages, 64)
        assert result.get("timings", {}).get("draft_n", 0) > 0, "MTP must be enabled"
        if i:
            assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, "restore was not exercised"
        messages.append(result["choices"][0]["message"])
    # Cancel during decoding, rather than before the request has reached the engine.
    chunks = 0
    with post(base, [dict(role="user", content="Write a long essay about the ocean.")], 1024, True) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            assert "error" not in event, event
            delta = event.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                chunks += 1
            if chunks >= 8:
                break
    assert chunks >= 8, "stream ended before cancellation"
    print("cancelled after 8 text chunks", flush=True)
    time.sleep(1)
    chat(base, [dict(role="user", content="Say hello again.")], 16)
    # Small limits exercise the first-anchor exit and T=1 token path as well as window exit.
    for limit in (1, 2, 3):
        result = chat(base, [dict(role="user", content="Count to ten.")], limit)
        assert result["usage"]["completion_tokens"] <= limit, result
    with urllib.request.urlopen(base + "/health", timeout=5) as response:
        health = json.load(response)
    assert health["tp"]["peer"] == "up", health
    print("PASS: three turns with reuse, cancel/follow-up, short limits, peer up", flush=True)


if __name__ == "__main__":
    main()
