#!/usr/bin/env python3
"""Live TP2 collective regression: large prompts, reuse and streaming, MTP on/off.

Start a matching pair at context 32768; use seven drafts for the MTP run.
Set MAYA_TP_TEST_WINDOW_RETRY=8 on rank 1 alone to exercise asymmetric retries.
This sends requests to the running server and never starts/stops processes.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(base, body):
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://192.168.1.22:18080')
    parser.add_argument('--mode', choices=('simple', 'mtp'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []

    def chat(label, messages, limit, minimum_prompt=0, minimum_cache=0):
        started = time.monotonic()
        with request(args.url, dict(model='maya', messages=messages, max_tokens=limit, temperature=0)) as r:
            out = json.load(r)
        assert 'error' not in out, out
        usage = out['usage']
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        assert usage['prompt_tokens'] >= minimum_prompt, out
        assert cached >= minimum_cache, out
        assert 0 < usage['completion_tokens'] <= limit, out
        drafts = out.get('timings', {}).get('draft_n', 0)
        if args.mode == 'simple':
            assert drafts == 0, out
        else:
            assert drafts > 0, out
        row = dict(case=label, mode=args.mode, wall_s=round(time.monotonic()-started, 3),
                   usage=usage, timings=out.get('timings', {}),
                   message=out['choices'][0]['message'])
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        return out['choices'][0]['message']

    def document(repeats):
        return [dict(role='system', content='You are a helpful assistant. This is a transport stress-test document:\n'
                     + 'alpha ' * repeats),
                dict(role='user', content='Summarize the document, then explain why network framing needs consistent '
                     'batch sizes. Use at least two paragraphs.')]

    # Approximately the user's failing 980-token request, then >48MiB and multiple-window prompts.
    chat('seven-draft-window', document(940), 64, minimum_prompt=900)
    messages = document(7240)
    answer = chat('over-48MiB', messages, 32, minimum_prompt=7234)
    messages += [answer, dict(role='user', content='Explain the same idea more simply.')]
    chat('reuse-after-large-prompt', messages, 24, minimum_prompt=7234, minimum_cache=7000)
    messages = document(14500)
    answer = chat('multiple-prefill-chunks', messages, 32, minimum_prompt=14000)
    messages += [answer, dict(role='user', content='Give one final short example.')]
    chunks = 0
    complete = False
    started = time.monotonic()
    with request(args.url, dict(model='maya', messages=messages, max_tokens=24,
                               temperature=0, stream=True)) as r:
        for line in r:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                complete = True
                break
            event = json.loads(payload)
            assert 'error' not in event, event
            chunks += 1
    assert complete and chunks > 0, 'stream did not complete'
    results.append(dict(case='stream-after-large-prompt', mode=args.mode, chunks=chunks,
                        wall_s=round(time.monotonic()-started,3)))
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    with urllib.request.urlopen(args.url + '/health', timeout=5) as r:
        health = json.load(r)
    assert health['tp']['world_size'] == 2 and health['tp']['peer'] == 'up', health
    print('PASS: large prompts, reuse, stream, peer up ('+args.mode+')', flush=True)


if __name__ == '__main__':
    main()
