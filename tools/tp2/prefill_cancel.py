#!/usr/bin/env python3
"""Live TP2 partial-prefill disconnect and retry regression (no process management).

Start both ranks with context32768, MAYA_GLM_CKPTS=16, MAYA_GLM_CKPT_EVERY=4096,
and MAYA_GLM_KEEP_CANCELLED=1. Run in simple and MTP modes. The client disconnects
after a completed checkpoint and partway through the next mirrored prefill call.
The retry must start fresh with both ranks healthy; TP2 checkpoint restore remains gated.
"""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://192.168.1.22:18080')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()

    def request(path, body=None):
        req = urllib.request.Request(args.url + path,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            return urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as error:
            with error:
                raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from error

    body = {'messages': [{'role': 'system', 'content':
                         'Paired prefill cancellation probe, independent fresh history.\n' + 'alpha ' * 18000},
                         {'role': 'user', 'content': 'Explain the document in one sentence.'}],
            'max_tokens': 8, 'temperature': 0, 'stream': True, 'return_progress': True}
    progress = None
    with request('/v1/chat/completions', body) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            assert 'error' not in event, event
            pp = event.get('prompt_progress')
            if pp and 5000 <= pp.get('processed', 0) < pp.get('total', 0) - 1000:
                progress = pp
                break
    assert progress, 'no partial prefill progress after the first checkpoint was observed'
    result = {'progress_at_disconnect': progress}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    with request('/v1/chat/completions', {**body, 'stream': False}) as response:
        retry = json.load(response)
    cached = retry['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
    assert cached == 0, retry
    assert 0 < retry['usage']['completion_tokens'] <= body['max_tokens'], retry
    with request('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Write one sentence about RDMA.'}],
                                        'max_tokens': 32, 'temperature': 0}) as response:
        followup = json.load(response)
    assert 0 < followup['usage']['completion_tokens'] <= 32, followup
    with request('/health') as response:
        health = json.load(response)
    assert health['tp']['world_size'] == 2 and health['tp']['peer'] == 'up', health
    result.update(retry_cached=cached, retry=retry, followup=followup, health=health)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: partial-prefill disconnect, fresh paired retry, follow-up and peer up')


if __name__ == '__main__':
    main()
