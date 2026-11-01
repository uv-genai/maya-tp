#!/usr/bin/env python3
"""Live GLM TP2 long-loop prefix reuse probe, against an already running pair.

Run each policy on a fresh matching pair at context32768: collapse limit0 versus32000.
The synthetic tool history is supplied as data; no tool is executed. No model/path is changed.
"""
import argparse
import json
from pathlib import Path
import random
import urllib.request


def fixture(turns):
    rng = random.Random(904)
    data = ' '.join(''.join(rng.choices('abcdefghijkmnpqrstuvwxyz', k=24)) for _ in range(1500))
    messages = [dict(role='user', content='Inspect the supplied synthetic transport frame and explain it.')]
    for i in range(turns):
        reasoning = 'I need to inspect the same transport frame.\n\nI am stuck repeating the same thing.'
        if i == 0:
            reasoning += '\n\n' + data
        messages += [dict(role='assistant', content='', reasoning_content=reasoning, tool_calls=[
            dict(id=f'fixture_{i}', type='function', function=dict(name='inspect_frame', arguments='{"case":"same"}'))]),
            dict(role='tool', tool_call_id=f'fixture_{i}', content='{"frame":"unchanged"}')]
    return messages


TOOLS = [dict(type='function', function=dict(name='inspect_frame', description='Read a synthetic frame fixture',
         parameters=dict(type='object', properties={'case': {'type': 'string'}}, required=['case'])))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://192.168.1.22:18080')
    ap.add_argument('--model', default='maya')
    ap.add_argument('--output', type=Path, required=True)
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument('--expect-preserved', action='store_true')
    choice.add_argument('--expect-collapsed', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('output exists; use a new path')
    base = args.url.rstrip('/')
    def request(path, body=None):
        q = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(),
                                   headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(q, timeout=600) as response:
            return json.load(response)
    before = request('/health')
    assert before['tp']['world_size'] == 2 and before['tp']['peer'] == 'up', before
    assert before['max_context'] >= 32768, before
    rows = []
    for turns in (2, 3):
        out = request('/v1/chat/completions', dict(model=args.model, messages=fixture(turns), tools=TOOLS,
                                                 max_tokens=8, temperature=0, reasoning_effort='max'))
        assert 0 < out['usage']['completion_tokens'] <= 8, out
        rows.append(out)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(policy='keep' if args.expect_preserved else 'collapse',
                                               responses=rows), indent=2)+'\n')
    initial = rows[0]['usage']['prompt_tokens']
    cached = rows[1]['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
    assert initial > 15000, initial
    assert rows[0]['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0, rows[0]
    metrics = request('/metrics')
    collapsed = metrics['loop_guard']['collapsed']
    if args.expect_preserved:
        assert cached >= initial - 256 and collapsed == 0, (cached, initial, collapsed)
    else:
        assert cached < 512 and collapsed > 0, (cached, initial, collapsed)
    after = request('/health')
    assert after['tp']['peer'] == 'up', after
    result = dict(policy='keep' if args.expect_preserved else 'collapse', initial_tokens=initial,
                  follow_cached=cached, collapsed=collapsed, responses=rows, health=after)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('PASS '+result['policy']+': initial='+str(initial)+' follow_cached='+str(cached), flush=True)


if __name__ == '__main__':
    main()
