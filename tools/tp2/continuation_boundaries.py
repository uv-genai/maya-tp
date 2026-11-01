#!/usr/bin/env python3
"""Live GLM TP2 continuation/context guard (#66), on an already running pair.

Use matching ranks at context4096 with MTP enabled; pass that pack's engine EOS id.
Forces EOS while thinking near the context boundary, then requires a normal MTP follow-up.
The script neither launches nor stops processes and does not select a different model.
"""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.server import CTX_SLACK


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://192.168.1.22:18080')
    ap.add_argument('--model', default='maya')
    ap.add_argument('--eos-id', type=int, required=True)
    ap.add_argument('--context', type=int, default=4096)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('output exists; use a new path')
    if args.eos_id < 0:
        ap.error('EOS id must be nonnegative')
    base = args.url.rstrip('/')

    def request(path, body=None):
        q = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(q, timeout=600) as response:
            return json.load(response)

    health = request('/health')
    assert health['tp']['world_size'] == 2 and health['tp']['peer'] == 'up', health
    assert health['max_context'] == args.context, health
    maximum = 64
    target = args.context - CTX_SLACK - maximum - 4
    assert target > 512
    rows = []
    for mode in ('mtp', 'simple'):
        def body(repeats):
            value = dict(model=args.model, messages=[dict(role='user', content='alpha ' * repeats)],
                         max_tokens=maximum, temperature=0, reasoning_effort='max',
                         logit_bias={str(args.eos_id): 100})
            if mode == 'simple':
                value['strata_tune'] = {'spec': 1}
            return value
        lo, hi = 1, target * 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if request('/api/tokens', body(mid))['tokens'] <= target:
                lo = mid
            else:
                hi = mid - 1
        payload = body(lo)
        count = request('/api/tokens', payload)['tokens']
        assert 0 <= target - count < 4, (count, target)
        out = request('/v1/chat/completions', payload)
        assert out['choices'][0]['finish_reason'] == 'length', out
        assert 0 < out['usage']['completion_tokens'] <= 2, out  # Must hit the continuation guard, not the64-token cap.
        if mode == 'simple':
            assert out.get('timings', {}).get('draft_n', 0) == 0, out
        follow = request('/v1/chat/completions', dict(model=args.model, messages=[dict(role='user', content=
                         'Write three paragraphs explaining how RDMA transports tensor slices.')],
                         max_tokens=64, temperature=0))
        assert 0 < follow['usage']['completion_tokens'] <= 64, follow
        assert follow.get('timings', {}).get('draft_n', 0) > 0, follow
        after = request('/health')
        assert after['tp']['peer'] == 'up' and after['max_context'] == args.context, after
        rows.append(dict(mode=mode, prompt_tokens=count, target=target, response=out,
                         followup=follow, health=after))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(eos_id=args.eos_id, context=args.context, cases=rows), indent=2)+'\n')
        print('PASS '+mode+': near-context EOS continuation ends length; MTP follow-up and peer up', flush=True)


if __name__ == '__main__':
    main()
