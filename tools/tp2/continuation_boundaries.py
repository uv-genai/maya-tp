#!/usr/bin/env python3
"""Live GLM TP2 continuation/context guard (#66), on an already running pair.

Use matching ranks at context4096 with MTP enabled; pass that pack's engine EOS id.
Normal mode checks rejection of unsupported MTP bias, then simple-loop biased EOS and MTP follow-up.
--forced-follow requires the engine's explicit follow-token diagnostic set to EOS for both ranks;
it exercises unbiased MTP/simple EOS continuations, with a forced-EOS follow-up rather than fidelity claims.
The script neither launches nor stops processes and does not select a different model.
"""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.server import CTX_SLACK


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://192.168.1.22:18080')
    ap.add_argument('--model', default='maya')
    ap.add_argument('--eos-id', type=int, required=True)
    ap.add_argument('--context', type=int, default=4096)
    ap.add_argument('--forced-follow', action='store_true')
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
                         max_tokens=maximum, temperature=0, reasoning_effort='max')
            if not args.forced_follow:
                value['logit_bias'] = {str(args.eos_id): 100}
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
        if mode == 'mtp' and not args.forced_follow:
            try:
                request('/v1/chat/completions', payload)
            except urllib.error.HTTPError as e:
                with e:
                    rejection = json.load(e)
                    assert e.code == 400 and 'TP2 MTP' in rejection['error']['message'], rejection
            else:
                raise AssertionError('unsupported TP2 MTP bias was accepted')
            out = {'rejected': rejection, 'status': 400}
        else:
            out = request('/v1/chat/completions', payload)
            assert out['choices'][0]['finish_reason'] == 'length', out
            assert 0 < out['usage']['completion_tokens'] <= 2, out  # Must hit the context guard, not the64-token cap.
            if mode == 'simple':
                assert out.get('timings', {}).get('draft_n', 0) == 0, out
        follow = request('/v1/chat/completions', dict(model=args.model, messages=[dict(role='user', content=
                         'Write three paragraphs explaining how RDMA transports tensor slices.')],
                         max_tokens=64, temperature=0))
        assert 0 < follow['usage']['completion_tokens'] <= 64, follow
        if args.forced_follow:
            assert follow['choices'][0]['finish_reason'] == 'stop' and follow['usage']['completion_tokens'] == 3, follow
        else:
            assert follow.get('timings', {}).get('draft_n', 0) > 0, follow
        after = request('/health')
        assert after['tp']['peer'] == 'up' and after['max_context'] == args.context, after
        rows.append(dict(mode=mode, prompt_tokens=count, target=target, response=out,
                         followup=follow, health=after))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(eos_id=args.eos_id, context=args.context, forced_follow=args.forced_follow, cases=rows), indent=2)+'\n')
        print('PASS '+mode+': continuation/rejection boundary, follow-up and peer up', flush=True)


if __name__ == '__main__':
    main()
