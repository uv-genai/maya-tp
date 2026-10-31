#!/usr/bin/env python3
"""Measure fresh TP2 API prefill at several prompt lengths on an idle pair.

Start matching ranks with the desired model/config, context 32768 and MTP fixed/cap
2. Restart the pair between arms; this probe never starts or stops services.
Keep settings and request order identical. Save both engine logs with the JSON.
The synthetic repeated-word document measures prompt processing, not answer quality.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://192.168.1.22:18080')
    parser.add_argument('--model', default='maya', help='API model id; select the pack on both rank launchers')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--lengths', default='960,4032,8128', help='document word counts, approximately 1K/4K/8K tokens')
    args = parser.parse_args()
    lengths = [int(n) for n in args.lengths.split(',')]
    if args.runs < 1 or not lengths or min(lengths) < 1:
        parser.error('runs and lengths must be positive')
    if args.output.exists():
        parser.error('output already exists; use a new path for each arm')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in lengths * args.runs:
        trial = len(rows)
        body = dict(model=args.model, messages=[
            dict(role='system', content=f'CASE {trial} transport test. ' + 'alpha ' * n),
            dict(role='user', content='Explain why network frames need consistent batch sizes.')],
            max_tokens=8, temperature=0)
        started = time.monotonic()
        req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            out = json.load(response)
        if 'error' in out:
            raise RuntimeError(out['error'])
        cached = out['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if cached:
            raise RuntimeError(f'fresh prompt reused {cached} tokens; restart the pair before comparison')
        if not out.get('timings', {}).get('prompt_ms', 0) > 0:
            raise RuntimeError('response has no prefill timing')
        row = dict(trial=trial, repeats=n, wall_s=time.monotonic() - started, response=out)
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(dict(trial=trial, repeats=n, usage=out['usage'], timings=out['timings'])), flush=True)
    with urllib.request.urlopen(args.url.rstrip('/') + '/health', timeout=10) as response:
        health = json.load(response)
    if health.get('tp', {}).get('peer') != 'up' or health.get('tp', {}).get('world_size') != 2:
        raise RuntimeError(f'pair unhealthy after probe: {health}')


if __name__ == '__main__':
    main()
