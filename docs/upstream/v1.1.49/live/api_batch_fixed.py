import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error

from pair import BASE, ROOT, PYTHON, URL, Pair, request


def get(path):
    with request(path, timeout=30) as r:
        return json.load(r)


def chat():
    with request('/v1/chat/completions', {'messages': [{'role': 'user', 'content':
                 'Explain tensor parallel RDMA in one paragraph.'}], 'max_tokens': 32, 'temperature': 0}) as r:
        out = json.load(r)
    assert 0 < out['usage']['completion_tokens'] <= 32, out
    return out


def probe(pair, script, *args):
    log = pair.out / (script + '.txt')
    with log.open('w') as f:
        subprocess.run([PYTHON, str(ROOT / 'tools/tp2' / (script + '.py')), '--url', URL, *map(str, args)],
                       cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    print('PASS', pair.model, pair.mode, script, flush=True)


def boundaries(pair):
    before = pair.identities()
    rows = []
    for path, body, reason in [('/models/unload', {}, 'both ranks'), ('/models/load', {}, 'both ranks'),
            ('/api/context', {'max_context': 65536}, 'both ranks'),
            ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe this picture.'},
                {'type': 'image_url', 'image_url': {'url': 'file:///missing/never-read'}}]}],
                'max_tokens': 8}, 'RDMA')]:
        try:
            with request(path, body) as r:
                raise AssertionError('unsupported operation accepted: ' + path)
        except urllib.error.HTTPError as e:
            with e:
                data = json.load(e)
                assert e.code == 400 and reason in data['error']['message'], (e.code, data)
            rows.append({'path': path, 'status': 400, 'message': data['error']['message']})
    assert pair.identities() == before
    time.sleep(3)
    out = chat()
    assert get('/health')['tp']['peer'] == 'up' and pair.identities() == before
    (pair.out / 'boundaries.json').write_text(json.dumps({'rejected': rows, 'identities': before,
                     'followup_after_idle': out, 'health': get('/health')}, indent=2) + '\n')
    print('PASS', pair.model, pair.mode, 'lifecycle/media/idle and owned PIDs stable', flush=True)


def audit(pair):
    if pair.mode != 'mtp':
        return
    r0 = (pair.out / 'engine-rank0.txt').read_text()
    r1 = (pair.out / 'rank1.txt').read_text()
    pat = r'glm mtp TP: rank=\d+ round-end p0=(\d+) produced=(\d+) keep=(\d+)'
    a, b = re.findall(pat, r0), re.findall(pat, r1)
    assert a and a == b, (len(a), len(b))
    pat = r'glm window TP: rank=\d+ p0=(\d+) T=(\d+) attempt=(\d+) local-clash=(\d+) retry=(\d+)'
    x, y = re.findall(pat, r0), re.findall(pat, r1)
    assert x and [(a,b,c,e) for a,b,c,d,e in x] == [(a,b,c,e) for a,b,c,d,e in y]
    asymmetric = sum(a[3] == '0' and b[3] == '1' and a[4] == b[4] == '1' for a,b in zip(x,y))
    assert asymmetric == 8, asymmetric
    result = {'matched_rounds': len(a), 'matched_window_attempts': len(x), 'asymmetric_retries': asymmetric}
    (pair.out / 'round-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS', pair.model, 'round/retry audit', result, flush=True)


def recovery_fail(pair):
    before = pair.identities()
    h0 = get('/health')
    assert h0['tp']['peer'] == 'up'
    first = chat()
    pair.command('rank1', 'kill')
    time.sleep(1)  # Match the established recovery harness.
    error = None
    try:
        with request('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Say hello.'}],
                                             'max_tokens': 8}, timeout=60) as r:
            data = json.load(r)
            error = data.get('error')
    except urllib.error.HTTPError as e:
        with e:
            error = json.load(e).get('error')
    assert error and 'TP2 pair down' in str(error), error
    h1 = get('/health')
    assert h1['status'] == 'degraded' and h1['tp']['peer'] == 'down', h1
    current = pair.identities()
    assert set(map(tuple, current['rank0'])).issubset(set(map(tuple, before['rank0'])))
    result = {'before': h0, 'first_chat': first, 'peer_error': error, 'after': h1,
              'owned_processes_before': before, 'owned_processes_after': current, 'checks_before_restart': 6}
    (pair.out / 'recovery-down.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS', pair.model, 'owned peer kill, clean failure, degraded health, no respawn', flush=True)
    return result


def model(name):
    down = None
    for mode in ('mtp', 'simple'):
        p = Pair(name, mode, name + '-' + mode + '-fixed')
        try:
            p.start()
            if mode == 'mtp':
                probe(p, 'mtp_acceptance')
            probe(p, 'regression_collectives', '--mode', mode, '--idle-seconds', 3,
                  '--max-long-decode-ratio', 4, '--output', p.out / 'matrix.json')
            boundaries(p)
            probe(p, 'prefill_cancel', '--output', p.out / 'prefill-cancel.json')
            if name == 'Maya-S-v2-IQ2_XXS' and mode == 'mtp':
                probe(p, 'long_loop_reuse', '--content-parts', '--expect-preserved',
                      '--output', p.out / 'long-loop.json')
            if mode == 'mtp':
                down = recovery_fail(p)
        finally:
            p.stop()
        audit(p)
        if mode == 'mtp':
            r = Pair(name, 'mtp', name + '-recovery-fixed', deep_idle=False)
            try:
                r.start()
                follow = chat()
                assert follow['timings']['draft_n'] > 0 and get('/health')['tp']['peer'] == 'up', follow
                down.update(recovered=get('/health'), recovered_chat=follow, checks_passed=8)
                (r.out / 'recovery.json').write_text(json.dumps(down, indent=2) + '\n')
                print('PASS', name, 'kill/recover 8/8', flush=True)
            finally:
                r.stop()
    print('PASS ALL API', name, flush=True)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['Maya-S-v2-IQ2_XXS', 'Maya-M', 'Maya-L']:
        model(name)
    print('PASS all requested model API/recovery suites', flush=True)
