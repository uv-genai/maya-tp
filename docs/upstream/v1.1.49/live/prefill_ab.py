import json
from pathlib import Path
import statistics
import subprocess
import urllib.error

import pair
from pair import BASE, Pair, PYTHON, HOST, URL

OUT = BASE / 'prefill-ab'
OUT.mkdir(exist_ok=False)
for arm, source in [('candidate', 'fixed-source'), ('baseline', 'control-source')]:
    pair.ROOT = BASE / source
    p = Pair('Maya-S-v2-IQ2_XXS', 'mtp', 'prefill-' + arm, drafts=2, deep_idle=False)
    p.server['argv'][p.server['argv'].index('--engine') + 1] = 'strata'
    cfg = json.loads(p.config.read_text())
    cfg['model_management'] = False
    cfg.pop('vision', None)
    for key in list(cfg['env']):
        if key.endswith(('GLM_MTP_TRACE', 'GLM_KEEP_CANCELLED', 'GLM_CKPT_EVERY', 'GLM_CKPTS')):
            del cfg['env'][key]
    for key in list(p.worker['env']):
        if key.endswith(('GLM_MTP_TRACE', 'GLM_KEEP_CANCELLED', 'GLM_CKPT_EVERY', 'GLM_CKPTS')):
            del p.worker['env'][key]
    p.config.write_text(json.dumps(cfg, indent=2) + '\n')
    subprocess.run(['scp', '-q', str(p.config), HOST + ':' + str(p.config)], check=True)
    try:
        p.start()
        with (OUT/(arm+'.txt')).open('w') as log:
            subprocess.run([PYTHON, str(pair.ROOT/'tools/tp2/bench_prefill.py'), '--url', URL,
                    '--runs', '3', '--output', str(OUT/(arm+'.json'))], cwd=pair.ROOT,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        if arm == 'candidate':
            records = {}
            body = {'messages': [{'role': 'user', 'content': 'Explain RDMA in one sentence.'}],
                    'max_tokens': 32, 'temperature': 0, 'logit_bias': {'0': 1.0}}
            before = p.identities()
            try:
                with pair.request('/v1/chat/completions', body):
                    raise AssertionError('nonempty TP2 MTP bias was accepted')
            except urllib.error.HTTPError as e:
                with e:
                    data = json.load(e)
                    assert e.code == 400 and 'logit_bias' in data['error']['message'], data
                    records['rejected_mtp_bias'] = data
            assert p.identities() == before
            with pair.request('/v1/chat/completions', {**body, 'strata_tune': {'spec': 1}}) as r:
                data = json.load(r)
            assert 0 < data['usage']['completion_tokens'] <= 32 and data['timings']['draft_n'] == 0, data
            records['ordinary_bias_spec1'] = data
            for path, payload, key in [
                    ('/v1/responses', {'model': 'maya', 'input': 'Explain RDMA briefly.',
                                       'max_output_tokens': 32}, 'output_tokens'),
                    ('/v1/messages', {'model': 'maya', 'messages': body['messages'],
                                      'max_tokens': 32, 'temperature': 0}, 'output_tokens')]:
                with pair.request(path, payload) as r:
                    data = json.load(r)
                assert 0 < data['usage'][key] <= 32, data
                records[path] = data
            with pair.request('/v1/chat/completions', {k:v for k,v in body.items() if k != 'logit_bias'}) as r:
                data = json.load(r)
            assert data['timings']['draft_n'] > 0 and 0 < data['usage']['completion_tokens'] <= 32, data
            records['mtp_followup'] = data
            with pair.request('/health') as r:
                data = json.load(r)
            assert data['tp']['peer'] == 'up' and p.identities() == before, data
            records['health'] = data
            (OUT/'extra-api.json').write_text(json.dumps(records, indent=2)+'\n')
            print('PASS MTP bias rejection/spec1, Responses/Anthropic limits and MTP follow-up', flush=True)
        print('PASS prefill arm', arm, flush=True)
    finally:
        p.stop()
a = json.loads((OUT/'baseline.json').read_text())
b = json.loads((OUT/'candidate.json').read_text())
assert len(a) == len(b) == 9
pairs = []
for x, y in zip(a, b):
    assert (x['trial'], x['repeats']) == (y['trial'], y['repeats'])
    rx, ry = x['response'], y['response']
    for r in (rx, ry):
        assert r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0
    row = {'trial': x['trial'], 'prompt_tokens': rx['usage']['prompt_tokens'],
           'messages_equal': rx['choices'][0]['message'] == ry['choices'][0]['message'],
           'usage_equal': rx['usage'] == ry['usage']}
    pairs.append(row)
summary = []
for n in (960, 4032, 8128):
    aa = [x['response']['timings']['prompt_ms'] for x in a if x['repeats'] == n and x['trial'] >= 3]
    bb = [x['response']['timings']['prompt_ms'] for x in b if x['repeats'] == n and x['trial'] >= 3]
    am, bm = statistics.median(aa), statistics.median(bb)
    summary.append({'repeats': n, 'baseline_prompt_ms': am, 'candidate_prompt_ms': bm,
                    'throughput_delta_percent': 100*(am/bm-1),
                    'baseline_samples_ms': aa, 'candidate_samples_ms': bb})
out = {'order': ['candidate', 'baseline'], 'engine_md5_both': 'b398d4a0711430edd6a28b95264c7293',
       'baseline_server': 'd7626ac', 'candidate_server': 'e1682cb',
       'warmup': 'first full 1K/4K/8K cycle retained and excluded from medians',
       'pairs': pairs, 'summary': summary}
(OUT/'comparison.json').write_text(json.dumps(out, indent=2)+'\n')
assert all(x['messages_equal'] and x['usage_equal'] for x in pairs)
print(json.dumps(out, indent=2), flush=True)
print('PASS all 9 prefill responses/usage/cache0; warm timing comparison recorded', flush=True)
