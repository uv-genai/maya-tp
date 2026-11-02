import json
from pathlib import Path
import statistics
import subprocess
import urllib.error

import pair
from pair import BASE, Pair, PYTHON, HOST, URL

OUT = BASE / 'prefill-ab-reverse'
OUT.mkdir(exist_ok=False)
for arm, source in [('baseline', 'control-source'), ('candidate', 'fixed-source')]:
    pair.ROOT = BASE / source
    p = Pair('Maya-S-v2-IQ2_XXS', 'mtp', 'prefill-reverse-' + arm, drafts=2, deep_idle=False)
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
out = {'order': ['baseline', 'candidate'], 'engine_md5_both': 'b398d4a0711430edd6a28b95264c7293',
       'baseline_server': 'd7626ac', 'candidate_server': 'e1682cb',
       'warmup': 'first full 1K/4K/8K cycle retained and excluded from medians',
       'pairs': pairs, 'summary': summary}
(OUT/'comparison.json').write_text(json.dumps(out, indent=2)+'\n')
assert all(x['messages_equal'] and x['usage_equal'] for x in pairs)
print(json.dumps(out, indent=2), flush=True)
print('PASS all 9 prefill responses/usage/cache0; warm timing comparison recorded', flush=True)
