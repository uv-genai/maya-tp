import json
from pathlib import Path
import subprocess

from pair import BASE, ROOT, PYTHON

REPO = Path('/home/ugo/software/project-maya')
cases = [
    ('S-technical', 'Maya-S-v2-IQ2_XXS', 256, 1,
     REPO/'docs/benchmarks/tp2-mtp-review-2026-10-30/tp2-technical-reference.txt', BASE/'technical-prompt-ids.txt'),
    ('S-code', 'Maya-S-v2-IQ2_XXS', 256, 2,
     REPO/'docs/benchmarks/tp2-mtp-review-2026-10-30/tp2-code-reference.txt', BASE/'code-prompt-ids.txt'),
    ('M', 'Maya-M', 64, 2, REPO/'docs/upstream/v1.1.39/Maya-M/reference.txt', BASE/'technical-prompt-ids.txt'),
    ('L', 'Maya-L', 64, 2, REPO/'docs/upstream/v1.1.39/Maya-L/reference.txt', BASE/'technical-prompt-ids.txt'),
]
for label, model, tokens, drafts, ref, prompt in cases:
    with (BASE/('strict-'+label+'.txt')).open('w') as log:
        subprocess.run([PYTHON, str(BASE/'bench_mtp_owned.py'), '--output', str(BASE/('strict-'+label)),
              '--model', model, '--pack', f'/home/ugo/software/Maya-data/models/{model}/pack', '--tokens', str(tokens), '--runs', '2', '--drafts', str(drafts),
              '--reference', str(ref), '--prompt-file', str(prompt)], cwd=ROOT,
              stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads((BASE/('strict-'+label)/'results.json').read_text())
    assert len(data['warmups']) == len(data['results']) == 4
    assert all(r['model_pick_differences'] == 0 for r in data['warmups'] + data['results'])
    print('PASS strict', label, '8 warm/measured records, raw pick differences0', flush=True)
print('PASS all S/M/L strict historical references', flush=True)
