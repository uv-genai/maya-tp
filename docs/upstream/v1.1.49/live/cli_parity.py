import json
import os
from pathlib import Path
import shlex
import subprocess
import uuid

import owned
from pair import BASE, ROOT, HOST, PYTHON, OWNER_TOOL, remote

OUT = BASE / 'cli-fixed'
OUT.mkdir(exist_ok=True)
PROMPT = '154822,25062,287,29905,371,25,7487,785,6722,315,9621,374'
REPO = Path('/home/ugo/software/project-maya')


def test(model, mode):
    label = model + '-' + mode
    owner = uuid.uuid4().hex
    reg0, reg1 = OUT / (label + '-rank0-processes.json'), OUT / (label + '-rank1-processes.json')
    common = {'MAYA_GLM_TIMING': '1', 'MAYA_GLM_LOOP_THINK': '0', 'MAYA_GLM_LOOP_ANSWER': '0'}
    if model == 'Maya-L':
        common['STRATA_GLM_WARM'] = '0'
    if mode == 'mtp':
        common.update(MAYA_TP_MTP='1', MAYA_GLM_MTP_FIXED='7', MAYA_GLM_MTP_DRAFT='7')
    else:
        common['MAYA_GLM_NO_MTP'] = '1'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('MAYA_', 'STRATA_'))}
    env.update(common, MAYA_VALIDATION_OWNER=owner)
    args = ['--glm-pack', f'/home/ugo/software/Maya-data/models/{model}/pack', '--tokens', PROMPT,
            '--max-new', '32', '--greedy', '--max-context', '4096', '--tp-world-size', '2',
            '--tp-bootstrap-port', '18587', '--tp-rdma-device', 'usb4_rdma5']
    remote([PYTHON, OWNER_TOOL, 'ports', reg0, owner, '18587'])
    r0ids, r1ids = OUT/(label+'-rank0-ids.txt'), OUT/(label+'-rank1-ids.txt')
    p0 = p1 = None
    with (OUT/(label+'-rank1.txt')).open('w') as f1, (OUT/(label+'-rank0.txt')).open('w') as f0:
        try:
            p1 = subprocess.Popen([str(ROOT/'build-hip/maya'), *args, '--tp-rank', '1',
                    '--tp-bootstrap-host', HOST, '--spec-follow-save', str(r1ids)], cwd=ROOT,
                    env=env, stdin=subprocess.DEVNULL, stdout=f1, stderr=subprocess.STDOUT, start_new_session=True)
            owned.save(reg1, {'owner': owner, 'root': p1.pid, 'processes': [owned.identity(p1.pid)]})
            cmd = [PYTHON, OWNER_TOOL, 'exec', str(reg0), owner, 'env',
                   *[k+'='+v for k,v in common.items()], str(ROOT/'build-hip/maya'), *args,
                   '--tp-rank', '0', '--spec-follow-save', str(r0ids)]
            p0 = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', HOST, 'exec '+shlex.join(cmd)],
                    stdin=subprocess.DEVNULL, stdout=f0, stderr=subprocess.STDOUT)
            c0, c1 = p0.wait(timeout=600), p1.wait(timeout=60)
            assert (c0, c1) == (0, 0), (label, c0, c1)
            text = remote(['cat', r0ids])
            r0ids.write_text(text)
            a = [int(x) for x in text.replace(',', ' ').split()]
            b = [int(x) for x in r1ids.read_text().replace(',', ' ').split()]
            assert len(a) == len(b) == 32 and a == b, (label, a, b)
            result = {'model': model, 'mode': mode, 'pair_matches': 32, 'rank0_ids': a, 'rank1_ids': b}
            if model == 'Maya-S-v2-IQ2_XXS':
                old = REPO/'docs/upstream/v1.1.42/cli'/(label+'-rank0-ids.txt')
                reference = [int(x) for x in old.read_text().replace(',', ' ').split()]
                assert a == reference, (label, 'changed from accepted v42 CLI trajectory')
                result['accepted_v42_exact'] = True
            (OUT/(label+'-result.json')).write_text(json.dumps(result, indent=2)+'\n')
            print('PASS', label, 'CLI32/32', flush=True)
        finally:
            if p0 is not None:
                if p0.poll() is None:
                    p0.terminate()
                p0.wait(timeout=20)
                cleanup = remote([PYTHON, OWNER_TOOL, 'stop', reg0, owner])
                (OUT/(label+'-rank0-cleanup.json')).write_text(cleanup)
            if p1 is not None:
                subprocess.run([PYTHON, OWNER_TOOL, 'stop', str(reg1), owner], check=True, capture_output=True)
                p1.wait(timeout=20)


for model in ('Maya-S-v2-IQ2_XXS', 'Maya-M', 'Maya-L'):
    for mode in ('simple', 'mtp'):
        test(model, mode)
print('PASS all S/M/L paired CLI modes', flush=True)
