import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
import uuid

BASE = Path('/tmp/maya-v49-live-87')
ROOT = BASE / 'fixed-source'
HOST = 'strix-halo-1.local'
PYTHON = '/home/ugo/software/project-maya/.venv/bin/python'
OWNER_TOOL = str(BASE / 'owned.py')
URL = 'http://192.168.1.22:18087'
BOOT = '18587'


def remote(argv):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', HOST, shlex.join(list(map(str, argv)))], text=True)


class Pair:
    def __init__(self, model, mode, label, drafts=7, deep_idle=True):
        self.model, self.mode = model, mode
        self.out = BASE / label
        self.out.mkdir(parents=True, exist_ok=False)
        self.owner = uuid.uuid4().hex
        self.started = []
        cfg = json.loads(Path('/home/ugo/software/project-maya-upstream-review/maya-maya-s-v2-iq2_xxs-hip.json').read_text())
        pack = f'/home/ugo/software/Maya-data/models/{model}/pack'
        cfg['exe'] = str(ROOT / 'build-hip/maya')
        cfg['args'][cfg['args'].index('--glm-pack') + 1] = pack
        cfg['args'][cfg['args'].index('--expert-profile') + 1] = str(ROOT / 'data/expert-profile-glm.bin')
        cfg['tokenizer'] = pack + '/tokenizer'
        cfg['cwd'] = str(ROOT)
        cfg['log'] = str(self.out / 'engine-rank0.txt')
        cfg['model_management'] = True
        cfg['vision'] = {'exe': '/missing/encoder-must-not-start', 'gpu': True}
        cfg['env']['MAYA_HIPBLASLT_TUNING'] = cfg['env']['MAYA_HIPBLASLT_TUNING'].replace(
            '/home/ugo/software/project-maya-upstream-review', str(ROOT))
        cfg['env'].update(MAYA_GLM_KEEP_CANCELLED='1', MAYA_GLM_CKPT_EVERY='4096', MAYA_GLM_CKPTS='16')
        if deep_idle:
            cfg['env'].update(MAYA_GLM_SERVICE_IDLE_MS='10', MAYA_GLM_SERVICE_DEEP_S='1', MAYA_GLM_SERVICE_DEEP_MS='3000')
        if mode == 'mtp':
            cfg['env'].update(MAYA_TP_MTP='1', MAYA_GLM_MTP_FIXED=str(drafts), MAYA_GLM_MTP_DRAFT=str(drafts),
                              MAYA_GLM_MTP_TRACE='1')
        else:
            cfg['env'].update(MAYA_GLM_NO_MTP='1', MAYA_TP_MTP='0')
        common = ['--tp-world-size', '2', '--tp-bootstrap-port', BOOT, '--tp-rdma-device', 'usb4_rdma5']
        args = cfg['args'][:]
        cfg['args'] += common + ['--tp-rank', '0', '--serve']
        self.config = self.out / 'config.json'
        self.config.write_text(json.dumps(cfg, indent=2) + '\n')
        remote(['mkdir', '-p', self.out])
        subprocess.run(['scp', '-q', str(self.config), HOST + ':' + str(self.config)], check=True)
        worker_env = dict(cfg['env'])
        for key, value in list(worker_env.items()):
            if key.startswith('MAYA_'):
                worker_env['STRATA_' + key[5:]] = value
        if mode == 'mtp' and drafts == 7:
            worker_env.update(MAYA_TP_TEST_WINDOW_RETRY='8', STRATA_TP_TEST_WINDOW_RETRY='8')
        self.worker = {'argv': [cfg['exe'], *args, *common, '--tp-rank', '1', '--tp-bootstrap-host', HOST],
                       'cwd': str(ROOT), 'env': worker_env, 'log': str(self.out / 'rank1.txt')}
        self.server = {'argv': [PYTHON, str(ROOT / 'serve/server.py'), '--engine', 'maya' if mode == 'mtp' else 'strata',
                                '--config', str(self.config), '--port', '18087', '--host', '0.0.0.0'],
                       'cwd': str(ROOT), 'env': {}, 'log': str(self.out / 'rank0.txt')}
        (self.out / 'owner.txt').write_text(self.owner)

    def command(self, side, action, extra=()):
        registry = self.out / (side + '-processes.json')
        argv = [PYTHON, OWNER_TOOL, action, str(registry), self.owner, *map(str, extra)]
        result = remote(argv) if side == 'rank0' else subprocess.check_output(argv, text=True)
        if action != 'ports':
            data = json.loads(result)
            if side == 'rank0':
                (self.out / 'rank0-processes.json').write_text(json.dumps(data, indent=2) + '\n')
            return data

    def start(self):
        self.command('rank0', 'ports', ['18087', BOOT])
        for side, spec in [('rank1', self.worker), ('rank0', self.server)]:
            path = self.out / (side + '-spec.json')
            path.write_text(json.dumps(spec, indent=2))
            if side == 'rank0':
                subprocess.run(['scp', '-q', str(path), HOST + ':' + str(path)], check=True)
            self.command(side, 'start', [path])
            self.started.append(side)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            status = self.command('rank1', 'status')
            if not status['alive']:
                raise RuntimeError('owned worker exited: ' + str(self.out / 'rank1.txt'))
            try:
                with urllib.request.urlopen(URL + '/health', timeout=2) as r:
                    h = json.load(r)
                if h.get('tp', {}).get('peer') == 'up':
                    self.command('rank0', 'status')
                    (self.out / 'ready.json').write_text(json.dumps(h, indent=2) + '\n')
                    print('READY', self.model, self.mode, flush=True)
                    return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError('pair startup exceeded unchanged 900s bound')

    def identities(self):
        return {side: [(x['pid'], x['start']) for x in self.command(side, 'status')['alive']]
                for side in self.started}

    def stop(self):
        errors = []
        for side in ('rank0', 'rank1'):
            if side not in self.started:
                continue
            try:
                self.command(side, 'stop')
            except Exception as e:
                errors.append(str(e))
        for name in (('rank0.txt', 'engine-rank0.txt') if 'rank0' in self.started else ()):
            try:
                subprocess.run(['scp', '-q', HOST + ':' + str(self.out / name), str(self.out / name)], check=True)
            except Exception as e:
                errors.append(str(e))
        if errors:
            raise RuntimeError('owned cleanup/log errors: ' + '; '.join(errors))


def request(path, body=None, timeout=600):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Origin': URL})
    return urllib.request.urlopen(req, timeout=timeout)
