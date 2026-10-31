"""TP2 launcher wiring tests: fake all SSH/process/GPU commands, no hosts required."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Tp2LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for directory in ('tools/llama-swap', 'tools/tp2', '.venv/bin'):
            (self.root / directory).mkdir(parents=True)
        for script in ('tools/llama-swap/tp2.sh', 'tools/tp2/rank0.sh', 'tools/tp2/rank1.sh',
                       'tools/tp2/model_env.sh'):
            shutil.copy(ROOT / script, self.root / script)
        self.config = self.root / 'maya-maya-s-v2-iq2_xxs-hip.json'
        self.config.write_text(json.dumps(dict(
            args=['--glm-pack', '/model/pack', '--max-context', '32768'],
            tokenizer='/model/pack/tokenizer')))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('MAYA_', 'STRATA_'))}
        self.env.update(PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        TP_TEST_LOG=str(self.root / 'events.jsonl'), TP_TEST_RESULT=str(self.root / 'result.json'))
        self.executable('ssh', '''import os,sys,json
with open(os.environ['TP_TEST_LOG'],'a') as f: f.write(json.dumps(['ssh',sys.argv[1:]])+'\\n')
cmd=sys.argv[-1]
if 'pgrep' in cmd: sys.exit(1)
if 'nohup' in cmd and os.environ.get('TP_TEST_SSH_FAIL'): sys.exit(77)
''')
        self.executable('pkill', '''import os,sys,json
with open(os.environ['TP_TEST_LOG'],'a') as f: f.write(json.dumps(['pkill',sys.argv[1:]])+'\\n')
''')
        self.executable('pgrep', 'import sys; sys.exit(1)')
        self.executable('ss', 'print("State Recv-Q Send-Q Local Address:Port Peer Address:Port")')
        self.executable('tee', 'import sys; sys.stdout.write(sys.stdin.read())')
        self.executable('ibv_devices', 'print("usb4_rdma5 000001")')

    def executable(self, name, body, path=None):
        p = path or self.bin / name
        p.write_text(f'#!{sys.executable}\n' + body + '\n')
        p.chmod(0o755)
        return p

    def run_script(self, script, *args, **extra_env):
        return subprocess.run(['bash', str(self.root / script), *args], env={**self.env, **extra_env},
                              capture_output=True, text=True, timeout=15)

    def stub_rank0(self):
        self.executable('rank0', '''import os,sys,json
keys=('MAYA_TP_MTP','MAYA_GLM_MTP_FIXED','MAYA_GLM_MTP_DRAFT','CONTEXT','SEQS','MAYA_MODEL','MAYA_PACK','MAYA_CONFIG','MAYA_MODELS_DIR')
with open(os.environ['TP_TEST_RESULT'],'w') as f: json.dump({'env':{k:os.environ.get(k) for k in keys},'args':sys.argv[1:]},f)
''', self.root / 'tools/tp2/rank0.sh')

    def test_mtp_four_drafts_reaches_both_ranks_and_stops_server_first(self):
        self.stub_rank0()
        r = self.run_script('tools/llama-swap/tp2.sh', 'run', '18080', '--mtp', '--drafts', '4', '--context', '8192')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        local = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(local['args'], ['18080'])
        self.assertEqual(local['env']['MAYA_TP_MTP'], '1')
        self.assertEqual(local['env']['MAYA_GLM_MTP_FIXED'], '4')
        self.assertEqual(local['env']['MAYA_GLM_MTP_DRAFT'], '4')
        self.assertEqual(local['env']['CONTEXT'], '8192')
        self.assertEqual(local['env']['SEQS'], '1')
        events = [json.loads(line) for line in (self.root / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(events[0][0], 'pkill')
        self.assertIn('[s]erver.py --engine strata --config /tmp/maya-tp2', events[0][1])
        command = next(e[1][-1] for e in events if e[0] == 'ssh' and 'nohup' in e[1][-1])
        for setting in ('MAYA_TP_MTP=1','MAYA_GLM_MTP_FIXED=4','MAYA_GLM_MTP_DRAFT=4','CONTEXT=8192','SEQS=1'):
            self.assertIn(setting, shlex.split(command))
        self.assertNotIn('&&', command)
        self.assertIn('-u MAYA_GLM_NO_MTP', command)

    def test_default_uses_simple_loop_and_configured_context(self):
        self.stub_rank0()
        r = self.run_script('tools/llama-swap/tp2.sh', 'run', '18080')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        env = json.loads((self.root / 'result.json').read_text())['env']
        self.assertEqual(env['MAYA_TP_MTP'], '0')
        self.assertEqual(env['CONTEXT'], '32768')

    def test_invalid_draft_count_does_not_stop_or_start_processes(self):
        r = self.run_script('tools/llama-swap/tp2.sh', 'run', '18080', '--mtp', '--drafts', '9')
        self.assertEqual(r.returncode, 2)
        self.assertFalse((self.root / 'events.jsonl').exists())

    def test_worker_launch_failure_is_reported(self):
        self.stub_rank0()
        r = self.run_script('tools/llama-swap/tp2.sh', 'run', '18080', TP_TEST_SSH_FAIL='1')
        self.assertEqual(r.returncode, 77)
        self.assertFalse((self.root / 'result.json').exists())

    def test_rank0_context_override_reaches_engine_config(self):
        self.executable('server', '''import os,sys,json
p=sys.argv[sys.argv.index('--config')+1]
with open(p) as f: cfg=json.load(f)
with open(os.environ['TP_TEST_RESULT'],'w') as f: json.dump(cfg,f)
os.unlink(p)
''', self.root / '.venv/bin/python')
        r = self.run_script('tools/tp2/rank0.sh', '18080', CONTEXT='8192')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        args = json.loads((self.root / 'result.json').read_text())['args']
        self.assertEqual(args[-2:], ['--max-context','8192'])

    def test_rank1_defaults_to_same_configured_context(self):
        engine = self.executable('engine', '''import os,sys,json
with open(os.environ['TP_TEST_RESULT'],'w') as f: json.dump(sys.argv[1:],f)
''')
        r = self.run_script('tools/tp2/rank1.sh', ENGINE=str(engine))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        args = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(args[-2:], ['--max-context','32768'])

    def test_rank1_model_flag_overrides_pack(self):
        engine = self.executable('engine', '''import os,sys,json
with open(os.environ['TP_TEST_RESULT'],'w') as f: json.dump(sys.argv[1:],f)
''')
        models = self.root / 'models'
        (models / 'Maya-M' / 'pack').mkdir(parents=True)
        r = self.run_script('tools/tp2/rank1.sh', ENGINE=str(engine),
                            MAYA_MODELS_DIR=str(models), MAYA_MODEL='Maya-M')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        args = json.loads((self.root / 'result.json').read_text())
        i = args.index('--glm-pack')
        self.assertEqual(args[i + 1], str(models / 'Maya-M' / 'pack'))

    def test_rank1_cli_pack_beats_env_model(self):
        engine = self.executable('engine', '''import os,sys,json
with open(os.environ['TP_TEST_RESULT'],'w') as f: json.dump(sys.argv[1:],f)
''')
        pack = self.root / 'explicit-pack'
        pack.mkdir()
        r = self.run_script('tools/tp2/rank1.sh', '--pack', str(pack),
                            ENGINE=str(engine), MAYA_MODEL='Maya-M')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        args = json.loads((self.root / 'result.json').read_text())
        i = args.index('--glm-pack')
        self.assertEqual(args[i + 1], str(pack))

    def test_rank0_model_flag_rewrites_config_pack_and_tokenizer(self):
        self.executable('server', '''import os,sys,json
p=sys.argv[sys.argv.index('--config')+1]
with open(p) as f: cfg=json.load(f)
with open(os.environ['TP_TEST_RESULT'],'w') as f: json.dump(cfg,f)
os.unlink(p)
''', self.root / '.venv/bin/python')
        models = self.root / 'models'
        pack = models / 'Maya-M' / 'pack'
        pack.mkdir(parents=True)
        r = self.run_script('tools/tp2/rank0.sh', '18080', '--model', 'Maya-M',
                            MAYA_MODELS_DIR=str(models))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        cfg = json.loads((self.root / 'result.json').read_text())
        self.assertIn('--glm-pack', cfg['args'])
        self.assertEqual(cfg['args'][cfg['args'].index('--glm-pack') + 1], str(pack))
        self.assertEqual(cfg['tokenizer'], str(pack / 'tokenizer'))

    def test_tp2_swap_forwards_model_to_both_ranks(self):
        self.stub_rank0()
        models = self.root / 'models'
        (models / 'Maya-M' / 'pack').mkdir(parents=True)
        r = self.run_script('tools/llama-swap/tp2.sh', 'run', '18080', '--model', 'Maya-M',
                            MAYA_MODELS_DIR=str(models))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        local = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(local['env']['MAYA_MODEL'], 'Maya-M')
        self.assertEqual(local['env']['MAYA_MODELS_DIR'], str(models))
        events = [json.loads(line) for line in (self.root / 'events.jsonl').read_text().splitlines()]
        command = next(e[1][-1] for e in events if e[0] == 'ssh' and 'nohup' in e[1][-1])
        tokens = shlex.split(command)
        self.assertIn('MAYA_MODEL=Maya-M', tokens)
        self.assertIn(f'MAYA_MODELS_DIR={models}', tokens)


if __name__ == '__main__':
    unittest.main()
