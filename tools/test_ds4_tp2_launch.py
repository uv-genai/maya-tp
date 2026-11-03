"""DeepSeek launcher configuration and process ownership regressions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.ds4 import tp2_launch
from tools.ds4 import tp2_rank

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {'model': 'a' * 64, 'partition': 'b' * 64, 'build': 'c' * 64}


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.state = self.directory / 'state'
        self.events = self.directory / 'events'
        self.config = self.directory / 'launch.json'
        self.settings = {
            'pack': 'models/XS', 'manifest': 'xs.manifest', 'engine': 'maya-ds4',
            'host': 'rank0.local', 'rdma_device': 'usb4_rdma5', 'context': 32768,
            'rank1': {'pack': '/remote/XS', 'manifest': '/remote/id.json', 'gpu': 1},
        }
        self.config.write_text(json.dumps(self.settings))
        self.bin = self.directory / 'bin'
        self.bin.mkdir()
        self.child = self.directory / 'child.py'
        self.child.write_text('''import os,signal,subprocess,sys,time
from pathlib import Path
name=sys.argv[1]
def record(event):
 with open(os.environ['EVENT_LOG'],'a') as f: f.write(event+'\\n')
def stop(*args):
 record(name+'_stopped');sys.exit(0)
signal.signal(signal.SIGTERM,stop)
record(name+'_started')
if name=='server' and os.environ.get('SERVER_CHILD'):
 subprocess.Popen([sys.executable,__file__,'engine'])
if os.environ.get('FAIL_CHILD')==name:
 time.sleep(.3);record(name+'_failed');sys.exit(7)
while True:time.sleep(.1)
''')
        self.runner = self.directory / 'runner.py'
        self.runner.write_text('''import os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
from tools.ds4 import tp2_launch
identity={'model':'a'*64,'partition':'b'*64,'build':'c'*64}
def api(settings,port,directory,output=None):
 return subprocess.Popen([sys.executable,os.environ['CHILD'],'server'],start_new_session=True)
with patch.object(tp2_launch,'prepare',return_value=(None,[],dict(os.environ),identity)), patch.object(tp2_launch,'start_api',side_effect=api):
 sys.exit(tp2_launch.main(sys.argv[1:]))
''')
        ssh = self.bin / 'ssh'
        ssh.write_text('#!' + sys.executable + '\n' + '''import json,os,sys
from unittest.mock import patch
from tools.ds4 import tp2_launch
settings=json.loads(sys.stdin.readline())
identity={'model':'a'*64,'partition':'b'*64,'build':'c'*64}
if os.environ.get('BAD_IDENTITY'):identity['build']='d'*64
command=[sys.executable,os.environ['CHILD'],'worker']
with patch.object(tp2_launch,'prepare',return_value=(None,command,dict(os.environ),identity)):
 sys.exit(tp2_launch.run_worker(settings,managed=True))
''')
        ssh.chmod(0o755)
        self.environment = dict(os.environ, PYTHONPATH=str(ROOT),
                                PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                                EVENT_LOG=str(self.events), CHILD=str(self.child))
        self.processes = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.communicate(timeout=5)

    def launch(self, action='run', extra_env=None, port='18090'):
        environment = dict(self.environment, **(extra_env or {}))
        process = subprocess.Popen(
            [sys.executable, str(self.runner), action, port, '--config', str(self.config),
             '--state-dir', str(self.state)],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.processes.append(process)
        return process

    def wait_for(self, predicate, process, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if process.poll() is not None:
                self.fail('launcher exited ' + repr(process.communicate(timeout=2)))
            time.sleep(.03)
        self.fail('launcher condition timed out')

    def recorded(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def wait_started(self, process):
        self.wait_for(lambda: {'server_started', 'worker_started'} <= set(self.recorded()), process)

    def assert_cleanup_order(self):
        events = self.recorded()
        self.assertLess(events.index('server_stopped'), events.index('worker_stopped'))
        self.assertEqual(events.count('server_started'), 1)
        self.assertEqual(events.count('worker_started'), 1)

    def test_settings_resolve_paths_and_keep_rank_choices(self):
        options = type('Options', (), {'config': str(self.config), 'context': 65536})()
        settings = tp2_launch.load_settings(options)
        local = tp2_launch.rank_settings(settings, 0)
        remote = tp2_launch.rank_settings(settings, 1)
        self.assertEqual(local['pack'], str(self.directory / 'models/XS'))
        self.assertEqual(remote['pack'], '/remote/XS')
        self.assertEqual(local['manifest'], str(self.directory / 'xs.manifest'))
        self.assertEqual((local['context'], remote['context']), (65536, 65536))
        self.assertEqual((local['gpu'], remote['gpu']), (0, 1))
        self.assertEqual((local['dspark'], remote['dspark']), ('on', 'on'))
        self.assertEqual((local['prefill_chunk'], remote['prefill_chunk']), (256, 256))

    def test_rank_cannot_override_shared_feature(self):
        self.settings['rank1']['context'] = 123
        self.config.write_text(json.dumps(self.settings))
        options = type('Options', (), {'config': str(self.config)})()
        with self.assertRaisesRegex(ValueError, 'overrides'):
            tp2_launch.load_settings(options)

    def test_stop_owns_only_pair_and_orders_cleanup(self):
        outsider = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        self.processes.append(outsider)
        process = self.launch()
        self.wait_started(process)
        stopped = self.launch('stop')
        self.assertEqual(stopped.wait(timeout=10), 0)
        self.assertEqual(process.wait(timeout=10), 0)
        self.assertIsNone(outsider.poll())
        self.assert_cleanup_order()
        self.assertFalse((self.state / '18090/control.sock').exists())

    def test_duplicate_run_and_fresh_restart(self):
        process = self.launch()
        self.wait_started(process)
        duplicate = self.launch()
        self.assertEqual(duplicate.wait(timeout=8), 1)
        self.assertIn('already owns', duplicate.communicate(timeout=2)[1])
        self.assertIsNone(process.poll())
        process.terminate()
        self.assertEqual(process.wait(timeout=10), 0)
        self.assert_cleanup_order()
        self.events.unlink()
        restarted = self.launch()
        self.wait_started(restarted)
        restarted.terminate()
        self.assertEqual(restarted.wait(timeout=10), 0)
        self.assert_cleanup_order()

    def test_peer_identity_mismatch_starts_no_engines(self):
        process = self.launch(extra_env={'BAD_IDENTITY': '1'})
        self.assertEqual(process.wait(timeout=8), 1)
        self.assertEqual(self.recorded(), [])
        self.assertIn('identity differs', process.communicate(timeout=2)[1])

    def test_same_bootstrap_pair_cannot_start_on_second_api_port(self):
        process = self.launch()
        self.wait_started(process)
        duplicate = self.launch(port='18091')
        self.assertEqual(duplicate.wait(timeout=8), 1)
        self.assertIsNone(process.poll())
        self.assertEqual(self.recorded().count('worker_started'), 1)

    def test_worker_failure_stops_server_without_restart(self):
        process = self.launch(extra_env={'FAIL_CHILD': 'worker'})
        self.assertEqual(process.wait(timeout=10), 1)
        events = self.recorded()
        self.assertIn('worker_failed', events)
        self.assertIn('server_stopped', events)
        self.assertEqual(events.count('worker_started'), 1)
        self.assertEqual(events.count('server_started'), 1)

    def test_server_failure_closes_remote_control_and_stops_worker(self):
        process = self.launch(extra_env={'FAIL_CHILD': 'server'})
        self.assertEqual(process.wait(timeout=10), 7)
        events = self.recorded()
        self.assertIn('server_failed', events)
        self.assertIn('worker_stopped', events)
        self.assertEqual(events.count('worker_started'), 1)

    def test_stop_missing_pair_does_not_create_state(self):
        self.assertEqual(tp2_launch.stop_pair(self.state / 'missing'), 0)
        self.assertFalse(self.state.exists())

    def test_stop_reaps_recorded_engine_after_server_exits(self):
        process = self.launch(extra_env={'SERVER_CHILD': '1'})
        self.wait_started(process)
        self.wait_for(lambda: 'engine_started' in self.recorded(), process)
        time.sleep(.25)
        stopped = self.launch('stop')
        self.assertEqual(stopped.wait(timeout=10), 0)
        self.assertEqual(process.wait(timeout=10), 0)
        events = self.recorded()
        self.assertLess(events.index('server_stopped'), events.index('engine_stopped'))
        self.assertLess(events.index('engine_stopped'), events.index('worker_stopped'))

    def test_model_selection_has_separate_config_files(self):
        (self.directory / 'maya-ds4-tp2-maya-xs.json').write_text(json.dumps(self.settings))
        options = type('Options', (), {'config': None, 'model': 'Maya-XS'})()
        with patch.object(tp2_launch, 'ROOT', self.directory):
            settings = tp2_launch.load_settings(options)
        self.assertEqual(settings['pack'], str(self.directory / 'models/XS'))
        options.model = '../../wrong'
        with self.assertRaises(ValueError):
            tp2_launch.load_settings(options)

    def test_llama_swap_pid_stop_checks_recorded_identity(self):
        process = self.launch()
        self.wait_started(process)
        tp2_launch.stop_pid(self.state, os.getpid())
        self.assertIsNone(process.poll())
        stopped = subprocess.run(
            [sys.executable, str(self.runner), 'stop-pid', str(process.pid), '--state-dir', str(self.state)],
            env=self.environment, capture_output=True, text=True, timeout=12,
        )
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertEqual(process.wait(timeout=10), 0)
        self.assert_cleanup_order()

    def test_api_generation_uses_checked_rank_config_and_parent_pipe(self):
        settings = tp2_launch.rank_settings(self.settings, 0)
        settings['manifest'] = str(self.directory / 'id.json')
        options = tp2_rank.parser().parse_args([
            '--serve', '--rank', '0', '--pack', '/model/XS', '--manifest', settings['manifest'],
            '--engine', '/review/maya-ds4', '--host', 'rank0.local', '--port', '18525',
            '--rdma-device', 'usb4_rdma5', '--context', '32768', '--expert-budget-mib', '92160',
            '--prefill-chunk', '256', '--dspark', 'on',
        ])
        with patch.object(tp2_launch, 'prepare', return_value=(options, [], {}, IDENTITY)), \
             patch.object(tp2_launch.subprocess, 'Popen') as popen:
            tp2_launch.start_api(settings, 18090, self.directory)
        command = popen.call_args.args[0]
        self.assertIn('--parent-pipe', command)
        self.assertEqual(command[command.index('--port') + 1], '18090')
        generated = json.loads((self.directory / 'server.json').read_text())
        parsed = tp2_rank.parser().parse_args(generated['args'])
        self.assertEqual((parsed.rank, parsed.pack, parsed.prefill_chunk, parsed.dspark),
                         (0, '/model/XS', 256, 'on'))

    def test_owned_child_exec_transition_and_reused_pid(self):
        process = SimpleNamespace(
            pid=12345, returncode=0, poll=lambda: 0, stdin=None,
            maya_children={23456: ('10', '/python')}, maya_child_executables={'/engine'},
        )
        with patch.object(tp2_launch, 'process_identity', side_effect=[('10', '/engine'), None]), \
             patch.object(tp2_launch.os, 'kill') as kill:
            tp2_launch.stop_process(process)
        kill.assert_called_once_with(23456, signal.SIGTERM)
        with patch.object(tp2_launch, 'process_identity', return_value=('11', '/engine')), \
             patch.object(tp2_launch.os, 'kill') as kill:
            tp2_launch.stop_process(process)
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
