"""The first fatal RDMA error must make TP2 health sticky-down, before another GEN."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, serve

ROOT = Path(__file__).resolve().parents[1]
POISON = 'verbs communicator is poisoned: verbs peer closed the bootstrap connection'


class PeerLoss(unittest.TestCase):
    def engine(self, world=2, multi=0, error=POISON, down_cause=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        seen = folder / 'requests.txt'
        script = folder / 'engine.py'
        script.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
print('INFO engine_kind=glm-fast tp_world={world} tp_rank=0', flush=True)
print('READY 4096 stop' + (' multi={multi}' if {multi} else ''), flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith('GEN '):
        with Path({str(seen)!r}).open('a') as out:
            out.write(line)
        rid = line.split('rid=', 1)[1].split()[0] + ' ' if 'rid=' in line else ''
        if {down_cause!r} is not None:
            print('TP2_DOWN ' + {down_cause!r}, flush=True)
        print('ERR ' + rid + {error!r}, flush=True)
''')
        script.chmod(0o755)
        engine = StrataEngine(str(script), [])
        def close_engine():
            engine.close()
            engine.proc.wait(timeout=5)
            engine.proc.stdin.close()
            engine.proc.stdout.close()
        self.addCleanup(close_engine)
        return engine, seen

    @staticmethod
    def generate(engine):
        return list(engine.generate([1, 2], 8, {}, threading.Event()))

    def test_first_poisoned_error_is_sticky_and_later_request_never_reaches_engine(self):
        for multi in (0, 2):
            with self.subTest(multi=multi):
                engine, seen = self.engine(multi=multi)
                with self.assertRaisesRegex(ValueError, 'TP2 pair down.*verbs communicator is poisoned'):
                    self.generate(engine)
                self.assertEqual(engine.info.get('tp_pair'), 'down')
                with self.assertRaisesRegex(ValueError, 'TP2 pair down.*restart both ranks'):
                    self.generate(engine)
                self.assertIn(POISON, engine.info['tp_error'])
                self.assertEqual(len(seen.read_text().splitlines()), 1)
                self.assertIsNone(engine.proc.poll())

    def test_first_pair_down_cause_survives_a_later_engine_error(self):
        for multi in (0, 2):
            with self.subTest(multi=multi):
                engine, seen = self.engine(multi=multi, down_cause=POISON,
                                           error='TP2 pair down: later shutdown error')
                with self.assertRaisesRegex(ValueError, 'TP2 pair down.*verbs communicator is poisoned'):
                    self.generate(engine)
                with self.assertRaisesRegex(ValueError, 'TP2 pair down.*verbs communicator is poisoned'):
                    self.generate(engine)
                self.assertNotIn('later shutdown error', engine.info['tp_error'])
                self.assertEqual(len(seen.read_text().splitlines()), 1)

    def test_unpaired_error_and_unrelated_tp_error_keep_their_existing_behavior(self):
        for world, error in ((1, POISON), (2, 'out of GPU memory')):
            with self.subTest(world=world, error=error):
                engine, _ = self.engine(world=world, error=error)
                with self.assertRaisesRegex(ValueError, error):
                    self.generate(engine)
                self.assertNotEqual(engine.info.get('tp_pair'), 'down')

    def test_health_is_degraded_immediately_after_first_http_failure(self):
        engine, seen = self.engine()
        service = Service(engine, ByteTokenizer(), ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        httpd = serve(service, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        body = {'messages': [{'role': 'user', 'content': 'Hello.'}], 'max_tokens': 8}
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as failed:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(failed.exception.code, 400)
        with failed.exception as response:
            self.assertIn('TP2 pair down', json.load(response)['error']['message'])
        with urllib.request.urlopen(base + '/health', timeout=10) as response:
            health = json.load(response)
        self.assertEqual((health['status'], health['tp']['peer']), ('degraded', 'down'))
        self.assertEqual(len(seen.read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
