"""The first fatal RDMA error must make TP2 health sticky-down, before another GEN."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, serve

ROOT = Path(__file__).resolve().parents[1]
POISON = 'verbs communicator is poisoned: verbs peer closed the bootstrap connection'


class PeerLoss(unittest.TestCase):
    def engine(self, world=2, multi=0, error=POISON, down_cause=None, end_cause=None,
               fail_after=1, notice_only=False, eof=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        seen = folder / 'requests.txt'
        script = folder / 'engine.py'
        script.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
print('INFO engine_kind=glm-fast tp_world={world} tp_rank=0 tp_prefill_chunk=256', flush=True)
print('READY 4096 stop' + (' multi={multi}' if {multi} else ''), flush=True)
requests = 0
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith('GEN '):
        with Path({str(seen)!r}).open('a') as out:
            out.write(line)
        requests += 1
        if requests < {fail_after}:
            continue
        if {eof!r}:
            break
        rid = line.split('rid=', 1)[1].split()[0] + ' ' if 'rid=' in line else ''
        if {down_cause!r} is not None:
            print('TP2_DOWN ' + {down_cause!r}, flush=True)
        if {end_cause!r} is not None:
            print('END ' + {end_cause!r}, flush=True)
        if not {notice_only!r}:
            print('ERR ' + rid + {error!r}, flush=True)
''')
        script.chmod(0o755)
        engine = StrataEngine(str(script), [])
        def close_engine():
            engine.close()
            engine.proc.wait(timeout=5)
            try:
                engine.proc.stdin.close()
            except BrokenPipeError:
                pass
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

    def test_raw_peer_close_is_sticky_even_before_the_pair_down_notice(self):
        engine, seen = self.engine(error='verbs peer closed the bootstrap connection')
        with self.assertRaisesRegex(ValueError, 'TP2 pair down.*verbs peer closed the bootstrap connection'):
            self.generate(engine)
        self.assertEqual(engine.info.get('tp_pair'), 'down')
        with self.assertRaisesRegex(ValueError, 'TP2 pair down.*verbs peer closed the bootstrap connection'):
            self.generate(engine)
        self.assertEqual(len(seen.read_text().splitlines()), 1)

    def test_every_active_route_wakes_on_fatal_notices_or_unexpected_eof(self):
        cases = ({'down_cause': POISON, 'notice_only': True},
                 {'end_cause': 'GPU state unknown', 'notice_only': True},
                 {'error': POISON}, {'eof': True})
        for options in cases:
            with self.subTest(options=options):
                engine, seen = self.engine(multi=3, fail_after=3, **options)
                errors = []

                def request():
                    try:
                        self.generate(engine)
                    except Exception as error:
                        errors.append(error)

                threads = [threading.Thread(target=request, daemon=True) for _ in range(3)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(3)
                self.assertEqual(len(errors), 3)
                self.assertTrue(all(isinstance(error, ValueError) for error in errors), errors)
                self.assertTrue(all(str(error) == engine.info['tp_error'] for error in errors))
                self.assertEqual(engine._routes, {})
                self.assertEqual(len(seen.read_text().splitlines()), 3)
                with self.assertRaisesRegex(ValueError, 'restart both ranks'):
                    self.generate(engine)
                self.assertEqual(len(seen.read_text().splitlines()), 3)

    def test_failure_while_waiting_for_writer_blocks_gen_and_releases_its_route(self):
        engine, seen = self.engine(multi=2)
        errors = []

        def request():
            try:
                self.generate(engine)
            except Exception as error:
                errors.append(error)

        with engine._write_lock:
            thread = threading.Thread(target=request, daemon=True)
            thread.start()
            deadline = time.monotonic() + 2
            while not engine._routes and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(len(engine._routes), 1)
            engine._tp_pair_down(POISON)
        thread.join(3)
        self.assertEqual(len(errors), 1)
        self.assertIn(POISON, str(errors[0]))
        self.assertEqual(engine._routes, {})
        self.assertFalse(seen.exists())

    def test_gpu_fault_end_requires_both_ranks_and_retains_the_first_cause(self):
        for multi in (0, 2):
            for prior in (None, POISON):
                with self.subTest(multi=multi, prior=prior):
                    cause = "a GPU's state is unknown after this request"
                    engine, seen = self.engine(multi=multi, error='HIP kernel failed',
                                               down_cause=prior, end_cause=cause)
                    with self.assertRaisesRegex(ValueError, 'TP2 pair down.*restart both ranks'):
                        self.generate(engine)
                    deadline = time.monotonic() + 2
                    while not engine.ended and time.monotonic() < deadline:
                        time.sleep(0.001)
                    self.assertTrue(engine.ended)
                    self.assertEqual(engine.info.get('tp_pair'), 'down')
                    self.assertIn(prior or cause, engine.info['tp_error'])
                    with self.assertRaisesRegex(ValueError, 'TP2 pair down.*restart both ranks'):
                        self.generate(engine)
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
        self.assertEqual(health['tp']['prefill_chunk'], 256)
        self.assertIn(POISON, health['tp']['error'])
        self.assertEqual(len(seen.read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
