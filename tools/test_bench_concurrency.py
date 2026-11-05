"""Concurrency benchmark accounting and streamed-response regressions."""
import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from tools import bench_concurrency as bench


class ConcurrencyBenchmarkTest(unittest.TestCase):
    def stream(self, chunks):
        response = mock.MagicMock()
        response.__enter__.return_value = iter(chunks)
        return response

    def test_stream_uses_content_for_ttft_and_keeps_usage(self):
        chunks = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                  b': keepalive\n', b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
                  b'data: {"choices":[{"delta":{},"finish_reason":"length"}],'
                  b'"usage":{"completion_tokens":256},"timings":{"predicted_per_second":32}}\n',
                  b'data: [DONE]\n']
        with mock.patch.object(bench.urllib.request, 'urlopen', return_value=self.stream(chunks)) as opened:
            with mock.patch.object(bench.time, 'perf_counter', side_effect=[10, 13, 18]):
                result = bench.request('http://localhost/v1', 'chosen-model', 'prompt', 'id', 256, 30)
        self.assertIsNone(result['error'])
        self.assertEqual(result['ttft_s'], 3)
        self.assertEqual(result['elapsed_s'], 8)
        self.assertEqual(result['content'], 'hello')
        self.assertEqual(result['usage']['completion_tokens'], 256)
        payload = json.loads(opened.call_args.args[0].data)
        self.assertEqual(payload['model'], 'chosen-model')
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})

    def test_incomplete_and_error_streams_fail(self):
        for chunks in ([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'],
                       [b'data: {"error":{"message":"pair down"}}\n']):
            with self.subTest(chunks=chunks):
                with mock.patch.object(bench.urllib.request, 'urlopen', return_value=self.stream(chunks)):
                    result = bench.request('http://localhost/v1', 'model', 'prompt', 'id', 256, 30)
                self.assertIsNotNone(result['error'])

    def test_wave_releases_all_clients(self):
        entered = threading.Barrier(4)

        def complete(*args):
            entered.wait(timeout=5)
            return {'request_id': args[3], 'error': None}

        options = SimpleNamespace(url='http://localhost/v1', model='model', max_tokens=256,
                                  timeout=30, api_key=None)
        with mock.patch.object(bench, 'request', side_effect=complete):
            wave = bench.wave(options, 'prompt', 'run', 4, 1)
        self.assertEqual(len(wave['requests']), 4)
        self.assertEqual(len({item['request_id'] for item in wave['requests']}), 4)

    def test_throughput_uses_entire_wave_duration(self):
        result = {'error': None, 'usage': {'completion_tokens': 256},
                  'timings': {'predicted_per_second': 40}, 'ttft_s': 2, 'elapsed_s': 10}
        waves = [{'concurrency': 2, 'elapsed_s': 10, 'requests': [result, result]},
                 {'concurrency': 2, 'elapsed_s': 30, 'requests': [result, result]}]
        summary = bench.summarize(waves)[0]
        self.assertEqual(summary['requests'], 4)
        self.assertEqual(summary['aggregate_tok_s'], 1024 / 40)
        self.assertEqual(summary['median_server_decode_tok_s'], 40)

    def test_replay_overrides_prompt_but_keeps_stream_contract(self):
        chunks=[b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}],"usage":{"completion_tokens":1}}\n',b'data: [DONE]\n']
        payload={'messages':[{'role':'user','content':'fixed'}],'strata_tune':{'spec':1},'stream':False}
        with mock.patch.object(bench.urllib.request,'urlopen',return_value=self.stream(chunks)) as opened:
            result=bench.request('http://local/v1','model','ignored','id',1,10,payload=payload)
        self.assertIsNone(result['error'])
        sent=json.loads(opened.call_args.args[0].data)
        self.assertEqual(sent['messages'],payload['messages'])
        self.assertEqual(sent['strata_tune'],{'spec':1})
        self.assertTrue(sent['stream'])

    def test_replenished_clients_keep_replay_order_and_stagger(self):
        options=SimpleNamespace(url='url',model='model',max_tokens=8,timeout=30,api_key=None,
                                stagger=.01,requests_per_client=2,requests=[{'messages':[{'content':str(i)}]} for i in range(4)])
        def complete(*args):return {'request_id':args[3],'payload':args[7],'error':None}
        with mock.patch.object(bench,'request',side_effect=complete), mock.patch.object(bench.time,'sleep') as sleep:
            wave=bench.wave(options,'ignored','fixed',2,1)
        self.assertEqual(len(wave['requests']),4)
        self.assertEqual([r['payload'] for r in wave['requests']],options.requests)
        self.assertEqual(len({r['request_id'] for r in wave['requests']}),4)
        sleep.assert_any_call(.01)


if __name__ == '__main__':
    unittest.main()
