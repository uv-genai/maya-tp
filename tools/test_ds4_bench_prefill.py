"""Fresh-prefill benchmarks support explicit long lengths and distinct inputs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.ds4 import bench_tp2_prefill as bench


class FakeTokenizer:
    def encode(self, _text):
        return list(range(10000))

    def decode(self, ids):
        return ','.join(map(str, ids))


class PrefillTest(unittest.TestCase):
    def test_explicit_lengths_and_fresh_offsets(self):
        prompts = []
        def call(_url, path, payload=None, timeout=900):
            if path == '/metrics':
                return {'engine': {'model_label': 'test', 'model': 'ds4', 'dspark': True}, 'live': {}}
            prompts.append((payload['messages'][0]['content'], timeout, payload['max_tokens']))
            return {'usage': {}, 'timings': {}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bench.Tokenizer, 'from_dir', return_value=FakeTokenizer()), \
                patch.object(bench, 'call', side_effect=call):
            out = Path(tmp) / 'result.json'
            args = ['bench', '--url', 'http://unused', '--label', 'test', '--dspark', 'on',
                    '--tokenizer', tmp, '--output', str(out), '--runs', '1',
                    '--source-tokens', '3', '--source-tokens', '5', '--timeout', '1200', '--output-tokens', '32']
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                bench.main()
            result = json.loads(out.read_text())
            self.assertEqual((result['source_lengths'], result['max_tokens'], result['timeout_s']), ([3, 5], 32, 1200))
            samples = result['samples']
            self.assertEqual([s['source_tokens'] for s in samples], [3, 5])
            self.assertEqual(prompts, [('0,1,2', 1200, 32), ('1027,1028,1029,1030,1031', 1200, 32)])
            with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                bench.main()


if __name__ == '__main__':
    unittest.main()
