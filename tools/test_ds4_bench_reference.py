"""Reference measurements save answers only when explicitly requested."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.ds4 import bench_tp2_reference as bench


class ReferenceTest(unittest.TestCase):
    def test_optional_answer_capture_and_existing_file_guard(self):
        def call(_url, path, payload=None, timeout=900):
            if path == '/metrics':
                return {'engine': {'model_label': 'test', 'model': 'ds4', 'dspark': True}, 'live': {}}
            return {'choices': [{'message': {'content': 'example answer'}, 'finish_reason': 'length'}],
                    'usage': {'completion_tokens': 2}, 'timings': {}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, 'call', side_effect=call):
            for save in (False, True):
                path = Path(tmp) / f'{save}.json'
                args = ['bench', '--url', 'http://unused', '--label', 'test', '--dspark', 'on',
                        '--output', str(path), '--runs', '1', '--tokens', '2']
                if save: args.append('--save-answers')
                with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                    bench.main()
                samples = json.loads(path.read_text())['samples']
                self.assertEqual(len(samples), 3)
                self.assertEqual(all(s.get('answer') == 'example answer' for s in samples), save)
                with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                    bench.main()


if __name__ == '__main__':
    unittest.main()
