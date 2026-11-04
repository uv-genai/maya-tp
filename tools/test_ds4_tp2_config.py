"""Dedicated DeepSeek TP2 server config retains the selected rank settings."""
import contextlib
import io
import json
from pathlib import Path
import unittest

from tools.ds4 import tp2_config
from tools.ds4 import tp2_rank


class ConfigTest(unittest.TestCase):
    def test_explicit_model_and_feature_settings(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(tp2_config.main([
                '--engine-log', 'tmp/engine.log', '--model-name', 'my-ds4', '--model-label', 'DeepSeek V4.1 · Maya-XS',
                '--pack', 'models/XS', '--manifest', 'tmp/id.json', '--engine', 'build-hip/maya-ds4',
                '--host', 'rank0.example', '--port', '17777', '--rdma-device', 'usb4_rdma5',
                '--context', '32768', '--expert-budget-mib', '90000', '--dspark', 'on',
                '--prefill-chunk', '256', '--suffix-draft', '3']), 0)
        config = json.loads(out.getvalue())
        args = config['args']
        def value(flag): return args[args.index(flag) + 1]
        self.assertEqual((value('--rank'), value('--prefill-chunk'), value('--suffix-draft')), ('0', '256', '3'))
        self.assertIn('--serve', args)
        self.assertEqual(value('--pack'), str(Path('models/XS').resolve()))
        self.assertEqual(config['tokenizer'], str(Path('models/XS/tokenizer').resolve()))
        self.assertEqual(config['log'], str(Path('tmp/engine.log').resolve()))
        self.assertEqual(config['model_name'], 'my-ds4')
        self.assertEqual(config['model_label'], 'DeepSeek V4.1 · Maya-XS')
        self.assertEqual(config['backend'], 'hip')
        parsed = tp2_rank.parser().parse_args(args)
        self.assertTrue(parsed.serve)
        self.assertEqual((parsed.rank, parsed.prefill_chunk, parsed.suffix_draft), (0, 256, 3))

    def test_cannot_override_rank_or_serve(self):
        for override in (['--rank', '1'], ['--rank=1'], ['--serve']):
            with self.subTest(override=override), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    tp2_config.main(['--engine-log', '/tmp/engine.log', *override])


if __name__ == '__main__':
    unittest.main()
