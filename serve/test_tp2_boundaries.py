"""TP2 must reject local-only media work before encoding or changing request state."""
from pathlib import Path
import unittest
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, vision_config

ROOT = Path(__file__).resolve().parents[1]


class MediaBoundary(unittest.TestCase):
    def test_media_rejected_before_render_or_encoding_and_text_still_works(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok")
        engine.info = {"tp_world": "2"}
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.vision = mock.Mock()
        svc.embeddings.path = "existing-state"
        for kind in ("image", "video"):
            with self.subTest(kind=kind), mock.patch.object(svc, "render_ids") as render:
                with self.assertRaisesRegex(ValueError, "TP2.*RDMA"):
                    svc.prepare([{"role": "user", "content": [{"type": kind, "source": "/missing"}]}], [], {})
                render.assert_not_called()
                svc.vision.encode.assert_not_called()
                self.assertEqual(svc.embeddings.path, "existing-state")
        ids, _, n = svc.prepare([{"role": "user", "content": "hello"}], [], {}, max_new=8)
        self.assertTrue(ids)
        self.assertEqual(n, 8)


class EncoderStartupBoundary(unittest.TestCase):
    def test_last_world_option_controls_encoder_without_changing_config(self):
        import copy
        vision = {"exe": "/missing/must-not-start", "gpu": True}
        for args, paired in (([], False), (["--tp-world-size", "2"], True),
                             (["--tp-world-size", "1", "--tp-world-size", "2"], True),
                             (["--tp-world-size", "2", "--tp-world-size", "1"], False)):
            with self.subTest(args=args):
                cfg = {"args": args, "vision": vision}
                before = copy.deepcopy(cfg)
                self.assertIs(vision_config(cfg), None if paired else vision)
                self.assertEqual(cfg, before)
