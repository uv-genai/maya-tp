"""TP2 must reject local-only media work before encoding or changing request state."""
from pathlib import Path
import unittest
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

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
