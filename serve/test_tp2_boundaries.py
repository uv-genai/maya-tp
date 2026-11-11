"""TP2 must reject local-only media work before encoding or changing request state."""
from pathlib import Path
import json
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request

from serve import glm_dsa_tune
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve, vision_config

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


class RequestTuningBoundary(unittest.TestCase):
    def test_every_new_knob_is_refused_before_single_or_batch_engine_write(self):
        for kind in ("glm-fast", "ds4"):
            for multi in (False, True):
                engine = StrataEngine.__new__(StrataEngine)
                engine.info = {"tp_world": 2, "engine_kind": kind, "tp_pair": "ready"}
                engine.multi = multi
                for knob in glm_dsa_tune.KEYS:
                    with self.subTest(kind=kind, multi=multi, knob=knob), \
                            mock.patch.object(engine, "_write_command") as write, \
                            mock.patch.object(engine, "_generate_multi") as batch:
                        with self.assertRaisesRegex(ValueError, "TP2.*unmirrored.*" + knob):
                            list(engine.generate([1, 2], 8, {"strata_tune": {knob: 0}}, threading.Event()))
                        write.assert_not_called()
                        batch.assert_not_called()
                # A poisoned pair always retains its original first fatal cause, even with a new unsupported knob.
                engine.info.update(tp_pair="down", tp_error="TP2 pair down: original peer loss")
                with self.assertRaisesRegex(ValueError, "original peer loss"):
                    list(engine.generate([1], 8, {"strata_tune": {"graphs": False}}, threading.Event()))

    def test_existing_spec_off_values_and_family_serialization_are_preserved(self):
        ds4 = StrataEngine.sampling_keys({"temperature": 0, "strata_tune": {"spec": 0, "mtp_keep": 2}}, "ds4")
        self.assertIn(" spec=0", ds4)
        self.assertNotIn("mtp_keep", ds4)
        glm = StrataEngine.sampling_keys({"temperature": 0, "strata_tune": {"spec": 1, "mtp_keep": 2}}, "glm-fast")
        self.assertIn(" spec=1", glm)
        self.assertIn(" mtp_keep=2", glm)
        for kind in ("qwen", "mm3"):
            self.assertNotIn("mtp_keep", StrataEngine.sampling_keys({"strata_tune": {"mtp_keep": 2}}, kind))

    def test_all_http_apis_refuse_tuning_before_render_or_generate(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok")
        engine.info = {"tp_world": 2, "engine_kind": "glm-fast"}
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        messages = [{"role": "user", "content": "hello"}]
        for path, body in (("/v1/chat/completions", {"messages": messages, "max_tokens": 8}),
                           ("/v1/messages", {"messages": messages, "max_tokens": 8}),
                           ("/v1/responses", {"input": "hello", "max_output_tokens": 8})):
            with self.subTest(path=path), mock.patch.object(svc, "render_ids") as render, \
                    mock.patch.object(engine, "generate") as generate:
                req = urllib.request.Request(base + path,
                    data=json.dumps({"model": "m", **body, "strata_tune": {"graphs": False}}).encode(),
                    headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    # Responses already refuses this extension at schema validation, before paired validation.
                    reason = ("unsupported Responses parameter: strata_tune" if path == "/v1/responses" else
                              "TP2 does not support unmirrored")
                    self.assertIn(reason, response.read().decode())
                render.assert_not_called()
                generate.assert_not_called()
