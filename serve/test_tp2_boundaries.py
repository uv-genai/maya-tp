"""TP2 must reject local-only media work before encoding or changing request state."""
from pathlib import Path
import copy
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request

from serve import glm_dsa_tune, models
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
    def test_ablation_is_refused_before_single_or_batch_engine_write(self):
        for kind in ("glm-fast", "ds4"):
            for multi in (False, True):
                engine = StrataEngine.__new__(StrataEngine)
                engine.info = {"tp_world": 2, "engine_kind": kind, "ablate": 1}
                engine.multi = multi
                for value in (True, False):
                    with self.subTest(kind=kind, multi=multi, value=value), \
                            mock.patch.object(engine, "_write_command") as write, \
                            mock.patch.object(engine, "_generate_multi") as batch:
                        with self.assertRaisesRegex(ValueError, "TP2.*ablation"):
                            list(engine.generate([1, 2], 8, {"ablate": value}, threading.Event()))
                        write.assert_not_called()
                        batch.assert_not_called()
                tok = ByteTokenizer()
                svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
                self.assertEqual(svc.ablate_state()["available"], False)
                self.assertEqual(svc.ablate_state()["default"], False)

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

    def test_ablation_is_refused_by_all_http_apis_for_both_families(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok")
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        messages = [{"role": "user", "content": "hello"}]
        for kind in ("glm-fast", "ds4"):
            engine.info = {"tp_world": 2, "engine_kind": kind, "ablate": 1}
            for value in (True, False):
                for path, body in (("/v1/chat/completions", {"messages": messages, "max_tokens": 8}),
                                   ("/v1/messages", {"messages": messages, "max_tokens": 8}),
                                   ("/v1/responses", {"input": "hello", "max_output_tokens": 8})):
                    with self.subTest(kind=kind, value=value, path=path), \
                            mock.patch.object(svc, "render_ids") as render, \
                            mock.patch.object(engine, "generate") as generate:
                        req = urllib.request.Request(base + path,
                            data=json.dumps({"model": "m", **body, "ablate": value}).encode(),
                            headers={"Content-Type": "application/json"})
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            urllib.request.urlopen(req, timeout=10)
                        with caught.exception as response:
                            self.assertEqual(response.code, 400)
                            self.assertIn("TP2 does not support unmirrored refusal ablation", response.read().decode())
                        render.assert_not_called()
                        generate.assert_not_called()


class AblationConfigBoundary(unittest.TestCase):
    def test_pair_detection_matches_both_launchers_and_last_repeated_glm_option(self):
        for cfg, paired in (({"args": []}, False),
                            ({"args": ["--tp-world-size", "2", "--tp-world-size", "1"]}, False),
                            ({"args": ["--tp-world-size", "1", "--tp-world-size=2"]}, True),
                            ({"args": ["--tp2-rank", "0"]}, True),
                            ({"args": ["--tp2-rank=1"]}, True),
                            ({"exe": "/review/tools/ds4/tp2_rank.py", "args": ["--rank", "0"]}, True)):
            with self.subTest(cfg=cfg):
                self.assertEqual(models.is_tp2(cfg), paired)

    def test_no_capability_or_config_injection_for_either_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            direction = Path(directory) / "direction.gguf"
            direction.write_bytes(b"GGUF")
            for exe, args in (("maya", ["--tp-world-size", "2"]),
                              ("maya-ds4", ["--tp2-rank", "0"]),
                              ("/review/tools/ds4/tp2_rank.py", ["--rank", "0"])):
                base = {"exe": exe, "args": args, "backend": "hip"}
                models.check_tp2_ablation_config(base)
                self.assertEqual(models.ablate_args(base, args), args)
                for value in (True, False):
                    with self.assertRaisesRegex(ValueError, "TP2 refusal ablation"):
                        models.apply(base, {"ablate": value})
                for choice in ({"ablate": {"file": str(direction)}}, {"ablate": {"default": True}},
                               {"sampling": {"ablate": False}},
                               {"args": args + ["--ablate", str(direction)]},
                               {"args": args + ["--ablate-default=off"]}):
                    cfg = {**base, **choice}
                    before = copy.deepcopy(cfg)
                    with self.subTest(exe=exe, choice=choice):
                        self.assertFalse(models.ablate_choice(cfg)["available"])
                        self.assertIn("TP2", models.ablate_choice(cfg)["why"])
                        with self.assertRaisesRegex(ValueError, "TP2 refusal ablation"):
                            models.check_tp2_ablation_config(cfg)
                        with self.assertRaisesRegex(ValueError, "TP2 refusal ablation"):
                            models.ablate_args(cfg, cfg["args"])
                        self.assertEqual(cfg, before)

    def test_server_refuses_config_before_tokenizer_or_engine_start(self):
        from serve import server
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maya-paired.json"
            path.write_text(json.dumps({"exe": "/must-not-start", "args": ["--tp-world-size", "2"],
                                        "ablate": {"file": "/must-not-read.gguf"}}))
            errors = io.StringIO()
            with mock.patch.dict(os.environ), mock.patch.object(sys, "argv", ["server", "--engine", "maya",
                    "--config", str(path)]), mock.patch.object(sys, "stderr", errors), \
                    mock.patch.object(server, "StrataEngine") as start:
                with self.assertRaises(SystemExit) as stopped:
                    server.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("TP2 refusal ablation", errors.getvalue())
                start.assert_not_called()
