import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ds4 import tp2_identity, tp2_rank


class DeepSeekTP2RankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.pack = root / "pack"
        self.pack.mkdir()
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps({}))
        self.engine = root / "maya-ds4"
        self.engine.write_text("engine")
        self.identity = {"model_digest": "a" * 64, "partition_digest": "b" * 64, "build_digest": "c" * 64}
        self.base = ["--pack", str(self.pack), "--manifest", str(self.manifest), "--engine", str(self.engine),
                     "--host", "node.local", "--port", "18515", "--rdma-device", "usb4_rdma5",
                     "--context", "32768", "--expert-budget-mib", "90000"]

    def build(self, flags, environ=None):
        opts = tp2_rank.parser().parse_args([*flags, *self.base])
        with patch.object(tp2_rank, "prepare_identity", return_value=self.identity) as checked:
            args, env = tp2_rank.prepare_launch(opts, {} if environ is None else environ)
        checked.assert_called_once_with({}, str(self.pack), str(self.engine))
        return args, env

    def test_rank_commands_bind_the_same_pack_and_identity(self):
        root, a = self.build(["--serve", "--rank", "0"])
        worker, b = self.build(["--rank", "1"])
        self.assertIn("--serve", root)
        self.assertIn("--tp2-worker", worker)
        self.assertEqual(root[root.index("--dspark") + 1], "on")
        self.assertEqual(worker[worker.index("--dspark") + 1], "none")
        for name in ("PACK", "DSPARK", "CHUNK", "LOOKUP", "MODEL_DIGEST", "PARTITION_DIGEST", "BUILD_DIGEST"):
            self.assertEqual(a["MAYA_DS4_TP2_" + name], b["MAYA_DS4_TP2_" + name])
        self.assertEqual(a["DS4_STREAM"], "0")
        self.assertEqual(a["MAYA_DS4_TP2_CHUNK"], "0")
        self.assertEqual(a["MAYA_DS4_TP2_LOOKUP"], "0")

    def test_lookup_is_owned_by_rank_zero_and_bound_on_both_ranks(self):
        root, a = self.build(["--serve", "--rank", "0", "--suffix-draft", "3", "--dspark", "off"])
        worker, b = self.build(["--rank", "1", "--suffix-draft", "3", "--dspark", "off"])
        self.assertEqual(root[root.index("--suffix-draft") + 1], "3")
        self.assertEqual(worker[worker.index("--suffix-draft") + 1], "0")
        self.assertEqual(a["MAYA_DS4_TP2_LOOKUP"], "3")
        self.assertEqual(a["MAYA_DS4_TP2_LOOKUP"], b["MAYA_DS4_TP2_LOOKUP"])

    def test_chunk_prefill_binds_both_ranks_and_rejects_conflicting_environment(self):
        root, a = self.build(["--serve", "--rank", "0", "--prefill-chunk", "256"])
        worker, b = self.build(["--rank", "1", "--prefill-chunk", "256"])
        for args in (root, worker):
            self.assertEqual(args[args.index("--prefill-chunk") + 1], "256")
        for env in (a, b):
            self.assertEqual(env["MAYA_DS4_TP2_CHUNK"], "256")
            self.assertEqual(env["DS4_FLOW"], "0")
            self.assertEqual(env["DS4_YOCO"], "0")
        for conflict in ({"DS4_FLOW": "1"}, {"DS4_YOCO": "1"}, {"DS4_CHUNK": "256"}):
            with self.subTest(conflict=conflict), self.assertRaises(tp2_identity.IdentityError):
                self.build(["--serve", "--rank", "0", "--prefill-chunk", "256"], conflict)

    def test_invalid_rank_role_and_stream_are_rejected(self):
        for flags in (["--rank", "0"], ["--serve", "--rank", "1"]):
            with self.subTest(flags=flags), self.assertRaises(tp2_identity.IdentityError):
                self.build(flags)
        with self.assertRaisesRegex(tp2_identity.IdentityError, "DS4_STREAM"):
            self.build(["--serve", "--rank", "0"], {"DS4_STREAM": "1"})

    def test_drafter_can_be_disabled_for_both_ranks(self):
        root, a = self.build(["--serve", "--rank", "0", "--dspark", "off"])
        worker, b = self.build(["--rank", "1", "--dspark", "off"])
        self.assertEqual(root[root.index("--dspark") + 1], "none")
        self.assertEqual(worker[worker.index("--dspark") + 1], "none")
        self.assertEqual(a["MAYA_DS4_TP2_DSPARK"], "0")
        self.assertEqual(a["MAYA_DS4_TP2_DSPARK"], b["MAYA_DS4_TP2_DSPARK"])


if __name__ == "__main__":
    unittest.main()
