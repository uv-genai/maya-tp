"""tools/test_tp_partition.py - unit tests for the two-host TP split planner and pack auditor (no GPU)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tp2"))
import audit_split  # noqa: E402


class TestShare(unittest.TestCase):
    def test_world1(self):
        self.assertEqual(audit_split.share(2048, 0, 1), (0, 2048))

    def test_halves(self):
        self.assertEqual(audit_split.share(2048, 0, 2), (0, 1024))
        self.assertEqual(audit_split.share(2048, 1, 2), (1024, 1024))

    def test_refuses_remainder(self):
        with self.assertRaises(ValueError):
            audit_split.share(2047, 0, 2)


class TestCheckSplit(unittest.TestCase):
    def test_green(self):
        self.assertEqual(audit_split.check_split("x", 2048, 256, 2, "ff"), [])

    def test_block(self):
        bad = audit_split.check_split("x", 1024, 256, 8, "ff")
        self.assertEqual(len(bad), 1)
        self.assertIn("128", bad[0])

    def test_world(self):
        bad = audit_split.check_split("x", 2047, 256, 2, "ff")
        self.assertEqual(len(bad), 1)


class TestIndexRow(unittest.TestCase):
    def test_output_weight(self):
        row = "output.weight 0 0 0 0 0 0 4096 154880 8 0 32 0 0 0 0 0 0 0".split()
        self.assertEqual(audit_split.parse_index_row(row), ("output.weight", 4096, 154880, 8))

    def test_short(self):
        self.assertIsNone(audit_split.parse_index_row("a b".split()))

    def test_bad_int(self):
        self.assertIsNone(audit_split.parse_index_row(["n", "x", "x", "x", "x", "x", "x", "x", "x", "x"]))


class TestAuditPack(unittest.TestCase):
    def _write(self, d: Path, native: str, index: str) -> Path:
        (d / "native_experts.txt").write_text(native)
        (d / "index.txt").write_text(index)
        return d

    def test_green_synthetic(self):
        # gu_row = 4096/256*34 = 544; d_row IQ3_XXS = 2048/256*162 = 1296
        # blob = 2*2048*544 + 4096*1296 = 7536640
        native = (
            "# hdr\n"
            "3 16 18 0 7536640 0 0 0\n"
        )
        index = (
            "# hdr\n"
            "blk.0.ffn_gate.weight 0 0 0 0 0 0 4096 12288 14 0 0 0 0 0 0 0 0 0\n"
            "blk.0.ffn_down.weight 0 0 0 0 0 0 12288 4096 14 0 0 0 0 0 0 0 0 0\n"
            "blk.3.ffn_gate.weight 0 0 0 0 0 0 4096 2048 16 0 0 0 0 0 0 0 0 0\n"
            "blk.3.ffn_down.weight 0 0 0 0 0 0 2048 4096 18 0 0 0 0 0 0 0 0 0\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = self._write(Path(td), native, index)
            self.assertEqual(audit_split.audit_pack(p, 2, 4096, 2048, 12288), 0)

    def test_bad_ff(self):
        native = "3 16 18 0 7536640 0 0 0\n"
        index = "blk.3.ffn_down.weight 0 0 0 0 0 0 2048 4096 18 0 0 0 0 0 0 0 0 0\n"
        with tempfile.TemporaryDirectory() as td:
            p = self._write(Path(td), native, index)
            self.assertEqual(audit_split.audit_pack(p, 2, 4096, 2047, 12288), 1)

    def test_missing_native(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "index.txt").write_text("#\n")
            self.assertEqual(audit_split.audit_pack(p, 2, 4096, 2048, 12288), 1)

    def test_blob_size_mismatch(self):
        native = "3 16 18 0 123 0 0 0\n"
        with tempfile.TemporaryDirectory() as td:
            p = self._write(Path(td), native, "#\n")
            self.assertEqual(audit_split.audit_pack(p, 2, 4096, 2048, 12288), 1)


if __name__ == "__main__":
    unittest.main()
