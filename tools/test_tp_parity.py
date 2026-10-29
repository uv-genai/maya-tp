"""tools/test_tp_parity.py - unit tests for the greedy token-dump comparator (no GPU)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tp2"))
import greedy_parity  # noqa: E402


class TestLoad(unittest.TestCase):
    def test_commas_and_whitespace(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.txt"
            p.write_text("1,2 3\n4, 5")
            self.assertEqual(greedy_parity.load_ids(p), [1, 2, 3, 4, 5])

    def test_skips_junk(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.txt"
            p.write_text("1 x 2")
            self.assertEqual(greedy_parity.load_ids(p), [1, 2])


class TestCompare(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(greedy_parity.compare([1, 2, 3], [1, 2, 3], 1.0, 5), 0)

    def test_one_flip(self):
        ids = list(range(100))
        b = ids[:]
        b[50] = 999
        self.assertEqual(greedy_parity.compare(ids, b, 0.95, 5), 0)

    def test_too_many_flips(self):
        a = list(range(10))
        b = list(range(10, 20))
        self.assertEqual(greedy_parity.compare(a, b, 0.95, 5), 1)

    def test_empty(self):
        self.assertEqual(greedy_parity.compare([], [], 0.95, 5), 1)


if __name__ == "__main__":
    unittest.main()
