"""The benchmark must reject incorrect raw picks even when output IDs are forced."""
import unittest

from tools.tp2.bench_mtp import check_fidelity


def diagnostic(count):
    return f"glm follow: 256 tokens of the continuation decoded, {count} of them differ from the model's own pick"


class Tp2BenchmarkFidelityTests(unittest.TestCase):
    def test_matching_raw_picks(self):
        self.assertEqual(check_fidelity(["glm stat: decode 38.9 ms/tok", diagnostic(0)]), 0)

    def test_old_verifier_disagreement_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "55 raw model picks"):
            check_fidelity([diagnostic(55)])

    def test_numeric_tolerance_must_be_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "maximum 0"):
            check_fidelity([diagnostic(1)])
        self.assertEqual(check_fidelity([diagnostic(1)], maximum=1), 1)

    def test_missing_or_ambiguous_diagnostic_is_rejected(self):
        for lines in ([], ["glm stat: decode 38.9 ms/tok"], [diagnostic(0), diagnostic(0)]):
            with self.subTest(lines=lines), self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
                check_fidelity(lines)


if __name__ == "__main__":
    unittest.main()
