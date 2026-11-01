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


class Tp2BenchmarkFramingTests(unittest.TestCase):
    def test_separate_streams_preserve_done_and_early_diagnostics(self):
        """Actual pipes reproduce stderr fragments and either reader scheduling order."""
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from unittest import mock
        from tools.tp2 import bench_mtp as bench

        real_popen, real_until = subprocess.Popen, bench.Pair.until
        program = r'''
import sys
print("READY fixture", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    print("T 7", flush=True)
    sys.stderr.write("glm mtp: a fragmented diagnostic on the whole h")
    sys.stderr.flush()
    done = "DONE 1 8 3.0 4.0 length 0 0 0 1 1"
    if sys.argv[1] == "after":
        print(done, flush=True)
    sys.stderr.write("ead\n")
    sys.stderr.write("glm stat: decode 4 ms/tok\n")
    sys.stderr.write("glm follow: 1 tokens of the continuation decoded, 0 of them differ from the model's own pick\n")
    sys.stderr.flush()
    if sys.argv[1] == "before":
        print(done, flush=True)
'''
        def bounded_until(pair, predicate, timeout, *args, **kwargs):
            return real_until(pair, predicate, min(timeout, 2), *args, **kwargs)
        for order in ("after", "before"):
            with self.subTest(footer=order), tempfile.TemporaryDirectory() as tmp:
                def spawn(_command, **kwargs):
                    return real_popen([sys.executable, "-u", "-c", program, order], **kwargs)
                with mock.patch.object(bench.subprocess, "Popen", side_effect=spawn), \
                        mock.patch.object(bench, "cleanup"), \
                        mock.patch.object(bench.Pair, "until", bounded_until):
                    pair = bench.Pair(Path(tmp), "fixture", 4096, follow="fixture", single_host=True)
                    try:
                        for _ in range(2):
                            result, ids = pair.request("simple", 1, follow=True)
                            self.assertEqual(ids, [7])
                            self.assertEqual(result["decode_ms"], 4)
                            self.assertEqual(bench.check_fidelity(result["diagnostics"]), 0)
                    finally:
                        pair.close()
                self.assertIn("DONE 1 8", (Path(tmp)/"fixture-rank0.log").read_text())
                self.assertIn("whole head\n", (Path(tmp)/"fixture-rank0-stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
