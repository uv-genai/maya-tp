"""Host-only row/batch bounds; no HIP SDK or GPU needed."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TpCollectiveBoundsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_receive_window_and_user_payload_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = str(Path(directory) / 'tp_collective_test')
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'include'), str(ROOT / 'tests/tp_collective_test.cpp'),
                            '-o', exe], check=True, timeout=60)
            subprocess.run([exe], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
