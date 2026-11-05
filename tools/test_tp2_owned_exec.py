"""EOF cleanup reaps its recorded child while leaving unrelated jobs alive."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]

class OwnedExecTests(unittest.TestCase):
    def test_eof_stops_only_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity=Path(tmp)/'identity.json'
            sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
            process=subprocess.Popen([sys.executable,str(ROOT/'tools/tp2/owned_exec.py'),'--identity',str(identity),
                                      '--',sys.executable,'-c','import time;time.sleep(60)'],stdin=subprocess.PIPE)
            try:
                deadline=time.monotonic()+5
                while not identity.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(identity.exists())
                child=json.loads(identity.read_text())['pid']
                process.stdin.close();process.wait(timeout=10)
                self.assertIsNone(sentinel.poll())
                self.assertFalse(Path('/proc',str(child)).exists())
            finally:
                if process.poll() is None:process.kill();process.wait()
                sentinel.terminate();sentinel.wait()

if __name__=='__main__':unittest.main()
