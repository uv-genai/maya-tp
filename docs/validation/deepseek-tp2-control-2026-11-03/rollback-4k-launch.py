import json
import os
import sys
from pathlib import Path

rank, pack, manifest, binary, port = sys.argv[1:]
identity = __import__('tools.ds4.tp2_identity', fromlist=['prepare_identity']).prepare_identity(
    json.loads(Path(manifest).read_text()), pack, binary)
for kind in ('model', 'partition', 'build'):
    os.environ['DS4_TP_TEST_' + kind.upper() + '_DIGEST'] = identity[kind + '_digest']
os.environ['DS4_TP_TEST_CHUNK'] = '256'
os.environ['DS4_TP_TEST_PROMPT_LEN'] = '4095'
os.environ['DS4_TP_TEST_VERIFY_ROWS'] = '16'
os.environ['MAYA_TP_BOOTSTRAP_TIMEOUT'] = '900'
os.execv(binary, [binary, '--sharded-rollback-dspark', rank, '192.168.1.92', port,
                  'usb4_rdma5', pack])
