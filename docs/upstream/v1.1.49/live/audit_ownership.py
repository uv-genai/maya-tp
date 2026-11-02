"""Read only registered validation identities; never inspect unrelated processes."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

BASE = Path('/tmp/maya-v49-live-87')
PREFIXES = ('Maya-', 'peer-candidate-', 'peer-control-', 'cli-fixed', 'strict-', 'prefill-')


def audit():
    import owned
    out = []
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(PREFIXES):
            continue
        for path in sorted(folder.rglob('*.json')):
            try:
                data = json.loads(path.read_text())
            except ValueError:
                continue
            if not isinstance(data, dict) or not all(k in data for k in ('owner', 'root', 'processes')):
                continue
            alive = [x for x in data['processes'] if owned.valid(x, data['owner'])]
            assert not alive, (str(path), alive)
            assert data.get('stopped') and data.get('alive') == [], (str(path), data)
            out.append({'path': str(path.relative_to(BASE)), 'root': data['root'],
                        'recorded_identities': len(data['processes']), 'alive': [], 'stopped': True})
    return out


if __name__ == '__main__':
    local = audit()
    if '--remote' in sys.argv:
        print(json.dumps(local))
    else:
        command = shlex.join(['/home/ugo/software/project-maya/.venv/bin/python',
                              str(BASE/'audit_ownership.py'), '--remote'])
        peer = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'strix-halo-1.local', command], text=True))
        result = {'rank1': local, 'rank0': peer, 'all_registered_processes_ended': True}
        (BASE/'ownership-final.json').write_text(json.dumps(result, indent=2)+'\n')
        print('PASS', len(local), 'local and', len(peer), 'remote saved ownership records: all stopped/alive=[]')
