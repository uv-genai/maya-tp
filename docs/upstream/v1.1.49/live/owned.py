"""Start, inspect and stop only this validation's registered processes (Linux)."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

KEY = b'MAYA_VALIDATION_OWNER='


def identity(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'start': int(data[19]), 'state': data[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def valid(item, owner):
    now = identity(item['pid'])
    if not now or now['start'] != item['start'] or now['state'] == 'Z':
        return False
    try:
        return KEY + owner.encode() in Path(f"/proc/{item['pid']}/environ").read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        return False


def save(path, record):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def refresh(record):
    items = {x['pid']: x for x in record['processes']}
    pending = list(items.values())
    while pending:
        item = pending.pop()
        if not valid(item, record['owner']):
            continue
        pid = item['pid']
        try:
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for child in children:
            kid = identity(int(child))
            if kid and kid['pid'] not in items and valid(kid, record['owner']):
                items[kid['pid']] = kid
                pending.append(kid)
    record['processes'] = list(items.values())
    record['alive'] = [x for x in items.values() if valid(x, record['owner'])]
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['start', 'exec', 'status', 'stop', 'kill', 'ports'])
    ap.add_argument('registry', type=Path)
    ap.add_argument('owner')
    ap.add_argument('argv', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.action == 'ports':
        for port in a.argv:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', int(port)))
        print('ports available')
        return
    if a.action in ('start', 'exec'):
        assert not a.registry.exists(), f'registry already exists: {a.registry}'
        a.registry.parent.mkdir(parents=True, exist_ok=True)
        if a.action == 'exec':
            env = {k: v for k, v in os.environ.items() if not k.startswith(('MAYA_', 'STRATA_'))}
            env['MAYA_VALIDATION_OWNER'] = a.owner
            pid = os.getpid()
            record = {'owner': a.owner, 'root': pid, 'processes': [identity(pid)], 'argv': a.argv}
            save(a.registry, record)
            os.execvpe(a.argv[0], a.argv, env)
        spec = json.loads(Path(a.argv[0]).read_text())
        env = {k: v for k, v in os.environ.items() if not k.startswith(('MAYA_', 'STRATA_'))}
        env.update(spec.get('env', {}), MAYA_VALIDATION_OWNER=a.owner)
        with Path(spec['log']).open('w') as log:
            p = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'owner': a.owner, 'root': p.pid, 'processes': [identity(p.pid)], 'argv': spec['argv']}
    else:
        record = json.loads(a.registry.read_text())
        assert record['owner'] == a.owner
    refresh(record)
    if a.action in ('stop', 'kill'):
        root = next(x for x in record['processes'] if x['pid'] == record['root'])
        if valid(root, a.owner):
            os.kill(root['pid'], signal.SIGKILL if a.action == 'kill' else signal.SIGTERM)
        end = time.monotonic() + (0 if a.action == 'kill' else 60)
        while time.monotonic() < end and refresh(record)['alive']:
            time.sleep(.2)
        for item in refresh(record)['alive']:
            if valid(item, a.owner):
                os.kill(item['pid'], signal.SIGKILL)
        for _ in range(100):
            if not refresh(record)['alive']:
                break
            time.sleep(.1)
        assert not record['alive'], record
        record['stopped'] = True
    save(a.registry, record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
