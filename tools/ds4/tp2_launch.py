#!/usr/bin/env python3
"""Launch a DeepSeek TP2 API rank, worker, or SSH-managed pair."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ds4 import tp2_config, tp2_rank

RANK_DEFAULTS = {
    'engine': str(ROOT / 'build-hip/maya-ds4'), 'port': 18525,
    'context': 32768, 'expert_budget_mib': 92160, 'reserve_mib': 1024,
    'gpu': 0, 'cpu_threads': 14, 'dspark': 'on', 'prefill_chunk': 256,
    'suffix_draft': 0,
}
PATH_KEYS = ('pack', 'manifest', 'engine')


def load_settings(options):
    if options.config:
        path = Path(options.config).resolve()
    elif options.model:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', options.model):
            raise ValueError('model names may contain letters, digits, underscores and hyphens')
        path = ROOT / ('maya-ds4-tp2-' + options.model.lower() + '.json')
    else:
        path = Path(os.environ.get('MAYA_DS4_TP2_CONFIG', ROOT / 'maya-ds4-tp2.json')).resolve()
    settings = json.loads(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError('launcher config must be a JSON object')
    for key in ('context', 'dspark', 'prefill_chunk', 'suffix_draft', 'peer'):
        value = getattr(options, key, None)
        if value is not None:
            settings[key] = value
    # Resolve paths here, before forwarding the peer settings over SSH.
    for values in (settings, settings.get('rank0', {}), settings.get('rank1', {})):
        if not isinstance(values, dict):
            raise ValueError('rank settings must be JSON objects')
        for key in PATH_KEYS:
            if key in values:
                candidate = Path(values[key]).expanduser()
                values[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
    local_keys = set(PATH_KEYS) | {'rdma_device', 'gpu', 'cpu_threads', 'expert_budget_mib', 'reserve_mib', 'env'}
    for rank in ('rank0', 'rank1'):
        if set(settings.get(rank, {})) - local_keys:
            raise ValueError(rank + ' overrides may contain only local paths, device, memory, threads and env')
    return settings


def rank_settings(settings, rank):
    selected = dict(RANK_DEFAULTS)
    selected.update({key: value for key, value in settings.items() if key not in ('rank0', 'rank1')})
    selected.update(settings.get('rank' + str(rank), {}))
    return selected


def prepare(settings, rank):
    arguments = ['--rank', str(rank)]
    if rank == 0:
        arguments.append('--serve')
    for key in ('pack', 'manifest', 'engine', 'host', 'port', 'rdma_device', 'context',
                'expert_budget_mib', 'reserve_mib', 'gpu', 'cpu_threads', 'dspark',
                'prefill_chunk', 'suffix_draft'):
        if key not in settings:
            raise ValueError('launcher config requires ' + key)
        arguments.extend(['--' + key.replace('_', '-'), str(settings[key])])
    options = tp2_rank.parser().parse_args(arguments)
    environment = dict(os.environ)
    extra = settings.get('env', {})
    if not isinstance(extra, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in extra.items()):
        raise ValueError('env must map names to strings')
    environment.update(extra)
    command, environment = tp2_rank.prepare_launch(options, environment)
    identity = {key: environment['MAYA_DS4_TP2_' + key.upper() + '_DIGEST']
                for key in ('model', 'partition', 'build')}
    return options, command, environment, identity


def process_identity(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        return (fields[19], os.readlink(directory / 'exe'))
    except (OSError, IndexError):
        return None


def remember_children(process):
    owned = getattr(process, 'maya_children', {})
    if process.returncode is not None:
        process.maya_children = owned
        return
    previous = dict(owned)
    pending = [process.pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
        except OSError:
            continue
        for value in children:
            pid = int(value)
            identity = process_identity(pid)
            if identity is not None:
                owned[pid] = identity
                pending.append(pid)
    process.maya_children = owned
    path = getattr(process, 'maya_children_path', None)
    if path is not None and owned != previous:
        path.write_text(json.dumps(owned, indent=2) + '\n')


def stop_process(process, timeout=45):
    if process is None:
        return
    remember_children(process)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            remember_children(process)
            process.kill()
            process.wait(timeout=10)
    # Loading can precede the API server's signal-handler installation.
    # Retain child identities while it runs, then clean up verified children
    # after the server itself has exited.
    def owned_child(pid, identity):
        current = process_identity(pid)
        if current is None or current[0] != identity[0]:
            return False
        # The checked rank launcher execs the engine under the same PID.
        return current[1] == identity[1] or current[1] in getattr(process, 'maya_child_executables', ())

    for pid, identity in process.maya_children.items():
        if owned_child(pid, identity):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(owned_child(pid, identity) for pid, identity in process.maya_children.items()):
        if time.monotonic() >= deadline:
            for pid, identity in process.maya_children.items():
                if owned_child(pid, identity):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            break
        time.sleep(.1)
    if process.stdin is not None:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass


def stop_event():
    event = threading.Event()
    for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
        signal.signal(getattr(signal, name), lambda *_: event.set())
    return event


def run_worker(settings, managed=False):
    options, command, environment, identity = prepare(settings, 1)
    if managed:
        print(json.dumps({'preflight': identity}), flush=True)
        if sys.stdin.readline().strip() != 'start':
            raise ValueError('peer start handshake was interrupted')
    event = stop_event()
    process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, start_new_session=True)
    print('DeepSeek owned worker ' + json.dumps({'pid': process.pid, 'identity': process_identity(process.pid)}),
          file=sys.stderr, flush=True)
    if managed:
        def control_closed():
            sys.stdin.read()
            event.set()
        threading.Thread(target=control_closed, daemon=True).start()
    try:
        while process.poll() is None and not event.wait(.2):
            pass
        return process.returncode if process.poll() is not None else 0
    finally:
        stop_process(process)


def start_api(settings, port, directory, output=None):
    options, command, environment, identity = prepare(settings, 0)
    config = tp2_config.make_config(options, str(directory / 'engine.log'),
                                   settings.get('model_name', 'deepseek-v4.1-flash'),
                                   settings.get('model_label'))
    config['env'] = settings.get('env', {})
    config_path = directory / 'server.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    process = subprocess.Popen(
        [sys.executable, '-m', 'serve.server', '--engine', 'maya', '--config', str(config_path),
         '--host', settings.get('listen', '0.0.0.0'), '--port', str(port), '--parent-pipe'],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
    )
    process.maya_children_path = directory / 'server-children.json'
    process.maya_child_executables = {str(Path(sys.executable).resolve()), str(Path(settings['engine']).resolve())}
    return process


def runtime_root(options):
    return Path(options.state_dir).expanduser().resolve()


def run_api(settings, port, state):
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = Path(tempfile.mkdtemp(prefix='rank0-', dir=state))
    print('DeepSeek rank 0 logs ' + str(directory), flush=True)
    event = stop_event()
    process = start_api(rank_settings(settings, 0), port, directory)
    try:
        while process.poll() is None and not event.wait(.2):
            remember_children(process)
        return process.returncode if process.poll() is not None else 0
    finally:
        stop_process(process)


def peer_preflight(process, expected, log):
    data = bytearray()
    deadline = time.monotonic() + 60
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while b'\n' not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(timeout=remaining):
                raise ValueError('peer preflight timed out')
            chunk = os.read(process.stdout.fileno(), 16384)
            if not chunk:
                raise ValueError('peer exited during preflight')
            data.extend(chunk)
            if len(data) > 16384:
                raise ValueError('peer preflight response is too large')
    line = data.decode('utf-8')
    log.write(line)
    log.flush()
    received = json.loads(line)
    if received.get('preflight') != expected:
        raise ValueError('peer model, partition or engine identity differs')


def run_pair(settings, port, state):
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    bootstrap = int(settings.get('port', RANK_DEFAULTS['port']))
    with (state / 'pair.lock').open('a') as lock, \
         (state.parent / f'bootstrap-{bootstrap}.lock').open('a') as resource:
        try:
            for handle in (lock, resource):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('a DeepSeek pair already owns this API or bootstrap port') from exc
        directory = Path(tempfile.mkdtemp(prefix='run-', dir=state))
        control_path = state / 'control.sock'
        control_path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX) as control:
            control.bind(str(control_path))
            control.listen(1)
            control.settimeout(.2)
            event = stop_event()
            server = worker = None
            server_log = worker_log = None
            drainer = None
            try:
                local = rank_settings(settings, 0)
                peer = rank_settings(settings, 1)
                _, _, _, expected = prepare(local, 0)
                peer_host = settings.get('peer', 'strix-halo-2.local')
                if not isinstance(peer_host, str) or not peer_host or peer_host.startswith('-'):
                    raise ValueError('invalid peer SSH host')
                peer_root = settings.get('peer_root', str(ROOT))
                peer_python = settings.get('peer_python', peer_root + '/.venv/bin/python')
                remote_command = shlex.join([peer_python, peer_root + '/tools/ds4/tp2_launch.py', 'worker', '--managed'])
                print('DeepSeek pair logs ' + str(directory), flush=True)
                worker_log = (directory / 'worker.log').open('w', buffering=1)
                server_log = (directory / 'server.log').open('w', buffering=1)
                worker = subprocess.Popen(
                    ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                     '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3', peer_host, remote_command],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log,
                    text=True, start_new_session=True,
                )
                worker.stdin.write(json.dumps(peer) + '\n')
                worker.stdin.flush()
                peer_preflight(worker, expected, worker_log)
                worker.stdin.write('start\n')
                worker.stdin.flush()
                # Worker output after preflight is copied without blocking
                # the lifecycle loop or keeping an extra control-pipe writer.
                def drain_worker():
                    for line in worker.stdout:
                        worker_log.write(line)
                drainer = threading.Thread(target=drain_worker, daemon=True)
                drainer.start()
                server = start_api(local, port, directory, server_log)
                (directory / 'owned.json').write_text(json.dumps({
                    'supervisor_pid': os.getpid(), 'server_pid': server.pid,
                    'supervisor_identity': process_identity(os.getpid()),
                    'ssh_pid': worker.pid, 'identity': expected, 'api_port': port,
                }, indent=2) + '\n')
                (state / 'latest').write_text(str(directory) + '\n')
                while not event.is_set():
                    remember_children(server)
                    if server.poll() is not None:
                        return server.returncode
                    if worker.poll() is not None:
                        raise ValueError('DeepSeek worker exited, see ' + str(directory / 'worker.log'))
                    try:
                        client, _ = control.accept()
                    except socket.timeout:
                        continue
                    with client:
                        client.settimeout(1)
                        if client.recv(16).strip() == b'stop':
                            event.set()
                            client.sendall(b'stopping\n')
                return 0
            finally:
                # Stop the API supervisor before closing the remote control
                # pipe, so its engine cannot be respawned during worker cleanup.
                stop_process(server)
                if worker is not None:
                    if worker.stdin is not None:
                        try:
                            worker.stdin.close()
                        except BrokenPipeError:
                            pass
                    try:
                        worker.wait(timeout=55)
                    except subprocess.TimeoutExpired:
                        stop_process(worker, timeout=5)
                if drainer is not None:
                    drainer.join(timeout=5)
                if server_log is not None:
                    server_log.close()
                if worker_log is not None:
                    worker_log.close()
                control_path.unlink(missing_ok=True)


def stop_pair(state):
    path = state / 'control.sock'
    if not path.exists():
        return 0
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(60)
        try:
            client.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            return 0
        client.sendall(b'stop\n')
        client.recv(32)
    # cmdStop completes only after the owner releases the lock.
    with (state / 'pair.lock').open('a') as lock:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError('pair shutdown timed out')
                time.sleep(.1)
    return 0


def stop_pid(root, pid):
    for latest in root.glob('*/latest'):
        directory = Path(latest.read_text().strip()).resolve()
        if directory.parent != latest.parent.resolve():
            continue
        try:
            owned = json.loads((directory / 'owned.json').read_text())
        except FileNotFoundError:
            continue
        identity = process_identity(pid)
        if owned.get('supervisor_pid') == pid and identity is not None and \
                owned.get('supervisor_identity') == list(identity):
            return stop_pair(latest.parent)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('rank0', 'rank1', 'run', 'stop', 'stop-pid', 'worker'))
    parser.add_argument('number', type=tp2_rank.bounded(1, 2147483647), nargs='?', default=8000,
                        help='HTTP port, or supervisor PID for stop-pid')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--config')
    choice.add_argument('--model')
    parser.add_argument('--context', type=tp2_rank.bounded(1, 1048576))
    parser.add_argument('--dspark', choices=('on', 'off'))
    parser.add_argument('--prefill-chunk', type=int, choices=(0, 256))
    parser.add_argument('--suffix-draft', type=int, choices=(0, 3))
    parser.add_argument('--peer')
    parser.add_argument('--state-dir', default=str(Path.home() / '.cache/maya/deepseek-tp2'))
    parser.add_argument('--managed', action='store_true', help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    try:
        root = runtime_root(options)
        if options.action == 'stop-pid':
            return stop_pid(root, options.number)
        if options.number > 65535:
            parser.error('HTTP port must be from 1 to 65535')
        state = root / str(options.number)
        if options.action == 'stop':
            return stop_pair(state)
        if options.action == 'worker' and options.managed:
            settings = json.loads(sys.stdin.readline())
            return run_worker(settings, managed=True)
        settings = load_settings(options)
        if options.action == 'rank1':
            return run_worker(rank_settings(settings, 1))
        if options.action == 'rank0':
            return run_api(settings, options.number, state)
        if options.action == 'run':
            return run_pair(settings, options.number, state)
        parser.error('worker requires --managed')
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print('DeepSeek TP2 launcher ' + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
