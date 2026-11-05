#!/usr/bin/env python3
"""Owned GLM TP2 pair launcher for llama-swap; engine data remains on RDMA."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ds4 import tp2_launch as owned
from serve.server import child_env


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(settings, rank):
    config = settings['base']
    env = child_env(config)
    env.update(settings.get('env', {}))
    slots = int(settings['parallel']); drafts = int(settings['drafts']); mtp = int(settings['mtp'])
    if not 1 <= slots <= 4 or not 1 <= drafts <= (7 if slots > 1 else 8) or mtp not in (0, 1):
        raise ValueError('parallel must be 1..4, drafts 1..7 (1..8 with parallel 1), mtp 0 or 1')
    context = int(settings['context'])
    if not 1 <= context <= 1048576:
        raise ValueError('context must be 1..1048576')
    for key, value in {'GLM_SEQS': slots, 'TP_MTP': mtp, 'GLM_MTP_FIXED': drafts,
                       'GLM_MTP_DRAFT': drafts}.items():
        env['MAYA_'+key] = env['STRATA_'+key] = str(value)
    if slots > 1:
        env['MAYA_GLM_MULTI_MTP'] = env['STRATA_GLM_MULTI_MTP'] = str(drafts if mtp else 0)
    if mtp:
        for key in ('GLM_NO_MTP', 'GLM_NO_SPEC'):
            for prefix in ('MAYA_', 'STRATA_'): env.pop(prefix+key, None)
    args = list(config['args'])
    pack = None
    for flag in ('--glm-pack', '--pack'):
        if flag in args: pack = args[args.index(flag)+1]
    if pack is None:
        raise ValueError('base config requires --glm-pack')
    # Explicit paired overrides are last, matching the direct rank scripts.
    args += ['--parallel', str(slots), '--max-context', str(context),
             '--tp-world-size', '2', '--tp-rank', str(rank),
             '--tp-bootstrap-port', str(settings['port']), '--tp-rdma-device', rdma_device(settings)]
    if rank == 1: args += ['--tp-bootstrap-host', settings['host']]
    else: args += ['--serve']
    engine = str(Path(settings.get('engine') or config['exe']).expanduser().resolve())
    identity = {'build': digest(engine), 'model': digest(Path(pack)/'index.txt'),
                'execution': hashlib.sha256(json.dumps([slots, drafts, mtp, context], separators=(',', ':')).encode()).hexdigest()}
    native = Path(pack)/'native_experts.txt'
    if native.exists(): identity['native'] = digest(native)
    return None, [engine, *args], env, identity


def rdma_device(settings):
    if settings.get('rdma_device'): return settings['rdma_device']
    rails = []
    for line in subprocess.check_output(['ibv_devices'], text=True).splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].startswith('usb4_rdma'): rails.append((fields[1], fields[0]))
    if not rails: raise ValueError('no usb4_rdma device found')
    return min(rails)[1]


def start_api(settings, port, directory, output=None):
    _, command, env, _ = prepare(settings, 0)
    config = dict(settings['base'])
    config.update(exe=command[0], args=[x for x in command[1:] if x != '--serve'],
                  env={k:v for k,v in env.items() if k.startswith(('MAYA_', 'STRATA_'))},
                  cwd=str(ROOT), log=str(directory/'engine.log'),
                  tp={'world_size': 2, 'rank': 0, 'bootstrap_port': settings['port'], 'rdma_device': rdma_device(settings)})
    config_path = directory/'server.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    process = subprocess.Popen([sys.executable, '-m', 'serve.server', '--engine', 'strata',
        '--config', str(config_path), '--host', settings.get('listen', '0.0.0.0'),
        '--port', str(port), '--parent-pipe'], cwd=ROOT, stdin=subprocess.PIPE,
        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    process.maya_children_path = directory/'server-children.json'
    process.maya_child_executables = {str(Path(sys.executable).resolve()), command[0]}
    return process


def load_settings(options):
    env = dict(os.environ)
    if options.config: env['MAYA_CONFIG'] = str(Path(options.config).resolve())
    if options.model:
        env['MAYA_MODEL'] = options.model; env.pop('MAYA_PACK', None)
    if options.pack: env['MAYA_PACK'] = str(Path(options.pack).resolve())
    # Reuse the fork's model selection rules, without evaluating config values.
    command = ['bash', '-c', 'source "$1"; config=$(maya_resolve_config "$2"); pack=""; '
               'if [[ -n "${MAYA_PACK:-}${MAYA_MODEL:-}" ]]; then pack=$(maya_resolve_pack); fi; '
               'ROOT="$2" maya_config_with_pack "$config" "$pack"',
               '_', str(ROOT/'tools/tp2/model_env.sh'), str(ROOT)]
    base = json.loads(subprocess.check_output(command, env=env, text=True))
    args = base['args']; context = 4096
    for i, value in enumerate(args[:-1]):
        if value == '--max-context': context = int(args[i+1])
    settings = dict(family='GLM',base=base, parallel=options.parallel, context=options.context or context,
                    mtp=options.mtp, drafts=options.drafts, port=options.bootstrap_port,
                    host=options.bootstrap_host, peer=options.peer,
                    peer_root=options.peer_root or str(ROOT), rdma_device=options.rdma_device,
                    engine=options.engine)
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','rank0','rank1','worker','stop','stop-pid'))
    parser.add_argument('number', type=int, nargs='?', default=None, help='HTTP port or owned supervisor PID')
    parser.add_argument('--config'); parser.add_argument('--model'); parser.add_argument('--pack')
    parser.add_argument('--engine'); parser.add_argument('--context', type=int)
    parser.add_argument('--parallel', type=int, default=os.getenv('SEQS', os.getenv('MAYA_GLM_SEQS',os.getenv('STRATA_GLM_SEQS','1'))))
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--mtp', dest='mtp', action='store_const', const=1)
    choice.add_argument('--no-mtp', dest='mtp', action='store_const', const=0)
    parser.set_defaults(mtp=int(os.getenv('MAYA_TP_MTP',os.getenv('STRATA_TP_MTP','0'))))
    parser.add_argument('--drafts', type=int, default=os.getenv('MAYA_GLM_MTP_FIXED',os.getenv('STRATA_GLM_MTP_FIXED','2')))
    parser.add_argument('--peer', default=os.getenv('RANK1_HOST','strix-halo-2.local'))
    parser.add_argument('--peer-root'); parser.add_argument('--rdma-device')
    parser.add_argument('--bootstrap-port', type=int, default=18515)
    parser.add_argument('--bootstrap-host', default=os.getenv('BOOTSTRAP_HOST','192.168.1.22'))
    parser.add_argument('--state-dir', type=Path, default=Path.home()/'.cache/maya/glm-tp2')
    parser.add_argument('--managed', action='store_true', help=argparse.SUPPRESS)
    options=parser.parse_args(argv)
    try:
        if options.action=='stop-pid':
            if options.number is None: raise ValueError('stop-pid requires a supervisor PID')
            return owned.stop_pid(options.state_dir.resolve(),options.number)
        if options.action=='stop' and options.number is None:
            active=list(options.state_dir.resolve().glob('*/control.sock'))
            if len(active)>1: raise ValueError('multiple owned pairs: stop requires PORT')
            return owned.stop_pair(active[0].parent) if active else 0
        options.number=options.number or 8000
        if not 1 <= options.number <= 65535 or not 1 <= options.bootstrap_port <= 65535:
            raise ValueError('ports must be 1..65535')
        state=options.state_dir.resolve()/str(options.number)
        if options.action=='stop': return owned.stop_pair(state)
        if options.action=='worker' and options.managed:
            return owned.run_worker(json.loads(sys.stdin.readline()), True, prepare)
        settings=load_settings(options)
        prepare(settings,0)  # Validate before creating or stopping any process.
        if options.action=='rank1': return owned.run_worker(settings, False, prepare)
        if options.action=='rank0': return owned.run_api(settings,options.number,state,start_api)
        if options.action=='run': return owned.run_pair(settings,options.number,state,prepare,start_api,'tools/tp2/launch.py')
        raise ValueError('worker requires --managed')
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print('GLM TP2 launcher '+str(exc), file=sys.stderr, flush=True); return 1

if __name__=='__main__': raise SystemExit(main())
