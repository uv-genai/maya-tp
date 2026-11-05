#!/usr/bin/env python3
"""Owned two-host DeepSeek batching correctness/lifecycle validation (requires GPUs)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from serve.server import StrataEngine
from serve.ds41_encoding import encode_messages
from tools.ds4 import tp2_launch
from strata_tokenizer import Tokenizer

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--peer', default='strix-halo-1.local')
p.add_argument('--lead-address', default='192.168.1.92')
p.add_argument('--rdma-device', default='usb4_rdma5')
p.add_argument('--output', type=Path, default=Path('/tmp/maya-batching-validation'))
p.add_argument('--seqs', type=int, default=1)
p.add_argument('--clients', type=int, default=1)
p.add_argument('--peer-seqs', type=int)
p.add_argument('--mixed', action='store_true')
p.add_argument('--scenario', choices=['normal','lifecycle','peer-loss'], default='normal')
p.add_argument('--context', type=int, default=32768)
p.add_argument('--dspark', type=int, default=1)
p.add_argument('--engine', required=True)
p.add_argument('--peer-root',default='/home/ugo/software/project-maya-batching-runtime')
p.add_argument('--port', type=int, default=19151)
p.add_argument('--tokens', type=int, default=32)
a = p.parse_args()
if not 1 <= a.seqs <= 4 or not 1 <= a.clients <= 16 or a.dspark not in (0,1):
    p.error('require 1..4 slots, 1..16 clients and dspark 0 or 1')
if a.scenario != 'normal' and a.seqs < 2: p.error('lifecycle/peer-loss scenarios require multiple slots')
if not 1 <= a.port <= 65535 or a.tokens <= 0 or a.context <= 0:
    p.error('require a valid port and positive token/context limits')
if Path(a.name).name != a.name or a.name in ('.','..'): p.error('name must be a directory name')
folder = a.output.resolve()/a.name
folder.mkdir(parents=True, exist_ok=False)
remote_folder = str(folder)
cfg = json.loads(a.config.read_text())
cfg.update(host=a.lead_address,port=a.port,rdma_device=a.rdma_device,context=a.context,
           parallel=a.seqs,engine=a.engine,dspark='on' if a.dspark else 'off',suffix_draft=0)
# Explicit test configs may use rank-local paths. Only override the executable.
for rank in ('rank0','rank1'):
    if rank in cfg and 'engine' in cfg[rank]: cfg[rank]['engine']=a.engine
root_settings=tp2_launch.rank_settings(cfg,0)
peer_settings=tp2_launch.rank_settings(cfg,1)
if a.peer_seqs is not None: peer_settings['parallel']=a.peer_seqs
options,root_command,env,checked=tp2_launch.prepare(root_settings,0)
pack=root_settings['pack']
common=root_command[1:]
remote_env={k:v for k,v in cfg.get('env',{}).items()}
remote_env.update(MAYA_TP_BOOTSTRAP_TIMEOUT='900',STRATA_TP_BOOTSTRAP_TIMEOUT='900')
remote_launch = """import json,os,pathlib,subprocess,sys
p=pathlib.Path(FOLDER)
p.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
env.update(ENV)
os.environ.update(env)
sys.path.insert(0,PEER_ROOT)
from tools.ds4 import tp2_launch
_,args,env,_=tp2_launch.prepare(SETTINGS,1)
with (p/'rank1.log').open('x') as log:
 child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 (p/'identity.json').write_text(json.dumps({'pid':child.pid,'start':pathlib.Path('/proc',str(child.pid),'stat').read_text().split()[21],'args':args}))
 sys.exit(child.wait())
""".replace('FOLDER',repr(remote_folder)).replace('SETTINGS',repr(peer_settings)).replace('PEER_ROOT',repr(a.peer_root)).replace('ENV',repr(remote_env))
remote_cmd = 'python3 -c '+shlex.quote(remote_launch)
remote_log = (folder/'ssh.log').open('w')
ssh = subprocess.Popen(['ssh','-o','BatchMode=yes',a.peer,remote_cmd], stdout=remote_log, stderr=subprocess.STDOUT, start_new_session=True)
engine = StrataEngine.__new__(StrataEngine)
result = {'name':a.name,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'engine_sha256':hashlib.sha256(Path(a.engine).read_bytes()).hexdigest(),
          'ssh_pid':ssh.pid,'started':time.time()}
(folder/'ownership.json').write_text(json.dumps(result,indent=2))
try:
    print('loading paired DeepSeek',json.dumps(result['args']),flush=True)
    loaded=threading.Event()
    load_errors=[]
    def load():
        try:
            engine.__init__(a.engine,common,log=str(folder/'rank0.log'),env=env)
        except BaseException as error:
            load_errors.append(error)
        finally:
            loaded.set()
    loader=threading.Thread(target=load,daemon=True)
    loader.start()
    while not loaded.wait(0.1):
        if hasattr(engine,'proc'):
            result['rank0_pid']=engine.proc.pid
            result['rank0_identity']=Path('/proc',str(engine.proc.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
            (folder/'ownership.json').write_text(json.dumps(result,indent=2))
        if ssh.poll() not in (None,0):
            raise RuntimeError('remote coordinator failed; inspect ssh.log')
    if load_errors:
        raise load_errors[0]

    if a.seqs > 1: assert engine.multi==a.seqs,engine.info
    result['rank0_pid']=engine.proc.pid
    (folder/'ownership.json').write_text(json.dumps(result,indent=2))
    tok=Tokenizer.from_dir(Path(pack)/'tokenizer')
    def render(messages,enable_thinking=False):
        return encode_messages(messages,thinking_mode='thinking' if enable_thinking else 'chat')
    prompts=['Explain what an eBPF verifier checks in a Linux tracing program.',
             'Give a short example of an eBPF tracing use case.']
    rows=[]
    texts=prompts if a.clients==1 else [prompts[i%len(prompts)]+' Include example '+str(i)+'.' for i in range(a.clients)]
    encoded=[tok.encode(render([{'role':'user','content':text}],enable_thinking=bool(a.mixed and index==2)),parse_special=True) for index,text in enumerate(texts)]
    barrier=threading.Barrier(a.clients) if a.clients>1 else None
    def request(index):
        text=texts[index]; ids=encoded[index]
        if barrier is not None: barrier.wait(timeout=5)
        started=time.monotonic(); first=None; out=[]
        sampling={'temperature':0}
        if a.mixed:
            if index==0: sampling['strata_tune']={'spec':0}
            if index==1: sampling.update(temperature=0.7,top_p=0.9,seed=1234,frequency_penalty=0.2,presence_penalty=0.1,repetition_penalty=1.03,penalty_last_n=64)
            if index==2: sampling.update(_think_budget=4,_think_end=tok.encode('</think>',parse_special=True)[0])
            if index==3: sampling['logit_bias']={2:2.0}
        for token in engine.generate(ids,a.tokens,sampling,threading.Event()):
            if token is not None:
                if first is None: first=time.monotonic()-started
                out.append(int(token))
        row={'prompt':text,'ids':ids,'tokens':out,'text':tok.decode(out),'last':engine.last.copy(),
             'wall_s':time.monotonic()-started,'first_s':first,'index':index}
        if a.mixed and index==0: assert row['last']['drafts_offered']==0,row
        if a.mixed and index==2: assert tok.encode('</think>',parse_special=True)[0] in out,row
        assert out and len(out)<=a.tokens
        assert engine.info.get('tp_pair')!='down',engine.info
        return row
    if a.scenario!='normal':
        events=[]; event_lock=threading.Lock(); first0=threading.Event(); cancels=[threading.Event() for _ in range(8)]
        def run_one(i,ids,budget,cancel_after=None):
            out=[]; start=time.monotonic()
            try:
                for t in engine.generate(ids,budget,{'temperature':0, **({'strata_tune':{'spec':0}} if i==2 else {})},cancels[i]):
                    if t is not None:
                        out.append(int(t))
                        with event_lock:events.append({'index':i,'t':time.monotonic(),'token':int(t)})
                        if i==0:first0.set()
                        if cancel_after and len(out)==cancel_after:cancels[i].set()
                return {'index':i,'tokens':out,'ids':ids,'last':engine.last.copy(),'wall_s':time.monotonic()-start}
            except Exception as e:
                return {'index':i,'tokens':out,'error':str(e),'wall_s':time.monotonic()-start}
        with ThreadPoolExecutor(max_workers=8) as pool:
            f0=pool.submit(run_one,0,encoded[0],192)
            assert first0.wait(120)
            if a.scenario=='lifecycle':
                long_ids=tok.encode(render([{'role':'user','content':('Explain Linux kernel tracing, verifier safety and map sharing. '*100)+' Give a concise answer.'}],enable_thinking=False),parse_special=True)
                f1=pool.submit(run_one,1,long_ids,24)
                f2=pool.submit(run_one,2,encoded[1],96,8)
                f3=pool.submit(run_one,3,encoded[0],4)
                short=f3.result(timeout=120)
                f4=pool.submit(run_one,4,encoded[0],8)
                rows=[f0.result(timeout=180),f1.result(timeout=180),f2.result(timeout=180),short,f4.result(timeout=180)]
                reused=run_one(5,rows[0]['ids']+rows[0]['tokens']+tok.encode(' Continue.',parse_special=True),8)
                rows.append(reused)
                assert all('error' not in r for r in rows),rows
                assert rows[2]['last']['finish']=='cancel',rows[2]
                assert rows[2]['last']['drafts_offered']==0,rows[2]
                assert rows[5]['last']['reused']>=len(encoded[0]),rows[5]
                assert len(rows[0]['tokens'])==192 and len(rows[1]['tokens'])==24,rows
                assert engine.info.get('tp_pair')!='down',engine.info
            else:
                fs=[f0]+[pool.submit(run_one,i,encoded[i%len(encoded)],192) for i in range(1,4)]
                deadline=time.monotonic()+120
                while len(engine._routes)<4 and time.monotonic()<deadline:time.sleep(.01)
                kill_start=time.monotonic()
                cleanup="import json,os,pathlib,signal;d=json.loads(pathlib.Path(%r).read_text());p=pathlib.Path('/proc')/str(d['pid']);assert (p/'stat').read_text().split()[21]==d['start'];os.killpg(d['pid'],signal.SIGKILL)" % str(folder/'identity.json')
                subprocess.run(['ssh','-o','BatchMode=yes',a.peer,'python3 -c '+shlex.quote(cleanup)],check=True)
                rows=[f.result(timeout=30) for f in fs]
                assert all('error' in r for r in rows),rows
                cause=engine.info.get('tp_pair_error') or getattr(engine,'_tp_error',None)
                later=run_one(6,encoded[0],8);rows.append(later)
                assert 'error' in later and engine.info.get('tp_pair')=='down',rows
                result['failure_fanout_s']=time.monotonic()-kill_start
        (folder/'events.json').write_text(json.dumps(events))
    elif a.clients==1:
        rows=[request(index) for index in range(len(texts))]
    else:
        assert engine.multi==a.seqs,engine.info
        with ThreadPoolExecutor(max_workers=a.clients) as executor:
            rows=list(executor.map(request,range(len(texts))))
    for row in rows: print(json.dumps({k:v for k,v in row.items() if k not in ('ids','tokens','text')}),flush=True)
    result['info']=engine.info.copy()
    assert not engine._routes
    (folder/'results.json').write_text(json.dumps(rows,indent=2))
    result['success']=True
finally:
    if hasattr(engine,'proc'):
        result['rank0_pid']=engine.proc.pid
        os.environ['STRATA_ENGINE_QUIT_S']='5'
        if hasattr(engine,'lines') and engine.max_context>0:
            engine.close()
        else:
            engine.proc.kill()
        try:
            engine.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            engine.proc.kill()
            engine.proc.wait(timeout=10)
        result['rank0_exit']=engine.proc.returncode
    try:
        ssh.wait(timeout=15)
    except subprocess.TimeoutExpired:
        cleanup='''import json,os,pathlib,signal
p=pathlib.Path(FOLDER)/'identity.json'
if p.is_file():
 d=json.loads(p.read_text()); proc=pathlib.Path('/proc')/str(d['pid'])
 if proc.exists() and (proc/'stat').read_text().split()[21]==d['start']:
  os.killpg(d['pid'],signal.SIGTERM)
'''.replace('FOLDER',repr(remote_folder))
        subprocess.run(['ssh','-o','BatchMode=yes',a.peer,'python3 -c '+shlex.quote(cleanup)],timeout=20,check=True)
        ssh.wait(timeout=20)
    result['ssh_exit']=ssh.returncode
    (folder/'ownership.json').write_text(json.dumps(result,indent=2))
    subprocess.run(['scp','-q',f'{a.peer}:{remote_folder}/rank1.log',str(folder/'rank1.log')],timeout=30,check=True)
    (folder/'ownership.json').write_text(json.dumps(result,indent=2))
    remote_log.close()
    print('completed',json.dumps(result),flush=True)
