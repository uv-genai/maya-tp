#!/usr/bin/env python3
"""Scoped owned TP2 checkpoint regression. No model downloads/deployment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,default=Path('/home/ugo/software/project-maya-upstream-20261111-runtime'))
p.add_argument('--name',required=True)
p.add_argument('--family',choices=('glm','ds4'),required=True)
p.add_argument('--model',choices=('s','xs'),default='s')
p.add_argument('--engine',required=True)
p.add_argument('--slots',type=int,choices=(1,2,4),required=True)
p.add_argument('--context',type=int)
p.add_argument('--tokens',type=int,default=64)
p.add_argument('--peer',default='strix-halo-1.local')
p.add_argument('--lead-address',default='192.168.1.92')
p.add_argument('--port',type=int,default=19351)
p.add_argument('--lifecycle',action='store_true')
p.add_argument('--fatal',action='store_true')
p.add_argument('--unicode',action='store_true',help='GLM single-slot non-Latin MTP regression')
p.add_argument('--http-smoke',action='store_true',help='public HTTP checks on this already-owned four-slot pair')
p.add_argument('--performance-only',action='store_true',help='one warmup and three matched four-client waves')
p.add_argument('--transition-waves',type=int,choices=(1,2,3),default=1,help='retain first original mixed wave and repeat warm waves in the same pair')
p.add_argument('--env',action='append',default=[],help='matched KEY=VALUE on both ranks')
p.add_argument('--output',type=Path,default=Path('/tmp/maya-upstream-20261111-validation'))
a=p.parse_args()
assert Path(a.name).name==a.name and a.name not in ('.','..')
assert not ((a.lifecycle or a.fatal) and a.slots==1)
assert 1<=a.port<=65535 and 1<=a.tokens<=1024
sys.path[:0]=[str(a.runtime),str(a.runtime/'tools')]
from serve.server import StrataEngine,child_env
from strata_tokenizer import Tokenizer

folder=a.output/a.name
folder.mkdir(parents=True,exist_ok=False)
cfg=json.loads((a.runtime/f'configs/{a.family}-{a.model}.json').read_text())
for assignment in a.env:
 key,value=assignment.split('=',1)
 assert key and all(c.isalnum() or c=='_' for c in key)
 cfg.setdefault('env',{})[key]=value
context=a.context or (32768 if a.family=='glm' else 262144)
if a.family=='glm':
 from serve.frontend import ChatTemplate
 pack=cfg['args'][cfg['args'].index('--glm-pack')+1]
 env=child_env(cfg)
 for key,value in {'GLM_SEQS':str(a.slots),'TP_MTP':'1','GLM_MTP_FIXED':'2','GLM_MTP_DRAFT':'3','TP_BOOTSTRAP_TIMEOUT':'900','ENGINE_QUIT_S':'5'}.items():
  env['MAYA_'+key]=env['STRATA_'+key]=value
 common=['--glm-pack',pack,'--max-context',str(context),'--parallel',str(a.slots),'--expert-cache','auto','--prefill','auto','--kv-streaming','on','--tp-world-size','2','--tp-bootstrap-port',str(a.port),'--tp-rdma-device','usb4_rdma5']
 root_args=[*common,'--tp-rank','0']
 peer_args=[a.engine,'--serve',*common,'--tp-rank','1','--tp-bootstrap-host',a.lead_address]
 template_path=Path(pack)/'tokenizer/chat_template.jinja'
 template=ChatTemplate(template_path if template_path.is_file() else a.runtime/'serve/chat_template.jinja')
 def render(text,thinking=False):return template.render([{'role':'user','content':text}],enable_thinking=thinking)
 off_value=1
else:
 from tools.ds4 import tp2_launch
 from serve.ds41_encoding import encode_messages
 cfg.update(host=a.lead_address,port=a.port,rdma_device='usb4_rdma5',context=context,parallel=a.slots,engine=a.engine,dspark='on',suffix_draft=0)
 for rank in ('rank0','rank1'):
  if rank in cfg and 'engine' in cfg[rank]:cfg[rank]['engine']=a.engine
 root_settings=tp2_launch.rank_settings(cfg,0)
 peer_settings=tp2_launch.rank_settings(cfg,1)
 _,root_command,env,_=tp2_launch.prepare(root_settings,0)
 pack=root_settings['pack'];root_args=root_command[1:];peer_args=None
 def render(text,thinking=False):return encode_messages([{'role':'user','content':text}],thinking_mode='thinking' if thinking else 'chat')
 off_value=0
tok=Tokenizer.from_dir(Path(pack)/'tokenizer')
remote_env={k:v for k,v in env.items() if k.startswith(('MAYA_','STRATA_','DS4_','HIP_','LD_LIBRARY_PATH'))}
remote_env.update(MAYA_TP_BOOTSTRAP_TIMEOUT='900',STRATA_TP_BOOTSTRAP_TIMEOUT='900')
remote_launch='''import hashlib,json,os,pathlib,subprocess,sys
folder=pathlib.Path(DATA['folder']);folder.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update(DATA['env']);os.environ.update(env)
sys.path[:0]=[DATA['runtime'],str(pathlib.Path(DATA['runtime'])/'tools')]
if DATA['family']=='ds4':
 from tools.ds4 import tp2_launch
 _,args,env,_=tp2_launch.prepare(DATA['settings'],1)
else:args=DATA['args']
with (folder/'rank1.log').open('x') as log:
 child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 proc=pathlib.Path('/proc')/str(child.pid)
 (folder/'identity.json').write_text(json.dumps({'pid':child.pid,'start':(proc/'stat').read_text().rsplit(')',1)[1].split()[19],'exe':os.readlink(proc/'exe'),'args':args,'sha256':hashlib.sha256(pathlib.Path(args[0]).read_bytes()).hexdigest()}))
 sys.exit(child.wait())
'''
data={'folder':str(folder),'env':remote_env,'runtime':str(a.runtime),'family':a.family,'args':peer_args,'settings':peer_settings if a.family=='ds4' else None}
remote_script='DATA='+repr(data)+'\n'+remote_launch
ssh_log=(folder/'ssh.log').open('x')
ssh=subprocess.Popen(['ssh','-o','BatchMode=yes',a.peer,'python3 -c '+shlex.quote(remote_script)],stdout=ssh_log,stderr=subprocess.STDOUT,start_new_session=True)
engine=StrataEngine.__new__(StrataEngine)
result={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'sha256':hashlib.sha256(Path(a.engine).read_bytes()).hexdigest(),'started':time.time(),'ssh_pid':ssh.pid,'cases':[]}
result['configured_env']=cfg.get('env',{}).copy()
result['effective_env']={k:v for k,v in env.items() if k.startswith(('MAYA_','STRATA_','DS4_'))}
result['clock_wall_minus_monotonic']=time.time()-time.monotonic()
context_sampler=r'''import glob,json,os,pathlib,select,sys,time
os.nice(10)
p=pathlib.Path('/proc')/str(os.getpid())
print(json.dumps({'sampler_identity':{'pid':os.getpid(),'start':(p/'stat').read_text().rsplit(')',1)[1].split()[19],'exe':os.readlink(p/'exe')},'scope':'1Hz GPU and validated owned-engine IO/CPU/fault/pressure context; no unrelated process inspection'}),flush=True)
def read(path):
 try:return pathlib.Path(path).read_text().strip()
 except OSError:return None
while not select.select([sys.stdin],[],[],1.)[0]:
 row={'time':time.time()}
 for pattern in ['/sys/class/drm/card0/device/pp_dpm_sclk','/sys/class/drm/card0/device/gpu_busy_percent','/sys/class/drm/card0/device/hwmon/hwmon*/temp1_input','/sys/class/drm/card0/device/hwmon/hwmon*/power1_average']:
  for name in glob.glob(pattern):
   value=read(name)
   if value is not None:row[name]=value
 row['pressure']={kind:read('/proc/pressure/'+kind) for kind in ('cpu','memory','io')}
 row['mem_available_kib']=next((int(line.split()[1]) for line in (read('/proc/meminfo') or '').splitlines() if line.startswith('MemAvailable:')),None)
 try:
  owner=json.loads(pathlib.Path(DATA['owner']).read_text())
  if DATA['rank']==0:pid=owner['rank0_pid'];start=owner['rank0_start'];exe=owner['rank0_exe']
  else:pid=owner['pid'];start=owner['start'];exe=owner['exe']
  proc=pathlib.Path('/proc')/str(pid);fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if fields[19]==start and os.readlink(proc/'exe')==exe:
   row['owned']={'pid':pid,'utime':int(fields[11]),'stime':int(fields[12]),'minflt':int(fields[7]),'majflt':int(fields[9]),'threads':int(fields[17]),'io':read(proc/'io'),'schedstat':read(proc/'schedstat')}
 except (OSError,ValueError,KeyError):pass
 print(json.dumps(row,separators=(',',':')),flush=True)
'''
context_processes=[];context_files=[]
def remember():
 if hasattr(engine,'proc'):
  result['rank0_pid']=engine.proc.pid
  try:
   proc=Path('/proc')/str(engine.proc.pid)
   result['rank0_start']=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
   result['rank0_exe']=os.readlink(proc/'exe')
  except OSError:pass
 (folder/'ownership.json').write_text(json.dumps(result,indent=2)+'\n')
remember()
def start_context_samplers():
 result['context_controls']=[]
 for host in ('local','peer'):
  data={'owner':str(folder/('ownership.json' if host=='local' else 'identity.json')),'rank':0 if host=='local' else 1}
  code='DATA='+repr(data)+'\n'+context_sampler
  log=(folder/('context-'+host+'.jsonl')).open('x');context_files.append(log)
  command=[sys.executable,'-u','-c',code] if host=='local' else ['ssh','-T','-o','BatchMode=yes',a.peer,'python3 -u -c '+shlex.quote(code)]
  process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);context_processes.append(process)
  identity={'host':host,'pid':process.pid,'exe':str(Path(shutil.which(command[0]) or command[0]).resolve())}
  try:identity['start']=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
  except OSError:identity['start']=None
  process.maya_context_identity=identity;result['context_controls'].append(identity);remember()
def stop_context_samplers():
 errors=[]
 for process in context_processes:
  try:process.stdin.close()
  except OSError:pass
 try:
  for process in context_processes:
   identity=process.maya_context_identity
   def owned():
    try:
     p=Path('/proc')/str(identity['pid'])
     return (p/'stat').read_text().rsplit(')',1)[1].split()[19]==identity['start'] and os.readlink(p/'exe')==identity['exe']
    except OSError:return False
   try:
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
     identity['forced_shutdown']=True
     if owned():os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=5)
     except subprocess.TimeoutExpired:
      if owned():os.killpg(process.pid,signal.SIGKILL)
      process.wait(timeout=5)
     if identity['host']=='peer':
      path=folder/'context-peer.jsonl'
      peer=json.loads(path.read_text().splitlines()[0])['sampler_identity']
      code="import os,pathlib,signal; d=DATA; p=pathlib.Path('/proc')/str(d['pid']); alive=p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==d['start'] and os.readlink(p/'exe')==d['exe']; os.kill(d['pid'],signal.SIGTERM) if alive else None"
      subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',a.peer,'python3 -c '+shlex.quote('DATA='+repr(peer)+'\n'+code)],timeout=10,check=True)
   except (OSError,ValueError,KeyError,IndexError,subprocess.SubprocessError) as error:errors.append(str(error))
 finally:
  for log in context_files:log.close()
 result['context_sampler_exits']=[process.returncode for process in context_processes]
 result['context_cleanup_errors']=errors
 if errors:raise RuntimeError('context sampler cleanup: '+repr(errors))
 assert all(code==0 for code in result['context_sampler_exits']),result['context_sampler_exits']
def encode(text,thinking=False):return tok.encode(render(text,thinking),parse_special=True)
def request(label,text,*,spec=True,budget=None,sampling=None,cancel_after=None,first_event=None,thinking=False,ids=None):
 ids=ids if ids is not None else encode(text,thinking)
 params={'temperature':0,'seed':1234,**(sampling or {})}
 if not spec:params['strata_tune']={'spec':off_value}
 cancelled=threading.Event();tokens=[];start=time.monotonic();first=None
 try:
  for token in engine.generate(ids,budget or a.tokens,params,cancelled):
   if token is not None:
    if first is None:
     first=time.monotonic()-start
     if first_event:first_event.set()
    tokens.append(int(token))
    if cancel_after and len(tokens)==cancel_after:cancelled.set()
  finished=time.monotonic()
  row={'label':label,'spec':spec,'ids':ids,'tokens':tokens,'text':tok.decode(tokens),'last':engine.last.copy(),'started_s':start,'finished_s':finished,'wall_s':finished-start,'first_s':first}
  row['engine_stat_snapshot']={'scope':'engine-wide latest STAT, not request-local','captured_time':time.time(),'values':dict(getattr(engine,'stat',{}))}
  assert tokens and len(tokens)<=(budget or a.tokens),row
  if not spec:assert row['last'].get('drafts_offered',0)==0,row
  if cancel_after:assert row['last'].get('finish')=='cancel',row
  else:assert engine.info.get('tp_pair')!='down',engine.info
  return row
 except Exception as error:
  return {'label':label,'spec':spec,'ids':ids,'tokens':tokens,'error':str(error),'wall_s':time.monotonic()-start}
def require(rows):
 assert all('error' not in row for row in rows),rows
 result['cases'].extend(rows);remember()
 for row in rows:print(json.dumps({k:v for k,v in row.items() if k not in ('ids','tokens','text')}),flush=True)
def peer_stop(sig='SIGTERM'):
 code='''import json,os,pathlib,signal
p=pathlib.Path(DATA)/'identity.json'
if p.exists():
 d=json.loads(p.read_text());proc=pathlib.Path('/proc')/str(d['pid'])
 if proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==d['start'] and os.readlink(proc/'exe')==d['exe']:
  os.killpg(d['pid'],getattr(signal,SIGNAL))
'''
 script='DATA='+repr(str(folder))+'\nSIGNAL='+repr(sig)+'\n'+code
 subprocess.run(['ssh','-o','BatchMode=yes',a.peer,'python3 -c '+shlex.quote(script)],timeout=30,check=True)
try:
 start_context_samplers()
 loaded=threading.Event();errors=[]
 def load():
  try:engine.__init__(a.engine,root_args,log=str(folder/'rank0.log'),env=env)
  except BaseException as error:errors.append(error)
  finally:loaded.set()
 threading.Thread(target=load,daemon=True).start()
 deadline=time.monotonic()+1200
 while not loaded.wait(.2):
  remember()
  if ssh.poll() not in (None,0):raise RuntimeError('remote coordinator failed; inspect ssh.log')
  if time.monotonic()>deadline:raise RuntimeError('engine load timeout')
 if errors:raise errors[0]
 remember();assert engine.info.get('tp_world')==2,engine.info
 if a.slots>1:assert engine.multi==a.slots,engine.info
 result['loaded_info']=engine.info.copy()
 if a.family=='glm':assert engine.info.get('kv')=='fp16',engine.info
 if a.performance_only:
  assert a.slots==2 and not a.lifecycle and not a.fatal
  result['performance_waves']=[]
  for wave in range(4):
   barrier=threading.Barrier(4)
   def perf_request(index):
    barrier.wait(timeout=10)
    return request('perf-'+str(wave)+'-'+str(index),'Explain eBPF tracing example '+str(index)+' with maps and safety.',spec=index!=0,sampling=({'temperature':.7,'top_p':.9,'seed':1234,'frequency_penalty':.2,'presence_penalty':.1} if index==1 else {}))
   with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(perf_request,range(4)))
   require(rows)
   assert all(row['last'].get('drafts_offered',0)>0 for row in rows[1:])
   seconds=max(row['finished_s'] for row in rows)-min(row['started_s'] for row in rows)
   point={'wave':wave,'warmup':wave==0,'wall_s':seconds,'tokens':sum(len(row['tokens']) for row in rows),'aggregate_tok_s':sum(len(row['tokens']) for row in rows)/seconds}
   result['performance_waves'].append(point);remember();print('PERF '+json.dumps(point),flush=True)
  assert not engine._routes
  result['success']=True
  raise SystemExit(0)
 prompts=[('short','Write a short eBPF tutorial explaining verifier safety, maps and tracing.'),('prefill','Explain this Linux tracing scenario with a concise practical example. '+('A tracing program observes a syscall, checks its arguments and updates a per-CPU map. '*75))]
 for label,text in prompts:
  off=request(label+'-off',text,spec=False)
  on=request(label+'-on',text,spec=True)
  require([off,on])
  assert off['tokens']==on['tokens'],{'same_topology_spec_difference':label,'off':off,'on':on}
  assert on['last'].get('drafts_offered',0)>0,{'no_real_proposals':on}
 if a.unicode:
  assert a.family=='glm' and a.slots==1
  import re
  log_offset=(folder/'rank0.log').stat().st_size
  text='请以“验证器安全检查”开头，用中文介绍 eBPF 验证器如何保证 Linux 内核安全，并给出一个跟踪系统调用的例子。请只使用中文说明。'
  off=request('unicode-off',text,spec=False,budget=96)
  on=request('unicode-on',text,spec=True,budget=96)
  require([off,on]);assert off['tokens']==on['tokens'],{'unicode_spec_difference':True}
  assert on['last'].get('drafts_offered',0)>0,on
  lines=(folder/'rank0.log').read_bytes()[log_offset:].decode(errors='replace').splitlines()
  evidence=[]
  for line in lines:
   match=re.search(r'(\d+) of (\d+) emitted tokens outside it, (\d+) of (\d+) drafting rounds on the whole head',line)
   if match:evidence.append({'line':line,'outside':int(match[1]),'emitted':int(match[2]),'whole_head_rounds':int(match[3]),'draft_rounds':int(match[4])})
  assert any(row['outside']>0 and row['whole_head_rounds']>0 and row['draft_rounds']>0 for row in evidence),{'missing_actual_unicode_fallback':evidence}
  result['unicode_fallback']={'actual_whole_head':True,'log_evidence':evidence,'off_on_raw_equal':True}
 if a.slots>1:
  result['transition_waves']=[]
  for wave in range(a.transition_waves):
   prefix='mixed-' if wave==0 else 'mixed-warm-'+str(wave)+'-'
   with ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(lambda i:request(prefix+str(i),'Explain eBPF tracing example '+str(i)+' with maps and safety.',spec=i!=0,sampling=({'temperature':.7,'top_p':.9,'seed':1234,'frequency_penalty':.2,'presence_penalty':.1} if i==1 else {})),range(4)))
   require(rows)
   assert all(row['last'].get('drafts_offered',0)>0 for row in rows[1:]),rows
   seconds=max(row['finished_s'] for row in rows)-min(row['started_s'] for row in rows)
   point={'wave':wave,'post_prefill_transition':wave==0,'wall_s':seconds,'tokens':sum(len(row['tokens']) for row in rows),'aggregate_tok_s':sum(len(row['tokens']) for row in rows)/seconds,'started_s':min(row['started_s'] for row in rows),'finished_s':max(row['finished_s'] for row in rows)}
   result['transition_waves'].append(point);remember();print('TRANSITION '+json.dumps(point),flush=True)
 if a.http_smoke:
  assert a.slots==4
  spec=importlib.util.spec_from_file_location('maya_owned_http_smoke','/tmp/maya-upstream-20261111-http-smoke.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  http=module.run_http_smoke(engine=engine,pack=pack,tok=tok,runtime=a.runtime,family=a.family,folder=folder,cfg=cfg,render=render)
  result['http_smoke']={'path':str(folder/'http-smoke.json'),'success':http['success'],'context':http['context'],'slots':http['slots'],'wave':http['wave'],'cleanup':http['cleanup']}
  remember()
 if a.lifecycle:
  first=threading.Event()
  with ThreadPoolExecutor(max_workers=4) as pool:
   f0=pool.submit(request,'life-long','Explain Linux eBPF verification step by step.',budget=192,first_event=first)
   assert first.wait(120)
   f1=pool.submit(request,'life-cancel','Describe eBPF map sharing.',spec=False,budget=96,cancel_after=8)
   f2=pool.submit(request,'life-prefill','Explain these tracing events. '+('An eBPF program observes a kernel event and updates a map. '*100),budget=24)
   require([f1.result(timeout=180)])
   f3=pool.submit(request,'life-refill','Define eBPF in one short sentence.',budget=8)
   require([f0.result(timeout=180),f2.result(timeout=180),f3.result(timeout=180)])
  # Reuse the most recently completed arena. An earlier completion may already
  # have been evicted by the deliberately queued refill request.
  previous=max(result['cases'][-3:],key=lambda row:row['finished_s'])
  row=request('life-reuse','',budget=8,ids=previous['ids']+previous['tokens']+tok.encode(' Continue.',parse_special=True))
  require([row])
  # GLM's recurrent prompt snapshot is deliberately taken before its final
  # input token; that anchor is recomputed on a continuation. DS4 reuses the
  # in-place prefix. Require each engine's documented retained boundary.
  result['reuse_expected_min']=len(previous['ids'])-(1 if a.family=='glm' else 0)
  assert row['last'].get('reused',0)>=result['reuse_expected_min'],row
 if a.fatal:
  first=threading.Event()
  with ThreadPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(request,'fatal-'+str(i),'Explain eBPF tracer '+str(i)+' with full detail.',budget=192,first_event=first) for i in range(4)]
   assert first.wait(120)
   deadline=time.monotonic()+20
   while len(engine._routes)<4 and time.monotonic()<deadline:time.sleep(.01)
   assert len(engine._routes)==4
   start=time.monotonic();peer_stop('SIGKILL');result['injected_peer_loss']=True
   rows=[f.result(timeout=30) for f in futures]
   assert all('error' in row for row in rows),rows
   cause=engine.info.get('tp_error')
   assert cause and engine.info.get('tp_pair')=='down',engine.info
   later=request('fatal-later','Return hello.',budget=8)
   assert 'error' in later and all(str(cause) in row['error'] for row in rows+[later]),{'cause':cause,'rows':rows+[later]}
   result['failure_fanout_s']=time.monotonic()-start
   result['cases'].extend(rows+[later]);remember()
 assert not engine._routes
 result['success']=True
finally:
 history=getattr(engine,'stat_history',None)
 result['stat_history']=list(history.copy()) if history is not None else []
 result['pre_shutdown_info']=getattr(engine,'info',{}).copy()
 result['pre_shutdown_error']=str(getattr(engine,'info',{}).get('tp_error',''))
 remember()
 try:
  if hasattr(engine,'proc'):
   os.environ['STRATA_ENGINE_QUIT_S']='5'
   if hasattr(engine,'lines') and getattr(engine,'max_context',0)>0:engine.close()
   elif engine.proc.poll() is None:engine.proc.kill()
   try:engine.proc.wait(timeout=10)
   except subprocess.TimeoutExpired:engine.proc.kill();engine.proc.wait(timeout=10)
   result['rank0_exit']=engine.proc.returncode
  try:ssh.wait(timeout=15)
  except subprocess.TimeoutExpired:
   peer_stop();
   try:ssh.wait(timeout=15)
   except subprocess.TimeoutExpired:peer_stop('SIGKILL');ssh.wait(timeout=15)
 finally:
  stop_context_samplers()
 result['ssh_exit']=ssh.returncode
 result['finished']=time.time()
 subprocess.run(['scp','-q',a.peer+':'+str(folder/'rank1.log'),str(folder/'rank1.log')],timeout=30,check=True)
 subprocess.run(['scp','-q',a.peer+':'+str(folder/'identity.json'),str(folder/'peer-identity.json')],timeout=30,check=True)
 peer=json.loads((folder/'peer-identity.json').read_text())
 assert peer['sha256']==result['sha256'],{'binary_mismatch':peer,'local':result['sha256']}
 legacy_eof=(a.family=='glm' and a.slots==1 and ssh.returncode==1 and result.get('success') and not result['pre_shutdown_error'] and result['pre_shutdown_info'].get('tp_pair')!='down' and 'TP2 worker recv: verbs peer closed the bootstrap connection' in (folder/'rank1.log').read_text(errors='replace'))
 result['peer_shutdown']='legacy_control_eof' if legacy_eof else 'injected_loss' if result.get('injected_peer_loss') else 'clean'
 (folder/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 remember();ssh_log.close()
 print('completed '+json.dumps({k:v for k,v in result.items() if k not in ('cases','stat_history')}),flush=True)
 assert result.get('success') and (result.get('rank0_exit')==0 or a.fatal) and (result.get('ssh_exit')==0 or a.fatal or legacy_eof),{k:v for k,v in result.items() if k not in ('cases','stat_history')}
