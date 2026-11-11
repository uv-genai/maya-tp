#!/usr/bin/env python3
"""#165: two owned S2 runs, preserve first post-prefill mixed wave."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

raw=Path('/tmp/maya-upstream-20261111-validation')
runtime=Path('/home/ugo/software/project-maya-upstream-20261111-runtime')
run=raw/'issue165-s2-transition'
run.mkdir(exist_ok=False)
sampler=r'''import glob,json,os,pathlib,select,sys,time
os.nice(10)
me=pathlib.Path('/proc')/str(os.getpid())
print(json.dumps({'sampler_identity':{'pid':os.getpid(),'start':(me/'stat').read_text().rsplit(')',1)[1].split()[19],'exe':os.readlink(me/'exe')},'hz':os.sysconf('SC_CLK_TCK'),'page_size':os.sysconf('SC_PAGE_SIZE'),'config':DATA}),flush=True)
def read(path):
 try:return pathlib.Path(path).read_text().strip()
 except OSError:return None
def fields(path,names):
 text=read(path)
 if text is None:return {}
 out={}
 for line in text.splitlines():
  parts=line.replace(':',' ').split()
  if parts and parts[0] in names:
   try:out[parts[0]]=int(parts[1])
   except (ValueError,IndexError):pass
 return out
def stat(path):
 text=read(path)
 if not text:return None
 v=text.rsplit(')',1)[1].split()
 out={'state':v[0],'minflt':int(v[7]),'majflt':int(v[9]),'utime':int(v[11]),'stime':int(v[12]),'threads':int(v[17]),'start':v[19],'vsize':int(v[20]),'rss_pages':int(v[21])}
 if len(v)>39:out.update(last_cpu=int(v[36]),delayacct_blkio_ticks=int(v[39]))
 return out
io_names={'rchar','wchar','syscr','syscw','read_bytes','write_bytes','cancelled_write_bytes'}
while not select.select([sys.stdin],[],[],1.)[0]:
 row={'time':time.time(),'monotonic':time.monotonic()}
 for pattern in ['/sys/class/drm/card0/device/pp_dpm_sclk','/sys/class/drm/card0/device/gpu_busy_percent','/sys/class/drm/card0/device/hwmon/hwmon*/temp1_input','/sys/class/drm/card0/device/hwmon/hwmon*/power1_average']:
  for name in glob.glob(pattern):
   value=read(name)
   if value is not None:row[name]=value
 row['pressure']={kind:read('/proc/pressure/'+kind) for kind in ('cpu','memory','io')}
 row['loadavg']=read('/proc/loadavg')
 row['memory_kib']=fields('/proc/meminfo',{'MemAvailable','MemFree','Cached','SwapTotal','SwapFree','Dirty','Writeback','AnonPages','Slab','PageTables'})
 row['vm']=fields('/proc/vmstat',{'pgfault','pgmajfault','pswpin','pswpout','pgscan_direct','pgscan_kswapd','pgsteal_direct','pgsteal_kswapd','allocstall_normal','allocstall_movable'})
 system_stat=read('/proc/stat') or ''
 row['cpu_ticks']=[int(x) for x in next((line for line in system_stat.splitlines() if line.startswith('cpu ')),'cpu').split()[1:]]
 row['system_counts']=fields('/proc/stat',{'ctxt','processes','procs_running','procs_blocked'})
 try:
  owner=json.loads(pathlib.Path(DATA['owner_file']).read_text())
  if DATA['rank']==0:pid=owner['rank0_pid'];start=owner['rank0_start'];exe=owner['rank0_exe']
  else:pid=owner['pid'];start=owner['start'];exe=owner['exe']
  proc=pathlib.Path('/proc')/str(pid);status=stat(proc/'stat')
  if status and status['start']==start and os.readlink(proc/'exe')==exe:
   row['owned']={'pid':pid,'stat':status,'io':fields(proc/'io',io_names),'status':fields(proc/'status',{'VmRSS','VmHWM','RssAnon','RssFile','RssShmem','Threads','voluntary_ctxt_switches','nonvoluntary_ctxt_switches'}),'schedstat':read(proc/'schedstat'),'threads':{}}
   for thread in (proc/'task').iterdir():
    t=stat(thread/'stat')
    if t:row['owned']['threads'][thread.name]={'stat':t,'schedstat':read(thread/'schedstat'),'io':fields(thread/'io',io_names)}
  else:row['owned_status']='recorded identity exited or reused'
 except (OSError,ValueError,KeyError):row['owned_status']='owned record not yet available'
 print(json.dumps(row,separators=(',',':')),flush=True)
'''
(run/'sampler.py').write_text(sampler)
(run/'ownership.json').write_text(json.dumps({'pid':os.getpid(),'start':Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19],'exe':os.readlink(Path('/proc',str(os.getpid()),'exe')),'started':time.time(),'task':'#165 two S2 transition loads'},indent=2)+'\n')
summary={'issue':165,'scope':'Accepted v54 vs v55 defaultoff; original short off/on,1441-token prefill off/on,first mixed wave retained then2 warmed waves. No unrelated process/service inspection.','started':time.time(),'clock_wall_minus_monotonic':time.time()-time.monotonic(),'runs':{}}
for kind,label in [('control','v1.1.54'),('candidate','v1.1.55')]:
 name='issue165-s2-'+kind+'-slots2';folder=raw/name
 sampler_processes=[];files=[]
 for host in ('local','peer'):
  owner_file=str(folder/('ownership.json' if host=='local' else 'identity.json'))
  data={'owner_file':owner_file,'rank':0 if host=='local' else 1,'kind':kind,'expected_engine':str(runtime/'engines'/label/'maya-ds4')}
  code='DATA='+repr(data)+'\n'+sampler
  log=(run/(kind+'-'+host+'-metrics.jsonl')).open('x');files.append(log)
  command=[sys.executable,'-u','-c',code] if host=='local' else ['ssh','-T','-o','BatchMode=yes','strix-halo-1.local','python3 -u -c '+shlex.quote(code)]
  proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);sampler_processes.append(proc)
 try:
  with (raw/(name+'.log')).open('x') as log:
   subprocess.run([sys.executable,'/tmp/maya-upstream-20261111-case.py','--name',name,'--family','ds4','--model','s','--engine',str(runtime/'engines'/label/'maya-ds4'),'--slots','2','--transition-waves','3'],stdout=log,stderr=subprocess.STDOUT,check=True)
 finally:
  for proc in sampler_processes:
   proc.stdin.close();proc.wait(timeout=25)
  for file in files:file.close()
 d=json.loads((folder/'results.json').read_text());summary['runs'][kind]={'sha256':d['sha256'],'transition_waves':d['transition_waves'],'sampler_exits':[p.returncode for p in sampler_processes],'rank_exits':[d['rank0_exit'],d['ssh_exit']]}
 (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(kind,json.dumps(summary['runs'][kind]),flush=True)
before=json.loads((raw/'issue165-s2-control-slots2/results.json').read_text())
after=json.loads((raw/'issue165-s2-candidate-slots2/results.json').read_text())
summary['raw_tokens_equal']=all(a['ids']==b['ids'] and a['tokens']==b['tokens'] for a,b in zip(before['cases'],after['cases']))
assert summary['raw_tokens_equal']
summary['finished']=time.time()
(run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('TRANSITION DIAGNOSTIC COMPLETE',json.dumps(summary),flush=True)
