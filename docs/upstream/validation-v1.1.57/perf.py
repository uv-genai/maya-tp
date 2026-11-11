#!/usr/bin/env python3
"""Bounded XS2 warmed control/candidate repeat with GPU sysfs telemetry."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import threading
import time

root=Path('/tmp/maya-upstream-20261111-validation')
runtime=Path('/home/ugo/software/project-maya-upstream-20261111-runtime')
sampler='''import glob,json,os,pathlib,select,sys,time
p=pathlib.Path('/proc')/str(os.getpid())
print(json.dumps({'identity':{'pid':os.getpid(),'start':(p/'stat').read_text().rsplit(')',1)[1].split()[19],'exe':os.readlink(p/'exe')}}),flush=True)
while not select.select([sys.stdin],[],[],.5)[0]:
 row={'time':time.time()}
 for pattern in ['/sys/class/drm/card0/device/pp_dpm_sclk','/sys/class/drm/card0/device/gpu_busy_percent','/sys/class/drm/card0/device/hwmon/hwmon*/temp1_input','/sys/class/drm/card0/device/hwmon/hwmon*/power1_average']:
  for name in glob.glob(pattern):
   try:row[name]=pathlib.Path(name).read_text().strip()
   except OSError:pass
 print(json.dumps(row),flush=True)
'''
summary={}
for kind,label in [('baseline','baseline'),('candidate','v1.1.54')]:
 name='perf-v54-xs-'+kind+'-slots2'
 paths=[];processes=[]
 for host in ('local','peer'):
  path=root/(name+'-'+host+'-gpu.jsonl');file=path.open('x');paths.append((path,file))
  command=[sys.executable,'-u','-c',sampler] if host=='local' else ['ssh','-T','-o','BatchMode=yes','strix-halo-1.local','python3 -u -c '+shlex.quote(sampler)]
  proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
  processes.append(proc)
 try:
  with (root/(name+'.log')).open('x') as log:
   subprocess.run([sys.executable,'/tmp/maya-upstream-20261111-case.py','--name',name,'--family','ds4','--model','xs','--engine',str(runtime/'engines'/label/'maya-ds4'),'--slots','2','--performance-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
 finally:
  for proc in processes:
   proc.stdin.close();proc.wait(timeout=20)
  for path,file in paths:file.close()
 d=json.loads((root/name/'results.json').read_text())
 summary[kind]={'sha256':d['sha256'],'waves':d['performance_waves'],'median_aggregate_tok_s':statistics.median(r['aggregate_tok_s'] for r in d['performance_waves'] if not r['warmup']),'sampler_exits':[proc.returncode for proc in processes]}
 (root/'perf-v54-xs-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(kind,json.dumps(summary[kind]),flush=True)
before=json.loads((root/'perf-v54-xs-baseline-slots2/results.json').read_text())
after=json.loads((root/'perf-v54-xs-candidate-slots2/results.json').read_text())
summary['raw_tokens_equal']=all(a['ids']==b['ids'] and a['tokens']==b['tokens'] for a,b in zip(before['cases'],after['cases']))
assert summary['raw_tokens_equal']
summary['ratio']=summary['candidate']['median_aggregate_tok_s']/summary['baseline']['median_aggregate_tok_s']
(root/'perf-v54-xs-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PERF COMPLETE',json.dumps(summary),flush=True)
