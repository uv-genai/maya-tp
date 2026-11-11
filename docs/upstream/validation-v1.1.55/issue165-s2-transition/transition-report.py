#!/usr/bin/env python3
"""Collect #165 lossless compressed telemetry and bounded per-stage summaries."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

raw=Path('/tmp/maya-upstream-20261111-validation');run=raw/'issue165-s2-transition'
dest=Path('/home/ugo/software/project-maya/docs/upstream/validation-v1.1.55/issue165-s2-transition')
dest.mkdir(parents=True,exist_ok=True)
summary=json.loads((run/'summary.json').read_text());offset=summary['clock_wall_minus_monotonic']
records=[];telemetry={'scope':'1Hz windows during the retained original sequence and three mixed waves. Deltas use first/last valid samples within each window (outer samples only for windows with fewer than2samples), and expose sample span. No telemetry exists for the original slow waves, so their cause cannot be inferred from this replay.','stages':{}}
def stats(values):
 return {'min':min(values),'median':statistics.median(values),'max':max(values)} if values else None
def delta(a,b):return max(0,b-a)
def psitotal(row,kind):
 return {line.split()[0]:int(re.search(r'\btotal=(\d+)',line).group(1)) for line in (row.get('pressure',{}).get(kind) or '').splitlines() if re.search(r'\btotal=(\d+)',line)}
def stage_metrics(entries,lo,hi,hz):
 entries=[row for row in entries if 'time' in row]
 chosen=[row for row in entries if lo<=row['time']<=hi]
 if len(chosen)<2:
  chosen=sorted([*([max((r for r in entries if r['time']<lo),key=lambda r:r['time'])] if any(r['time']<lo for r in entries) else []),*chosen,*([min((r for r in entries if r['time']>hi),key=lambda r:r['time'])] if any(r['time']>hi for r in entries) else [])],key=lambda r:r['time'])
 out={'window':[lo,hi],'sample_count':len(chosen)}
 if not chosen:return out
 out['sample_interval']=[chosen[0]['time'],chosen[-1]['time']]
 for metric,factor in [('temp1_input',.001),('power1_average',.000001),('gpu_busy_percent',1)]:
  out[metric]=stats([float(value)*factor for row in chosen for key,value in row.items() if key.endswith(metric)])
 out['gpu_MHz']=stats([int(match.group(1)) for row in chosen for key,value in row.items() if key.endswith('pp_dpm_sclk') for match in [re.search(r'(\d+)Mhz \*',value)] if match])
 out['mem_available_MiB']=stats([row['memory_kib']['MemAvailable']/1024 for row in chosen if 'MemAvailable' in row.get('memory_kib',{})])
 out['swap_free_MiB']=stats([row['memory_kib']['SwapFree']/1024 for row in chosen if 'SwapFree' in row.get('memory_kib',{})])
 first,last=chosen[0],chosen[-1];seconds=last['time']-first['time'];out['sample_span_s']=seconds
 if seconds>0:
  cpu=[b-a for a,b in zip(first['cpu_ticks'][:8],last['cpu_ticks'][:8])];total=sum(cpu)
  if total>0:out['system_cpu_busy_pct']=100*(total-cpu[3]-cpu[4])/total;out['system_cpu_iowait_pct']=100*cpu[4]/total
  out['pressure_pct']={}
  for kind in ('cpu','memory','io'):
   a=psitotal(first,kind);b=psitotal(last,kind)
   out['pressure_pct'][kind]={key:delta(value,b[key])/1e6/seconds*100 for key,value in a.items() if key in b}
  out['vm_deltas']={key:delta(value,last.get('vm',{}).get(key,value)) for key,value in first.get('vm',{}).items()}
 owned=[row for row in chosen if 'owned' in row]
 if len(owned)>=2:
  a,b=owned[0]['owned'],owned[-1]['owned'];out['owned_pid']=a['pid'];out['owned_cpu_s']=(delta(a['stat']['utime'],b['stat']['utime'])+delta(a['stat']['stime'],b['stat']['stime']))/hz
  out['owned_io_deltas']={key:delta(value,b['io'].get(key,value)) for key,value in a['io'].items()};out['owned_fault_deltas']={key:delta(a['stat'][key],b['stat'][key]) for key in ('minflt','majflt')}
  threads=[]
  for tid,old in a['threads'].items():
   if tid not in b['threads']:continue
   new=b['threads'][tid];item={'tid':int(tid),'cpu_s':(delta(old['stat']['utime'],new['stat']['utime'])+delta(old['stat']['stime'],new['stat']['stime']))/hz,'minflt':delta(old['stat']['minflt'],new['stat']['minflt']),'majflt':delta(old['stat']['majflt'],new['stat']['majflt']),'io':{key:delta(value,new['io'].get(key,value)) for key,value in old['io'].items()}}
   if old.get('schedstat') and new.get('schedstat'):
    old_sched=list(map(int,old['schedstat'].split()));new_sched=list(map(int,new['schedstat'].split()));item['run_s']=delta(old_sched[0],new_sched[0])/1e9;item['runnable_wait_s']=delta(old_sched[1],new_sched[1])/1e9
   threads.append(item)
  out['thread_cpu_s_total']=sum(row['cpu_s'] for row in threads);out['thread_runnable_wait_s_total']=sum(row.get('runnable_wait_s',0) for row in threads)
  out['top_thread_io']=sorted(threads,key=lambda row:(row['io'].get('read_bytes',0),row['io'].get('rchar',0)),reverse=True)[:8]
  out['top_thread_cpu']=sorted(threads,key=lambda row:row['cpu_s'],reverse=True)[:8]
 return out

for kind in ('control','candidate'):
 folder=raw/('issue165-s2-'+kind+'-slots2');case=dest/kind;case.mkdir(exist_ok=True)
 d=json.loads((folder/'results.json').read_text());assert d.get('success')
 for name in ('results.json','ownership.json','peer-identity.json'):
  value=json.loads((folder/name).read_text())
  if name=='ownership.json':value.pop('cases',None);value['request_results']='results.json'
  (case/name).write_text(json.dumps(value,indent=2)+'\n')
 peer=json.loads((folder/'peer-identity.json').read_text());records.extend([{'host':'local','case':kind,'pid':d['rank0_pid'],'start':d['rank0_start'],'exe':d['rank0_exe']},{'host':'peer','case':kind,'pid':peer['pid'],'start':peer['start'],'exe':peer['exe']}])
 for name in ('rank0.log','rank1.log'):
  lines=(folder/name).read_text(errors='replace').splitlines();selected=[x for x in lines if any(k in x for k in ('ds4 tp2:','maya ds4 serve:','ERROR','poison','pair down'))]
  (case/(name+'.excerpt')).write_text('\n'.join(selected[:30]+(['[repetitive request summaries omitted]'] if len(selected)>90 else [])+selected[-60:] if len(selected)>90 else selected)+'\n')
 stages=[(row['label'],row['started_s']+offset,row['finished_s']+offset) for row in d['cases'] if row['label'] in ('short-off','short-on','prefill-off','prefill-on')]
 stages.extend(('mixed-wave-'+str(wave['wave']),wave['started_s']+offset,wave['finished_s']+offset) for wave in d['transition_waves'])
 telemetry['stages'][kind]={}
 for host in ('local','peer'):
  path=run/(kind+'-'+host+'-metrics.jsonl');entries=[json.loads(line) for line in path.read_text().splitlines()];identity=entries[0]['sampler_identity'];records.append({'host':host,'case':kind+'-sampler',**identity})
  with (case/(host+'-metrics.jsonl.gz')).open('wb') as target:
   with gzip.GzipFile(fileobj=target,mode='wb',compresslevel=6,mtime=0) as zipped:zipped.write(path.read_bytes())
  telemetry['stages'][kind][host]={name:stage_metrics(entries,lo,hi,entries[0]['hz']) for name,lo,hi in stages}

driver=json.loads((run/'ownership.json').read_text());records.append({'host':'local','case':'diagnostic-driver','pid':driver['pid'],'start':driver['start'],'exe':driver['exe']})
audit_code='''import json,os,pathlib
out=[]
for d in DATA:
 p=pathlib.Path('/proc')/str(d['pid']);item=dict(d)
 try:
  start=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
  item.update(owned_alive=start==d['start'] and os.readlink(p/'exe')==d['exe'],status='same_start' if start==d['start'] else 'pid_reused')
 except OSError:item.update(owned_alive=False,status='exited')
 out.append(item)
print(json.dumps(out))
'''
scope={};local=[row for row in records if row['host']=='local'];exec('DATA='+repr(local)+'\n'+audit_code.replace('print(json.dumps(out))','RESULT=out'),scope)
remote=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','strix-halo-1.local','python3','-'],input='DATA='+repr([row for row in records if row['host']=='peer'])+'\n'+audit_code,text=True));audit={'checked':time.time(),'identities':scope['RESULT']+remote};assert not any(row['owned_alive'] for row in audit['identities'])
(dest/'stopped-owned-processes.json').write_text(json.dumps(audit,indent=2)+'\n')
(dest/'telemetry-summary.json').write_text(json.dumps(telemetry,indent=2)+'\n');(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for script in ('case','transition','transition-report'):shutil.copy2('/tmp/maya-upstream-20261111-'+script+'.py',dest/(script+'.py'))
shutil.copy2(run/'sampler.py',dest/'sampler.py')
def compact(v,depth=0):
 pad='  '*depth
 if isinstance(v,dict):return '{\n'+',\n'.join('  '*(depth+1)+json.dumps(k)+': '+compact(x,depth+1) for k,x in v.items())+'\n'+pad+'}' if v else '{}'
 if isinstance(v,list):
  if all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v):return json.dumps(v,separators=(',',':'))
  return '[\n'+',\n'.join('  '*(depth+1)+compact(x,depth+1) for x in v)+'\n'+pad+']' if v else '[]'
 return json.dumps(v)
for path in dest.rglob('*.json'):path.write_text(compact(json.loads(path.read_text()))+'\n')
print('Collected',dest,'stopped identities',len(records),flush=True)
