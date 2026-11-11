#!/usr/bin/env python3
"""Collect bounded, reproducible checkpoint evidence; audit only recorded PIDs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('--previous',required=True)
p.add_argument('--source',type=Path,default=Path('/home/ugo/software/project-maya'))
p.add_argument('--runtime',type=Path,default=Path('/home/ugo/software/project-maya-upstream-20261111-runtime'))
p.add_argument('--raw',type=Path,default=Path('/tmp/maya-upstream-20261111-validation'))
p.add_argument('--peer',default='strix-halo-1.local')
a=p.parse_args()
destination=a.source/'docs/upstream'/('validation-'+a.label)
destination.mkdir(parents=True,exist_ok=True)
records=[]
for label,kind in [(a.previous,'previous'),(a.label,'candidate')]:
 for folder in sorted(a.raw.glob(label+'-*-slots*')):
  if not folder.is_dir() or not (folder/'results.json').is_file():continue
  if kind=='previous' and folder.name not in {f'{label}-{model}-slots{slots}' for model in ('glm-s','ds4-xs','ds4-s') for slots in (1,2)}:continue
  target=destination/kind/folder.name.removeprefix(label+'-');target.mkdir(parents=True,exist_ok=True)
  d=json.loads((folder/'results.json').read_text())
  assert d.get('success'),folder
  for name in ('results.json','ownership.json','peer-identity.json'):
   shutil.copy2(folder/name,target/name)
   if name=='ownership.json':
    ownership=json.loads((target/name).read_text());ownership.pop('cases',None);ownership.pop('stat_history',None)
    ownership['request_results']='results.json'
    (target/name).write_text(json.dumps(ownership,indent=2)+'\n')
  if (folder/'http-smoke.json').exists():shutil.copy2(folder/'http-smoke.json',target/'http-smoke.json')
  for name in ('rank0.log','rank1.log'):
   lines=(folder/name).read_text(errors='replace').splitlines()
   selected=[line for line in lines if any(s in line for s in ('glm stat:','glm spec:','glm mtp: ','glm multi:','glm multi mtp:','glm prefill:','ds4 tp2:','maya ds4 serve:','TP2 worker recv:','control recv:','TP2_DOWN','poison','pair down','ERROR','ERROR:','FAIL'))]
   # Keep all failure/control summaries and bounded first/last timing summaries.
   if len(selected)>90:selected=selected[:35]+['[intermediate repetitive statistics omitted]']+selected[-55:]
   (target/(name+'.excerpt')).write_text('\n'.join(selected)+'\n')
  peer=json.loads((folder/'peer-identity.json').read_text())
  records.extend([{'case':folder.name,'host':'strix-halo-2.local','pid':d['rank0_pid'],'start':d.get('rank0_start'),'exe':d['rank0_exe']},{'case':folder.name,'host':a.peer,'pid':peer['pid'],'start':peer['start'],'exe':peer['exe']}])
  for host in ('local','peer'):
   metrics=folder/('context-'+host+'.jsonl')
   if metrics.exists():
    identity=json.loads(metrics.read_text().splitlines()[0])['sampler_identity']
    records.append({'case':folder.name+'-context','host':a.peer if host=='peer' else 'strix-halo-2.local',**identity})
    with (target/('context-'+host+'.jsonl.gz')).open('wb') as compressed:
     with gzip.GzipFile(fileobj=compressed,mode='wb',compresslevel=6,mtime=0) as stream:stream.write(metrics.read_bytes())
 if (a.raw/(label+'-summary.json')).exists():shutil.copy2(a.raw/(label+'-summary.json'),destination/(kind+'-summary.json'))
scripts=['case','checkpoint','build','collect','perf']
if a.label=='v1.1.58':scripts.extend(['finalize','components','final-http'])
if Path('/tmp/maya-upstream-20261111-ai-run.py').exists():scripts.append('ai-run')
if Path('/tmp/maya-upstream-20261111-http-smoke.py').exists() and a.label=='v1.1.58':scripts.append('http-smoke')
for script in scripts:
 shutil.copy2('/tmp/maya-upstream-20261111-'+script+'.py',destination/(script+'.py'))
for name in ('build-source.json','runtime-source.json','source.json'):
 file=a.runtime/'engines'/a.label/name
 if file.exists():shutil.copy2(file,destination/name)
for name in ('glm-s.json','ds4-xs.json','ds4-s.json'):
 shutil.copy2(a.runtime/'configs'/name,destination/name)
if a.label=='v1.1.58':
 manifest_dir=destination/'manifests';manifest_dir.mkdir(exist_ok=True)
 for name in ('maya-xs.json','maya-s.json'):
  shutil.copy2(a.runtime/'manifests'/name,manifest_dir/('lead-'+name))
  subprocess.run(['scp','-q',a.peer+':'+str(a.runtime/'manifests'/name),str(manifest_dir/('peer-'+name))],check=True)
components=a.raw/(a.label+'-components')
if (components/'results.json').exists():
 target=destination/'components';target.mkdir(exist_ok=True)
 entries=json.loads((components/'results.json').read_text());(target/'results.json').write_text(json.dumps(entries,indent=2)+'\n')
 for entry in entries:
  assert entry.get('passed',entry['exit_code']==0),entry
  shutil.copy2(components/(entry['name']+'.log'),target/(entry['name']+'.log'))
  records.append({'case':entry['name'],'host':'strix-halo-2.local','pid':entry['pid'],'start':entry['start'],'exe':entry['exe']})
if a.label=='v1.1.58':
 host=Path('/tmp/maya-v58-host-gates');target=destination/'host-gates';target.mkdir(exist_ok=True)
 for file in sorted(host.glob('*.log')):shutil.copy2(file,target/file.name)
 if Path('/tmp/maya-v58-host-gates.py').exists():shutil.copy2('/tmp/maya-v58-host-gates.py',target/'host-gates.py')
 for name in ('v58-components-build.log','v58-components.log'):
  file=Path('/tmp/maya-upstream-20261111-'+name)
  if file.exists():shutil.copy2(file,destination/file.name)
 commands=Path('/tmp/maya-upstream-20261111-v58-commands.txt')
 if commands.exists():shutil.copy2(commands,destination/'commands.txt')
build_environment=a.raw/(a.label+'-build-environment.json')
if build_environment.exists():shutil.copy2(build_environment,destination/'build-environment.json')
http_summary=a.raw/(a.label+'-http-summary.json')
if http_summary.exists():shutil.copy2(http_summary,destination/'http-summary.json')
probe=a.raw/(a.label+'-ai-run-probe.json')
if probe.exists():shutil.copy2(probe,destination/'ai-run-probe.json')
if a.label=='v1.1.54' and (a.raw/'perf-v54-xs-summary.json').exists():
 target=destination/'performance-repeat';target.mkdir(exist_ok=True)
 shutil.copy2(a.raw/'perf-v54-xs-summary.json',target/'summary.json')
 for kind in ('baseline','candidate'):
  folder=a.raw/('perf-v54-xs-'+kind+'-slots2');d=json.loads((folder/'results.json').read_text());case=target/kind;case.mkdir(exist_ok=True)
  for name in ('results.json','ownership.json','peer-identity.json'):
   shutil.copy2(folder/name,case/name)
   if name=='ownership.json':
    ownership=json.loads((case/name).read_text());ownership.pop('cases',None)
    ownership['request_results']='results.json'
    (case/name).write_text(json.dumps(ownership,indent=2)+'\n')
  peer=json.loads((folder/'peer-identity.json').read_text())
  records.extend([{'case':folder.name,'host':'strix-halo-2.local','pid':d['rank0_pid'],'start':d['rank0_start'],'exe':d['rank0_exe']},{'case':folder.name,'host':a.peer,'pid':peer['pid'],'start':peer['start'],'exe':peer['exe']}])
  for host in ('local','peer'):
   path=a.raw/(folder.name+'-'+host+'-gpu.jsonl');shutil.copy2(path,case/(host+'-gpu.jsonl'))
   identity=json.loads(path.read_text().splitlines()[0])['identity'];records.append({'case':folder.name+'-sampler','host':a.peer if host=='peer' else 'strix-halo-2.local',**identity})
failures=[]
for failed in ('failed-reuse-assumption','failed-output-directory-reuse','failed-last-prompt-snapshot-assumption'):
 folder=a.raw/failed/('v1.1.54-glm-s-slots2')
 if a.label!='v1.1.54' or not (folder/'results.json').exists():continue
 d=json.loads((folder/'results.json').read_text());failures.append({'attempt':failed,'success':d.get('success',False),'rank0_pid':d['rank0_pid'],'rank0_exit':d.get('rank0_exit'),'ssh_exit':d.get('ssh_exit'),'case_summaries':[{'label':row['label'],'last':row.get('last'),'error':row.get('error')} for row in d['cases']]})
 records.append({'case':failed,'host':'strix-halo-2.local','pid':d['rank0_pid'],'start':d['rank0_start'],'exe':d['rank0_exe']})
 if failed!='failed-output-directory-reuse':
  peer=json.loads((folder/'peer-identity.json').read_text());records.append({'case':failed,'host':a.peer,'pid':peer['pid'],'start':peer['start'],'exe':peer['exe']})
if failures:(destination/'harness-precondition-attempts.json').write_text(json.dumps(failures,indent=2)+'\n')
audit_code='''from pathlib import Path
import json,os
out=[]
for d in DATA:
 p=Path('/proc')/str(d['pid']);entry=dict(d)
 try:
  start=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
  if start!=d['start']:entry.update(owned_alive=False,status='pid_reused')
  else:entry.update(owned_alive=os.readlink(p/'exe')==d['exe'],status='same_start_identity')
 except OSError:entry.update(owned_alive=False,status='exited')
 out.append(entry)
print(json.dumps(out))
'''
local=[d for d in records if d['host']!=a.peer]
scope={};exec('DATA='+repr(local)+'\n'+audit_code.replace('print(json.dumps(out))','RESULT=out'),scope)
peer=[d for d in records if d['host']==a.peer]
remote=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',a.peer,'python3','-'],input='DATA='+repr(peer)+'\n'+audit_code,text=True))
audit={'checked':time.time(),'identities':scope['RESULT']+remote}
assert not any(d['owned_alive'] for d in audit['identities']),audit
(destination/'stopped-owned-processes.json').write_text(json.dumps(audit,indent=2)+'\n')
(destination/'scope.json').write_text(json.dumps({'label':a.label,'previous':a.previous,'raw_directory':str(a.raw),'runtime':str(a.runtime),'scope':('HIP gfx1151 / APU RDMA Flash models; six raw-engine StrataEngine cases plus two same-pair four-slot real public HTTP/Service smoke cases; synthetic GPU parity and pre-bootstrap rejection gates; no model downloads or GLM 744B execution' if a.label=='v1.1.58' else 'HIP gfx1151 / APU RDMA Flash models; raw-engine StrataEngine gates, not public HTTP; no model downloads or GLM 744B execution'),'config_notes':'Archived configs are base templates; case commands override engine/context/parallel/host/port. Effective argv, executable hashes, env and INFO in per-case ownership/results are authoritative.', 'audited_identity_count':len(records),'harness_sha256':{script:hashlib.sha256((destination/(script+'.py')).read_bytes()).hexdigest() for script in scripts}},indent=2)+'\n')
def compact_numeric_arrays(value,depth=0):
 pad='  '*depth
 if isinstance(value,dict):
  return '{\n'+',\n'.join('  '*(depth+1)+json.dumps(k)+': '+compact_numeric_arrays(v,depth+1) for k,v in value.items())+'\n'+pad+'}' if value else '{}'
 if isinstance(value,list):
  if all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in value):return json.dumps(value,separators=(',',':'))
  return '[\n'+',\n'.join('  '*(depth+1)+compact_numeric_arrays(v,depth+1) for v in value)+'\n'+pad+']' if value else '[]'
 return json.dumps(value)
for file in destination.rglob('*.json'):
 file.write_text(compact_numeric_arrays(json.loads(file.read_text()))+'\n')
print('Collected',destination,'recorded owned identities stopped',len(records),flush=True)
