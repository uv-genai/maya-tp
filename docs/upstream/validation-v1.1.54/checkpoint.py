#!/usr/bin/env python3
"""Run six owned Flash TP2 cases and compare previous frozen checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('--previous')
p.add_argument('--runtime',type=Path,default=Path('/home/ugo/software/project-maya-upstream-20261111-runtime'))
p.add_argument('--output',type=Path,default=Path('/tmp/maya-upstream-20261111-validation'))
p.add_argument('--models',nargs='+',choices=('glm-s','ds4-xs','ds4-s'),default=['glm-s','ds4-xs','ds4-s'])
p.add_argument('--slots',nargs='+',type=int,choices=(1,2,4),default=[1,2])
p.add_argument('--tokens',type=int,default=64)
p.add_argument('--lifecycle',action='store_true')
p.add_argument('--fatal',action='store_true')
p.add_argument('--unicode',action='store_true')
p.add_argument('--fatal-models',nargs='+',choices=('glm-s','ds4-xs','ds4-s'),default=['glm-s','ds4-xs'])
p.add_argument('--resume',action='store_true')
a=p.parse_args()
assert Path(a.label).name==a.label
a.output.mkdir(parents=True,exist_ok=True)
summary={'label':a.label,'previous':a.previous,'started':time.time(),'cases':{},'differences':[],'dynamic_input_changes':[]}
for model in a.models:
 family,quant=model.split('-')
 engine=a.runtime/'engines'/a.label/('maya' if family=='glm' else 'maya-ds4')
 for slots in a.slots:
  case=f'{a.label}-{model}-slots{slots}'
  folder=a.output/case
  command=[sys.executable,'/tmp/maya-upstream-20261111-case.py','--runtime',str(a.runtime),'--output',str(a.output),'--name',case,'--family',family,'--model',quant,'--engine',str(engine),'--slots',str(slots),'--tokens',str(a.tokens)]
  if slots>1:
   if a.lifecycle:command.append('--lifecycle')
   if a.fatal and model in a.fatal_models:command.append('--fatal')
  if a.unicode and family=='glm' and slots==1:command.append('--unicode')
  if not (a.resume and (folder/'results.json').exists()):
   with (a.output/(case+'.log')).open('x') as log:
    print('RUN',case,flush=True)
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  data=json.loads((folder/'results.json').read_text())
  assert data.get('success'),folder
  rows=[]
  for row in data['cases']:
   metrics=row.get('last',{})
   decode_ms=metrics.get('decode_ms',0)
   prefill_ms=metrics.get('prompt_ms',0)
   rows.append({'label':row['label'],'generated':len(row.get('tokens',[])),'decode_tok_s':len(row.get('tokens',[]))*1000/decode_ms if decode_ms else None,'prompt_tokens':metrics.get('prompt_tokens'),'reused':metrics.get('reused'),'prefill_ms':prefill_ms,'drafts_offered':metrics.get('drafts_offered'),'drafts_accepted':metrics.get('drafts_accepted'),'wall_s':row['wall_s'],'error':row.get('error')})
  summary['cases'][model+'-slots'+str(slots)]={'sha256':data['sha256'],'metrics':rows}
  if a.previous:
   before=json.loads((a.output/f'{a.previous}-{model}-slots{slots}'/'results.json').read_text())
   prior={r['label']:r for r in before['cases']}
   for row in data['cases']:
    old=prior.get(row['label'])
    if old and 'error' not in row and 'error' not in old:
     if old['ids']!=row['ids'] and row['label']=='life-reuse':
      summary['dynamic_input_changes'].append({'model':model,'slots':slots,'label':row['label'],'reason':'latest completed arena differs; retained-prefix gate is checked independently'})
      continue
     if old['ids']!=row['ids'] or old['tokens']!=row['tokens']:
      summary['differences'].append({'model':model,'slots':slots,'label':row['label'],'inputs_equal':old['ids']==row['ids'],'old_tokens':old['tokens'],'new_tokens':row['tokens']})
  (a.output/(a.label+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
  print('PASS',case,'differences',len(summary['differences']),flush=True)
summary['finished']=time.time()
(a.output/(a.label+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
assert not summary['differences'],summary['differences']
print('CHECKPOINT COMPLETE',a.label,'seconds',summary['finished']-summary['started'],flush=True)
