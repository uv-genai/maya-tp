#!/usr/bin/env python3
"""Exactly two final 4-slot owned loads, only after six checkpoint cases pass."""
import json,subprocess,sys,time
from pathlib import Path
out=Path('/tmp/maya-upstream-20261111-validation');runtime=Path('/home/ugo/software/project-maya-upstream-20261111-runtime');label='v1.1.58'
summary=json.loads((out/(label+'-summary.json')).read_text());assert summary.get('finished') and not summary['differences'] and len(summary['cases'])==6,summary
records=[]
for model,context,binary in (('glm-s',32768,'maya'),('ds4-xs',8192,'maya-ds4')):
 family,quant=model.split('-');case=f'{label}-{model}-http-slots4'
 command=[sys.executable,'/tmp/maya-upstream-20261111-case.py','--name',case,'--family',family,'--model',quant,'--engine',str(runtime/'engines'/label/binary),'--slots','4','--context',str(context),'--http-smoke']
 print('RUN',case,flush=True);start=time.time()
 with (out/(case+'.log')).open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((out/case/'results.json').read_text());http=json.loads((out/case/'http-smoke.json').read_text())
 assert result['success'] and http['success'] and http['cleanup']['listener_exited']
 records.append({'case':case,'command':command,'started':start,'finished':time.time(),'success':True,'wave':http['wave'],'cleanup':http['cleanup']})
 (out/(label+'-http-summary.json')).write_text(json.dumps(records,indent=2)+'\n')
 print('PASS',case,flush=True)
print('FINAL HTTP COMPLETE',flush=True)
