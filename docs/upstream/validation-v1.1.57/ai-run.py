#!/usr/bin/env python3
"""Matched-rank v55 opt-in indexer workspace probe; existing Flash XS only."""
import json
from pathlib import Path
import subprocess
import sys

root=Path('/tmp/maya-upstream-20261111-validation')
runtime=Path('/home/ugo/software/project-maya-upstream-20261111-runtime')
label='v1.1.55';name=label+'-ds4-xs-ai-run-slots2'
with (root/(name+'.log')).open('x') as log:
 subprocess.run([sys.executable,'/tmp/maya-upstream-20261111-case.py','--name',name,'--family','ds4','--model','xs','--engine',str(runtime/'engines'/label/'maya-ds4'),'--slots','2','--env','MAYA_DS4_AI_RUN=1'],stdout=log,stderr=subprocess.STDOUT,check=True)
before=json.loads((root/(label+'-ds4-xs-slots2')/'results.json').read_text())
after=json.loads((root/name/'results.json').read_text())
assert after['effective_env']['MAYA_DS4_AI_RUN']=='1'
assert before['sha256']==after['sha256']
assert before['loaded_info']['context']==after['loaded_info']['context']==262144
assert before['effective_env']['MAYA_DS4_TP2_EXECUTION_DIGEST']!=after['effective_env']['MAYA_DS4_TP2_EXECUTION_DIGEST']
prior={row['label']:row for row in before['cases']};comparisons=[]
for row in after['cases']:
 old=prior[row['label']]
 assert old['ids']==row['ids'] and old['tokens']==row['tokens'],row['label']
 assert old['last']['drafts_accepted']==row['last']['drafts_accepted'] and old['last']['drafts_offered']==row['last']['drafts_offered'],row['label']
 comparisons.append({'label':row['label'],'input_tokens':len(row['ids']),'output_tokens':len(row['tokens']),'raw_equal':True,'before':old['last'],'after':row['last']})
result={'label':label,'flag':'MAYA_DS4_AI_RUN=1','context':262144,'slots':2,'long_prompt_tokens':1441,'boundaries':'Six256-row prompt pieces; input exceeds2*index_topk512 for compressor ratio2, exercising indexer selection. The >16K compressed-key candidate-block reduction path is outside this bounded probe.','execution_identity_changed':True,'comparisons':comparisons,'success':True}
(root/(label+'-ai-run-probe.json')).write_text(json.dumps(result,indent=2)+'\n')
print('AI_RUN PROBE PASSED',len(comparisons),'requests',sum(row['output_tokens'] for row in comparisons),'output tokens',flush=True)
