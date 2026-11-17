#!/usr/bin/env python3
"""Pre-bootstrap paired-ablation rejection probes (label argv[1]); no model files."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
rt=Path('/home/ugo/software/project-maya-upstream-20261111-runtime');import sys;label=sys.argv[1];dest=rt/'engines'/label;out=Path('/home/ugo/software/maya-upstream-20261117/validation/'+label+'-components');out.mkdir(exist_ok=False)
entries=[]
def run(name,exe,args,expected=0,message=None,timeout=60,env=None):
 logpath=out/(name+'.log');started=time.time()
 with logpath.open('x') as log:
  process=subprocess.Popen([str(exe),*args],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,**(env or {})})
  row={'name':name,'pid':process.pid,'start':None,'exe':str(exe),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'args':args,'started':started,'expected_exit_code':expected}
  try:row['start']=(Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
  except OSError:row['identity_note']='process exited before proc stat capture; own direct Popen reaped below'
  try:row['exit_code']=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGKILL);process.wait();row.update(exit_code=process.returncode,timeout=True)
 row['finished']=time.time();row['expected_message']=message
 text=logpath.read_text(errors='replace');row['passed']=row['exit_code']==expected and (message is None or message in text)
 if message:row['before_model_or_bootstrap']=not any(word in text for word in ('cannot open','READY','bootstrap connection','TP2_DOWN','hipError','Loading'))
 entries.append(row);(out/'results.json').write_text(json.dumps(entries,indent=2)+'\n')
 print(json.dumps(row),flush=True)
 assert row['passed'],row
 if message:assert row['before_model_or_bootstrap'],(row,text)
for family,binary,base,code,message in (
 ('glm','maya',['--glm-pack','/tmp/maya-upstream-no-model-or-ablation-file','--tp-world-size','2'],1,'GLM refusal ablation is unavailable with TP2'),
 ('ds4','maya-ds4',['--pack','/tmp/maya-upstream-no-model-or-ablation-file','--serve','--tp2-rank','0'],2,'ablation is unsupported for DeepSeek TP2')):
 for flag,value in (('--ablate','/tmp/maya-upstream-no-direction.gguf'),('--ablate-layer','1'),('--ablate-scale','0.35'),('--ablate-default','off')):
  run(family+'-reject-'+flag.removeprefix('--'),dest/binary,[*base,flag,value],expected=code,message=message,timeout=5)
if (dest/'maya-ds4').exists() and label not in ('v1.1.59','v1.1.60'):   # (the cone refusal exists from v1.1.61 on)
 run('ds4-reject-yoco-cone',dest/'maya-ds4',['--pack','/tmp/maya-upstream-no-model-or-ablation-file','--serve','--tp2-rank','0'],expected=2,message='fast prompt reading (DS4_YOCO_CONE) is unsupported for DeepSeek TP2',timeout=5,env={'DS4_YOCO_CONE':'1'})
print('All synthetic parity / early rejection gates passed',flush=True)
