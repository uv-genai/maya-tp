from pathlib import Path
import argparse,hashlib,json,os,shutil,socket,stat,subprocess,tempfile,time

p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','commit','verify','rollback']);a=p.parse_args()
root=Path('/home/ugo/software/project-maya');review=Path('/home/ugo/software/project-maya-ds4-tp2-review')
src=Path('/home/ugo/software/project-maya-upstream-20261111-runtime/engines/v1.1.58')
backup=Path('/home/ugo/software/maya-binary-backups/issue-169-20261111-v58')
expected={'maya':'dd37dbd84cabcfce75c4072a8fdac95b08a065ae47481c8bba77fbb9ab07f965','maya-ds4':'17f42fa16d52dd59c2843138a23a2a2dbcbecc40826d623272e2456f82a11076'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def snapshot():
 files=set()
 for r in (root,review):
  files.update(x for x in r.glob('maya-*.json') if x.is_file())
  for d in ('tools/tp2','tools/ds4','tools/llama-swap'):
   files.update(x for x in (r/d).glob('*') if x.is_file() and x.suffix in ('.sh','.py','.json'))
 conf=Path('/home/ugo/.config/llama-swap/config.yaml')
 if conf.is_file():files.add(conf)
 return {str(x):{'sha256':sha(x),'size':x.stat().st_size} for x in sorted(files)}
def write(path,obj):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(temp,path)
def links(path):
 r=subprocess.run(['ldd',str(path)],capture_output=True,text=True,timeout=20)
 assert r.returncode==0 and 'not found' not in r.stdout+r.stderr,(str(path),r.stdout,r.stderr)
 return r.stdout

def readplan():return json.loads((backup/'plan.json').read_text())
if a.phase=='prepare':
 assert not backup.exists(),f'Existing backup/plan requires inspection: {backup}'
 for name,digest in expected.items():assert sha(src/name)==digest,(name,'source mismatch')
 targets=[root/'build-hip/maya',root/'build-hip/maya-ds4',review/'build-hip/queued-a950e7e9350b/maya-ds4']
 plan={'issue':169,'source_commit':'2485cb609da0c436e1b7d9012d31a136656dd013','host':socket.gethostname(),'created':time.time(),'backup_dir':str(backup),'selectors_before':snapshot(),'targets':[],'absent_targets':[]}
 backup.mkdir(parents=True)
 for i,target in enumerate(targets):
  if not target.exists():plan['absent_targets'].append(str(target));continue
  assert target.is_file() and not target.is_symlink(),str(target)
  st=target.stat();name=target.name;old=sha(target)
  saved=backup/f'{i}-{name}.before';shutil.copy2(target,saved);assert sha(saved)==old
  fd,tmp=tempfile.mkstemp(prefix=f'.{name}.issue169-',dir=target.parent);os.close(fd);tmp=Path(tmp)
  shutil.copy2(src/name,tmp);os.chmod(tmp,stat.S_IMODE(st.st_mode));assert sha(tmp)==expected[name]
  linked=links(tmp)
  with tmp.open('rb') as f:os.fsync(f.fileno())
  plan['targets'].append({'path':str(target),'source':str(src/name),'old_sha256':old,'new_sha256':expected[name],'backup':str(saved),'stage':str(tmp),'mode':stat.S_IMODE(st.st_mode),'uid':st.st_uid,'gid':st.st_gid,'ldd':linked})
 assert len(plan['targets'])>=2
 assert snapshot()==plan['selectors_before'],'launcher/config changed during prepare'
 write(backup/'plan.json',plan)
 print(json.dumps({'host':plan['host'],'phase':'prepared','backup':str(backup),'paths':[x['path'] for x in plan['targets']],'selector_count':len(plan['selectors_before']),'absent':plan['absent_targets']}))
elif a.phase=='commit':
 plan=readplan();assert snapshot()==plan['selectors_before'],'launcher/config changed before commit'
 for x in plan['targets']:
  target=Path(x['path']);tmp=Path(x['stage']);assert sha(target)==x['old_sha256'];assert sha(tmp)==x['new_sha256']
  assert target.stat().st_uid==x['uid'] and tmp.stat().st_uid==x['uid']
 for x in plan['targets']:
  os.replace(x['stage'],x['path'])
  fd=os.open(str(Path(x['path']).parent),os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 plan['committed']=time.time();write(backup/'plan.json',plan)
 assert snapshot()==plan['selectors_before'],'launcher/config changed during replacement'
 print(json.dumps({'host':plan['host'],'phase':'installed','targets':len(plan['targets'])}))
elif a.phase=='verify':
 plan=readplan();checks=[]
 assert snapshot()==plan['selectors_before'],'launcher/config fingerprints changed'
 for x in plan['targets']:
  target=Path(x['path']);assert sha(target)==x['new_sha256'];assert sha(x['backup'])==x['old_sha256']
  st=target.stat();assert (stat.S_IMODE(st.st_mode),st.st_uid,st.st_gid)==(x['mode'],x['uid'],x['gid'])
  linked=links(target)
  r=subprocess.run([str(target),'--help'],capture_output=True,text=True,timeout=20)
  assert r.returncode==0,(str(target),r.returncode,r.stdout,r.stderr)
  (backup/(target.name+'-'+str(len(checks))+'-help.txt')).write_text(r.stdout+r.stderr)
  checks.append({'path':str(target),'sha256':sha(target),'ldd_ok':True,'help_exit':r.returncode})
 assert snapshot()==plan['selectors_before'],'launcher/config fingerprints changed during verify'
 result={'host':plan['host'],'verified':time.time(),'source_commit':plan['source_commit'],'targets':checks,'selector_count':len(plan['selectors_before']),'selectors_unchanged':True,'backup_dir':str(backup),'processes_stopped':0,'models_started':0}
 write(backup/'verified.json',result);print(json.dumps(result))
else:
 plan=readplan()
 for x in plan['targets']:
  target=Path(x['path']);saved=Path(x['backup']);assert sha(saved)==x['old_sha256']
  fd,tmp=tempfile.mkstemp(prefix=f'.{target.name}.rollback169-',dir=target.parent);os.close(fd);shutil.copy2(saved,tmp);os.replace(tmp,target);assert sha(target)==x['old_sha256']
 print(json.dumps({'host':plan['host'],'phase':'rolled-back'}))
