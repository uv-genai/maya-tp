#!/usr/bin/env python3
"""Verify root's final commit matches compiled inputs, sync runtime, freeze fixtures."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--commit',required=True);p.add_argument('--fixtures',nargs='*',default=[]);a=p.parse_args()
src=Path('/home/ugo/software/project-maya');rt=Path('/home/ugo/software/project-maya-upstream-20261111-runtime');build=Path('/home/ugo/software/project-maya-upstream-20261111-build');peer='strix-halo-1.local';dest=rt/'engines'/a.label
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==a.commit
assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=src,text=True).strip()
record=json.loads((dest/'source.json').read_text());old=json.loads(json.dumps(record));compiled={k:v for k,v in record['files'].items() if k.startswith(('src/','include/','cmake/')) or k=='CMakeLists.txt'}
for name,digest in compiled.items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,('compiled input changed',name)
for name in a.fixtures:
 relative='tests/'+name+'.cu'
 assert hashlib.sha256((src/relative).read_bytes()).hexdigest()==hashlib.sha256((rt/relative).read_bytes()).hexdigest(),relative
(dest/'build-source.json').write_text(json.dumps(old,indent=2)+'\n')
files=subprocess.check_output(['git','ls-files','-z'],cwd=src);listing=Path('/tmp/maya-upstream-20261111-final-files.z');listing.write_bytes(files)
for target in (str(rt)+'/',peer+':'+str(rt)+'/'):subprocess.run(['rsync','-a','--from0','--files-from='+str(listing),str(src)+'/',target],check=True)
record.update(source_commit=a.commit,tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=src,text=True).strip(),index_still_unmerged=[],build_snapshot_parent_head=old['source_commit'],compiled_inputs_verified_at=time.time(),compiled_input_count=len(compiled),build_snapshot_had_unmerged_index=bool(old.get('index_still_unmerged')),runtime_synced_commit=a.commit)
record['files']={name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in old['files']}
record['fixtures']={}
for name in a.fixtures:
 shutil.copy2(build/name,dest/name);subprocess.run(['scp','-q',str(dest/name),peer+':'+str(dest)+'/'],check=True)
 record['fixtures'][name]={'sha256':hashlib.sha256((dest/name).read_bytes()).hexdigest(),'source':'tests/'+name+'.cu','source_sha256':hashlib.sha256((src/'tests'/(name+'.cu')).read_bytes()).hexdigest()}
for name in ('source.json','runtime-source.json'):(dest/name).write_text(json.dumps(record,indent=2)+'\n')
(rt/('.source-'+a.label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
for name in ('source.json','runtime-source.json','build-source.json'):subprocess.run(['scp','-q',str(dest/name),peer+':'+str(dest)+'/'],check=True)
subprocess.run(['scp','-q',str(rt/('.source-'+a.label+'.json')),peer+':'+str(rt)+'/'],check=True)
hashes={**record['binaries'],**{k:v['sha256'] for k,v in record['fixtures'].items()}}
for line in subprocess.check_output(['ssh','-o','BatchMode=yes',peer,'sha256sum',*[str(dest/name) for name in hashes]],text=True).splitlines():
 digest,path=line.split();assert hashes[Path(path).name]==digest,(path,digest)
print(json.dumps({'label':a.label,'source_commit':a.commit,'compiled_inputs_verified':len(compiled),'runtime_synced':True,'both_hosts_identical':hashes},indent=2))
