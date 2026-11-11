#!/usr/bin/env python3
"""Snapshot explicitly approved source into isolated roots, build/freeze gfx1151."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('--source',type=Path,default=Path('/home/ugo/software/project-maya'))
p.add_argument('--runtime',type=Path,default=Path('/home/ugo/software/project-maya-upstream-20261111-runtime'))
p.add_argument('--build',type=Path,default=Path('/home/ugo/software/project-maya-upstream-20261111-build'))
p.add_argument('--peer',default='strix-halo-1.local')
p.add_argument('--jobs',type=int,default=6)
p.add_argument('--sync-only',action='store_true')
a=p.parse_args()
assert Path(a.label).name==a.label
files=subprocess.check_output(['git','ls-files','-z'],cwd=a.source)
listing=Path('/tmp/maya-upstream-20261111-files.z');listing.write_bytes(files)
for dest in (str(a.runtime)+'/',a.peer+':'+str(a.runtime)+'/'):
 subprocess.run(['rsync','-a','--from0','--files-from='+str(listing),str(a.source)+'/',dest],check=True)
record={'label':a.label,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip(),'tree':subprocess.check_output(['git','write-tree'],cwd=a.source,text=True).strip(),'created':time.time(),'files':{}}
for relative in files.decode().split('\0'):
 if relative and (a.source/relative).is_file() and (relative.startswith(('src/','include/','cmake/','serve/','tools/')) or relative in ('CMakeLists.txt','VERSION','setup.py','maya.py')):
  record['files'][relative]=hashlib.sha256((a.source/relative).read_bytes()).hexdigest()
source_file=a.runtime/('.source-'+a.label+'.json');source_file.write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(['scp','-q',str(source_file),a.peer+':'+str(source_file)],check=True)
print('SNAPSHOT',record['source_commit'],record['tree'],flush=True)
if a.sync_only:raise SystemExit(0)
environment={**os.environ,'HIP_PATH':'/opt/rocm/core'}
cmake=str(a.source/'.venv/bin/cmake')
prefix='/tmp/maya-upstream-20261111-'+a.label
with open(prefix+'-configure.log','w') as log:
 subprocess.run([cmake,'-S',str(a.runtime),'-B',str(a.build),'-DCMAKE_BUILD_TYPE=Release','-DSTRATA_ENABLE_HIP=ON','-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_PREFILL_MMQ=ON','-DSTRATA_ENABLE_TP2_RDMA=ON','-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_BUILD_TESTS=ON','-DCMAKE_HIP_ARCHITECTURES=gfx1151','-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++','-DCMAKE_PREFIX_PATH=/opt/rocm','-DSTRATA_GGML_DIR='+str(a.source/'third_party/llama.cpp')],stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
with open(prefix+'-build.log','w') as log:
 subprocess.run([cmake,'--build',str(a.build),'--target','maya','maya-ds4','maya-mm3','-j',str(a.jobs)],stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
destination=a.runtime/'engines'/a.label;destination.mkdir(exist_ok=True)
subprocess.run(['ssh','-o','BatchMode=yes',a.peer,'mkdir','-p',str(destination)],check=True)
record['binaries']={}
for binary in ('maya','maya-ds4','maya-mm3'):
 shutil.copy2(a.build/binary,destination/binary)
 subprocess.run(['scp','-q',str(destination/binary),a.peer+':'+str(destination)+'/'],check=True)
 record['binaries'][binary]=hashlib.sha256((destination/binary).read_bytes()).hexdigest()
record['build_complete']=time.time()
(destination/'source.json').write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(['scp','-q',str(destination/'source.json'),a.peer+':'+str(destination)+'/'],check=True)
remote=subprocess.check_output(['ssh','-o','BatchMode=yes',a.peer,'sha256sum',*[str(destination/b) for b in record['binaries']]],text=True)
for line in remote.splitlines():
 digest,path=line.split();assert record['binaries'][Path(path).name]==digest
print('FROZEN',json.dumps(record['binaries']),flush=True)
