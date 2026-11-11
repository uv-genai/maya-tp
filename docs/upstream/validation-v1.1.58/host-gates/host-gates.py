from pathlib import Path
import subprocess,os,yaml
root=Path('/home/ugo/software/project-maya'); out=Path('/tmp/maya-v58-host-gates');out.mkdir(exist_ok=True)
env={**os.environ,'RUNNER_TEMP':str(out)}
commands=[('hip',[str(root/'.venv/bin/python'),'tools/check_hip_compat.py']),('serve',[str(root/'.venv/bin/python'),'-m','unittest','discover','-s','serve','-p','test_*.py','-t','.']),('tools',[str(root/'.venv/bin/python'),'-m','unittest','discover','-s','tools','-p','test_*.py'])]
for name,cmd in commands:
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
for step in yaml.safe_load((root/'.github/workflows/tests.yml').read_text())['jobs']['python']['steps']:
 if '(CPU only)' not in step.get('name',''):continue
 with (out/(step['name'].split(' ')[0]+str(len(step['name']))+'.log')).open('w') as f:r=subprocess.run(['bash','-e','-c',step['run']],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(step['name'],r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
