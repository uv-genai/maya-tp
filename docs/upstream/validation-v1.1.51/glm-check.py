import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

runtime=Path('/home/ugo/software/project-maya-upstream-validation')
tag,variant=sys.argv[1:3]
assert variant in ('baseline','candidate')
sys.path.insert(0,str(runtime))
from serve.server import child_env
engine=runtime/'engines'/(os.environ.get('MAYA_UPSTREAM_BASELINE', 'baseline') if variant=='baseline' else tag)/'maya'
prefix=Path('/tmp/maya-upstream-'+tag+'-glm-s-'+variant)
api_port=18251
base='http://127.0.0.1:'+str(api_port)
original=json.loads(Path('/home/ugo/software/project-maya/maya-maya-s-v2-iq2_xxs-hip.json').read_text())
args=list(original['args'])
for flag,value in (('--max-context','32768'),('--parallel','1'),('--tp-world-size','2'),('--tp-rank','0'),('--tp-bootstrap-port','18951'),('--tp-rdma-device','usb4_rdma5')):
    if flag in args:args[args.index(flag)+1]=value
    else:args.extend([flag,value])
environment={**original.get('env',{}),'MAYA_TP_MTP':'1','MAYA_GLM_MTP_DRAFT':'2','MAYA_GLM_MTP_FIXED':'2','MAYA_GLM_TP_AR_TRACE':'1','MAYA_TP_BOOTSTRAP_TIMEOUT':'900'}
config={'exe':str(engine),'args':args,'cwd':str(runtime),'tokenizer':original['tokenizer'],
        'backend':'hip','gpu':[0],'model_name':'glm-5.3-flash','lib_dirs':original.get('lib_dirs',[]),
        'env':environment,'log':str(prefix)+'-engine.log'}
config_path=Path(str(prefix)+'-config.json');config_path.write_text(json.dumps(config,indent=2)+'\n')
worker_args=[str(engine)]+args.copy()
worker_args[worker_args.index('--tp-rank')+1]='1'
worker_args.extend(['--tp-bootstrap-host','192.168.1.92'])
worker_env=child_env({'env':environment,'backend':'hip','gpu':[0],'lib_dirs':config['lib_dirs']})
worker_env={key:value for key,value in worker_env.items() if key.startswith(('MAYA_','STRATA_','DS4_','HIP_','LD_LIBRARY_PATH'))}
worker_meta=str(prefix)+'-worker-owned.json'
def remote(code,data):
    script='data = '+repr(data)+'\n'+code
    return subprocess.check_output(['ssh','strix-halo-1.local','python3','-'],input=script,text=True,timeout=50)

start_code='''
import json,os,subprocess
from pathlib import Path
log=open(data['log'],'w')
p=subprocess.Popen(data['args'],cwd=data['root'],env={**os.environ,**data['env']},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
record={'pid':p.pid,'start':start,'exe':data['args'][0]}
Path(data['meta']).write_text(json.dumps(record))
print(json.dumps(record))
'''
stop_code='''
import json,os,signal,time
from pathlib import Path
record=json.loads(Path(data['meta']).read_text())
def owned():
 try:
  path=Path('/proc')/str(record['pid'])
  return (path/'stat').read_text().rsplit(')',1)[1].split()[19]==record['start'] and os.readlink(path/'exe')==record['exe']
 except OSError:return False
if owned():
 os.kill(record['pid'],signal.SIGTERM)
 for _ in range(200):
  if not owned():break
  time.sleep(.1)
 if owned():os.kill(record['pid'],signal.SIGKILL)
print('recorded worker stopped')
'''

def call(path,body=None,timeout=240):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as response:
        with response:return response.code,json.load(response)

def body(prompt,stream=False):
    return {'model':'glm-5.3-flash','messages':[{'role':'user','content':prompt}],'temperature':0,
            'max_tokens':256,'stream':stream,'chat_template_kwargs':{'enable_thinking':False}}

def wait_prefill(thread=None):
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        _,metrics=call('/metrics',timeout=10);live=metrics.get('live',{})
        read,total=live.get('prompt_read'),live.get('prompt_total')
        if live.get('state')=='reading' and read is not None and total and 0<read<total:return [read,total]
        if thread is not None and not thread.is_alive():raise RuntimeError('request ended before active prefill')
        time.sleep(.05)
    raise RuntimeError('no active prefill')

with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    probe.bind(('127.0.0.1',api_port))
worker_record=json.loads(remote(start_code,{'args':worker_args,'env':worker_env,'root':str(runtime),'log':str(prefix)+'-worker.log','meta':worker_meta}))
log=Path(str(prefix)+'-server.log').open('w',buffering=1)
process=subprocess.Popen([str(runtime/'.venv/bin/python'),'-m','serve.server','--engine','maya','--config',str(config_path),'--host','127.0.0.1','--port',str(api_port)],cwd=runtime,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
Path(str(prefix)+'-server-owned.json').write_text(json.dumps({'pid':process.pid,'start':start,'command':process.args}))
stop=threading.Event();guard_error=[];owned_engines=[]
def guard():
    while not stop.wait(2) and process.poll() is None:
        try:
            local=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
            peer=int(subprocess.check_output(['ssh','strix-halo-1.local',"awk '/^MemAvailable:/ {print $2}' /proc/meminfo"],text=True,timeout=10))
            if min(local,peer)<3*1024*1024:
                guard_error.append('memory guard below 3 GiB')
                process.send_signal(signal.SIGINT);return
        except (OSError,ValueError,subprocess.SubprocessError):pass
threading.Thread(target=guard,daemon=True).start()
result={'tag':tag,'variant':variant,'model':'GLM Maya-S','context':32768,'engine_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),'runs':[],'worker':worker_record}
try:
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('server exited')
        try:
            code,health=call('/health',timeout=3)
            if code==200 and health.get('status')=='ok' and health.get('tp',{}).get('peer')=='up':break
        except OSError:pass
        time.sleep(.5)
    else:raise RuntimeError('startup timeout')
    for value in Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split():
        path=Path('/proc')/value
        if os.readlink(path/'exe')==str(engine):
            owned_engines.append((int(value),(path/'stat').read_text().rsplit(')',1)[1].split()[19]))
    assert len(owned_engines)==1,owned_engines
    result['health']=health
    for repeat in (1,2):
        request=body('Write an eBPF tutorial to explain how to use eBPF.')
        code,response=call('/v1/chat/completions',request)
        assert code==200 and response['choices'][0]['message'].get('content'),response
        result['runs'].append({'repeat':repeat,'request':request,'response':response})
        print('GLM',variant,repeat,response.get('timings'),flush=True)
    engine_text=Path(str(prefix)+'-engine.log').read_text(errors='replace')
    assert re.search(r'glm mtp TP: rank=0 .*keep=[2-9]',engine_text),'no accepted MTP drafts'
    result['mtp_acceptance']='positive'
    long_prompt='First independent request. '+('Explain how Linux eBPF verification and maps work. '*600)
    connection=http.client.HTTPConnection('127.0.0.1',api_port,timeout=15)
    connection.request('POST','/v1/chat/completions',json.dumps(body(long_prompt,True)),{'Content-Type':'application/json'})
    result['cancelled_prefill']=wait_prefill();connection.close()
    code,response=call('/v1/chat/completions',body('Give one sentence defining eBPF.'))
    assert code==200 and response.get('choices'),response
    _,health=call('/health');assert health['tp']['peer']=='up',health
    result['cancel_followup']=response
    if variant=='candidate':
        replies=[]
        def active_request():
            try:replies.append(call('/v1/chat/completions',body('Second independent request. '+long_prompt)))
            except Exception as error:replies.append({'exception':str(error)})
        thread=threading.Thread(target=active_request);thread.start()
        result['peer_loss_prefill']=wait_prefill(thread)
        remote(stop_code,{'meta':worker_meta})
        thread.join(180);assert not thread.is_alive(),'request hung after peer loss'
        _,health=call('/health');assert health['tp']['peer']=='down',health
        original_cause=health['tp'].get('error','');assert 'peer closed' in original_cause,health
        code,later=call('/v1/chat/completions',body('Return only hello.'))
        assert code>=400 and original_cause in json.dumps(later),(code,later,original_cause)
        result['peer_loss']={'responses':replies,'health':health,'later':later}
    Path(str(prefix)+'-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GLM actual MTP, cancellation, follow-up and applicable peer-loss gates passed',flush=True)
finally:
    stop.set()
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
    for pid,expected in owned_engines:
        path=Path('/proc')/str(pid)
        try:
            if (path/'stat').read_text().rsplit(')',1)[1].split()[19]==expected and os.readlink(path/'exe')==str(engine):os.kill(pid,signal.SIGTERM)
        except OSError:pass
    print(remote(stop_code,{'meta':worker_meta}).strip(),flush=True)
    log.close()
    print('Owned API server exited',process.returncode,flush=True)
if guard_error:raise RuntimeError(guard_error)
