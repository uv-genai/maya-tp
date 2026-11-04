import hashlib
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
tag, model, variant=sys.argv[1:4]
assert model in ('xs','s') and variant in ('baseline','candidate')
engine=runtime/'engines'/('baseline' if variant=='baseline' else tag)/'maya-ds4'
prefix=Path('/tmp/maya-upstream-'+tag+'-deepseek-'+model+'-'+variant)
state=Path(str(prefix)+'-state')
api_port=18250
base='http://127.0.0.1:'+str(api_port)
pack=Path('/home/ugo/software/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF')/('Maya-'+model.upper())/'maya-ds4'
settings={'pack':str(pack),'manifest':str(runtime/'manifests'/('maya-'+model+'.json')),
          'engine':str(engine),'model_label':'DeepSeek V4.1 Maya-'+model.upper(),
          'host':'192.168.1.92','peer':'strix-halo-1.local','peer_root':str(runtime),
          'port':18950,'rdma_device':'usb4_rdma5','context':262144,'expert_budget_mib':92160,
          'reserve_mib':1024,'cpu_threads':14,'prefill_chunk':256,'dspark':'on','suffix_draft':0,
          'env':{'MAYA_DS4_WARM_CAP_GB':'0','MAYA_DS4_ROUTER_CHECK':'1'}}
config=Path(str(prefix)+'-launch.json');config.write_text(json.dumps(settings,indent=2)+'\n')

def request(path, body=None, timeout=240, own_page=False):
    headers={'Content-Type':'application/json'}
    if own_page: headers['Origin']=base
    req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response: return response.status,json.load(response)
    except urllib.error.HTTPError as response:
        with response:return response.code,json.load(response)

def memory_available():
    local=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    peer=int(subprocess.check_output(['ssh','strix-halo-1.local',"awk '/^MemAvailable:/ {print $2}' /proc/meminfo"],text=True,timeout=10))
    return min(local,peer)

with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    probe.bind(('127.0.0.1',api_port))
log=Path(str(prefix)+'-supervisor.log').open('w',buffering=1)
process=subprocess.Popen([str(runtime/'.venv/bin/python'),str(runtime/'tools/ds4/tp2_launch.py'),
    'run',str(api_port),'--config',str(config),'--state-dir',str(state)],cwd=runtime,stdout=log,
    stderr=subprocess.STDOUT,start_new_session=True)
start=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
Path(str(prefix)+'-supervisor-owned.json').write_text(json.dumps({'pid':process.pid,'start':start,'command':process.args}))
stop=threading.Event();guard_error=[]
def guard():
    while not stop.wait(2) and process.poll() is None:
        try:
            available=memory_available()
            if available<3*1024*1024:
                guard_error.append('memory headroom below 3 GiB: '+str(available)+' KiB')
                process.terminate();return
        except (OSError,ValueError,subprocess.SubprocessError):pass
threading.Thread(target=guard,daemon=True).start()
result={'tag':tag,'model':model,'variant':variant,'engine_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),
        'settings':settings,'runs':[]}
try:
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('owned supervisor exited: '+str(prefix))
        try:
            code,health=request('/health',timeout=3)
            if code==200 and health.get('status')=='ok' and health.get('tp',{}).get('peer')=='up':break
        except OSError:pass
        time.sleep(.5)
    else:raise RuntimeError('startup timeout')
    assert not guard_error,guard_error
    assert health['tp']['world_size']==2 and health['tp']['prefill_chunk']==256,health
    result['health']=health
    prompt='Write an eBPF tutorial to explain how to use eBPF.'
    payload={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],
             'temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False},
             'strata_tune':{'suffix_draft':0,'spec':1}}
    for repeat in (1,2):
        code,response=request('/v1/chat/completions',payload)
        assert code==200 and response.get('choices'),response
        assert response['usage']['completion_tokens']==256,response
        assert response['timings'].get('draft_n',0)>0,response.get('timings')
        result['runs'].append({'repeat':repeat,'request':payload.copy(),'response':response})
        Path(str(prefix)+'-results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(model,variant,'repeat',repeat,response['timings']['predicted_per_second'],'tok/s',flush=True)
    plain_payload={**payload,'strata_tune':{'suffix_draft':0,'spec':0}}
    code,plain=request('/v1/chat/completions',plain_payload)
    assert code==200 and plain['timings'].get('draft_n',0)==0,plain
    assert plain['choices'][0]['message']['content']==result['runs'][-1]['response']['choices'][0]['message']['content'],'DSpark off/on text mismatch'
    result['speculation_off']=plain
    for endpoint,body in (('/api/turbo',{'on':True}),('/api/context',{'max_context':65536})):
        code,response=request(endpoint,body,own_page=True)
        assert code==400 and 'both ranks' in json.dumps(response),(endpoint,code,response)
    code,health=request('/health')
    assert code==200 and health.get('tp',{}).get('peer')=='up',health
    result['paired_restart_guards']='passed'
    directory=Path((state/str(api_port)/'latest').read_text().strip())
    result['run_directory']=str(directory)
    if variant=='candidate':
        text=(directory/'engine.log').read_text(errors='replace')
        assert re.search(r'router check: [1-9][0-9]* BF16 routers',text),'missing BF16 router equivalence check'
        result['router_check']='passed'
    if variant=='candidate':
        code,tokens=request('/tokenize',{'content':'hello','parse_special':False})
        assert code==200 and len(tokens['tokens'])==1,tokens
        code,decoded=request('/detokenize',{'tokens':tokens['tokens']})
        assert code==200 and decoded['content']=='hello',decoded
        biased={**payload,'messages':[{'role':'user','content':'Say hello.'}],'max_tokens':4,
                'logit_bias':{str(tokens['tokens'][0]):100}}
        code,answer=request('/v1/chat/completions',biased)
        assert code==200 and answer['usage']['completion_tokens']==4,answer
        assert answer['choices'][0]['message']['content']=='hello'*4,answer
        result['tokenize_roundtrip']={'tokens':tokens,'decoded':decoded}
        result['biased_speculative_response']=answer
    Path(str(prefix)+'-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DSpark off/on exact text, live drafts, paired guards and healthy follow-up passed',flush=True)
finally:
    stop.set()
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=100)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=10)
    log.close()
    print('Owned supervisor exited',process.returncode,flush=True)
if guard_error:raise RuntimeError(guard_error)
