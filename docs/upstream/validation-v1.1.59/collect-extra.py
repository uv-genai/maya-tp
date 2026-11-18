#!/usr/bin/env python3
"""v1.1.59 extra evidence: failed attempts (bounded), matched S2 repeat, and owned-identity audit for both."""
import gzip,json,shutil,subprocess,time
from pathlib import Path
raw=Path('/home/ugo/software/maya-upstream-20261117/validation')
dest=Path('/home/ugo/software/project-maya-upstream-v59-v65/docs/upstream/validation-v1.1.59')
peer='strix-halo-1.local';records=[]
def identities(folder,case):
 d=json.loads((folder/'ownership.json').read_text())
 if d.get('rank0_pid'):records.append({'case':case,'host':'strix-halo-2.local','pid':d['rank0_pid'],'start':d.get('rank0_start'),'exe':d.get('rank0_exe')})
 for c in d.get('context_controls',[]):records.append({'case':case+'-context','host':peer if c['host']=='peer' else 'strix-halo-2.local','pid':c['pid'],'start':c['start'],'exe':c['exe']})
 pi=folder/'peer-identity.json'
 if pi.exists():p=json.loads(pi.read_text());records.append({'case':case,'host':peer,'pid':p['pid'],'start':p['start'],'exe':p['exe']})
 return d
attempts={}
# attempt 1: f6bf54d1 engines, legacy-header tuning tables rejected on rank 1
a1=raw/'attempt1-tuning-header-mismatch';s=json.loads((a1/'v1.1.59-summary.json').read_text())
diffs=[{'model':x['model'],'slots':x['slots'],'label':x['label'],'inputs_equal':x['inputs_equal'],'first_difference':next((i for i,(p,q) in enumerate(zip(x['old_tokens'],x['new_tokens'])) if p!=q),None)} for x in s['differences']]
rej=[l for l in (a1/'v1.1.59-glm-s-slots1/rank1.log').read_text(errors='replace').splitlines() if 'prefill gemm' in l][:2] if (a1/'v1.1.59-glm-s-slots1/rank1.log').exists() else []
cases={}
for f in sorted(a1.glob('v1.1.59-*-slots*')):
 if f.is_dir() and (f/'results.json').exists():
  d=json.loads((f/'results.json').read_text());cases[f.name]={'success':d.get('success'),'sha256':d['sha256'],'rank0_exit':d.get('rank0_exit'),'ssh_exit':d.get('ssh_exit'),'peer_shutdown':d.get('peer_shutdown')}
  identities(f,'attempt1-'+f.name)
attempts['attempt1-tuning-header-mismatch']={'engines_commit':'f6bf54d109605f492ebec2e1a4517f625d7739e6','outcome':'GLM differs from v1.1.58 (13 labels), DeepSeek XS/S match; rank 1 rejected every legacy-header table on the peer checkout and used hipBLASEx; fixed by 147d2684','rank1_tuning_lines':rej,'differences':diffs,'cases':cases}
# attempt 2: peer host loss during ds4-xs-slots2 load
a2=raw/'attempt2-peer-host-loss/v1.1.59-ds4-xs-slots2';d=identities(a2,'attempt2-ds4-xs-slots2')
rows=[json.loads(l) for l in (a2/'context-peer.jsonl').read_text().splitlines() if l.startswith('{"time')]
with (dest/'attempt2-peer-context.jsonl.gz').open('wb') as out:
 with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as g:g.write((a2/'context-peer.jsonl').read_bytes())
attempts['attempt2-peer-host-loss']={'case':'v1.1.59-ds4-xs-slots2','engines_commit':'147d26845ee8121f34187bc8b3021cf774e752f6','case_started':d['started'],
 'peer_last_telemetry':rows[-1]['time'],'peer_last_mem_available_kib':rows[-1]['mem_available_kib'],'peer_min_mem_available_kib':min(r['mem_available_kib'] for r in rows),
 'peer_journal_last_entry':'2026-11-17 19:11:47 AWST (no panic/OOM/shutdown; kernel log empty for the window)','lead_kernel':'19:12:51 thunderbolt retimers and both XDomain hosts disconnected; hotplug worker blocked in thunderbolt_ibverbs peer removal until 19:24:08',
 'outcome':'peer host hard-stopped mid-load; user restarted it; same case later passed (0 differences)','telemetry':'attempt2-peer-context.jsonl.gz'}
(dest/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
# matched S2 repeat
rep=dest/'s2-mixed-repeat';rep.mkdir(exist_ok=True);summary={}
for kind in ('control','candidate'):
 f=raw/('perf-v59-s2-'+kind);t=rep/kind;t.mkdir(exist_ok=True)
 d=identities(f,'perf-v59-s2-'+kind)
 shutil.copy2(f/'results.json',t/'results.json');shutil.copy2(f/'peer-identity.json',t/'peer-identity.json')
 o=json.loads((f/'ownership.json').read_text());o.pop('cases',None);o.pop('stat_history',None);(t/'ownership.json').write_text(json.dumps(o,indent=2)+'\n')
 summary[kind]={'sha256':d['sha256'],'waves_aggregate_tok_s':[w['aggregate_tok_s'] for w in json.loads((f/'results.json').read_text())['transition_waves']]}
A=json.loads((raw/'perf-v59-s2-control/results.json').read_text());B=json.loads((raw/'perf-v59-s2-candidate/results.json').read_text())
summary['raw_tokens_equal']=len(A['cases'])==len(B['cases']) and all(x['ids']==y['ids'] and x['tokens']==y['tokens'] for x,y in zip(A['cases'],B['cases']))
summary['requests']=len(A['cases']);assert summary['raw_tokens_equal']
(rep/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
audit_code='''from pathlib import Path
import json,os
out=[]
for d in DATA:
 p=Path('/proc')/str(d['pid']);e=dict(d)
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
  e.update(owned_alive=(s==d['start'] and os.readlink(p/'exe')==d['exe']),status='same_start_identity' if s==d['start'] else 'pid_reused')
 except OSError:e.update(owned_alive=False,status='exited')
 out.append(e)
print(json.dumps(out))
'''
loc=[r for r in records if r['host']!=peer];rem=[r for r in records if r['host']==peer]
local=json.loads(subprocess.check_output(['python3','-'],input='DATA='+repr(loc)+'\n'+audit_code,text=True))
remote=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',peer,'python3','-'],input='DATA='+repr(rem)+'\n'+audit_code,text=True))
audit=json.loads((dest/'stopped-owned-processes.json').read_text());audit['extra_checked']=time.time();audit['extra_identities']=local+remote
assert not any(e['owned_alive'] for e in local+remote)
(dest/'stopped-owned-processes.json').write_text(json.dumps(audit,indent=2)+'\n')
print('extra evidence written; extra owned identities checked',len(local+remote),'all exited')
