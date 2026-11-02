import hashlib,json,statistics,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/home/ugo/software/project-maya'); BASE=Path('/tmp/maya-ds4-bench-100'); URL='http://127.0.0.1:8000'
sys.path.insert(0,str(ROOT/'tools'))
import strata_tokenizer
cfg=json.loads((ROOT/'maya-ds4-hip.json').read_text()); tok=strata_tokenizer.Tokenizer.from_dir(Path(cfg['tokenizer']))
files=[ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md')),*sorted((ROOT/'bench/results').glob('*.md'))]
text='\n\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in files if p.exists())
ids=tok.encode(text)
assert len(ids)>=9189
items=[('warmup-answer','Say hello in five languages.',64,None),
 ('python','Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.',256,None),
 ('prose','Explain how photosynthesis works, step by step, for a high-school student.',256,None),
 ('html','Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.',256,None),
 ('warmup-prefill',tok.decode(ids[-2048:]),1,2048),
 ('prefill-2k',tok.decode(ids[:2048]),1,2048),
 ('prefill-8k',tok.decode(ids[997:997+8192]),1,8192)]
results=[]
for label,prompt,limit,source in items:
 with urllib.request.urlopen(URL+'/metrics',timeout=5) as r: state=json.load(r)
 assert state['engine'].get('dspark') and state['engine'].get('model_label')=='DeepSeek V4.1 · Maya-XS',state['engine']
 if state['live'].get('active') or state['live'].get('queued'):
  raise RuntimeError('Server busy before '+label+'; stopped benchmark without sending a request')
 body={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],
       'max_tokens':limit,'temperature':0,'top_p':1,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(URL+'/v1/chat/completions',json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 start=time.monotonic(); print('START',label,flush=True)
 with urllib.request.urlopen(req,timeout=660) as r: out=json.load(r)
 msg=out['choices'][0]['message']; usage=out['usage']; timing=out['timings']
 assert not msg.get('reasoning_content'),(label,'thinking unexpectedly present')
 row={'case':label,'source_tokens':source,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
      'wall_s':time.monotonic()-start,'usage':usage,'timings':timing,'finish_reason':out['choices'][0].get('finish_reason')}
 results.append(row); (BASE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print('PASS',label,json.dumps(row),flush=True)
dec=[x['timings']['predicted_per_second'] for x in results if x['case'] in ('python','prose','html') and x['usage']['completion_tokens']>=64]
assert dec
print('PASS benchmark; eligible answer mean tok/s',round(statistics.mean(dec),2),flush=True)
