import json
from pathlib import Path
import time
import urllib.error

import pair
from pair import Pair, request


def capture(path, body=None):
    try:
        with request(path, body, timeout=60) as r:
            return {'status': r.status, 'body': json.load(r)}
    except urllib.error.HTTPError as e:
        with e:
            return {'status': e.code, 'body': json.load(e)}


for arm, source in [('control-rerun', 'control-source')]:
    pair.ROOT = pair.BASE / source
    p = Pair('Maya-S-v2-IQ2_XXS', 'mtp', 'peer-' + arm + '-delay1', deep_idle=False)
    p.server['argv'][p.server['argv'].index('--engine') + 1] = 'strata'
    result = {'arm': arm, 'source': str(pair.ROOT), 'delay_after_kill_s': 1}
    try:
        p.start()
        result['before'] = capture('/health')
        result['initial_chat'] = capture('/v1/chat/completions', {'messages': [
            {'role': 'user', 'content': 'Explain tensor parallel RDMA in one paragraph.'}],
            'max_tokens': 32, 'temperature': 0})
        p.command('rank1', 'kill')
        time.sleep(1)
        result['requests'] = []
        for _ in range(3):
            out = capture('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Say hello.'}],
                                                   'max_tokens': 8})
            result['requests'].append({'response': out, 'health': capture('/health'),
                                        'owned_processes': p.identities()})
            (p.out / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        p.stop()
