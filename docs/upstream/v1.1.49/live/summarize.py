"""Audit completed probe artifacts and write a compact acceptance summary."""
import json
import statistics
from pathlib import Path

BASE = Path('/tmp/maya-v49-live-87')
MODELS = ('Maya-S-v2-IQ2_XXS', 'Maya-M', 'Maya-L')


def read(path):
    return json.loads((BASE/path).read_text())


api = []
for model in MODELS:
    for mode in ('mtp', 'simple'):
        path = model+'-'+mode+'-fixed/'
        matrix, cancel, bounds = [read(path+f) for f in ('matrix.json', 'prefill-cancel.json', 'boundaries.json')]
        assert len(matrix) == 5
        for row, limit in zip(matrix[:4], (64, 32, 24, 32)):
            assert 0 < row['usage']['completion_tokens'] <= limit
            assert (row['timings']['draft_n'] > 0) == (mode == 'mtp')
        assert matrix[2]['usage']['prompt_tokens_details']['cached_tokens'] >= 7000
        assert matrix[4]['chunks'] > 0
        progress = cancel['progress_at_disconnect']
        assert 0 < progress['processed'] < progress['total'] and cancel['retry_cached'] == 0
        assert cancel['health']['tp']['peer'] == 'up'
        assert len(bounds['rejected']) == 4 and all(r['status'] == 400 for r in bounds['rejected'])
        short, long = [matrix[i]['timings']['predicted_ms']/matrix[i]['timings']['predicted_n'] for i in (0, 3)]
        assert long/short <= 4
        api.append({'model': model, 'mode': mode, 'matrix_cases': 5,
                    'partial_prefill_progress': progress, 'retry_cached': 0,
                    'long_short_decode_ratio': long/short, 'maximum_ratio': 4,
                    'lifecycle_media_context_rejections': 4})
rounds, recovery = {}, {}
for model in MODELS:
    rounds[model] = read(model+'-mtp-fixed/round-audit.json')
    assert rounds[model]['asymmetric_retries'] == rounds[model]['matched_window_attempts'] == 8
    r = read(model+'-recovery-fixed/recovery.json')
    assert r['checks_passed'] == 8 and r['after']['status'] == 'degraded' and r['after']['tp']['peer'] == 'down'
    assert 'TP2 pair down' in r['peer_error']['message']
    # Peer death can first reach either the queued-collective drain or the existing mirror guard.
    assert any(c in r['peer_error']['message'] for c in ('verbs communicator is poisoned', 'mirror send failed'))
    assert r['recovered']['tp']['peer'] == 'up' and r['recovered_chat']['timings']['draft_n'] > 0
    recovery[model] = {'checks_passed': 8, 'first_error': r['peer_error']['message'],
                        'fresh_pair_prompt_ms': r['recovered_chat']['timings']['prompt_ms'],
                        'fresh_pair_decode_ms': r['recovered_chat']['timings']['predicted_ms']}
cli = []
for model in MODELS:
    for mode in ('simple', 'mtp'):
        d = read('cli-fixed/'+model+'-'+mode+'-result.json')
        assert d['pair_matches'] == 32 and d['rank0_ids'] == d['rank1_ids']
        if model == MODELS[0]:
            assert d['accepted_v42_exact']
        cli.append({'model': model, 'mode': mode, 'pair_matches': 32,
                    'accepted_v42_exact': d.get('accepted_v42_exact')})
strict = []
for label in ('S-technical', 'S-code', 'M', 'L'):
    d = read('strict-'+label+'/results.json')
    assert len(d['warmups']) == len(d['results']) == 4
    assert d['max_pick_differences'] == 0
    assert all(r['model_pick_differences'] == 0 for r in d['warmups']+d['results'])
    strict.append({'case': label, 'tokens': d['tokens'], 'drafts': d['settings']['MAYA_GLM_MTP_DRAFT'],
                   'warmups': 4, 'measurements': 4, 'maximum_raw_pick_differences': 0, 'summary': d['summary']})
prefill = read('prefill-ab/comparison.json')
assert len(prefill['pairs']) == 9 and all(r['messages_equal'] and r['usage_equal'] for r in prefill['pairs'])
assert read('prefill-ab/extra-api.json')['health']['tp']['peer'] == 'up'
reverse = read('prefill-ab-reverse/comparison.json')
assert reverse['order'] == ['baseline', 'candidate']
assert len(reverse['pairs']) == 9 and all(r['messages_equal'] and r['usage_equal'] for r in reverse['pairs'])
pooled = []
for first, second in zip(prefill['summary'], reverse['summary']):
    assert first['repeats'] == second['repeats']
    aa = first['baseline_samples_ms'] + second['baseline_samples_ms']
    bb = first['candidate_samples_ms'] + second['candidate_samples_ms']
    am, bm = statistics.median(aa), statistics.median(bb)
    pooled.append({'repeats': first['repeats'], 'samples_per_server': 4,
                   'baseline_samples_ms': aa, 'candidate_samples_ms': bb,
                   'baseline_median_ms': am, 'candidate_median_ms': bm,
                   'throughput_delta_percent': 100*(am/bm-1)})

normalized_configs = []
for prefix in ('prefill-', 'prefill-reverse-'):
    for arm, root in (('candidate', 'fixed-source'), ('baseline', 'control-source')):
        cfg = read(prefix+arm+'/config.json')
        cfg.pop('log')
        normalized_configs.append(json.dumps(cfg, sort_keys=True).replace(str(BASE/root), '<SOURCE>'))
assert len(set(normalized_configs)) == 1, 'prefill semantic settings differ'
assert read('runtime-preservation.json')['production_binary_unchanged']
ownership = read('ownership-final.json')
assert ownership['all_registered_processes_ended']
for side in ('rank0', 'rank1'):
    assert ownership[side] and all(r['stopped'] and not r['alive'] for r in ownership[side])
longloop = read(MODELS[0]+'-mtp-fixed/long-loop.json')
assert longloop['initial_tokens'] == 20021 and longloop['follow_cached'] == 20020 and longloop['collapsed'] == 0
out = {'accepted': True, 'merged_source': '997bba7cc7672c5635a2ab11f92972d4a30f33c6',
       'fixed_runtime_source': 'e1682cb8e0782fb2a1103ba4511734c12cd1a6b3',
       'baseline_server': 'd7626ac36637388517331c323a40adbf4b315f01',
       'engine_md5_both': 'b398d4a0711430edd6a28b95264c7293',
       'cpu': {'hip_map': 'PASS', 'serve': 613, 'serve_skips': 11, 'tools': 686, 'tools_skips': 35,
               'final_peer_loss_tests': 3},
       'api': api, 'rounds': rounds, 'recovery': recovery, 'cli': cli, 'strict': strict, 'prefill': prefill,
       'prefill_reverse': reverse, 'prefill_pooled': pooled,
       'text_parts_prefix': {'initial_tokens': 20021, 'follow_cached': 20020, 'collapsed': 0},
       'all_owned_processes_ended': True, 'prefill_normalized_configs_equal': True,
       'limits': ['No production deployment/tag/release', 'No DeepSeek full-model GPU acceptance',
                  'Cold/first-request outliers retained; prefill and latency issues33/60 remain open',
                  'Temporary owned harness adaptations do not resolve83; L CLI retains WARM0 workaround42']}
(BASE/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
print('PASS final API/CLI32/strict32records/raw0/prefill18pairs/ownership artifact audit')
