#!/usr/bin/env bash
# Matched GLM repeat for the v1.1.65 checkpoint: frozen v1.1.64 (control) and v1.1.65 (candidate) engines alternate
# twice, on the one-slot case (speculative short decode) and the two-slot case with three mixed waves and the
# injected peer loss (mixed speed, first-fatal fanout). Outputs must stay identical across engines.
set -uo pipefail
H=/home/ugo/software/maya-upstream-20261117
R=/home/ugo/software/project-maya-upstream-20261111-runtime
PY=/home/ugo/software/project-maya/.venv/bin/python
cd /home/ugo/software/project-maya
for round in 1 2; do
  for kind in control:v1.1.64 candidate:v1.1.65; do
    k=${kind%:*}; lab=${kind#*:}
    for shape in g1 g2; do
      name=perf-v65-$shape-$k$round
      if [ $shape = g1 ]; then extra=(--slots 1); else extra=(--slots 2 --transition-waves 3 --fatal); fi
      "$PY" $H/case.py --name $name --family glm --model s --engine $R/engines/$lab/maya --tokens 64 "${extra[@]}" \
          > $H/validation/$name.log 2>&1 || { echo "FAIL $name (see $H/validation/$name.log)"; exit 1; }
      "$PY" - "$H/validation/$name/results.json" "$name" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); r = {x['label']: x for x in d['cases']}
f = lambda l: round(len(r[l]['tokens']) * 1000 / r[l]['last']['decode_ms'], 2)
waves = [round(w['aggregate_tok_s'], 2) for w in d.get('transition_waves', [])]
fan = d.get('failure_fanout_s')
print('DONE', sys.argv[2], d['sha256'][:12], 'short-on', f('short-on'), 'prefill-on', f('prefill-on'),
      'short-off', f('short-off'), 'mixed', waves, 'fanout', round(fan, 3) if fan else None, 'success', d.get('success'),
      flush=True)
EOF
    done
  done
done
"$PY" - <<'EOF'
import json
V = '/home/ugo/software/maya-upstream-20261117/validation/'
for shape in ('g1', 'g2'):
    runs = [json.load(open(V + 'perf-v65-%s-%s/results.json' % (shape, k)))
            for k in ('control1', 'candidate1', 'control2', 'candidate2')]
    ref = runs[0]['cases']
    eq = all(len(r['cases']) == len(ref) and all(x['ids'] == y['ids'] and x['tokens'] == y['tokens']
             and x.get('error') == y.get('error') for x, y in zip(ref, r['cases'])) for r in runs[1:])
    print('raw equal', shape, eq, len(ref), flush=True)
EOF
echo PERF-REPEAT-FINISHED
