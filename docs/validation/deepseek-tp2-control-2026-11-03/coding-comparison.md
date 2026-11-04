# DeepSeek cross-path coding checks

Issues #110 and #116. XS and S used the same engine built from source
`52cf519` (SHA-256 `a0dcc1cb91e9e52fa365291d68eef8203af714d8222e77e49d72741a4331ddc2`),
the same model files, a 32,768-token context, the same prompts, greedy sampling
and a 512-token output limit. DSpark was on; lookup and vision were off. TP2
used 256-row prefill and rank-sharded placement; single-host used an 8,192 MiB
reserve and zero expert read-ahead for memory headroom. Placement and reserve
thus differ between paths. Each configuration ran once.

Four prompts requested Python functions for merging intervals without modifying
the input, checking balanced brackets, finding a substring, and run-length
encoding. The generated code was evaluated on 23 fixed cases per configuration,
including empty inputs, overlapping or touching intervals, nested and invalid
brackets, repeated substrings, whitespace and an input-mutation check.

| Model | Single-host tasks / cases | TP2 tasks / cases | Identical generated answers |
| --- | ---: | ---: | ---: |
| XS | 4/4 tasks, 23/23 cases | 4/4 tasks, 23/23 cases | 2/4 |
| S | 4/4 tasks, 23/23 cases | 4/4 tasks, 23/23 cases | 3/4 |

The differences are alternative implementations that passed the same fixed
cases; this small sample does not establish general coding quality or numerical
equivalence. Longer answers and full logits have diverged in earlier checks.
All owned servers and workers exited.

The [evaluator](coding-evaluate.py) runs each generated program with `bwrap`
network and filesystem isolation, a 3-second CPU limit, 512 MiB address-space
limit, 1 MiB output limit and 6-second wall timeout. It requires bubblewrap;
there is no unsandboxed fallback. It was checked against known-good functions,
a mutating interval implementation and an infinite loop. For example:

```sh
python docs/validation/deepseek-tp2-control-2026-11-03/coding-evaluate.py \
  docs/validation/deepseek-tp2-control-2026-11-03/coding-xs-tp.json
```

Raw prompts, generated answers and timings: [XS single](coding-xs-single.json),
[XS TP2](coding-xs-tp.json), [S single](coding-s-single.json),
[S TP2](coding-s-tp.json). Evaluator results: [XS single](coding-xs-single-evaluation.json),
[XS TP2](coding-xs-tp-evaluation.json), [S single](coding-s-single-evaluation.json),
[S TP2](coding-s-tp-evaluation.json).
