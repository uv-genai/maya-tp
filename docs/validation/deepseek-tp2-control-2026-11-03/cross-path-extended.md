# Extended DeepSeek cross-path checks

Issues #110 and #116. The identical engine built at source `52cf519`
(SHA-256 `a0dcc1cb91e9e52fa365291d68eef8203af714d8222e77e49d72741a4331ddc2`),
model files, 32,768-token context, prompts, greedy sampling and 64-token output
limit were used on single-host and TP2 for XS and S. DSpark was on; lookup and
vision were off. The TP2 runs used 256-row chunked prefill and rank-sharded
expert placement; single-host runs used an 8,192 MiB reserve and capped expert
read-ahead at zero for memory headroom. Thus placement and reserve differ.

Twenty short, objective cases covered arithmetic, bit operations, sequences,
Python expressions, JSON, logic and units. The strict check parsed the requested
integer, array or object (accepting fenced JSON) or exact YES/NO. Outputs, prompts,
expected values, strict checks, usage and timings are recorded in the raw files.

| Model | Single-host strict | TP2 strict | Text-identical answers |
| --- | ---: | ---: | ---: |
| XS | 19/20 | 19/20 | 18/20 |
| S | 19/20 | 20/20 | 19/20 |

The single strict-format miss on each of XS single-host, XS TP2 and S
single-host was `45 & 27 = 9` rather than the requested bare integer `9`;
the value was correct. XS's two other cross-path text differences were JSON
whitespace and code fencing; the parsed objects were correct. S's only
cross-path text difference was the bitwise formatting. All owned processes
exited. [XS single](quality-extended-xs-single.json),
[XS TP2](quality-extended-xs-tp.json), [S single](quality-extended-s-single.json),
[S TP2](quality-extended-s-tp.json).

These are twenty short cases, one run per configuration; no general claim of
quality equivalence follows. Longer answers and full logits have diverged, as
recorded in [chunk review](chunk-review.md) and
[window/chunk current](window-chunk-current.md).
