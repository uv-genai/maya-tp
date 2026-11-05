# GLM lifecycle checkpoint — G4

Issue [#147](https://github.com/uv-genai/maya-tp/issues/147), 2026-11-05.

The existing scheduler now preserves queued per-request speculation settings
and uses completed mirrored prompt calls for TP2 prefill/decode interleaving.
An unset interleave setting defaults to 256-token pieces in the opt-in batch
mode. Rank-0-only in-call gap callbacks are disabled in that mode. Failed
mirror sends stop prompt recovery before further device mutation.

A two-slot run exposed an additional first-call-order memory fault: the MLA
q/context scratch allocated for sliced trunk heads was reused by wider,
replicated NextN. HIP now tracks its byte capacity and grows it when necessary.
The original failure is retained in `g4-final-queued-mtp`, with the failing
binary frozen under `g4-final/`. The corrected binary is `g4-scratch/maya`,
SHA256 `d97fd8d5ca73e6bb90776db97d04349f01221b2bf77028f68984af2f34fcecbe`.

| Gate | Outcome |
| --- | --- |
| Staggered admission and 1,120-token prefill beside decode | Pass with MTP on/off; the MTP survivor emits 11 tokens during the long prefill. |
| Cancel one conversation, keep survivors | Pass; remaining requests reach their own budgets. |
| Four-token retirement, immediate refill, later prefix reuse | Pass; 32 prompt tokens reused. |
| Four requests with only two physical slots | Pass with the scratch fix; queued `spec=1` offers zero drafts. |
| Four-slot mixed sampling, penalties, bias and thinking closure | Pass with final scratch candidate. |
| Owned peer loss, MTP on/off | Every active route and a later GEN receive the same retained cause; fanout 1.37/1.64 seconds. |
| Build and host checks | HIP target, HIP mapping, Python harness compilation and whitespace pass. Exact-head CI is recorded on the issue. |

The reusable GPU driver is [validate_batch.py](../../tools/tp2/validate_batch.py).
It requires an explicit config/binary, records owned identities and writes
results under a unique output directory. Example from host 2:

```sh
.venv/bin/python tools/tp2/validate_batch.py --name unique-g4-run \
  --config maya-maya-s-v2-iq2_xxs-hip.json --engine /absolute/frozen/maya \
  --peer strix-halo-1.local --lead-address 192.168.1.92 \
  --seqs 2 --clients 4 --mtp 1 --scenario lifecycle
```

Use `--scenario peer-loss` only with an owned isolated pair: it deliberately
kills its recorded worker. Use `--mixed --seqs 4 --clients 4` for request-local
controls. This driver never inspects or stops unrelated services.

[Raw outputs and process metadata](tp2-batching-g4-2026-11-05.json).
Logs remain under `/tmp/maya-batching-validation/g4-*`. All owned model
processes have exited. Rollout performance and launcher acceptance remain G5.
