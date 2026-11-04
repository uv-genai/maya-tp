# Sampled lookup proposal provenance

Issue #113. A sampled DSpark proposal can leave its q distribution cached if
it is not verified. A later lookup proposal can reuse the same seed, counter
and sampling parameters, but its q is a point at the lookup token. The engine
now explicitly marks lookup proposals as point distributions, preventing the
verifier from treating an unrelated cached drafter q as the lookup's q.

On a checked-identity gfx1151 XS pair, the rebuilt test executable SHA-256
`cc1736ab4be89d97a8897f7aadb0ece8be02c250ac04872712fd94c04bae2814`
compared 1,000 direct target samples with 50 speculative samples per mode.
Sampled drafter q yielded p=0.6119 with 35/250 drafts accepted; point q with
penalties yielded p=0.5272 with 8/250 accepted. The new case deliberately left
a matching sampled q cached, then verified an independently selected lookup
point: p=0.5444, with 15/50 accepted. A negative control that intentionally
reused the stale q differed in 23 of 32 paired verdicts. Different verdicts
alone do not establish distributional bias; the comparison exercises proposal
provenance. These are small statistical smoke tests, not a proof of equivalence.

The same 50-trial test passed on Maya-S: sampled drafter q p=0.0129 with
26/250 drafts accepted, point q with penalties p=0.3652 with 15/250 accepted,
and lookup after stale q p=0.3873 with 15/50 accepted. The stale-q control
changed 19 of 32 verdicts. The preset rejection threshold was p < 0.001; these
small samples have limited power.

A larger Maya-S run compared 10,000 direct target draws with 500 speculative
draws per mode. Sampled drafter q gave p=0.6686 (345/2,500 drafts accepted),
point q with penalties p=0.9782 (174/2,500), and lookup after a cached sampled
q p=0.9369 (172/500). Each comparison used 11 grouped bins. The stale-q
control again changed 19 of 32 verdicts. These results cover the selected
prompt and sampling chains and do not establish general distributional
equivalence.

The same rebuilt test executable also forced each accepted count 0–15 across
the 128-token boundary on XS and S. All next logits matched exactly, and the
next five DSpark drafts matched in every case. Logs: [XS lead](rollback-point-xs-rank0.txt),
[XS worker](rollback-point-xs-rank1.txt), [S lead](rollback-point-s-rank0.txt)
and [S worker](rollback-point-s-rank1.txt).

All owned ranks exited. Logs: [XS lead](sampled-lookup-xs-rank0.txt),
[XS worker](sampled-lookup-xs-rank1.txt), [S lead](sampled-lookup-s-rank0.txt)
and [S worker](sampled-lookup-s-rank1.txt); the larger S run's
[lead](sampled-lookup-s-500-rank0.txt) and [worker](sampled-lookup-s-500-rank1.txt).
