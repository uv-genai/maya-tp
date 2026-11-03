# Experimental DeepSeek TP2 serving protocol

Issues #111, #113 and #114. An isolated `maya-ds4` build and experimental
rank launcher prepared independent local pack, partition and binary hashes,
then agreed them over the two-host control channel. Rank 0 ran the ordinary
stdin/stdout serving protocol; rank 1 ran the typed worker. Tensors used RDMA.

With the XS pack at 32K context and window prefill, two greedy requests for the
same 19-token eBPF prompt each produced the earlier target-only pair's 64-token
sequence. Each accepted 30 of 37 DSpark drafts. The second request exercised
reset and replay; rank 0 then requested shutdown and both ranks exited cleanly.
The first and second requests reported 316.9 ms and 185.8 ms prompt time, and
2378.2 ms and 2642.3 ms generation time respectively. These are single-run
functional observations, not a performance comparison.

The launcher and engine hooks remain experimental. This test does not cover
HTTP/API serving, idle watcher failure injection, long context, chunked
prefill, prompt lookup, vision, concurrent requests or mirrored checkpoints.
Those gates remain open; no production executable was replaced.
