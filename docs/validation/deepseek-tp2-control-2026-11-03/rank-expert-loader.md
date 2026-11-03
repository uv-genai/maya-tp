# DeepSeek pack-backed rank expert loader

Issue #115, 2026-11-03. The isolated loader uses the engine's pack parser,
validates each selected expert's type, shape and mapped file bounds, derives the
two-rank block ranges, and loads the requested rank's packed expert slices into a
bounded GPU pool. It exposes entries only after every selected slice is uploaded.
It is not connected to model execution and does not load dense weights, attention,
KV, Engram or the drafter.

The real-pack test rejected an invalid rank, insufficient budget, an unsupported
tensor type and an out-of-file tensor offset. It loaded 385 experts from both XS
and S on each rank assignment, covering the layer boundary where the larger band
switches ranks; the first and last GPU slices matched independently reconstructed
source bytes. Allocation was 1,477,959,680/1,183,897,600 bytes for XS rank 0/1,
and 2,699,141,120/2,161,838,080 bytes for S rank 0/1. These are subset figures.

On the two gfx1151 hosts, the same review binary then loaded **all 15,360 XS
trunk experts** per rank with a 60 GiB budget. Rank 0 allocated 54,814,310,400
GPU bytes and rank 1 allocated 54,751,395,840; each used a 4,249,600-byte pinned
staging buffer. The first and final expert slices matched independent source
reconstruction. Wall times were 34.29 and 40.06 seconds in this run, with
different cache conditions and a concurrent model-file transfer; they are not
production model load times. Logs: `rank-load-xs-rank0.txt` and
`rank-load-xs-rank1.txt`. Both owned test processes exited.

The same test then loaded all 15,360 S expert slices for each rank, sequentially
on the source host. Each rank's GPU allocation was 84,509,982,720 bytes with
7,014,400 bytes of pinned staging. First and final slices matched source bytes;
wall times were 56.37 and 53.87 seconds under the test's cache and transfer
conditions. See `rank-load-s-rank0.txt` and `rank-load-s-rank1.txt`. Both owned
processes exited. The second host's S pack was still being copied.

The XS and S expert totals agree with the earlier logical estimates, but do not
establish complete rank fit or inferencing. An isolated single-host XS DSpark
verification regression also passed after the loader was added.
