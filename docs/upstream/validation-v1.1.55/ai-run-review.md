# AI_RUN read-only numerical review (#166)

Reviewed source cc5ea59 and the same da47391e... binary with AI_RUN disabled/enabled, without GPU runs or source edits.

## Observed scope

The eight fixed requests have identical input IDs. Six short/mixed responses remain exact. Both long-prompt responses (ordinary and DSpark, 1441 inputs, 64 output IDs) share seven matching IDs with disabled mode, then differ at zero-based index 7 (the eighth ID), 15840 -> 42199. Ordinary and DSpark agree with each other in each mode. The enabled pair stays up and exits 0/0. No quality/reference/full-logit margin measurement is available; aligned later token differences do not measure quality.

## Capacity and candidate audit

Allocation and runtime use the same ai_run_enabled(ds4_env(...)) predicate. At 262K context AI=32 and enabled AIM=128. iq and bm grow with AIM, scores remain AI*(context+64). The additional query storage is 1,572,864 bytes; the additional block-map storage is 12,610,560 bytes. Disabled mode preserves the original allocation.

Runtime ssr never exceeds ss=context+64. air is bounded by AIM and the floor-rounded score capacity; air*ssr <= AI*ss. All five index_score/topk/cand_blocks consumers use ssr. The direct branch's nlast is at most floor((p0+T)/R), covered by keys+padding. The candidate-limited branch caps score columns at candidate_topk*candidate_block_size, which the keys calculation includes; candidate indices remain in the same C-row buffers/offsets and bm uses its original maximum-context stride with AIM rows. 3,030 CPU arithmetic spot checks across bounded contexts/rows/ratios/candidate variants found no capacity violation; this is not a device-memory proof.

The actual model has candidate_topk=2048 and block_size=8, so the candidate-restricted branch begins above 16,384 compressed positions. A 1441-token prompt never takes that branch. For later candidate-using layers, enabled ssr is at least 16,448 here, and 128*16,448=2,105,344 floats fits within 32*262,208=8,390,656 floats. No missing stride consumer, candidate cutoff, or underallocation was found.

## Plausible numerical mechanism and limits

The index-query GEMM changes row count from 32 to 128 before finish(quant=3) quantizes its query output. Dense gemm calls hipblasGemmEx through the HIP compatibility mapping, whose implementation may select a different reduction/kernel for a different shape. If hipBLASLt is compiled/enabled, 128 also crosses its explicit minimum of 64 rows, whereas 32 bypasses it (src/ds4/ds4_lt.hpp/.cu). Probe logs contain no hipBLASLt selection lines, so this review does not claim that Lt actually ran. Shape-dependent vendor-BLAS rounding before low-bit query quantization is plausible; a small input change can change quantization or a top-k near tie and later greedy picks. It is not established as the cause.

Existing synthetic index-score tests feed query arrays directly and compare packed/BF16 keys or TC/non-TC score kernels; they do not prove equivalence of the preceding 32-row and 128-row query GEMMs. No component capture exists here to distinguish numerical ordering from a latent device defect.

## Recommendation

Retain only as explicitly experimental, default-off, with the observed answer-change caveat and #166 remaining open. Both ranks must agree via the new feature bit and batch execution identity. Do not claim same answers, quality parity, reference correctness or a measured speed benefit from this probe. Current evidence finds no memory/control defect requiring a paired-only prohibition; ordinary/DSpark agreement and healthy termination support bounded experimental use, while the mechanism remains unproven. Default-off exact Flash validation remains the merge gate. Do not expand this integration into a broad quality campaign; a future bounded query capture/logit comparison can qualify the option separately.
