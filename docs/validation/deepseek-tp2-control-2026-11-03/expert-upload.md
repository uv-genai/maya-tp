# DeepSeek TP2 rank-local expert upload

Issue #115. The HIP uploader validates the source and an aligned destination
layout, then copies a rank's gate/up rows and strided down blocks into a compact
device allocation. It has not been connected to the DeepSeek model loader.

On gfx1151, the `ds4_gguf_test --tp-slices` test compared uploaded synthetic
5120-by-2304 expert slices with CPU-packed bytes and with the full expert's
arithmetic. Both alternating 1280/1024 splits passed for IQ1_S/IQ1_S,
IQ1_S/IQ2_XXS, IQ2_S/IQ3_S and IQ2_XXS/IQ3_XXS. The test also checks short
buffers and overflowing layouts. A separate explicit-pack GPU test read expert
0 from layers 0 and 1 of both real XS and S packs and matched both ranks' device
bytes to CPU packing. No full model allocation or two-rank inference was tested.

The tested uploader's SHA-256 was
`b94843773fbef640142157998e7f651ea49fe6e79d9cf8a216d0fc8cac8baf66`.
