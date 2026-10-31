# Maya-TP

**Tensor parallel inference for very large MoE models on dual AMD Strix Halo systems.**

[![Tests](https://github.com/uv-genai/maya-tp/actions/workflows/tests.yml/badge.svg)](https://github.com/uv-genai/maya-tp/actions/workflows/tests.yml)

Maya-TP is a derived project built on [Project Maya](https://github.com/mw00/project-maya), focused on
**HIP/ROCm, gfx1151 and Linux**, with two Strix Halo APUs connected over **Thunderbolt/USB4 RDMA**.
It partitions model weights and computation across the pair to increase inference speed and make larger
models practical on local hardware.

The original Maya was created and led by **Peasant Smith ([@mw00](https://github.com/mw00))**, with
**[@needmorevram](https://github.com/needmorevram)** as co-maintainer and major contributor. Maya-TP builds on
their model engines, quantization work, expert memory management and serving stack. Development of this
derived project is tracked in [uv-genai/maya-tp](https://github.com/uv-genai/maya-tp).

## Why Maya-TP?

Very large MoE models put pressure on both memory capacity and memory bandwidth. A single APU has a finite
memory budget; streaming experts from slower storage can limit throughput. Two Strix Halo systems provide
more memory and two GPUs, but using them efficiently requires coordinated execution across machines.

Maya-TP adds that coordination to Maya:

- **More speed:** divide attention, expert and feed-forward computation between two APUs. The current
  technical-prompt control measures about **25.8 tokens/s on the pair versus 18.6 on one host**, roughly
  **39% higher throughput** with ordinary decoding.
- **More model capacity:** partition working weights across both systems, allowing model configurations
  that exceed one system's usable GPU memory. This opens the path to larger models and higher-quality
  quants. Replicated weights, context state and scratch buffers still consume memory on each host.
- **A transport suited to tensor parallelism:** model tensors travel over RDMA; TCP carries bootstrap,
  request coordination and token decisions. The measured Thunderbolt link reaches about 37 Gb/s for
  large transfers and 6 µs for tiny WRITE round trips; collective costs also include staging and driver work.

The current published model measurements use **GLM-5.3-Flash / Maya-S IQ2_XXS**. Larger GLM quant variants
are capacity targets enabled by pack-derived partitioning; they need their own fit and performance validation.

## What this project adds

- **Two-host GLM tensor parallelism:** attention heads, routed experts, shared experts and dense FFNs are
  partitioned using dimensions read from the model pack.
- **RDMA collectives and matching execution:** both ranks agree on collective shapes, prefill chunk sizes
  and verification retries. Large prompts stay within the transport's receive window.
- **MTP under TP2:** rank 0 coordinates draft inputs, verification, accepted tokens and stopping. Draft
  count is configurable; replicated draft-layer computation stays local while trunk partials are summed.
- **Serving and conversation reuse:** the Maya server drives the pair, including streaming, cached
  follow-ups, cancellation and thinking-budget handling. Peer health reports loss and recovery.
- **llama-swap integration:** one wrapper starts/stops the pair and forwards MTP draft count, capacity
  and context to both ranks.
- **Strix Halo kernel work and validation:** HIP optimizations for padded verification outputs,
  numerical parity checks, controlled benchmarks and API/peer-recovery regression harnesses.

Maya's browser interface and OpenAI- and Anthropic-compatible APIs remain the foundation for local chat
and coding-agent integrations. See [the TP2 runbook](docs/TP2-RUN.md) for operating the pair.

## Measured performance

Corrected engine, Maya-S IQ2_XXS, resident experts, 256-token answers, excluded warmups and balanced
simple/MTP run order, measured on 2026-10-30:

| Workload | Systems | Decode mode | Tokens/s |
| --- | ---: | --- | ---: |
| Technical | 1 | Simple | 18.55 |
| Technical | 2 | Simple | 25.82 |
| Technical | 2 | MTP, 1 draft | **28.60** |
| Code | 2 | Simple | 25.86 |
| Code | 2 | MTP, 2 drafts | **33.53** |

On the pair, MTP adds **10.8%** for the technical prompt with one draft and **29.6%** for code with two.
Acceptance depends on the workload: seven drafts slowed the technical case by 38.6%. One or two drafts
are useful starting points for the measured workloads.

Simple/MTP comparisons use the same continuation and agree on all 256 raw greedy picks, with zero measured
disk reads or CPU expert work. Single-host and TP2 controls use their own references because reduction
order can change greedy text. These are workload measurements, not a universal speedup guarantee.
The [full review, methodology and raw results](docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md)
include the draft-width sweep, single-host controls and kernel measurements.

## Current scope

| Area | Status |
| --- | --- |
| Platform | Two Strix Halo gfx1151 systems, Linux, HIP/ROCm, Thunderbolt/USB4 RDMA |
| Validated model | GLM-5.3-Flash / Maya-S IQ2_XXS, text inference |
| Larger GLM packs | Partition geometry comes from pack metadata; validate memory fit and results per pack |
| MTP | Opt-in, greedy TP2 verification, configurable fixed width of 1–8 drafts |
| Conversations | One active sequence; cached follow-ups and cancellation tested |
| Other model families | DeepSeek and MiniMax engines are inherited; this project's published TP2 validation covers GLM |
| Deferred work | [Multi-sequence TP2](https://github.com/uv-genai/maya-tp/issues/11), [vision](https://github.com/uv-genai/maya-tp/issues/16) and [gather-based reduction](https://github.com/uv-genai/maya-tp/issues/10); DFlash under TP2 remains gated |

## Set up the pair

Both hosts need a gfx1151-capable ROCm/HIP toolchain, C++20 compiler, CMake, Python 3.10+ with venv,
`jq`, libibverbs development files and a working `usb4_rdma*` device. Keep matching engine builds and
model packs on both systems. Use fast local storage for model files and initial loading.

Clone **this project** on each host and use Maya's installer to prepare the environment and model pack:

```sh
git clone https://github.com/uv-genai/maya-tp.git
cd maya-tp
./maya.sh --backend hip --setup
```

The installer starts the model when setup completes. Stop it before launching the TP2 pair. General
model/setup options are in [docs/SETUP.md](docs/SETUP.md) and [docs/MODELS.md](docs/MODELS.md).

For the custom two-host RDMA build, configure explicitly on each host:

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_ENABLE_TP2_RDMA=ON \
  -DSTRATA_BUILD_TESTS=OFF
cmake --build build-hip --target maya -j"$(nproc)"
```

The launchers in `tools/tp2/` currently use the development pair's presets:
`strix-halo-1.local` is rank 0/API server, and `strix-halo-2.local` is rank 1/worker.
Review their host/IP and path settings for your installation. Both hosts need the **same model pack**
(and a base `maya-<model>-hip.json` for serve, or let the rank scripts rewrite the default config).

**Choose the model** on every launcher with `--model NAME` / `MAYA_MODEL` (under
`MAYA_MODELS_DIR`, default `/home/ugo/software/Maya-data/models`), `--pack PATH` / `MAYA_PACK`,
or `--config PATH` / `MAYA_CONFIG` for a full base JSON. Default remains Maya-S.

Start the worker on rank 1, then the server on rank 0, with matching contexts and the **same model flags**:

```sh
# On rank 1, in its checkout:
CONTEXT=32768 bash tools/tp2/rank1.sh --model Maya-S-v2-IQ2_XXS
# or: MAYA_PACK=/path/to/pack  /  --pack /path/to/pack

# On rank 0, in its checkout:
CONTEXT=32768 bash tools/tp2/rank0.sh 18080 --model Maya-S-v2-IQ2_XXS
# explicit config still works:
MAYA_CONFIG=/path/to/maya-model-hip.json CONTEXT=32768 bash tools/tp2/rank0.sh 18080
```

Controlled pair bench (parity + ms/token) without serve:

```sh
bash tools/tp2/run_bench.sh                    # default Maya-S
bash tools/tp2/run_bench.sh --model Maya-M
```

Check `http://strix-halo-1.local:18080/health` for `tp.peer: up`. The API is at
`http://strix-halo-1.local:18080/v1`. After a peer or RDMA failure, restart both ranks.
The [runbook](docs/TP2-RUN.md) covers model selection, configuration, health checks, logs and recovery.

### MTP and llama-swap

Run the wrapper **on rank 0** after adapting its host/path presets:

```sh
./tools/llama-swap/tp2.sh run 18080 --mtp --drafts 1 --context 32768
./tools/llama-swap/tp2.sh run 18080 --model Maya-M --mtp --drafts 2 --context 32768
./tools/llama-swap/tp2.sh stop
```

For llama-swap, use the same `run ${PORT} ...` and `stop` commands as the model's `cmd` and `cmdStop`,
with `proxy: http://127.0.0.1:${PORT}`. The wrapper launches the worker over SSH and the local API server;
SSH access from rank 0 to rank 1 is required.

`--no-mtp` selects ordinary decoding. `--drafts N` sets both count and capacity on both ranks.
`--model` / `--pack` / `--config` select the pack and are forwarded to both ranks.
These flags belong to the wrapper. Direct engine/rank launches enable MTP with
`MAYA_TP_MTP=1 MAYA_GLM_MTP_FIXED=N MAYA_GLM_MTP_DRAFT=N` on both hosts, with `MAYA_GLM_NO_MTP` unset.
See [TP2 settings](docs/SETTINGS.md#glm-two-host-tensor-parallel-tp2) for runtime controls.

## Validation and development

Development and measurements are tracked in [Maya-TP issues](https://github.com/uv-genai/maya-tp/issues).
Reports should include the model/quant, both hosts' configuration, commit and binary hashes, context,
draft width, cache/disk counters and reproduction steps. `./maya.sh --report` collects local diagnostics.

GitHub CI runs CPU-only checks: the HIP compatibility-name gate, server tests and installer/tools tests.
GPU parity, large-prompt API, cancellation, forced retries and peer death/recovery are exercised on the
Strix Halo pair. Evidence is saved in the [MTP review](docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md)
and the [collective regression report](docs/validation/tp2-collectives-2026-10-30/RESULTS.md).

Useful references:

- [TP2 operations](docs/TP2-RUN.md) and [runtime settings](docs/SETTINGS.md)
- [HIP setup](docs/AMD_MAYA.md), [model files](docs/MODELS.md) and [Responses API](docs/RESPONSES.md)
- [Contributing](CONTRIBUTING.md), [changelog](CHANGELOG.md) and [releases](https://github.com/uv-genai/maya-tp/releases)

Some inherited documentation describes upstream Maya's broader platform and model support; the validated
Maya-TP scope is listed above. Release **v1.1.39** includes the corrected TP2/MTP computation and
measurements described here; consult the changelog when selecting a revision.

## Credits and license

Maya-TP is a derivative of **[Project Maya](https://github.com/mw00/project-maya)**. Credit for the original
project, model engines and Maya quants belongs to **Peasant Smith ([@mw00](https://github.com/mw00))**,
**[@needmorevram](https://github.com/needmorevram)** and the
[Project Maya contributors](https://github.com/mw00/project-maya/graphs/contributors).
Their work makes this project's focus on dual Strix Halo tensor parallelism possible.

- **Built on [Strata](https://github.com/Niko1221/Strata)** (MIT License, Copyright (c) 2026 Niko1221 and the Strata
  contributors): Maya's engine started from Strata's and was rewritten for GLM-5.3-Flash (the expert tiers across VRAM,
  RAM and SSD, the layer split, MTP decoding), and its server and dashboard grew from Strata's. **And on
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml
  authors): the quantization formats, the CPU dot products and the prompt path's MMQ kernels, built from a pinned
  commit (`third_party/ggml/LICENSE`).
- **EXL3 support** follows [exllamav3](https://github.com/turboderp-org/exllamav3)'s format, codebooks, CPU expert
  kernel and tensor-core decoding of a tile (MIT License, Copyright (c) 2025 Turboderp; `third_party/exllamav3/LICENSE`). **The DFlash2 drafter's
  forward** follows [beellama.cpp](https://github.com/Anbeeld/beellama.cpp)'s (MIT License, Copyright (c) 2023-2026
  The ggml authors, Anbeeld; `third_party/beellama.cpp/LICENSE`).
- The models: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by zai-org (MIT); Maya's quants and the
  image encoder file are made from zai-org's released weights and keep its license.
  [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by DeepSeek (MIT); Maya's
  quant, its Engram tables and the drafter are made from DeepSeek's released weights and keep its license; the vision
  tower comes from DeepSeek's checkpoint, the GGUF option from
  [unsloth's GGUFs](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF), the EXL3 options are
  LibertAIDAI's ([2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) and
  [3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)). The optional DFlash2 drafter
  ([GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF) by Anbeeld) is CC BY-NC-ND
  4.0, non-commercial use only, and is downloaded only when you choose it.
- The dashboard's font: Outfit (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`); its icons:
  [Lucide](https://lucide.dev) (ISC License, `serve/web/LICENSE.lucide.txt`); its UI library:
  [Preact](https://preactjs.com) (MIT License) with [htm](https://github.com/developit/htm) (Apache License 2.0),
  bundled in `serve/web/preact.js` with their notices.
- Maya-TP is distributed under the [MIT License](LICENSE); the original copyright notices and those of
  Strata, ggml, exllamav3 and beellama.cpp stay with every copy.
