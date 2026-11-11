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

The next upstream update batch (v1.1.54–v1.1.58) is being integrated in
[#159](https://github.com/uv-genai/maya-tp/issues/159), with paired GLM-5.3-Flash and DeepSeek-V4.1-Flash
validation before the final PR. Production selections remain unchanged.

## Why Maya-TP?

Very large MoE models put pressure on both memory capacity and memory bandwidth. A single APU has a finite
memory budget; streaming experts from slower storage can limit throughput. Two Strix Halo systems provide
more memory and two GPUs, but using them efficiently requires coordinated execution across machines.

Maya-TP adds that coordination to Maya:

- **More speed:** divide attention, expert and feed-forward computation between two APUs. The 2026-10-30
  technical-prompt control measures about **25.8 tokens/s on the pair versus 18.6 on one host**, roughly
  **39% higher throughput** with ordinary decoding.
- **More model capacity:** partition working weights across both systems, allowing model configurations
  that exceed one system's usable GPU memory. This opens the path to larger models and higher-quality
  quants. Replicated weights, context state and scratch buffers still consume memory on each host.
- **A transport suited to tensor parallelism:** model tensors travel over RDMA; TCP carries bootstrap,
  request coordination and token decisions. The measured Thunderbolt link reaches about 37 Gb/s for
  large transfers and 6 µs for tiny WRITE round trips; collective costs also include staging and driver work.

The published simple/MTP speed tables use **GLM-5.3-Flash / Maya-S IQ2_XXS**. **TP2 and MTP have been
validated end-to-end on Maya-S, Maya-M and Maya-L** (same API matrices, acceptance, CLI parity and
controlled bench). Reports: [Maya-S](docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md),
[Maya-M](docs/benchmarks/tp2-maya-m-mtp-2026-10-31/RESULTS.md),
[Maya-L](docs/benchmarks/tp2-maya-l-mtp-2026-10-31/RESULTS.md).

The user tested the prefill candidate on Maya-S, Maya-M and Maya-L through llama-swap and reported
working prefill and generation across all three. Their eBPF tutorial prompt averaged 30, 32 and
28 generation tokens/s respectively. [Results and validation limits](docs/benchmarks/tp2-prefill-2026-11-02/README.md).

**DeepSeek-V4.1-Flash Maya-XS and Maya-S also support TP2 with DSpark.** Queued
collectives measured 23–24% higher XS generation throughput than the same
single-host build on two prompts. The user reports Maya-S eBPF generation
slightly above 30 tok/s with prefill similar to XS, and hours of successful
DeepSeek and GLM use including DeepSWE benchmarks.
[DeepSeek measurements and user results](docs/benchmarks/deepseek-tp2-queued-2026-11-04/README.md).

## What this project adds

- **Two-host GLM tensor parallelism:** attention heads, routed experts, shared experts and dense FFNs are
  partitioned using dimensions read from the model pack.
- **Two-host DeepSeek tensor parallelism and DSpark** partition target computation,
  coordinate speculative verification and use GPU-queued RDMA collectives.
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

The 2026-10-30 simple/MTP comparisons used the same continuation and agreed on all 256 raw greedy picks, with zero measured
disk reads or CPU expert work. Single-host and TP2 controls use their own references because reduction
order can change greedy text. These are workload measurements, not a universal speedup guarantee.
The [full review, methodology and raw results](docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md)
include the draft-width sweep, single-host controls and kernel measurements.

**Maya-M** (same harness, 2026-10-31): simple median **41.12 ms/tok (24.32 tok/s)** vs two-draft MTP
**38.03 (26.29)** — **+8.11%** throughput on the 256-token reference, with pair parity and the full
MTP/API suite passing. Report:
[tp2-maya-m-mtp-2026-10-31](docs/benchmarks/tp2-maya-m-mtp-2026-10-31/RESULTS.md).

## Current scope

The v1.1.50 integration adds the GLM Maya-XS installer option, tokenizer API,
model-management APIs and UI controls. Model management stays opt-in. Settings
that restart a TP2 engine require the pair launcher, including Turbo. GLM
Maya-XS has installer coverage; the recorded paired model validation covers
the models listed below. [Integration checkpoint](docs/upstream/v1.1.50.md).

Upstream v1.1.50–v1.1.53 is integrated through separate sequential PRs with
isolated gfx1151 GLM TP2/MTP and DeepSeek TP2/DSpark validation, ending at
`de184d9` through PR #134 with merged-main CI passing. The later
checkpoints add serving, model-storage/UI, packing and GPU-fault handling fixes.
The new optional engine experiments are tracked for separate assessment in #133.
[Last merged-main upstream checkpoint](docs/upstream/v1.1.53.md).
The active v54–v58 branch includes full GLM-5.3 source/catalog additions, while
remote TP2 remains **Flash-only** and rejects the 744B architecture before
bootstrap. Full-model support is unqualified here; no 744B model is downloaded
or tested. Its new Q8 attention repacking and single-APU drafting calibration do
not repack Flash models or change the accepted Flash launch defaults. New
refusal-ablation controls are limited to unpaired engines: TP2 refuses their
startup/config settings and every explicit `ablate` request, including `false`,
and exposes no ablation UI capability. The branch also adds
Responses compatibility and inline document inputs;
text extraction works within paired text-only limits. Images/video remain outside
the validated TP2 scope. [v54](docs/upstream/v1.1.54.md), [v55](docs/upstream/v1.1.55.md),
[v56](docs/upstream/v1.1.56.md), [v57](docs/upstream/v1.1.57.md),
[v58](docs/upstream/v1.1.58.md). The final branch is still under validation;
source integration does not change production binaries or model selections.

DeepSeek TP2 and DSpark were merged into `main` through PR #122 at `75e18e8`
after extended user trials. HIP mapping, server and tools CI passed on the
feature head and merged commit. [Merged pull request](https://github.com/uv-genai/maya-tp/pull/122).

The `main` branch includes the sequential semantic integration of upstream **v1.1.37–v1.1.46** and
the cumulative **v1.1.47–v1.1.49** merge, with TP2/MTP preservation checks and CI. These changes are included in
**[tp-v1.1.44](https://github.com/uv-genai/maya-tp/releases/tag/tp-v1.1.44)**,
the first published fork release using the `tp-vMAJOR.MINOR.PATCH` naming scheme.
Fresh S/M/L ordinary/MTP API, CLI, strict historical-reference and peer-recovery checks passed after
the cumulative upstream merge on its then-current binary. The later prefill build was accepted in user
trials; its stricter fidelity limitations are documented in the [prefill report](docs/benchmarks/tp2-prefill-2026-11-02/README.md).
The [upstream v1.1.49 merge report](docs/upstream/v1.1.49.md) and
[earlier integration report](docs/upstream/COMPLETION-2026-11-02.md) distinguish accepted changes from
deferred DeepSeek work, including [WMMA parity](https://github.com/uv-genai/maya-tp/issues/68),
[tail draining](https://github.com/uv-genai/maya-tp/issues/75) and
[chunk-cap measurement](https://github.com/uv-genai/maya-tp/issues/79).

| Area | Status |
| --- | --- |
| Platform | Two Strix Halo gfx1151 systems, Linux, HIP/ROCm, Thunderbolt/USB4 RDMA |
| Validated models (TP2 + MTP) | **Maya-S**, **Maya-M**, **Maya-L** — API matrices, CLI pair parity, kill/recover, controlled bench ([S](docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md), [M](docs/benchmarks/tp2-maya-m-mtp-2026-10-31/RESULTS.md), [L](docs/benchmarks/tp2-maya-l-mtp-2026-10-31/RESULTS.md)) |
| Larger GLM packs | Partition geometry comes from pack metadata; validate memory fit and results per pack before claiming support |
| MTP | Opt-in, configurable draft width; legacy single-slot flow is greedy, batching preserves per-request sampling and speculation controls |
| Conversations | One sequence by default; explicit `--parallel 2..4` enables checked continuous batching for GLM/MTP and DeepSeek/DSpark; cached follow-ups and cancellation tested |
| Model selection | Launchers take `--model` / `--pack` / `--config` (or `MAYA_*` env) — [choosing a model](docs/TP2-RUN.md#choosing-a-model) |
| DeepSeek TP2 + DSpark | Maya-XS and Maya-S have paired validation and successful user trials including coding workloads. [Operations and validation scope](docs/DEEPSEEK-TP2-RUN.md). |
| MiniMax TP2 | Outside the current validation scope |
| Deferred work | [vision](https://github.com/uv-genai/maya-tp/issues/16) and [gather-based reduction](https://github.com/uv-genai/maya-tp/issues/10); DFlash under TP2 remains gated |

DeepSeek launchers are available as `tools/ds4/rank0.sh`,
`tools/ds4/rank1.sh` and `tools/llama-swap/tp2-ds4.sh run PORT --model NAME`.
Per-model configs and the PID-scoped llama-swap stop command are documented in
the DeepSeek guide. [DeepSeek launcher setup](docs/DEEPSEEK-TP2-RUN.md#rank-scripts-and-llama-swap).

## Set up the GLM pair

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
  -DSTRATA_ENABLE_CUDA=OFF \
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
GLM continuous batching is opt-in with `--parallel 2`–`4` on the paired wrapper,
with matching per-slot context and real MTP (`--mtp --drafts 2`). The default
remains one conversation. The wrapper owns its pair and stops by port or
recorded supervisor PID. See [the batch runbook](docs/TP2-RUN.md#opt-in-glm-tp2-continuous-batching)
and [TP2 settings](docs/SETTINGS.md#glm-two-host-tensor-parallel-tp2).

DeepSeek XS/S continuous batching is also opt-in with `--parallel 2`–`4` on
its pair wrapper, using matching current `maya-ds4` binaries and launchers.
DSpark remains active through fitting drafter groups and complete per-slot
verdicts. The default is one conversation. See [DeepSeek batching and runtime
selection](docs/DEEPSEEK-TP2-RUN.md#opt-in-continuous-batching); existing pinned
production binaries/configs require explicit selection of the new build.
The user has tested and accepted this mode; #11/#140 are closed. See the
[1.1.44 release notes](docs/releases/tp-v1.1.44.md) for measured scope and upgrade details.

## Validation and development

Development and measurements are tracked in [Maya-TP issues](https://github.com/uv-genai/maya-tp/issues).
Reports should include the model/quant, both hosts' configuration, commit and binary hashes, context,
draft width, cache/disk counters and reproduction steps. `./maya.sh --report` collects local diagnostics.

GitHub CI runs the HIP compatibility-name gate, server and installer/tools tests,
plus CPU memory-policy and row-preparation/stub checks. The latest source
checkpoint passed paired GLM Maya-S and DeepSeek XS/S controls. Its bounded
measurements and model coverage are in the [v1.1.53 report](docs/upstream/v1.1.53.md).
GPU parity, large-prompt API, cancellation, forced retries and peer death/recovery were exercised on the
Strix Halo pair for **Maya-S, Maya-M and Maya-L** (simple and MTP) on earlier builds. Evidence:

- [Maya-S MTP review](docs/benchmarks/tp2-mtp-review-2026-10-30/RESULTS.md)
- [Maya-M TP2/MTP suite](docs/benchmarks/tp2-maya-m-mtp-2026-10-31/RESULTS.md)
- [Maya-L TP2/MTP suite](docs/benchmarks/tp2-maya-l-mtp-2026-10-31/RESULTS.md)
- [Collective regression report](docs/validation/tp2-collectives-2026-10-30/RESULTS.md) (Maya-S)

Useful references:

- [Documentation index and scope](docs/README.md)
- [TP2 operations](docs/TP2-RUN.md) and [runtime settings](docs/SETTINGS.md)
- [HIP setup](docs/AMD_MAYA.md), [model files](docs/MODELS.md) and [Responses API](docs/RESPONSES.md)
- [Contributing](CONTRIBUTING.md), [changelog](CHANGELOG.md) and [releases](https://github.com/uv-genai/maya-tp/releases)

Some inherited documentation describes upstream Maya's broader platform and model support; the validated
Maya-TP scope is listed above. Release **v1.1.42** improves controlled Maya-S TP2 prefill throughput by
32–37% through worker configuration and gfx1151/hipBLASLt 100401 tuning. Cold Maya-L delays and the
prefill row cap remain limitations; #33 was closed after user acceptance. Consult the
[changelog](CHANGELOG.md) and [latest prefill report](docs/benchmarks/tp2-prefill-2026-11-02/README.md).

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
  [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) by zai-org (its own license); its GGUF is
  [unsloth's](https://huggingface.co/unsloth/GLM-5.3-GGUF).
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
