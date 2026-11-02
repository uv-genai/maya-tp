# DeepSeek Maya-XS API benchmark — 2026-11-02

With the user's permission, the existing DeepSeek V4.1 Flash Maya-XS server on one gfx1151 Strix Halo was measured via its OpenAI-compatible API. The model had a 262144-token context and DSpark enabled. Requests used temperature 0, thinking off, one warm-up answer, then the same three benchmark questions as upstream. A separate ~2K warm-up preceded fresh ~2K and ~8K prefill requests. The server was healthy and idle afterward; it was not restarted.

| Measurement | This host | Upstream report |
| --- | ---: | ---: |
| Python answer | 39.3 tok/s | 46.1 tok/s |
| Prose answer | 28.5 tok/s | 32.6 tok/s |
| HTML answer | 47.9 tok/s | 50.9 tok/s |
| Arithmetic mean, three answers | 38.6 tok/s | 43.2 tok/s |
| Prefill, ~2K tokens | 203.5 tok/s | 278 tok/s |
| Prefill, ~8K tokens | 298.1 tok/s | 453 tok/s |

All three measured answers produced 256 tokens. The API prefill prompts contained 2052 and 8196 tokens respectively, each with zero cached tokens; upstream's direct-engine benchmark used 2048 and 8192 tokens. These are single samples, and this is an API approximation, not an identical benchmark. The [upstream report](https://github.com/mw00/project-maya/issues/663) used orcarouter abliterated Mix-128G, 256K context and DSpark on a 192 GB Ryzen AI Max+ PRO 495 / 8065S under Windows. This host uses Maya-XS v2 on a 128 GB Ryzen AI Max+ 395 / 8060S under Linux. These differences mean the results do not identify a cause for the gap.

The saved `results.json` contains usage and timings, `output.txt` contains progress, and `bench_api.py` contains the synthetic prompts and corpus construction. No user conversation was saved. Earlier setup's DSpark calibration failed with a 600-second watchdog (#101); these requests passed with DSpark enabled but do not resolve that failure. Issue #100 tracks the performance comparison.
