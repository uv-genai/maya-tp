# Maya-TP security reporting and scope

Report issues for this fork in `uv-genai/maya-tp`. For a security-sensitive
report, use the repository's private reporting option when available; otherwise
contact its owner privately before posting sensitive reproduction details.
Describe the source/release, selected binaries/configs and affected host roles.
No response-time or backport commitment is implied by this inherited guide.

This fork targets HIP/ROCm gfx1151 on Linux. Its qualified two-host paths are
GLM-5.3-Flash TP2/MTP and DeepSeek-V4.1-Flash TP2/DSpark. Broader upstream
platform/model guides do not qualify those configurations here. Model weights
retain their authors' licenses; include the model/quant when reporting a problem.

The standalone server defaults to loopback; explicit config/launcher bind
settings can expose it to other hosts. API-key and CORS controls are documented
in [runtime settings](docs/SETTINGS.md). Pair operation also uses configured
peer SSH/control/bootstrap sockets and APU RDMA tensor transfers. Treat both
configured hosts as part of the inference system.

Setup may contact Hugging Face for explicitly selected model assets, PyPI and
ROCm package sources for dependencies, and GitHub for source/update checks.
Requests may also name external image/file URLs or configured MCP endpoints.
There is no added usage telemetry in Maya-TP. Prompts/answers can travel to the
configured pair or tools; they are not guaranteed to remain on one PC.
`./maya.sh --report` writes a local diagnostic report rather than uploading it.

Do not automatically update or restart an active user pair while reporting.
Agents follow the process-ownership and protected-checkout rules in
[AGENTS.md](AGENTS.md); production changes require the task's authorization.
