# Handoff-Prompt — restart prompt for a fresh session

Give this prompt to a new session to resume the Maya TP2 work. It points at `Handoff.md` as the source of truth.

---

Read `/home/ugo/software/project-maya/Handoff.md` in full before doing anything — it is the live handoff and overrides anything you infer. Also read `AGENTS.md` (platform rule, GitHub workflow, git remotes).

**Project:** two-host tensor parallel (TP2) for GLM-5.3-Flash Maya-S on twin Strix Halo gfx1151 hosts over Thunderbolt RDMA. Repo `/home/ugo/software/project-maya`, GitHub `uv-genai/maya-tp`, branch `main`.

**Hard rules (non-negotiable):**
1. **HIP/ROCm gfx1151 Linux only.** No CUDA/NVIDIA work.
2. **RDMA only** (`usb4_rdma*`), target ~37 Gb/s / ~6 µs. No TCP data path.
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. **Track every workstream as its own GitHub issue** on `uv-genai/maya-tp`. Comment while making progress. Link commits `refs #N`.
5. Push only to `origin` (`uv-genai/maya-tp`). Never push `upstream` (`mw00/project-maya`).
6. **NEVER use nested SSH.** You are ON strix-halo-2. Use SSH only to reach strix-halo-1.
7. Python uses `.venv/bin/python`. Shell uses `jq`. Always kill old processes before starting new ones (`pkill -9 maya; pkill -9 -f server.py`).

**Current state:** TP2 slicing is done — +32% vs single-host (56.81 → 38.50 ms/tok), pair parity 100% (32/32), spec-follow 31/32 (1 near-tie). The benchmark (`tools/tp2/run_bench.sh`) works perfectly.

**THE OPEN PROBLEM (do this):** Serve-mode TP2 token mirroring. Only rank 0's GPU works; tokens never reach rank 1. The fix is gufo's `TpInstruction` pattern: put `SendControl`/`RecvControl` at every `model.forward()` call site in `prep_prompt` and `run_request` (see Handoff.md §10 for exact line numbers and gufo reference files). Do NOT put the mirror inside `forward()` — `prep_prompt` blocks before `forward()` is called.

**Reference:** gufo at `/home/ugo/projects/gufo`, especially `src/cli/serve/tp_executor.cpp` and `tp_control.cpp`.

Verify every step: kill old processes, deploy binary, check timestamp, then test with a curl request.

---

_When you make progress, update `Handoff.md` (§4 issue map, §5 what works, §10 the open problem) and comment on the relevant issue. This file is the restart point — keep it current._
