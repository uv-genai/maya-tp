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
7. Python uses `.venv/bin/python`. Shell uses `jq`. Always kill old processes before starting new ones (`pkill -9 maya`; do **not** `pkill -f server.py` — that matches the ssh command line).

**Current state:** TP2 slicing done (+32%), bench perfect. **Serve-mode token mirroring FIXED** (2026-10-30): rank 0 sends TpMirrorHdr before every collective call; rank 1 worker executes it. Two curl chat/completions verified.

**NEXT (issue #6 remaining):**
1. `/health` pair-up: report peer down when rank 1 dies (today it only checks rank 0).
2. Kill rank 1 → clean pair-down; restart both → recover.
3. Optional: multi-seq mirror; snapshot/restore mirror for multi-turn reuse.

**Launch for serve test:** rank 1 first (`bash tools/tp2/rank1.sh` on this host), then rank 0 (`bash tools/tp2/rank0.sh 18080` on strix-halo-1 via SSH). Never probe 18515 with `/dev/tcp` — it steals rank 0's accept.

**Reference:** gufo at `/home/ugo/projects/gufo`, especially `src/cli/serve/tp_executor.cpp` and `tp_control.cpp`.

Verify every step: kill old processes, deploy binary, check timestamp, then test with a curl request.

---

_When you make progress, update `Handoff.md` (§4 issue map, §5 what works, §10) and comment on the relevant issue. This file is the restart point — keep it current._
