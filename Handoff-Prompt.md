# Handoff-Prompt — restart prompt for a fresh session

Give this prompt to a new session to resume the Maya TP2 work. It points at `Handoff.md` as the source of truth.

---

Read `/home/ugo/software/project-maya/Handoff.md` in full before doing anything — it is the live handoff and overrides anything you infer. Also read `AGENTS.md` (platform rule, GitHub workflow, git remotes).

**Project:** two-host tensor parallel (TP2) for GLM-5.3-Flash on twin Strix Halo gfx1151 hosts over Thunderbolt RDMA. Repo `/home/ugo/software/project-maya`, GitHub `uv-genai/maya-tp`, branch `main`, release **v1.1.37**.

**Hard rules (non-negotiable):**
1. **HIP/ROCm gfx1151 Linux only.** No CUDA/NVIDIA work.
2. **RDMA only** (`usb4_rdma*`) for tensors. Bootstrap TCP = handshake + serve mirror only.
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. **Track work as GitHub issues** on `uv-genai/maya-tp`. Comment while progressing. Link `refs #N`.
5. Push only to `origin`. Never push `upstream`.
6. **NEVER nested SSH.** You are ON strix-halo-2. SSH only to strix-halo-1.
7. Python: `.venv/bin/python`. Shell: `jq`. Kill with `pkill -9 -x maya` — **never** `pkill -f server.py` (kills the ssh shell).

**Done (do not redo):** full slice + RDMA; serve mirroring; peer `/health` + kill/recover; v1.1.37; CI green. Bench ~+26–32% vs single-host; pair parity 100%. Decode under TP2 is **simple loop only** (no MTP/DFlash).

**Open issues:** #1 umbrella · #10 bit-exact gather (optional) · #11 multi-seq + snapshot reuse · #12 pack-derived TP geometry · #13 MTP under TP2.

**Serve launch:** rank 1 first (`bash tools/tp2/rank1.sh`), then rank 0 (`bash tools/tp2/rank0.sh 18080` on strix-halo-1). **Never** probe 18515 with `/dev/tcp`. Ops: `docs/TP2-RUN.md`.

**Reference:** gufo `/home/ugo/projects/gufo` (`tp_executor.cpp`, `tp_control.hpp`).

Verify every step: kill old processes, deploy binary (both hosts may have separate `build-hip/`), check timestamp, then curl.

---

_When you make progress, update `Handoff.md` and the issue. This file is the restart point — keep it current._
