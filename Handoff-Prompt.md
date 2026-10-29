# Handoff-Prompt — restart prompt for a fresh session

Give this prompt to a new session to resume the Maya TP2 work. It points at `Handoff.md` as the source of truth.

---

Read `/home/ugo/software/project-maya/Handoff.md` in full before doing anything — it is the live handoff and overrides anything you infer. Also read `AGENTS.md` (platform rule, GitHub workflow, git remotes).

**Project:** two-host tensor parallel (TP2) for GLM-5.3-Flash Maya-S on twin Strix Halo gfx1151 hosts over Thunderbolt RDMA. Repo `/home/ugo/software/project-maya`, GitHub `uv-genai/maya-tp`, branch `main`.

**Hard rules (non-negotiable):**
1. **HIP/ROCm gfx1151 Linux only.** No CUDA/NVIDIA work.
2. **RDMA only** (`usb4_rdma*`), target ~37 Gb/s / ~6 µs as in gufo. No TCP data path.
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. **Track every workstream as its own GitHub issue** on `uv-genai/maya-tp` (not one umbrella with comments). Comment while making progress. Link commits `refs #N`. Open a new issue for any bug/gap not already covered.
5. Push only to `origin` (`uv-genai/maya-tp`). Never push `upstream` (`mw00/project-maya`).

**Reference implementation:** gufo at `/home/ugo/projects/gufo` (TP2/RDMA). See its `TP2-SETUP.md` and `docs/models/qwen3.8-flash-next/TP2.md`.

**Current state:** correctness is done — pair parity 100%, 31/32 tokens are the model's own greedy picks (1 near-tie). RDMA pair live. Expert pool sliced to 44.3 GB/rank. MTP window fixed.

**The open gap is performance:** experts-only slice gives +9.4% (with AR) to +16.8% (no AR) vs single-host; gufo gets 30%+. The mixer is still replicated and the all-reduce is on the critical path.

**Next work (issue #5 — do this):**
1. Slice the KDA mixer (heads 64→32): `attn_q/k/v` row-slice, `attn_output` column-slice, per-head recurrent state; DSA head-slice (MLA latent stays replicated).
2. Move the all-reduce to the overlapped `split_reduce` form (`StartPartial`/`FinishPartial`) to reclaim the 6.7%.

Then re-measure with the controlled protocol in `Handoff.md` §6/§9 (no-MTP, same prompt, report ms/tok and `--spec-follow` near-tie count). Target 30%+ vs single-host.

**Watch the gotchas in `Handoff.md` §8** — especially the shard-stride rule (sliced `Ly.gu_bytes` is NOT the GGUF stride) and the per-layer width (`ff_compute_for(il)`; the MTP draft layer stays full).

---

_When you make progress, update `Handoff.md` (§4 issue map, §5 what works, §6 performance, §10 next steps) and comment on the relevant issue. This file is the restart point — keep it current._
