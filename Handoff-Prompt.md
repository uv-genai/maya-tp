# Handoff-Prompt — restart prompt for a fresh session

Give this prompt verbatim to a new session to resume the Maya TP2 work. It points at `Handoff.md` as the source of truth.

---

Read `/home/ugo/software/project-maya/Handoff.md` in full before doing anything — it is the live handoff and overrides anything you infer. Also read `AGENTS.md` (platform rule, GitHub workflow, git remotes).

**Project:** two-host tensor parallel (TP2) for GLM-5.3-Flash on twin Strix Halo gfx1151 hosts over Thunderbolt RDMA. Repo `/home/ugo/software/project-maya`, GitHub `uv-genai/maya-tp`, branch `main`, release **v1.1.37**.

**Hard rules (non-negotiable):**
1. **HIP/ROCm gfx1151 Linux only.** No CUDA/NVIDIA work.
2. **RDMA only** (`usb4_rdma*`) for tensors. Bootstrap TCP = handshake + serve mirror + MTP accept wire only.
3. **GPU is the inference path.** CPU only for unit tests of slice math.
4. **Track work as GitHub issues** on `uv-genai/maya-tp`. Comment while progressing. Link `refs #N`.
5. Push only to `origin`. Never push `upstream`.
6. **NEVER nested SSH.** You are ON strix-halo-2. SSH only to strix-halo-1.
7. Python: `.venv/bin/python`. Shell: `jq`. Kill engines with `pkill -9 -x maya` — **never** `pkill -f server.py` (kills the ssh shell); for rank 0's server use `pkill -9 -f "[s]erver.py --engine"`.
8. **After every build, `scp` the binary to strix-halo-1** — both `build-hip/maya` **and** `/tmp/maya-tp2` — after killing engines **and** server.py there (a live server respawns the engine, holds the binary open → scp "text file busy", and loses your env vars). Confirm `md5sum` matches both hosts.

**Done (do not redo):** full slice + RDMA; serve mirroring + peer `/health` + kill/recover; snapshot reuse (multi-turn `cached_tokens`); pack-derived TP geometry (#12); v1.1.37; CI green; bench ~+26–32% vs single-host; pair parity 100%. **This session on #13:** slice `moe_gate_up` window fault **fixed** (window `FFs` was full-width vs sliced shared weights); slow verify **resolved** (73–77 ms/round); CLI TP2+MTP pair parity 100%; serve turn 1 works; **MTP accept wire implemented** (rank 0 wires `{keep, tokens}` per round, op 12; rank 1 commits wired tokens — verify near-ties had desynced `keep` between ranks).

**FIRST THING: commit the uncommitted WIP** (§11.1 lists the changed files; the tree is the result of a full day of verified fixes — do not lose it).

**Primary next work — #13 turn-2 desync (Handoff §11.3):** with the accept wire, turn-2 serve (restore+prefill+decode) stalls: rank1 blocks in round-13 `wire-recv` while rank0 runs one round ahead → AR timeout → poison. The wire-send/recv skew mechanism is unknown. Steps: deploy the traced local binary (`e668184c…`) to both hosts, rerun the 2-turn serve test with `STRATA_GLM_TP_AR_TRACE=1` on both ranks, diff `wire-send` vs `wire-recv wait`/`wire-got` to find the skew round; then fix cancel robustness (rank1 stuck in wire-recv misaligns the op stream). Exit criteria + gate decision in §11.5.

**Open issues:** #1 umbrella · #10 gather bit-exact (optional) · #11 multi-seq (**still gated**, `SEQS>1` prompt `mv_rdna`; note `glm_rows.cu` full-width `FFs` bug to fix when re-enabling) · #13 as above.

**Serve launch:** rank 1 first (`bash tools/tp2/rank1.sh`), then rank 0 (`bash tools/tp2/rank0.sh 18080` on strix-halo-1). **Never** probe 18515 with `/dev/tcp`. Ops: `docs/TP2-RUN.md`. Logs: rank1 `/tmp/tp2-r1.log`; rank0 engine `maya-maya-s-v2-iq2_xxs-hip.log` (append-mode — segment by last `glm prefill: CUDA0 chunks`), server `/tmp/tp2-r0.log`. An idle worker holds ~45 GB VRAM with no kernels — normal; kill it to free.

**Perf context:** TP2+MTP 44.2 ms/tok vs TP2 simple 41.1–43 — MTP not yet a win; the lever is the overlapped AR (gufo `StartPartial`/`FinishPartial`; no `MAYA_TP_OVERLAP` knob exists).

When you make progress: commit (`refs #N`), comment on the issue, and update `Handoff.md` (§11) — keep it the restart point.

---
