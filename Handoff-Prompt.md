# Handoff-Prompt — Maya TP2 restart

Read `/home/ugo/software/project-maya/Handoff.md` in full, especially §18 and §21, then `AGENTS.md`.

Project: **Maya-TP**, GLM-5.3-Flash TP2 across twin Strix Halo hosts (Maya-S default; Maya-M packed and
pair-benched). HIP/ROCm **gfx1151 Linux only**; GPU inference, RDMA tensor path only.
Bootstrap/control/token metadata may use TCP.
Repository `uv-genai/maya-tp`, local branch main. No subagents unless explicitly authorized.

## Current state

**v1.1.41 is published** (`a6d15a7`, CI41191735148); #32 closed (Handoff §24) — do not republish.
v1.1.40 = `d9248e8`; v1.1.39 = `20ff03a`. Do not repeat completed measurements or start a
new release without an explicit request.
