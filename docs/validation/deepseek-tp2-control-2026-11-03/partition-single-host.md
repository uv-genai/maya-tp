# DeepSeek uneven-partition helper: single-host guard

Issue #115. The two-way uneven local-group partition calculation was switched to the tested
`expert_ranges` helper. This branch is not entered by the following one-GPU check; the check guards
the existing DeepSeek single-host path and does not prove local two-device or two-host parity.

On strix-halo-1, isolated HIP gfx1151 builds used the same Maya-XS pack, `--tp 1`, 32768 context,
`--ids 5,6,7 --n 64 --dspark on --suffix-draft 0 --vision off --threads on --profile-readonly`.
Reference binary MD5 `40b27560ae45924879fed5c5ef2808a7`; candidate binary MD5
`2604b704c054652273615653ad4b41e5`. Candidate `ds4_model.cu` SHA256
`8d05ce349b8a444242d8e428d2c87b0d6bd2470ba1f004a89d7815965bcca034`.
Both completed 64 tokens with identical token-file SHA256
`381baeb7bfea390e05fa6488507a4c43dcd5b845ffc556799aed32b44df67c89` and 33/37 DSpark drafts
accepted. The CMake-registered host partition properties test passed in the candidate build.
The paired-host TP2 model path and actual uneven-group GPU behavior remain unvalidated.
