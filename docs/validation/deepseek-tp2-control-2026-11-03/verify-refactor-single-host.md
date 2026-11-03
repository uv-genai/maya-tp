# DeepSeek verify/commit refactor: single-host guard

Issue #113. An isolated gfx1151 build of `ds4_model.cu` SHA256
`68fc02f3d399fb7901905a5557622e784ac76fc21dda21751a14179baf3d058a` and executable MD5
`3d5a4d28e298be57566a7f9ea095de8e` was compared with the reference described in
[partition-single-host.md](partition-single-host.md), using the same single-host Maya-XS pack and
settings. The 64-token greedy token-file digest matched exactly
(`381baeb7bfea390e05fa6488507a4c43dcd5b845ffc556799aed32b44df67c89`), and both accepted 33 of
37 DSpark drafts. Candidate `--sampler-test 50` passed all four chains and its text output matched
the prior single-host candidate's output exactly (SHA256
`64c0b08a63f5e6f608ea44fbd76792f1bb7a411d01373b5a6f79aa9bef2b56e7`). This is a small fixed-seed
statistical smoke test, not a distribution proof or a two-host verification test.
