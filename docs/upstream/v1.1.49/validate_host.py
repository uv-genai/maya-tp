"""Validate actual DS4 request code and link only its changed host entry point; no GPU execution.

The reference build is read-only. Its headers/kernels must match this tree; only ds4_main.cpp
may differ. Output goes into a separate directory, never into a production or reference build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def run(argv, **kw):
    subprocess.run(argv, check=True, **kw)


def gpu_sections(path):
    sections = subprocess.check_output(["readelf", "-SW", str(path)], text=True)
    data = path.read_bytes()
    out = {}
    for name, offset, size in re.findall(
            r"\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)", sections):
        if "hip_fatbin" in name:
            out[name] = hashlib.sha256(data[int(offset, 16):int(offset, 16) + int(size, 16)]).hexdigest()
    assert out, "no HIP image found"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    ref, out = a.reference_build.resolve(), a.out.resolve()
    assert out != ref and not out.is_relative_to(ref)
    out.mkdir(parents=True, exist_ok=True)
    ref_root = ref.parent
    ref_head = subprocess.check_output(["git", "-C", str(ref_root), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", ref_head, "--", "src", "include", "CMakeLists.txt", "cmake", "third_party"],
        cwd=root, text=True).splitlines()
    assert changed == ["src/ds4/ds4_main.cpp"], changed
    source = (root / "src/ds4/ds4_main.cpp").read_text()

    def part(begin, end):
        return source[source.index(begin):source.index(end, source.index(begin))]

    policy = ('#include "src/ds4/ds4_logprobs.hpp"\n#include <cstdlib>\n#include <unordered_set>\n'
              + part("std::vector<int> parse_ids(", "void usage()")
              + part("struct Req {", "// A prompt's picture:")
              + part("bool parse_gen(", "// logprobs=<k>:") + r'''
int main() {
    int checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) std::exit(1); };
    Req r;
    check(parse_gen("GEN 64 think_budget=12 think_end=99 think_soft=8 think_stop=7 think_weak=3 think_dig=2,4 1,5", r, false));
    check(r.ids == std::vector<int>({1,5}) && r.think_end == 99 && r.think_budget == 12);
    check(!r.soft_check(2, 7));
    check(!r.soft_check(3, 8)); // A decimal/dot predecessor does not close a bare period.
    check(!r.soft_check(1, 9));
    check(r.soft_check(3, 10));
    check(r.soft_check(7, 11));
    check(r.soft_room(8, 7) == 1 && r.soft_room(8, 8) == 1);
    Req off;
    check(parse_gen("GEN 64 think_budget=12 think_end=99 1", off, false));
    check(!off.soft_check(7, 100) && off.soft_room(8, 100) == 8);
    for (long long soft : {0LL, 1LL, 8LL, 100LL, 512LL}) {
        Req q;
        q.think_soft = soft;
        for (long long produced = 0; produced < 1050; ++produced) {
            for (long long room = 1; room <= 32; ++room) {
                const auto n = q.soft_room(room, produced);
                check(n > 0 && n <= room);
                check(soft <= 0 ? n == room : produced >= soft ? n == 1 : produced + n <= soft);
            }
        }
    }
    std::printf("{\"policy_checks\":%d}\n", checks);
}
''')
    cpp = out / "soft_close.cpp"
    cpp.write_text(policy)
    run(["c++", "-std=c++20", "-O2", "-I", str(root), str(cpp), "-o", str(out / "soft_close")])
    result = json.loads(subprocess.check_output([str(out / "soft_close")], text=True))

    # Compile with the accepted CMake-generated HIP host flags. All dependencies remain read-only.
    flags = (ref / "CMakeFiles/maya-ds4.dir/flags.make").read_text()
    args = ["/usr/bin/c++"]
    for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        args += shlex.split(re.search(rf"^{name} = (.*)$", flags, re.M)[1])
    obj = out / "ds4_main.cpp.o"
    compile_cmd = args + ["-c", str(root / "src/ds4/ds4_main.cpp"), "-o", str(obj)]
    (out / "compile.json").write_text(json.dumps(compile_cmd, indent=2) + "\n")
    run(compile_cmd)
    cmd = shlex.split((ref / "CMakeFiles/maya-ds4.dir/link.txt").read_text())
    for i, arg in enumerate(cmd):
        if arg == "-o":
            cmd[i + 1] = str(out / "maya-ds4")
        elif arg.endswith("/ds4_main.cpp.o"):
            cmd[i] = str(obj)
        elif not arg.startswith("-") and not Path(arg).is_absolute() and (ref / arg).is_file():
            cmd[i] = str(ref / arg)
    (out / "link.json").write_text(json.dumps(cmd, indent=2) + "\n")
    run(cmd, cwd=out)
    baseline, candidate = gpu_sections(ref / "maya-ds4"), gpu_sections(out / "maya-ds4")
    assert baseline == candidate, (baseline, candidate)
    result.update(reference_head=ref_head, changed_engine_files=changed, gpu_sections=candidate,
                  reference_md5=hashlib.md5((ref / "maya-ds4").read_bytes()).hexdigest(),
                  candidate_md5=hashlib.md5((out / "maya-ds4").read_bytes()).hexdigest(),
                  gpu_execution=False)
    (out / "host-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
