#!/usr/bin/env python3
"""Run an owned test rank with independently checked local preflight values."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ds4.tp2_identity import MAX_META, IdentityError, prepare_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exe", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    try:
        args = opts.args[1:] if opts.args[:1] == ["--"] else opts.args
        if args and args[0].startswith("--sharded-"):
            if len(args) < 6 or Path(args[5]).resolve(strict=True) != Path(opts.pack).resolve(strict=True):
                raise IdentityError("model test pack differs from the checked pack")
        manifest = Path(opts.manifest)
        if manifest.stat().st_size > MAX_META:
            raise IdentityError("manifest is too large")
        content = manifest.read_bytes()
        if len(content) > MAX_META:
            raise IdentityError("manifest is too large")
        identity = prepare_identity(json.loads(content), opts.pack, opts.exe)
        env = os.environ.copy()
        for name in ("model", "partition", "build"):
            env["DS4_TP_TEST_" + name.upper() + "_DIGEST"] = identity[name + "_digest"]
        print("test rank preflight: " + json.dumps(identity, sort_keys=True), file=sys.stderr, flush=True)
        exe = str(Path(opts.exe).resolve(strict=True))
        os.execve(exe, [exe, *args], env)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"test rank preflight: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
