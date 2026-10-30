#!/usr/bin/env python3
"""Compare MTP and simple TP2 decode on a resident gfx1151 pair over RDMA.

Run on strix-halo-2: .venv/bin/python tools/tp2/bench_mtp.py --output /tmp/mtp-bench
Both hosts must have matching binaries. Stops existing Maya engines/remote server.
Generates a simple-loop reference, then replays it with --spec-follow. MTP stays
loaded in both modes; GEN spec=1/0 selects simple/MTP. Model load and warmups are
excluded. EOS stopping is disabled via an unreachable EOS ID for fixed-length runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HOST = "strix-halo-1.local"
PROMPT = "154822,25062,287,29905,371,25,7487,785,6722,315,9621,374"
PACK = "/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/pack"
# Explicitly neutralize inherited test/isolation settings on the remote host too.
CLEAR = ("GLM_NO_MTP", "GLM_SPEC_FOLLOW", "GLM_SPEC_FOLLOW_SAVE", "GLM_TP_AR_TRACE",
         "GLM_MTP_TRACE", "GLM_MTP_FIXED", "GLM_MTP_DRAFT", "TP_TEST_WINDOW_RETRY",
         "GLM_WIN_PROF", "GLM_NO_REUSE", "TP_NOAR", "TP_NOSLICE", "TP_SH_BOTH")
SETTINGS = dict(MAYA_TP_MTP="1", MAYA_GLM_MTP_FIXED="2", MAYA_GLM_LOOP_THINK="0", MAYA_GLM_LOOP_ANSWER="0")


def check_fidelity(diagnostics, maximum=0):
    """Forced output IDs alone do not establish agreement with the model's picks."""
    counts = [int(match.group(1)) for line in diagnostics
              if (match := re.search(r"(\d+) of them differ from the model's own pick", line))]
    if len(counts) != 1:
        raise RuntimeError("missing or ambiguous raw-pick agreement diagnostic")
    if counts[0] > maximum:
        raise RuntimeError(f"{counts[0]} raw model picks differ from reference (maximum {maximum})")
    return counts[0]


def remote(argv, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, shlex.join(argv)], check=True, **kwargs)


def cleanup():
    subprocess.run(["pkill", "-9", "-x", "maya"], check=False, capture_output=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", HOST,
                    'pkill -9 -f "[s]erver.py --engine"; pkill -9 -x maya; true'],
                   check=False, capture_output=True)


class Pair:
    def __init__(self, output, label, context, follow=None, single_host=False):
        self.output = output
        self.label = label
        self.lines = queue.Queue()
        self.r0 = self.r1 = None
        self.log0 = (output / f"{label}-rank0.log").open("w")
        self.log1 = (output / f"{label}-rank1.log").open("w")
        args = ["--glm-pack", PACK, "--max-context", str(context), "--greedy", "--parallel", "1"]
        tp_args = ["--tp-world-size", "2", "--tp-bootstrap-port", "18515", "--tp-rdma-device", "usb4_rdma5"]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("MAYA_", "STRATA_"))}
        env.update(SETTINGS)
        try:
            if single_host:
                cmd = [str(ROOT / "build-hip/maya"), *args, "--serve", "--eos-ids", "2147483647"]
            else:
                self.r1 = subprocess.Popen([str(ROOT / "build-hip/maya"), *args, *tp_args, "--tp-rank", "1",
                                            "--tp-bootstrap-host", HOST], stdin=subprocess.DEVNULL,
                                           stdout=self.log1, stderr=subprocess.STDOUT, env=env)
                remote_env = ["env"]
                for setting in CLEAR:
                    for prefix in ("MAYA_", "STRATA_"):
                        remote_env += ["-u", prefix + setting]
                remote_env += [f"{k}={v}" for k, v in SETTINGS.items()]
                cmd = [*remote_env, str(ROOT / "build-hip/maya"), *args, *tp_args, "--tp-rank", "0", "--serve",
                       "--eos-ids", "2147483647"]
            if follow:
                cmd += ["--spec-follow", follow]
            command = cmd if single_host else ["ssh", "-o", "BatchMode=yes", HOST, "exec " + shlex.join(cmd)]
            self.r0 = subprocess.Popen(command,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env=env if single_host else None)
            self.pump = threading.Thread(target=self._read, daemon=True)
            self.pump.start()
            self.until(lambda line: line.startswith("READY "), 240)
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.r0.stdout:
            self.log0.write(line)
            self.log0.flush()
            self.lines.put(line.rstrip())
        self.lines.put(None)

    def until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        collected = []
        while True:
            line = self.lines.get(timeout=max(0.001, deadline - time.monotonic()))
            if line is None:
                raise RuntimeError(f"rank 0 ended; see {self.label}-rank0.log")
            if line.startswith(("ERR ", "TP2_DOWN")):
                raise RuntimeError(line)
            collected.append(line)
            if predicate(line):
                return collected
            if time.monotonic() >= deadline:
                raise TimeoutError(f"rank 0 timeout; see {self.label}-rank0.log")

    def request(self, mode, tokens, follow=False):
        before = time.monotonic()
        self.r0.stdin.write(f"GEN {tokens} temperature=0 spec={0 if mode == 'mtp' else 1} {PROMPT}\n")
        self.r0.stdin.flush()
        lines = self.until(lambda line: line.startswith("DONE "), 180)
        wall_ms = 1000 * (time.monotonic() - before)
        done = lines[-1].split()
        ids = [int(line.split()[1]) for line in lines if line.startswith("T ")]
        if len(ids) != tokens or int(done[1]) != tokens:
            raise RuntimeError(f"expected {tokens} output tokens, got {done}")
        lines += self.until(lambda line: line.startswith("glm follow:") and "of the continuation decoded" in line
                            if follow else line.startswith("glm stat: decode"), 30)
        result = dict(mode=mode, tokens=len(ids), prompt_ms=float(done[3]), decode_ms=float(done[4]),
                      ms_per_token=float(done[4]) / tokens, tokens_per_second=1000 * tokens / float(done[4]),
                      wall_ms=wall_ms, accepted_drafts=int(done[6]), drafts=int(done[7]), cached_tokens=int(done[8]),
                      cache_hits=int(done[9]), cache_lookups=int(done[10]),
                      output_sha256=hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
                      diagnostics=[line for line in lines if line.startswith(("glm stat:", "glm mtp:", "glm follow:"))])
        if mode == "mtp" and result["drafts"] == 0:
            raise RuntimeError("MTP did not run")
        if mode == "simple" and result["drafts"] != 0:
            raise RuntimeError("simple loop unexpectedly drafted")
        return result, ids

    def close(self):
        if self.r0 is not None and self.r0.poll() is None:
            try:
                self.r0.stdin.write("QUIT\n")
                self.r0.stdin.flush()
                self.r0.stdin.close()
                self.r0.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                self.r0.kill()
                self.r0.wait()
        if self.r1 is not None and self.r1.poll() is None:
            self.r1.terminate()
            try:
                self.r1.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.r1.kill()
                self.r1.wait()
        if hasattr(self, "pump"):
            self.pump.join(timeout=5)
        self.log0.close()
        self.log1.close()
        cleanup()


def main():
    global PROMPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=4, help="measured runs per mode (positive, even)")
    parser.add_argument("--drafts", type=int, default=2, choices=range(1, 9), help="fixed MTP draft count and capacity")
    parser.add_argument("--reference", type=Path, help="reuse reference IDs from a prior sweep run")
    parser.add_argument("--prompt-file", type=Path, help="comma-separated prompt IDs for a representative workload")
    parser.add_argument("--window-profile", action="store_true", help="instrument verification; exclude from performance comparisons")
    parser.add_argument("--max-pick-differences", type=int, default=0,
                        help="raw model-pick differences allowed against the forced reference (default: 0)")
    parser.add_argument("--single-host", action="store_true", help="local gfx1151 control without TP or RDMA")
    args = parser.parse_args()
    if args.prompt_file:
        ids = [int(x) for x in args.prompt_file.read_text().strip().split(",")]
        if not ids or min(ids) < 0:
            parser.error("--prompt-file must contain nonnegative token IDs")
        PROMPT = ",".join(map(str, ids))
    if args.runs < 2 or args.runs % 2 or args.tokens < 4:
        parser.error("--runs must be positive/even and >=2; --tokens must be >=4")
    if args.max_pick_differences < 0:
        parser.error("--max-pick-differences must be nonnegative")
    if args.context <= args.tokens + len(PROMPT.split(",")) + 3:
        parser.error("--context must have room for the prompt, output and draft window")
    SETTINGS.update(MAYA_GLM_MTP_FIXED=str(args.drafts), MAYA_GLM_MTP_DRAFT=str(args.drafts))
    if args.window_profile:
        SETTINGS["MAYA_GLM_WIN_PROF"] = "1"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = ROOT / "build-hip/maya"
    md5 = hashlib.md5(binary.read_bytes()).hexdigest()
    if not args.single_host:
        peer_md5 = remote(["md5sum", str(binary), "/tmp/maya-tp2"], capture_output=True, text=True).stdout
        if any(line.split()[0] != md5 for line in peer_md5.splitlines()) or not peer_md5.strip():
            raise RuntimeError("host binary hashes do not match")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = dict(binary_md5=md5, source_head=head, host=os.uname().nodename if args.single_host else HOST,
                    world_size=1 if args.single_host else 2, pack=PACK, prompt=PROMPT,
                    context=args.context, tokens=args.tokens, runs_per_mode=args.runs,
                    settings=SETTINGS, timing="DONE decode_ms / output tokens; model load/prefill excluded",
                    max_pick_differences=args.max_pick_differences,
                    warmup_order=["simple", "mtp", "mtp", "simple"], results=[], warmups=[])
    (out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cleanup()
    pair = None
    try:
        if args.reference:
            reference = [int(x) for x in args.reference.read_text().strip().split(",")]
            if len(reference) != args.tokens:
                raise ValueError("--reference must contain exactly --tokens IDs")
            metadata["reference_source"] = str(args.reference.resolve())
        else:
            print("Generating common reference with the simple TP2 loop...", flush=True)
            pair = Pair(out, "reference", args.context, single_host=args.single_host)
            seed, reference = pair.request("simple", args.tokens)
            metadata["reference"] = seed
            pair.close()
            pair = None
        (out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
        reference_file = out / "reference.txt"
        reference_file.write_text(",".join(map(str, reference)) + "\n")
        remote_reference = str(reference_file) if args.single_host else f"/tmp/{out.name}-reference.txt"
        if not args.single_host:
            subprocess.run(["scp", str(reference_file), f"{HOST}:{remote_reference}"], check=True)
        print("Starting resident pair with fixed reference; warmups first...", flush=True)
        pair = Pair(out, "benchmark", args.context, remote_reference, single_host=args.single_host)
        order = ["simple", "mtp", "mtp", "simple"]
        # Each ABBA block gives both modes the same mean position in the run order.
        measured = order * (args.runs // 2)
        metadata["measured_order"] = measured
        for warm, modes in ((True, order), (False, measured)):
            for mode in modes:
                result, ids = pair.request(mode, args.tokens, follow=True)
                if ids != reference:
                    raise RuntimeError(f"{mode}: output differs from common reference")
                result["model_pick_differences"] = check_fidelity(result["diagnostics"], args.max_pick_differences)
                group = "warmups" if warm else "results"
                metadata[group].append(result)
                (out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"{'warmup' if warm else 'measure'} {mode:6s}: {result['ms_per_token']:.3f} ms/tok, "
                      f"{result['tokens_per_second']:.2f} tok/s, "
                      f"drafts={result['accepted_drafts']}/{result['drafts']}", flush=True)
        summary = {}
        for mode in ("simple", "mtp"):
            values = [r["ms_per_token"] for r in metadata["results"] if r["mode"] == mode]
            med = statistics.median(values)
            summary[mode] = dict(median_ms_per_token=med, median_tokens_per_second=1000 / med,
                                 min_ms_per_token=min(values), max_ms_per_token=max(values),
                                 mean_ms_per_token=statistics.mean(values), stdev_ms_per_token=statistics.stdev(values))
        summary["mtp_latency_change_percent"] = 100 * (summary["mtp"]["median_ms_per_token"] /
                                                         summary["simple"]["median_ms_per_token"] - 1)
        summary["mtp_throughput_change_percent"] = 100 * (summary["simple"]["median_ms_per_token"] /
                                                            summary["mtp"]["median_ms_per_token"] - 1)
        metadata["summary"] = summary
        (out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if pair is not None:
            pair.close()
        cleanup()


if __name__ == "__main__":
    main()
