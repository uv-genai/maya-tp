#!/usr/bin/env python3
"""Content-derived DeepSeek TP2 pack identity with local change detection.

Create a manifest once per host, then validate it immediately before launching
that host's rank. The exchanged digest excludes absolute paths and local inode
metadata, while the local manifest checks those metadata for every indexed file.
A changed source invalidates the manifest and requires another content hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

VERSION = 1
MAX_META = 64 * 1024 * 1024
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer.json", "token_type.json", "chat_template.jinja")
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


class IdentityError(ValueError):
    pass


def _signature(st):
    if not stat.S_ISREG(st.st_mode):
        raise IdentityError("identity input is not a regular file")
    return {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def _path_signature(path):
    try:
        return _signature(Path(path).stat())
    except OSError as exc:
        raise IdentityError(f"identity input unavailable: {path}: {exc.strerror}") from exc


def _stable_read(path: Path, keep=False, progress=None):
    """Hash an opened file and check both the descriptor and its path stayed put."""
    h = hashlib.sha256()
    chunks = [] if keep else None
    with path.open("rb") as f:
        before = _signature(os.fstat(f.fileno()))
        if keep and before["size"] > MAX_META:
            raise IdentityError(f"metadata file too large: {path}")
        done = 0
        last_report = time.monotonic()
        if progress:
            progress(done, before["size"])
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
            now = time.monotonic()
            if progress and now - last_report >= 5:
                progress(done, before["size"])
                last_report = now
        after = _signature(os.fstat(f.fileno()))
    if before != after or after != _path_signature(path):
        raise IdentityError(f"identity input changed while hashing: {path}")
    if progress:
        progress(done, before["size"])
    return h.hexdigest(), after, b"".join(chunks) if chunks is not None else None


def _digest(value):
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def _model_config(raw):
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise IdentityError("ds4.txt is not UTF-8") from exc
    model_dirs = [line.split(" ", 1)[1] for line in lines if line.startswith("model_dir ")]
    if len(model_dirs) != 1 or not Path(model_dirs[0]).is_absolute():
        raise IdentityError("ds4.txt must contain one absolute model_dir")
    formats = [line.split(" ", 1)[1] for line in lines if line.startswith("format ")]
    if formats != ["gguf"]:
        raise IdentityError("DeepSeek TP2 identity requires one GGUF format declaration")
    # Match the engine's key/value parser, but remove this host's root from the
    # shared fingerprint. Everything else, including line order, remains bound.
    canonical = "\n".join("model_dir <local>" if line.startswith("model_dir ") else line for line in lines) + "\n"
    return Path(model_dirs[0]), hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _index(raw):
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise IdentityError("tensors.txt is not UTF-8") from exc
    files = {}
    spans = []
    for line in lines:
        if line.startswith("file "):
            pieces = line.split()
            if len(pieces) != 3 or not pieces[1].isdigit():
                raise IdentityError("invalid file entry in tensors.txt")
            number = int(pieces[1])
            if number > 65535 or number in files or Path(pieces[2]).is_absolute():
                raise IdentityError("duplicate, invalid, or absolute file entry")
            files[number] = pieces[2]
        elif line.startswith("t "):
            pieces = line.split()
            if len(pieces) < 7:
                raise IdentityError("invalid tensor entry in tensors.txt")
            try:
                fi, offset, size = map(int, pieces[3:6])
                dims = [int(n) for n in pieces[6:]]
            except ValueError as exc:
                raise IdentityError("invalid tensor coordinates") from exc
            if fi < 0 or offset < 0 or size <= 0 or not dims or any(n <= 0 for n in dims):
                raise IdentityError("invalid tensor coordinates")
            spans.append((fi, offset, size))
    if not files or sorted(files) != list(range(len(files))) or not spans:
        raise IdentityError("missing or noncontiguous file entries or tensors")
    return [files[i] for i in range(len(files))], spans


def create_manifest(pack, progress=None):
    pack = Path(pack).resolve(strict=True)
    if not pack.is_dir():
        raise IdentityError("pack is not a directory")
    local = []

    def add(role, path, keep=False):
        path = Path(os.path.abspath(path))
        report = (lambda done, total: progress(role, done, total)) if progress and role.startswith("source/") else None
        digest, sig, content = _stable_read(path, keep, report)
        local.append({"role": role, "path": str(path), "sha256": digest, "stat": sig})
        return digest, content

    _, config = add("config", pack / "ds4.txt", True)
    root, config_digest = _model_config(config)
    index_digest, index = add("index", pack / "tensors.txt", True)
    names, spans = _index(index)
    engram_digest, _ = add("engram", pack / "engram.bin")
    tokens = {}
    for name in TOKENIZER_FILES:
        path = pack / "tokenizer" / name
        if path.exists():
            tokens[name], _ = add("tokenizer/" + name, path)
    if not all(name in tokens for name in TOKENIZER_FILES[:4]):
        raise IdentityError("incomplete tokenizer metadata")
    sources = []
    for i, name in enumerate(names):
        digest, _ = add("source/" + str(i), root / name)
        sources.append({"index": i, "name": name, "size": local[-1]["stat"]["size"], "sha256": digest})
    for fi, offset, size in spans:
        if fi >= len(sources) or offset > sources[fi]["size"] or size > sources[fi]["size"] - offset:
            raise IdentityError("tensor span is outside its indexed file")
    # Recheck every input after the last source is hashed, not only immediately
    # after its own read, so a concurrent change invalidates the whole result.
    for entry in local:
        if _path_signature(entry["path"]) != entry["stat"]:
            raise IdentityError(f"identity input changed during manifest creation: {entry['path']}")
    canonical = {"version": VERSION, "model_config_sha256": config_digest, "index_sha256": index_digest,
                 "engram_sha256": engram_digest, "tokenizer": tokens, "sources": sources}
    return {"version": VERSION, "identity": _digest(canonical), "canonical": canonical,
            "pack": str(pack), "local": local}


def check_manifest(manifest, pack):
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise IdentityError("invalid manifest version")
    canonical = manifest.get("canonical")
    identity = manifest.get("identity")
    if not isinstance(canonical, dict) or canonical.get("version") != VERSION or not isinstance(identity, str) or not HEX256.fullmatch(identity) or _digest(canonical) != identity:
        raise IdentityError("manifest identity is invalid")
    pack = Path(pack).resolve(strict=True)
    if str(pack) != manifest.get("pack"):
        raise IdentityError("manifest belongs to a different local pack")
    entries = manifest.get("local")
    if not isinstance(entries, list):
        raise IdentityError("manifest has no local inputs")
    roles = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str) or entry["role"] in roles:
            raise IdentityError("invalid or duplicate local input")
        roles[entry["role"]] = entry

    def entry_for(role, expected):
        entry = roles.get(role)
        path = str(Path(os.path.abspath(expected)))
        if not entry or entry.get("path") != path or not isinstance(entry.get("stat"), dict):
            raise IdentityError("manifest local input path mismatch")
        if not isinstance(entry.get("sha256"), str) or not HEX256.fullmatch(entry["sha256"]):
            raise IdentityError("invalid local input digest")
        if _path_signature(path) != entry["stat"]:
            raise IdentityError(f"identity input changed: {path}")
        return entry

    # Re-read metadata on every check; the large immutable source files are
    # rehashed when their local stat signature changes, not on every startup.
    def meta(role, path, keep=False):
        entry = entry_for(role, path)
        digest, sig, data = _stable_read(Path(path), keep)
        if digest != entry["sha256"] or sig != entry["stat"]:
            raise IdentityError(f"identity metadata changed: {path}")
        return digest, data

    _, config = meta("config", pack / "ds4.txt", True)
    root, config_digest = _model_config(config)
    index_digest, index = meta("index", pack / "tensors.txt", True)
    names, spans = _index(index)
    engram_digest, _ = meta("engram", pack / "engram.bin")
    if (config_digest != canonical.get("model_config_sha256") or index_digest != canonical.get("index_sha256")
            or engram_digest != canonical.get("engram_sha256")):
        raise IdentityError("manifest metadata digest mismatch")
    tokens = {}
    for name in TOKENIZER_FILES:
        path = pack / "tokenizer" / name
        if path.exists():
            tokens[name], _ = meta("tokenizer/" + name, path)
    if not all(name in tokens for name in TOKENIZER_FILES[:4]) or tokens != canonical.get("tokenizer"):
        raise IdentityError("manifest tokenizer digest mismatch")
    sources = canonical.get("sources")
    if not isinstance(sources, list) or len(sources) != len(names):
        raise IdentityError("manifest source count mismatch")
    required = {"config", "index", "engram"} | {"tokenizer/" + n for n in tokens}
    for i, name in enumerate(names):
        role = "source/" + str(i)
        required.add(role)
        entry = entry_for(role, root / name)
        src = sources[i]
        if (not isinstance(src, dict) or src.get("index") != i or src.get("name") != name
                or entry["sha256"] != src.get("sha256") or entry["stat"]["size"] != src.get("size")):
            raise IdentityError("manifest source digest mismatch")
    if set(roles) != required:
        raise IdentityError("manifest inputs are incomplete")
    for fi, offset, size in spans:
        if fi >= len(sources) or offset > sources[fi]["size"] or size > sources[fi]["size"] - offset:
            raise IdentityError("tensor span is outside its indexed file")
    # Catch a source changed while the other metadata was being checked.
    for entry in entries:
        if _path_signature(entry["path"]) != entry["stat"]:
            raise IdentityError(f"identity input changed: {entry['path']}")
    return identity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("create", "check"))
    p.add_argument("--pack", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args(argv)
    try:
        if a.action == "create":
            def progress(role, done, total):
                print(f"ds4 tp2 identity: {role} {done}/{total} bytes", file=sys.stderr, flush=True)
            m = create_manifest(a.pack, progress)
            target = Path(a.manifest)
            # Publish a complete file atomically without replacing an existing
            # cache, even if hashing or writing fails part way through.
            fd, tmp = tempfile.mkstemp(prefix="." + target.name + ".", dir=target.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(m, f, sort_keys=True, indent=2)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.link(tmp, target)
            finally:
                os.unlink(tmp)
            print(m["identity"])
        else:
            manifest_path = Path(a.manifest)
            if manifest_path.stat().st_size > MAX_META:
                raise IdentityError("manifest is too large")
            data = manifest_path.read_text(encoding="utf-8")
            if len(data.encode("utf-8")) > MAX_META:
                raise IdentityError("manifest is too large")
            print(check_manifest(json.loads(data), a.pack))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ds4 tp2 identity: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
