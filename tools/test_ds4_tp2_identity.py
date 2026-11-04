import copy
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.ds4 import tp2_identity as identity


class DeepSeekTP2IdentityTest(unittest.TestCase):
    def test_partition_identity_covers_both_alternating_rank_slices(self):
        config = (b"hidden_size 5120\nnum_hidden_layers 2\nnum_attention_heads 64\n"
                  b"o_groups 8\nn_routed_experts 384\nmoe_intermediate_size 2304\n")
        digest, plan = identity.partition_identity(config)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(plan["layers"][0], [[0, 1280, 0, 32, 0, 4], [1280, 1024, 32, 32, 4, 4]])
        self.assertEqual(plan["layers"][1], [[0, 1024, 0, 32, 0, 4], [1024, 1280, 32, 32, 4, 4]])
        self.assertNotEqual(digest, identity.partition_identity(config.replace(b"5120", b"4096"))[0])
        for bad in (config.replace(b"2304", b"2305"), config.replace(b"o_groups 8", b"o_groups 7"),
                    config + b"hidden_size 5120\n", config.replace(b"num_hidden_layers 2\n", b"")):
            with self.subTest(bad=bad), self.assertRaises(identity.IdentityError):
                identity.partition_identity(bad)

    def test_prepare_binds_pack_partition_and_executable(self):
        pack = self.make_pack("a")
        with (pack / "ds4.txt").open("a") as f:
            f.write("num_hidden_layers 2\nnum_attention_heads 64\no_groups 8\n"
                    "n_routed_experts 384\nmoe_intermediate_size 2304\n")
        manifest = identity.create_manifest(pack)
        exe = self.base / "engine"
        exe.write_bytes(b"one engine")
        exe.chmod(0o700)
        prepared = identity.prepare_identity(manifest, pack, exe)
        self.assertEqual(prepared["model_digest"], manifest["identity"])
        self.assertEqual(prepared["geometry"]["moe_intermediate_size"], 2304)
        self.assertRegex(prepared["partition_digest"], r"^[0-9a-f]{64}$")
        exe.write_bytes(b"other engine")
        self.assertNotEqual(prepared["build_digest"], identity.prepare_identity(manifest, pack, exe)["build_digest"])
        (pack.parent / "weights.gguf").write_bytes(b"different source")
        with self.assertRaises(identity.IdentityError):
            identity.prepare_identity(manifest, pack, exe)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def make_pack(self, name, source=b"abcdefghijklmnop", second=b"12345678"):
        root = self.base / name / "model"
        pack = root / "maya-ds4"
        (pack / "tokenizer").mkdir(parents=True)
        (root / "weights.gguf").write_bytes(source)
        (root / "draft.bin").write_bytes(second)
        (pack / "ds4.txt").write_text(f"format gguf\nhidden_size 5120\nmodel_dir {root}\n")
        (pack / "tensors.txt").write_text(
            "file 0 weights.gguf\nfile 1 draft.bin\n"
            "t trunk.weight Q8_0 0 1 8 2 4\n"
            "t mtp.weight BF16 1 0 8 2 2\n")
        (pack / "engram.bin").write_bytes(b"engram")
        for filename in identity.TOKENIZER_FILES:
            (pack / "tokenizer" / filename).write_text(filename)
        return pack

    def test_identical_bytes_different_paths_have_same_identity(self):
        a, b = self.make_pack("a"), self.make_pack("b")
        ma, mb = identity.create_manifest(a), identity.create_manifest(b)
        self.assertEqual(ma["identity"], mb["identity"])
        self.assertEqual(identity.check_manifest(ma, a), ma["identity"])
        self.assertEqual(identity.check_manifest(mb, b), mb["identity"])
        with self.assertRaises(identity.IdentityError):
            identity.check_manifest(ma, b)

    def test_different_source_and_metadata_change_identity(self):
        a = self.make_pack("a")
        b = self.make_pack("b", source=b"abcdefghijklmnoX")
        self.assertNotEqual(identity.create_manifest(a)["identity"], identity.create_manifest(b)["identity"])
        (b.parent / "weights.gguf").write_bytes(b"abcdefghijklmnop")
        (b / "tokenizer" / "vocab.json").write_text("different tokenizer")
        self.assertNotEqual(identity.create_manifest(a)["identity"], identity.create_manifest(b)["identity"])

    def test_changed_source_invalidates_even_if_size_and_mtime_restored(self):
        pack = self.make_pack("a")
        m = identity.create_manifest(pack)
        source = pack.parent / "weights.gguf"
        old = source.stat()
        source.write_bytes(b"abcdefghijklmnZZ")
        os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
        with self.assertRaisesRegex(identity.IdentityError, "changed"):
            identity.check_manifest(m, pack)

    def test_cache_survives_device_renumbering_with_same_filesystem(self):
        pack = self.make_pack("a")
        manifest = identity.create_manifest(pack)
        for entry in manifest["local"]:
            self.assertTrue(entry["stat"]["filesystem_id"])
            entry["stat"]["device"] += 1
        self.assertEqual(identity.check_manifest(manifest, pack), manifest["identity"])

    def test_stable_filesystem_cache_still_rejects_other_file_changes(self):
        pack = self.make_pack("a")
        manifest = identity.create_manifest(pack)
        for role in ("config", "source/0"):
            for field in ("filesystem_id", "inode", "size", "mtime_ns", "ctime_ns"):
                with self.subTest(role=role, field=field):
                    changed = copy.deepcopy(manifest)
                    entry = next(x for x in changed["local"] if x["role"] == role)
                    entry["stat"]["device"] += 1
                    entry["stat"][field] += 1
                    with self.assertRaisesRegex(identity.IdentityError, "changed"):
                        identity.check_manifest(changed, pack)

    def test_legacy_cache_keeps_strict_device_check(self):
        pack = self.make_pack("a")
        manifest = identity.create_manifest(pack)
        for entry in manifest["local"]:
            entry["stat"].pop("filesystem_id", None)
        self.assertEqual(identity.check_manifest(manifest, pack), manifest["identity"])
        for entry in manifest["local"]:
            entry["stat"]["device"] += 1
        with self.assertRaisesRegex(identity.IdentityError, "changed"):
            identity.check_manifest(manifest, pack)

    def test_zero_filesystem_id_keeps_device_identity(self):
        pack = self.make_pack("a")
        current = identity._signature((pack / "ds4.txt").stat(), 0)
        self.assertNotIn("filesystem_id", current)
        cached = dict(current, device=current["device"] + 1)
        self.assertFalse(identity._cached_signature_matches(current, cached))

    def test_device_change_during_hashing_is_rejected(self):
        pack = self.make_pack("a")
        path = pack / "ds4.txt"
        changed = identity._path_signature(path)
        changed["device"] += 1
        with mock.patch.object(identity, "_path_signature", return_value=changed):
            with self.assertRaisesRegex(identity.IdentityError, "changed while hashing"):
                identity._stable_read(path)

    def test_metadata_is_rehashed_and_paths_cannot_be_redirected(self):
        a, b = self.make_pack("a"), self.make_pack("b")
        m = identity.create_manifest(a)
        tampered = copy.deepcopy(m)
        entry = next(x for x in tampered["local"] if x["role"] == "source/0")
        entry["path"] = str(b.parent / "weights.gguf")
        entry["stat"] = identity._signature((b.parent / "weights.gguf").stat())
        with self.assertRaisesRegex(identity.IdentityError, "path mismatch"):
            identity.check_manifest(tampered, a)
        (a / "tokenizer" / "vocab.json").write_text("changed")
        with self.assertRaisesRegex(identity.IdentityError, "changed"):
            identity.check_manifest(m, a)

    def test_detects_symlink_retarget(self):
        pack = self.make_pack("a")
        target = pack.parent / "weights.gguf"
        alternate = pack.parent / "alternate.gguf"
        alternate.write_bytes(target.read_bytes())
        real = pack.parent / "real.gguf"
        target.rename(real)
        target.symlink_to(real.name)
        m = identity.create_manifest(pack)
        target.unlink()
        target.symlink_to(alternate.name)
        with self.assertRaisesRegex(identity.IdentityError, "changed"):
            identity.check_manifest(m, pack)

    def test_missing_source_invalidates_manifest(self):
        pack = self.make_pack("a")
        m = identity.create_manifest(pack)
        (pack.parent / "weights.gguf").unlink()
        with self.assertRaisesRegex(identity.IdentityError, "unavailable"):
            identity.check_manifest(m, pack)

    def test_rejects_missing_duplicate_or_invalid_index_and_spans(self):
        pack = self.make_pack("a")
        index = pack / "tensors.txt"
        cases = [
            "file 1 weights.gguf\nt x Q8_0 1 0 1 1\n",
            "file 0 weights.gguf\nfile 0 draft.bin\nt x Q8_0 0 0 1 1\n",
            "file 0 /tmp/weights.gguf\nt x Q8_0 0 0 1 1\n",
            "file 0 weights.gguf\nt x Q8_0 0 10 20 1\n",
            "file 0 weights.gguf\nt x Q8_0 0 -1 1 1\n",
        ]
        for case in cases:
            with self.subTest(case=case):
                index.write_text(case)
                with self.assertRaises(identity.IdentityError):
                    identity.create_manifest(pack)

    def test_manifest_tampering_is_rejected(self):
        pack = self.make_pack("a")
        m = identity.create_manifest(pack)
        for mutate in (lambda x: x["canonical"]["sources"][0].update(sha256="0" * 64),
                       lambda x: x["local"].pop(),
                       lambda x: x.update(identity="0" * 64)):
            changed = copy.deepcopy(m)
            mutate(changed)
            with self.assertRaises(identity.IdentityError):
                identity.check_manifest(changed, pack)

    def test_cli_creates_without_overwriting_and_checks(self):
        pack = self.make_pack("a")
        manifest = self.base / "manifest.json"
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(identity.main(["create", "--pack", str(pack), "--manifest", str(manifest)]), 0)
        digest = output.getvalue().strip()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(identity.main(["check", "--pack", str(pack), "--manifest", str(manifest)]), 0)
        self.assertEqual(output.getvalue().strip(), digest)
        before = manifest.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(identity.main(["create", "--pack", str(pack), "--manifest", str(manifest)]), 1)
        self.assertEqual(manifest.read_bytes(), before)
        self.assertFalse(list(self.base.glob(".manifest.json.*")))


if __name__ == "__main__":
    unittest.main()
