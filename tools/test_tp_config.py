"""TP2 config plumbing tests (CI, no GPU): the `tp` block in maya-*.json injects --tp-* flags."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maya import tp_inject_args, set_arg, flag_value, TP_FLAGS


class TestTpInjectArgs(unittest.TestCase):
    def test_no_tp_block(self):
        cfg = {"args": ["--glm-pack", "/p"]}
        tp_inject_args(cfg)
        self.assertEqual(cfg["args"], ["--glm-pack", "/p"])

    def test_tp_block_world_1(self):
        cfg = {"tp": {"world_size": 1}, "args": ["--glm-pack", "/p"]}
        tp_inject_args(cfg)
        self.assertEqual(cfg["args"], ["--glm-pack", "/p"])

    def test_tp_block_rank0(self):
        cfg = {
            "tp": {"world_size": 2, "rank": 0, "bootstrap_port": 18515, "rdma_device": "usb4_rdma5"},
            "args": ["--glm-pack", "/p"],
        }
        tp_inject_args(cfg)
        self.assertEqual(flag_value(cfg["args"], "--tp-world-size"), "2")
        self.assertEqual(flag_value(cfg["args"], "--tp-rank"), "0")
        self.assertEqual(flag_value(cfg["args"], "--tp-bootstrap-port"), "18515")
        self.assertEqual(flag_value(cfg["args"], "--tp-rdma-device"), "usb4_rdma5")
        self.assertIsNone(flag_value(cfg["args"], "--tp-bootstrap-host"))

    def test_tp_block_rank1(self):
        cfg = {
            "tp": {"world_size": 2, "rank": 1, "bootstrap_host": "strix-halo-1.local",
                   "bootstrap_port": 18515, "rdma_device": "usb4_rdma5"},
            "args": ["--glm-pack", "/p"],
        }
        tp_inject_args(cfg)
        self.assertEqual(flag_value(cfg["args"], "--tp-world-size"), "2")
        self.assertEqual(flag_value(cfg["args"], "--tp-rank"), "1")
        self.assertEqual(flag_value(cfg["args"], "--tp-bootstrap-host"), "strix-halo-1.local")

    def test_tp_overwrites_existing(self):
        cfg = {
            "tp": {"world_size": 2, "rank": 1},
            "args": ["--glm-pack", "/p", "--tp-world-size", "1", "--tp-rank", "0"],
        }
        tp_inject_args(cfg)
        self.assertEqual(flag_value(cfg["args"], "--tp-world-size"), "2")
        self.assertEqual(flag_value(cfg["args"], "--tp-rank"), "1")

    def test_no_args_key(self):
        cfg = {"tp": {"world_size": 2, "rank": 0}}
        tp_inject_args(cfg)
        self.assertIn("--tp-world-size", cfg["args"])

    def test_tp_flags_keys(self):
        self.assertEqual(TP_FLAGS["world_size"], "--tp-world-size")
        self.assertEqual(TP_FLAGS["rank"], "--tp-rank")
        self.assertEqual(TP_FLAGS["bootstrap_host"], "--tp-bootstrap-host")
        self.assertEqual(TP_FLAGS["bootstrap_port"], "--tp-bootstrap-port")
        self.assertEqual(TP_FLAGS["rdma_device"], "--tp-rdma-device")


class TestSetArg(unittest.TestCase):
    def test_add(self):
        args = []
        set_arg(args, "--foo", "bar")
        self.assertEqual(args, ["--foo", "bar"])

    def test_replace(self):
        args = ["--foo", "old"]
        set_arg(args, "--foo", "new")
        self.assertEqual(args, ["--foo", "new"])

    def test_flag_value(self):
        args = ["--a", "1", "--b", "2"]
        self.assertEqual(flag_value(args, "--a"), "1")
        self.assertEqual(flag_value(args, "--b"), "2")
        self.assertIsNone(flag_value(args, "--c"))


if __name__ == "__main__":
    unittest.main()
