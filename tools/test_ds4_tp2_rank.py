import json
import os
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ds4 import tp2_identity, tp2_rank, tp2_launch


class DeepSeekTP2RankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.pack = root / "pack"
        self.pack.mkdir()
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps({}))
        self.engine = root / "maya-ds4"
        self.engine.write_text("engine")
        self.identity = {"model_digest": "a" * 64, "partition_digest": "b" * 64, "build_digest": "c" * 64}
        self.base = ["--pack", str(self.pack), "--manifest", str(self.manifest), "--engine", str(self.engine),
                     "--host", "node.local", "--port", "18515", "--rdma-device", "usb4_rdma5",
                     "--context", "32768", "--expert-budget-mib", "90000"]

    def build(self, flags, environ=None):
        opts = tp2_rank.parser().parse_args([*flags, *self.base])
        with patch.object(tp2_rank, "prepare_identity", return_value=self.identity) as checked:
            args, env = tp2_rank.prepare_launch(opts, {} if environ is None else environ)
        checked.assert_called_once_with({}, str(self.pack), str(self.engine))
        return args, env

    def test_rank_commands_bind_the_same_pack_and_identity(self):
        root, a = self.build(["--serve", "--rank", "0"])
        worker, b = self.build(["--rank", "1"])
        self.assertIn("--serve", root)
        self.assertIn("--tp2-worker", worker)
        self.assertEqual(root[root.index("--dspark") + 1], "on")
        self.assertEqual(worker[worker.index("--dspark") + 1], "none")
        for name in ("PACK", "DSPARK", "CHUNK", "LOOKUP", "MODEL_DIGEST", "PARTITION_DIGEST", "BUILD_DIGEST"):
            self.assertEqual(a["MAYA_DS4_TP2_" + name], b["MAYA_DS4_TP2_" + name])
        self.assertEqual(a["DS4_STREAM"], "0")
        self.assertEqual(a["MAYA_DS4_TP2_CHUNK"], "0")
        self.assertEqual(a["MAYA_DS4_TP2_LOOKUP"], "0")

    def test_lookup_is_owned_by_rank_zero_and_bound_on_both_ranks(self):
        root, a = self.build(["--serve", "--rank", "0", "--suffix-draft", "3", "--dspark", "off"])
        worker, b = self.build(["--rank", "1", "--suffix-draft", "3", "--dspark", "off"])
        self.assertEqual(root[root.index("--suffix-draft") + 1], "3")
        self.assertEqual(worker[worker.index("--suffix-draft") + 1], "0")
        self.assertEqual(a["MAYA_DS4_TP2_LOOKUP"], "3")
        self.assertEqual(a["MAYA_DS4_TP2_LOOKUP"], b["MAYA_DS4_TP2_LOOKUP"])

    def test_chunk_prefill_binds_both_ranks_and_rejects_conflicting_environment(self):
        root, a = self.build(["--serve", "--rank", "0", "--prefill-chunk", "256"])
        worker, b = self.build(["--rank", "1", "--prefill-chunk", "256"])
        for args in (root, worker):
            self.assertEqual(args[args.index("--prefill-chunk") + 1], "256")
        for env in (a, b):
            self.assertEqual(env["MAYA_DS4_TP2_CHUNK"], "256")
            self.assertEqual(env["DS4_FLOW"], "0")
            self.assertEqual(env["DS4_YOCO"], "0")
        for conflict in ({"DS4_FLOW": "1"}, {"DS4_YOCO": "1"}, {"DS4_CHUNK": "256"}):
            with self.subTest(conflict=conflict), self.assertRaises(tp2_identity.IdentityError):
                self.build(["--serve", "--rank", "0", "--prefill-chunk", "256"], conflict)

    def test_invalid_rank_role_and_stream_are_rejected(self):
        for flags in (["--rank", "0"], ["--serve", "--rank", "1"]):
            with self.subTest(flags=flags), self.assertRaises(tp2_identity.IdentityError):
                self.build(flags)
        with self.assertRaisesRegex(tp2_identity.IdentityError, "DS4_STREAM"):
            self.build(["--serve", "--rank", "0"], {"DS4_STREAM": "1"})

    def test_parallel_is_shared_and_execution_identity_is_separate(self):
        root,a=self.build(['--serve','--rank','0','--parallel','4','--prefill-chunk','256'])
        peer,b=self.build(['--rank','1','--parallel','4','--prefill-chunk','256'])
        for command in (root,peer): self.assertEqual(command[command.index('--parallel')+1],'4')
        self.assertEqual(a['MAYA_DS4_TP2_EXECUTION_DIGEST'],b['MAYA_DS4_TP2_EXECUTION_DIGEST'])
        _,single=self.build(['--serve','--rank','0','--parallel','1','--prefill-chunk','256'])
        _,small=self.build(['--serve','--rank','0','--parallel','2','--prefill-chunk','256'])
        self.assertEqual(len({a['MAYA_DS4_TP2_EXECUTION_DIGEST'],single['MAYA_DS4_TP2_EXECUTION_DIGEST'],small['MAYA_DS4_TP2_EXECUTION_DIGEST']}),3)
        for env in (a,b,single,small): self.assertEqual(env['MAYA_DS4_TP2_MODEL_DIGEST'],'a'*64)

    def test_indexer_slab_identity_normalizes_aliases_and_disabled_defaults(self):
        flags = ["--serve", "--rank", "0", "--parallel", "2"]
        def digest(environment):
            return self.build(flags, environment)[1]["MAYA_DS4_TP2_EXECUTION_DIGEST"]
        disabled = digest({})
        enabled = digest({"DS4_AI_RUN": "1"})
        self.assertNotEqual(disabled, enabled)
        for values in ({"DS4_AI_RUN": "0"}, {"MAYA_DS4_AI_RUN": ""},
                       {"MAYA_DS4_AI_RUN": "0", "STRATA_DS4_AI_RUN": "1", "DS4_AI_RUN": "1"},
                       {"STRATA_DS4_AI_RUN": "0", "DS4_AI_RUN": "1"},
                       {"MAYA_DS4_AI_RUN": "", "STRATA_DS4_AI_RUN": "1"}):
            with self.subTest(disabled=values):
                self.assertEqual(digest(values), disabled)
        for values in ({"STRATA_DS4_AI_RUN": "1"}, {"MAYA_DS4_AI_RUN": "1"},
                       {"STRATA_DS4_AI_RUN": "1", "DS4_AI_RUN": "0"},
                       {"MAYA_DS4_AI_RUN": "1", "STRATA_DS4_AI_RUN": "0", "DS4_AI_RUN": "0"},
                       {"MAYA_DS4_AI_RUN": " +2 "}, {"DS4_AI_RUN": "-1"}):
            with self.subTest(enabled=values):
                self.assertEqual(digest(values), enabled)
        _, worker = self.build(["--rank", "1", "--parallel", "2"], {"MAYA_DS4_AI_RUN": "1"})
        self.assertEqual(worker["MAYA_DS4_TP2_EXECUTION_DIGEST"], enabled)

    def test_indexer_slab_rejects_ambiguous_or_out_of_range_values_before_launch(self):
        for value in ("on", "1x", "1.0", "1_0", "\u00a01", "2147483648", "-2147483649"):
            with self.subTest(value=value), self.assertRaisesRegex(tp2_identity.IdentityError, "DS4_AI_RUN"):
                self.build(["--serve", "--rank", "0", "--parallel", "2"], {"MAYA_DS4_AI_RUN": value})

    def test_default_preflight_retains_legacy_identity_format(self):
        settings=dict(tp2_launch.RANK_DEFAULTS,pack=str(self.pack),manifest=str(self.manifest),
                      engine=str(self.engine),host='rank0.local',rdma_device='usb4_rdma5')
        with patch.dict(os.environ,{},clear=True),patch.object(tp2_rank,'prepare_identity',return_value=self.identity):
            _,_,_,legacy=tp2_launch.prepare(settings,0)
            settings['parallel']=4
            _,_,_,batch=tp2_launch.prepare(settings,0)
            for value in ("0", "1"):
                settings.update(parallel=1, env={"MAYA_DS4_AI_RUN": value})
                self.assertEqual(tp2_launch.prepare(settings,0)[3], legacy)
        self.assertEqual(legacy,{'model':'a'*64,'partition':'b'*64,'build':'c'*64})
        self.assertEqual({k:batch[k] for k in legacy},legacy)
        self.assertEqual(set(batch)-set(legacy),{'execution'})

    def test_batch_proposal_modes_fail_before_launch(self):
        for extra,env in [(['--suffix-draft','3'],{}),([],{ 'DS4_DSPARK_SAMPLE':'1'})]:
            with self.subTest(extra=extra,env=env),self.assertRaises(tp2_identity.IdentityError):
                self.build(['--serve','--rank','0','--parallel','4',*extra],env)
        # Legacy one-sequence sampling remains available.
        _,env=self.build(['--serve','--rank','0'],{'DS4_DSPARK_SAMPLE':'1'})
        self.assertEqual(env['DS4_DSPARK_SAMPLE'],'1')
        for count in ('0','5'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                tp2_rank.parser().parse_args(['--serve','--rank','0','--parallel',count,*self.base])

    def test_drafter_can_be_disabled_for_both_ranks(self):
        root, a = self.build(["--serve", "--rank", "0", "--dspark", "off"])
        worker, b = self.build(["--rank", "1", "--dspark", "off"])
        self.assertEqual(root[root.index("--dspark") + 1], "none")
        self.assertEqual(worker[worker.index("--dspark") + 1], "none")
        self.assertEqual(a["MAYA_DS4_TP2_DSPARK"], "0")
        self.assertEqual(a["MAYA_DS4_TP2_DSPARK"], b["MAYA_DS4_TP2_DSPARK"])


if __name__ == "__main__":
    unittest.main()
