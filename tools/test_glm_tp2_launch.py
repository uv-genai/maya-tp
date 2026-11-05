"""Paired GLM execution choices and owned-launch admission, no GPU required."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from tools.tp2 import launch


class GlmPairedLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'index.txt').write_text('checked model index')
        self.engine=self.root/'maya';self.engine.write_bytes(b'engine identity')
        self.settings={'base':{'exe':str(self.engine),'args':['--glm-pack',str(self.root)],'env':{}},
                       'parallel':4,'drafts':2,'mtp':1,'context':32768,'port':18515,
                       'host':'lead','rdma_device':'usb4_rdma5'}

    def test_rank_choices_and_preflight_match(self):
        with mock.patch.dict(os.environ,{'MAYA_GLM_NO_MTP':'1','STRATA_GLM_NO_SPEC':'1'}):
            _,lead,env0,identity0=launch.prepare(self.settings,0)
            _,peer,env1,identity1=launch.prepare(self.settings,1)
        self.assertEqual(identity0,identity1)
        for command in (lead,peer):
            self.assertEqual(command[command.index('--parallel')+1],'4')
            self.assertEqual(command[command.index('--max-context')+1],'32768')
        self.assertEqual(peer[peer.index('--tp-bootstrap-host')+1],'lead')
        self.assertIn('--serve',lead)
        for env in (env0,env1):
            self.assertEqual(env['MAYA_GLM_MULTI_MTP'],'2')
            self.assertEqual(env['STRATA_GLM_MULTI_MTP'],'2')
            self.assertNotIn('MAYA_GLM_NO_MTP',env)
            self.assertNotIn('STRATA_GLM_NO_SPEC',env)

    def test_capacity_or_feature_mismatch_changes_preflight(self):
        expected=launch.prepare(self.settings,0)[3]
        for key,value in [('parallel',2),('context',16384),('mtp',0),('drafts',3)]:
            changed={**self.settings,key:value}
            self.assertNotEqual(expected,launch.prepare(changed,1)[3])

    def test_invalid_options_never_start_pair(self):
        for options in [{'parallel':5},{'drafts':8},{'context':0}]:
            with mock.patch.object(launch,'load_settings',return_value={**self.settings,**options}), \
                 mock.patch.object(launch.owned,'run_pair') as start:
                self.assertEqual(launch.main(['run','19161']),1)
                start.assert_not_called()

    def test_one_sequence_keeps_existing_protocol_features(self):
        settings={**self.settings,'parallel':1,'drafts':8,'mtp':0}
        _,command,env,_=launch.prepare(settings,0)
        self.assertEqual(command[command.index('--parallel')+1],'1')
        self.assertEqual(env['MAYA_TP_MTP'],'0')
        self.assertNotIn('MAYA_GLM_MULTI_MTP',env)

    def test_stop_unowned_port_creates_no_process(self):
        with mock.patch.object(launch.subprocess,'Popen') as start:
            self.assertEqual(launch.main(['stop','19161','--state-dir',str(self.root)]),0)
            start.assert_not_called()

    def test_bare_stop_refuses_multiple_owned_sockets(self):
        for port in ('19160','19161'):
            directory=self.root/port;directory.mkdir();(directory/'control.sock').touch()
        with mock.patch.object(launch.owned,'stop_pair') as stop:
            self.assertEqual(launch.main(['stop','--state-dir',str(self.root)]),1)
            stop.assert_not_called()

if __name__=='__main__':unittest.main()
