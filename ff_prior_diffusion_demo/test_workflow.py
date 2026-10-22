"""Software and frozen-scope checks only, not physical qualification."""
import ast
import shutil
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from .protocol import P,OUT,ROOT,PACKAGE
from .budget import Budget,SliceBudget,BudgetExhausted
from .experiment import assert_resume_safe

class Tests(unittest.TestCase):
    def test_original_single_diffusion_task(self):
        from .inputs import identity
        s=identity()['selection'];self.assertEqual(len(s['units']),1)
        self.assertEqual(s['entries'][0]['sample_id'],'capture_3_obstacles')
        self.assertEqual(s['entries'][0]['task']['condition']['duration'],27.)
        self.assertEqual(s['units'][0]['method'],'diffusion_geometry')
        self.assertEqual(len(s['units'][0]['proposal_seeds']),5)
    def test_budget_complete_not_second_run(self):
        self.assertLessEqual(40+2*P['repair_coupled_cap'],P['coupled_cap'])
        self.assertLessEqual(200000+2*P['repair_integration_cap'],P['integration_cap'])
        self.assertLessEqual(180+2*P['repair_wall_s']+P['execution_reserve_s'],P['unit_wall_s'])
        self.assertEqual(2*int(27.5/.002),P['native_steps_cap'])
        self.assertEqual(P['comparison_runs'],0);self.assertFalse(P['TEST'])
    def test_no_budget_transfer(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td),time.time()+60);b.begin_unit();s=SliceBudget(b,1,100,time.time()+60)
            s.consume(64)
            with self.assertRaises(BudgetExhausted):s.consume(1)
    def test_unknown_never_resumed(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);atomic_json(d/'ledger.json',dict(inflight={'unit':'u00'}))
            with self.assertRaisesRegex(ValueError,'UNKNOWN_WORK'):assert_resume_safe(d)
    def test_full_replay_gate(self):
        from ff_prior_static_goal_seed.physical_audit import replay_agrees
        self.assertTrue(replay_agrees({'qpos':0.,'integration':0.},1e-10))
        for x in ({'qpos':0.},{'integration':float('nan')},{'integration':1e-5}):self.assertFalse(replay_agrees(x,1e-10))
    def test_visualization_has_no_simulation_or_training_call(self):
        t=ast.parse((PACKAGE/'visuals.py').read_text(encoding='utf-8'))
        for n in ast.walk(t):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                self.assertNotIn(n.func.attr,('mj_step','backward','optimize'))
        self.assertIsNotNone(shutil.which('ffmpeg'))
    def test_actual_path_envelope(self):
        for rel in ('units/u00/candidates/04/repair/iterations/15/trial_06_pairs.json.4294967295.tmp',
                    'units/u00/candidates/04/validated/grid_16384_geometry_scans.json.4294967295.tmp'):
            self.assertLess(len(str(OUT/rel)),260)
    def test_public_scope_no_known_solution(self):
        for name in ('experiment.py','inputs.py'):
            s=(PACKAGE/name).read_text(encoding='utf-8')
            for forbidden in ('capture_trust_001','07_bow','optimizer.step','teachers/'):
                self.assertNotIn(forbidden,s)

if __name__=='__main__':unittest.main()
