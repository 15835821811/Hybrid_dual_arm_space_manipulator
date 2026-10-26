import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v6_4.teacher_prior import (BUDGET, FiniteTeacherPrior, build_task_center,
                              rate_project_coordinate, select_center, task_max_ratio)
from v6_4.teacher_planner import TeacherAttempt
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


class PriorRateTests(unittest.TestCase):
    def setUp(self):
        self.codec = CubicBSplineCodec(np.zeros(17), np.zeros(17))
        self.controls = self.codec.decode_free(np.zeros((30, 17)))

    def test_nonzero_initial_rate_and_fixed_derivative_contribution(self):
        q0 = np.full(17, .1); dq0 = np.zeros(17); dq0[0] = .05
        codec = CubicBSplineCodec(q0, dq0)
        full = codec.decode_free(np.tile(q0, (30, 1)))
        result = SimpleNamespace(x=full[2:, 0], success=True, status=0,
                                 message='known feasible', nit=0, nfev=1, njev=1)
        with patch('v6_4.teacher_prior.minimize', return_value=result) as solver:
            column, report = rate_project_coordinate(codec, full, 0, np.array([27.]), np.array([.1]),
                lower=-1., upper=1., velocity=.2, acceleration=4.)
        constraints = solver.call_args.kwargs['constraints']
        slack = constraints[0]['fun'](column)
        values = codec.sample(full, np.arange(1351)*.02)
        expected = np.r_[values['dq'][:,0]+.2, values['ddq'][:,0]+4.,
                         .2-values['dq'][:,0], 4.-values['ddq'][:,0]]
        np.testing.assert_allclose(slack, expected, atol=1e-14, rtol=0.)
        self.assertAlmostEqual(values['dq'][0,0], .05, places=14)
        self.assertTrue(report['accepted'])
        self.assertEqual(solver.call_args.kwargs['options']['maxiter'], 35)
        self.assertIsNotNone(solver.call_args.kwargs['jac'])
        np.testing.assert_allclose(constraints[1]['fun'](column), 0., atol=1e-14)
        np.testing.assert_array_equal(full[:2], codec.fixed_controls)

    def test_real_linear_projection_preserves_anchor_and_limits(self):
        full = self.controls.copy(); full[14,0] = .9
        column, report = rate_project_coordinate(self.codec, full, 0, np.array([27.]), np.array([0.]),
            lower=-1., upper=1., velocity=.14, acceleration=.4)
        self.assertTrue(report['accepted'], report)
        self.assertLessEqual(report['nit'], 35)
        changed = full.copy(); changed[2:,0] = column
        values = self.codec.sample(changed, np.arange(1351)*.02)
        self.assertLessEqual(np.max(abs(values['dq'][:,0])), .14+1e-12)
        self.assertLessEqual(np.max(abs(values['ddq'][:,0])), .4+1e-12)
        self.assertLess(abs(values['q'][-1,0]), 1e-10)
        np.testing.assert_array_equal(changed[:2], self.codec.fixed_controls)
        self.assertGreater(np.linalg.norm(changed-full), .1)

    def test_failed_and_infeasible_success_iterates_are_never_applied(self):
        for success in (False, True):
            result = SimpleNamespace(x=np.full(30,100.), success=success, status=9,
                                     message='failed', nit=35, nfev=35, njev=35)
            with patch('v6_4.teacher_prior.minimize', return_value=result) as solver:
                column, report = rate_project_coordinate(self.codec, self.controls, 0,
                    np.array([27.]), np.array([0.]), lower=-1., upper=1., velocity=.76, acceleration=2.5)
            np.testing.assert_array_equal(column, self.controls[2:,0])
            self.assertFalse(report['accepted'])
            self.assertEqual(report['candidate_free_column'], [100.]*30)
            self.assertFalse(report['failed_solution_applied'])
            self.assertEqual(solver.call_count, 1)

    def test_arm_specific_anchors_and_two_stage_budget(self):
        planner = FiniteTeacherPrior(default_v6_lite_robot_spec())
        task = SimpleNamespace(requirements=[SimpleNamespace(time_s=9., arm='continuum'),
                                            SimpleNamespace(time_s=27., arm='rigid')])
        fake = SimpleNamespace(prediction={'full_qpos': np.zeros((1351,81))})
        def ik(model, qpos, guess, points):
            value = np.zeros(17)
            if points[0].arm == 'continuum': value[0],value[10] = .1,.8
            else: value[0],value[10] = .8,.2
            return value, {'nfev':1, 'optimizer_success':True}
        with patch('v6_4.teacher_prior.SplineReferenceProvider') as provider, \
             patch.object(planner, '_anchor_configuration', side_effect=ik), \
             patch('v6_4.teacher_prior.requirement_results', return_value={'passed':False, 'requirements':[]}), \
             patch('mujoco.mj_step', side_effect=AssertionError('physics forbidden')):
            provider.return_value.prepare.return_value = fake
            result, logs = planner._optimize(task, self.codec, self.controls, None,None,None,None,None)
            result, post = planner._optimize(task, self.codec, result, None,None,None,None,None)
        sampled = self.codec.sample(result, np.array([9.,27.]))['q']
        self.assertAlmostEqual(sampled[0,0], .1, places=10)
        self.assertAlmostEqual(sampled[1,10], .2, places=10)
        self.assertLess(abs(sampled[0,10]), 1e-8)  # no rigid anchor at continuum point
        self.assertLess(abs(sampled[1,0]), 1e-8)   # no continuum anchor at rigid point
        self.assertEqual(len(logs),8); self.assertEqual(len(post),8)
        self.assertEqual(planner.costs['anchor_rate_rounds'],16)
        self.assertEqual(planner.costs['anchor_optimization_calls'],2)
        np.testing.assert_array_equal(result[:2], self.codec.fixed_controls)

    def test_early_stop_uses_fresh_prediction_and_rechecks_post_geometry_segment(self):
        planner = FiniteTeacherPrior(default_v6_lite_robot_spec())
        task = SimpleNamespace(requirements=[SimpleNamespace(time_s=27., arm='rigid')])
        fake = SimpleNamespace(prediction={'full_qpos':np.zeros((1351,81))})
        with patch('v6_4.teacher_prior.SplineReferenceProvider') as provider, \
             patch.object(planner, '_anchor_configuration', return_value=(np.zeros(17), {'nfev':1})), \
             patch('v6_4.teacher_prior.requirement_results', return_value={'passed':True, 'requirements':[]}) as requirement, \
             patch('mujoco.mj_step', side_effect=AssertionError('physics forbidden')):
            provider.return_value.prepare.return_value = fake
            result, pre = planner._optimize(task,self.codec,self.controls,None,None,None,None,None)
            result, post = planner._optimize(task,self.codec,result,None,None,None,None,None)
        self.assertEqual(len(pre),1); self.assertEqual(len(post),1)
        self.assertEqual(provider.return_value.prepare.call_count,4)
        self.assertEqual(requirement.call_count,2)
        self.assertEqual(planner.costs['anchor_rate_rounds'],2)
        self.assertTrue(pre[0]['early_stop']['passed'])
        self.assertTrue(post[0]['early_stop']['passed'])
        self.assertFalse(pre[0]['early_stop']['geometry_checked_here'])
        self.assertTrue(pre[0]['early_stop']['geometry_and_final_full_gate_still_required'])
        self.assertTrue(BUDGET['anchor_rounds_are_maxima'])
        np.testing.assert_array_equal(result[:2],self.codec.fixed_controls)

    def test_selection_preserves_finite_failed_centers_and_ties(self):
        rows=[{'attempt_index':i,'center_finite':True,'start_boundary_exact':True,
               'original_gate_passed':False,'task_max_ratio':2.,'static_score':1.} for i in range(8)]
        self.assertEqual(select_center(rows),0)
        rows[3]['task_max_ratio']=1.5;self.assertEqual(select_center(rows),3)
        rows[7]['original_gate_passed']=True;self.assertEqual(select_center(rows),7)
        rows[6]['original_gate_passed']=True;self.assertEqual(select_center(rows),6)
        rows[7]['static_score']=.9;self.assertEqual(select_center(rows),7)
        rows[7]['start_boundary_exact']=False;self.assertEqual(select_center(rows),6)
        self.assertIsNone(select_center([dict(r,center_finite=False) for r in rows]))
        self.assertEqual(task_max_ratio({'requirements':[{'position_error_m':.002,
            'orientation_error_rad':.008,'position_tolerance_m':.001,'orientation_tolerance_rad':.004}]}),2.)

    def test_finite_provider_failure_retains_all_slots_and_bound_center(self):
        spec = default_v6_lite_robot_spec()
        task = SimpleNamespace(task_id='fixture', group_id='fixture', split='train',
            sha256=lambda:'a'*64, initial_qpos=np.zeros(81), initial_qvel=np.zeros(79),
            initial_planner_q=np.zeros(17), initial_planner_dq=np.zeros(17),
            model_contract_sha256=spec.runtime_contract_sha256())
        model = SimpleNamespace(geom_contype=np.zeros(1), geom_conaffinity=np.zeros(1))
        initial = SimpleNamespace(qpos=np.zeros(81), qvel=np.zeros(79), ctrl=np.zeros(67))
        attempts = [TeacherAttempt(i,None,self.controls.copy(),None,False,
                                   {'failure':{'type':'bounded pilot failure'}}) for i in range(8)]
        planner = SimpleNamespace(propose=lambda *a,**k:attempts, costs={'physics_steps':0})
        with tempfile.TemporaryDirectory() as temporary, \
             patch('v6_4.teacher_prior.load_bootstrap', return_value=(np.arange(1,13501)*.002,np.zeros((13500,17)),{})), \
             patch('v6_4.teacher_prior.scenario_from_task', return_value=SimpleNamespace(obstacles=[])), \
             patch('model_test.whole_body_verifier_v5.WholeBodyCollisionVerifier', return_value=SimpleNamespace(model=model,pairs=[])), \
             patch('v6_4.teacher_prior.FiniteTeacherPrior', return_value=planner), \
             patch('v6_4.teacher_prior.SplineReferenceProvider') as provider, \
             patch('mujoco.MjData', return_value=initial), patch('mujoco.mj_forward'), \
             patch('mujoco.mj_step', side_effect=AssertionError('physics forbidden')):
            provider.return_value.prepare.side_effect = RuntimeError('finite failed forecast')
            row = build_task_center(task,Path(temporary)/'center',spec=spec)
            self.assertEqual(len(row['attempts']),8)
            self.assertEqual(row['chosen_attempt_index'],0)
            self.assertTrue(row['center_finite']);self.assertTrue(row['start_boundary_exact'])
            self.assertFalse(row['prior_only_gate_raw_passed']);self.assertFalse(row['prior_only_gate_passed'])
            gate=json.loads(Path(row['prior_only_gate_path']).read_text(encoding='utf-8'))
            self.assertEqual(gate['task_id'],'fixture');self.assertEqual(gate['task_sha256'],'a'*64)
            guard=json.loads(Path(row['prior_only_acceleration_guard_path']).read_text(encoding='utf-8'))
            self.assertEqual(guard['status'],'NOT_RUN');self.assertFalse(guard['passed'])
            self.assertTrue(row['model_assets_unchanged']);self.assertTrue(row['source_unchanged'])
            self.assertEqual(row['costs']['physics_steps'],0)


if __name__ == '__main__':
    unittest.main()
