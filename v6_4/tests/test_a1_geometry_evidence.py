"""Focused evidence/metric tests; mock geometry is not physical acceptance."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier,WholeBodyVerificationConfig
from v6_4.evaluate_planning import _save_native_geometry,_shape_error_metrics,evaluate_trial
from v6_4.tests.test_evaluate_planning import tiny_trace,fresh_state
from v6_4.tests.test_task_protocol import fixture_task


def pair(name='robot_target'):
    return SimpleNamespace(pair_class=name,geom_a=0,geom_b=1,geom_a_name='arm',geom_b_name='target',body_a_name='arm_body',body_b_name='target_body')


def geometry_fixture(task):
    config=WholeBodyVerificationConfig(minimum_clearance=.005,query_distance_max=2.5,
        adaptive_subdivisions=4,self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True)
    verifier=SimpleNamespace(config=config,pairs=(pair(),),robot_spec=SimpleNamespace(source_bundle_sha256=lambda:'asset-sha'))
    state=fresh_state(task);state['target_minimum_m']=np.full(11,.006);state['target_minimum_censored']=np.zeros(11,dtype=bool)
    whole={'feasible':True,'minimum_clearance':.006,'minimum_pair':{'signed_distance_m':.006},
        'violation_count':0,'negative_distance_query_count':0,'checked_state_count':5,'supplied_sample_count':2,
        'adaptive_subdivisions':4,'pair_count':1,'query_count':5,'truncated_query_count':0,
        'pair_policy_sha256':'policy-sha','continuous_time_certified':False}
    return state,whole,verifier


class A1GeometryEvidenceTests(unittest.TestCase):
    def test_geometry_survives_later_consumed_command_binding_failure(self):
        task=fixture_task();trace=tiny_trace(task);trace['task_selected_command'][0,0]=np.nan
        spec=SimpleNamespace(planner_lower=np.full(17,-np.pi),planner_upper=np.full(17,np.pi),torque_limits=np.full(67,4.))
        def replay(t,tr,cfg,out):
            state,whole,verifier=geometry_fixture(t)
            np.savez(out/'fresh_replay.npz',**state)
            native=_save_native_geometry(t,state,whole,verifier,verifier.pairs,
                {'below_threshold':0,'negative':0,'truncated':0},{'state_index':0},out)
            return state,spec,{'passed':True},native
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'trace.npz';np.savez(path,**trace)
            with patch('v6_4.evaluate_planning._replay',side_effect=replay):
                report=evaluate_trial(task,path,None,scenario_result={},qp_config=None,output_dir=root/'audit')
            self.assertFalse(report['evidence_valid']);self.assertFalse(report['task_success'])
            self.assertIn('consumed task_selected_command',report['errors'][0]['message'])
            native=json.loads((root/'audit/native_geometry.json').read_text())
            policy=json.loads((root/'audit/whole_body_policy.json').read_text())
            self.assertTrue(native['evidence_available']);self.assertTrue(native['passed'])
            self.assertEqual(native,report['native_geometry'])
            self.assertFalse(native['continuous_time_certified'])
            self.assertEqual(policy['configuration']['minimum_clearance'],.005)
            self.assertEqual(policy['configuration']['adaptive_subdivisions'],4)
            self.assertEqual(policy['intentional_target_contact_geom_names'],['collision_0072','collision_0073'])
            self.assertIn('not all 500Hz',native['whole_body_count_scope'])
            self.assertIn('native_geometry.json',json.loads((root/'audit/manifest.json').read_text()))

    def test_whole_body_negative_queries_count_once_without_more_native_queries(self):
        verifier=WholeBodyCollisionVerifier.__new__(WholeBodyCollisionVerifier)
        verifier.model=SimpleNamespace(nq=1)
        verifier.data=SimpleNamespace(qpos=np.zeros(1),qvel=np.zeros(1))
        verifier.config=WholeBodyVerificationConfig(minimum_clearance=.005,adaptive_subdivisions=1)
        verifier.pairs=(pair('arm_base'),pair('rigid_target'))
        verifier._pair_policy_sha256='unchanged-policy'
        verifier._interpolated_states=lambda q:(q.copy(),np.arange(len(q),dtype=float))
        with patch('model_test.whole_body_verifier_v5.mujoco.mj_forward') as forward,patch(
            'model_test.whole_body_verifier_v5.mujoco.mj_geomDistance',side_effect=[-.002,.003,.01,2.5]) as distance:
            result=verifier.verify_qpos_sequence(np.zeros((2,1))).to_dict()
        self.assertEqual(distance.call_count,4);self.assertEqual(forward.call_count,2)
        self.assertEqual(result['query_count'],4);self.assertEqual(result['negative_distance_query_count'],1)
        self.assertEqual(result['violation_count'],2);self.assertEqual(result['truncated_query_count'],1)
        self.assertEqual(result['pair_policy_sha256'],'unchanged-policy');self.assertFalse(result['feasible'])

    def test_aligned_17d_metrics_keep_group_norms_and_coordinate_rmse_separate(self):
        actual=np.zeros((2,17));actual[1,:10]=2.;actual[1,10:]=3.
        result=_shape_error_metrics(actual,np.zeros_like(actual),np.array([0.,.002]),
            [f'q{i}' for i in range(17)],reference_scope='frozen-selected')
        self.assertAlmostEqual(result['RMS_L2_17'],np.sqrt((10*4+7*9)/2))
        self.assertAlmostEqual(result['RMS_L2_continuum_10'],np.sqrt(20))
        self.assertAlmostEqual(result['RMS_L2_rigid_7'],np.sqrt(31.5))
        self.assertAlmostEqual(result['coordinate_rmse_17']['q0'],np.sqrt(2))
        self.assertAlmostEqual(result['coordinate_rmse_17']['q16'],3/np.sqrt(2))
        self.assertEqual(result['sample_count'],2);self.assertTrue(result['includes_initial_state'])
        self.assertFalse(result['is_end_effector_task_error'])

    def test_nonfinite_or_misaligned_shape_vectors_fail(self):
        for actual,reference,times in [(np.zeros((2,16)),np.zeros((2,16)),[0.,.002]),
            (np.full((2,17),np.nan),np.zeros((2,17)),[0.,.002]),
            (np.zeros((2,17)),np.zeros((1,17)),[0.,.002]),
            (np.zeros((2,17)),np.zeros((2,17)),[0.])]:
            with self.subTest(shape=actual.shape),self.assertRaises(ValueError):
                _shape_error_metrics(actual,reference,times,[f'q{i}' for i in range(17)],reference_scope='frozen-selected')


if __name__=='__main__':unittest.main()
