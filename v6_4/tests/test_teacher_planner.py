import unittest
import hashlib
from dataclasses import replace
from unittest.mock import patch
import numpy as np

from v6_4.teacher_planner import TeacherPlanner, load_bootstrap
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_4.tests import test_reference_adapter as _adapter_tests


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _adapter_tests.AdapterTests.setUpClass.__func__(cls)

    data=_adapter_tests.AdapterTests.data
    def test_multistart_retains_rejected_attempts_without_success_labels(self):
        codec=CubicBSplineCodec(self.task.initial_planner_q,self.task.initial_planner_dq)
        controls=codec.decode_free(np.tile(self.task.initial_planner_q,(30,1)))
        times=np.linspace(0,27,100);prior=np.tile(self.task.initial_planner_q,(100,1))
        planner=TeacherPlanner(self.spec,max_starts=2,optimization_rounds=1)
        with patch.object(planner,'_optimize',return_value=(controls,[])):
            attempts=planner.propose(self.task,times,prior,model=self.model,initial_data=self.data(),pairs=self.verifier.pairs)
        self.assertEqual(len(attempts),2)
        self.assertTrue(all(not item.screening_passed for item in attempts))
        self.assertTrue(all(not item.to_dict()['demonstration_eligible'] for item in attempts))
        self.assertTrue(all(item.proposal.postprocessing==() for item in attempts))
        self.assertTrue(all('constrained_spline_fit' in item.metadata['intrinsic_teacher_operations'] for item in attempts))
        self.assertEqual(attempts[0].metadata['path_offset_rad'],[0.]*6)
        self.assertNotEqual(attempts[1].metadata['path_offset_rad'],[0.]*6)

    def test_failure_saved_and_incomplete_prior_rejected(self):
        planner=TeacherPlanner(self.spec,max_starts=2)
        times=np.linspace(0,27,100);prior=np.tile(self.task.initial_planner_q,(100,1))
        with patch.object(planner,'_optimize',side_effect=RuntimeError('planned failure')):
            attempts=planner.propose(self.task,times,prior,model=self.model,initial_data=self.data(),pairs=self.verifier.pairs)
        self.assertEqual(len(attempts),2)
        self.assertTrue(all(item.metadata['failure']['message']=='planned failure' for item in attempts))
        with self.assertRaises(ValueError):planner.propose(self.task,times[:-1],prior[:-1])

    def test_test_prior_must_be_disjoint_and_bound_to_source_arrays(self):
        task=replace(self.task,split='test')
        planner=TeacherPlanner(self.spec,max_starts=1)
        times=np.linspace(0,27,100);prior=np.tile(task.initial_planner_q,(100,1))
        metadata={'source_split':'train','source_group_id':'independent_train',
            'time_array_sha256':hashlib.sha256(times.tobytes()).hexdigest(),
            'planner_q_array_sha256':hashlib.sha256(prior.tobytes()).hexdigest()}
        for bad in [None,dict(metadata,source_split='test'),dict(metadata,source_group_id=task.group_id),
                    dict(metadata,source_group_id=None),dict(metadata,planner_q_array_sha256='0'*64)]:
            with self.assertRaises(ValueError):planner.propose(task,times,prior,bootstrap_metadata=bad)
        with patch.object(planner,'_environment',side_effect=RuntimeError('validated source')):
            with self.assertRaisesRegex(RuntimeError,'validated source'):
                planner.propose(task,times,prior,bootstrap_metadata=metadata)

    def test_load_bootstrap_rejects_test_future_trace_before_read(self):
        from pathlib import Path
        with self.assertRaisesRegex(ValueError,'training TaskSpec'):
            load_bootstrap(Path('does_not_exist.npz'),source_task=replace(self.task,split='test'))

    def test_geometry_free_step_preserves_start_and_all_task_q_anchors(self):
        codec=CubicBSplineCodec(self.task.initial_planner_q,self.task.initial_planner_dq)
        times=np.array(sorted({point.time_s for point in self.task.requirements}))
        anchors=codec.basis(times)[:,2:]
        gradient=np.linspace(.1,1.,17)
        step=TeacherPlanner._geometry_free_direction(codec,[anchors]*17,1.,gradient,.03)
        self.assertIsNotNone(step);self.assertLessEqual(np.max(abs(step)),.25+1e-15)
        np.testing.assert_allclose(anchors@step,0.,atol=1e-13,rtol=0.)
        changed=codec.decode_free(np.tile(self.task.initial_planner_q,(30,1))+step)
        np.testing.assert_array_equal(changed[:2],codec.fixed_controls)
        self.assertGreater(float(codec.basis(np.array([1.]))[0,2:]@step@gradient),0.)
        self.assertIsNone(TeacherPlanner._geometry_free_direction(codec,[anchors]*17,1.,np.zeros(17),.03))

    def test_geometry_step_releases_unrequired_arm_at_other_arm_anchor(self):
        codec=CubicBSplineCodec(self.task.initial_planner_q,self.task.initial_planner_dq)
        continuum=codec.basis(np.array([9.,22.,27.]))[:,2:]
        rigid=codec.basis(np.array([27.]))[:,2:]
        gradient=np.ones(17)
        step=TeacherPlanner._geometry_free_direction(codec,[continuum]*10+[rigid]*7,22.,gradient,.03)
        self.assertIsNotNone(step)
        np.testing.assert_allclose(continuum@step[:,:10],0.,atol=1e-13,rtol=0.)
        np.testing.assert_allclose(rigid@step[:,10:],0.,atol=1e-13,rtol=0.)
        self.assertGreater(np.linalg.norm(codec.basis(np.array([22.]))[0,2:]@step[:,10:]),.001)
        changed=codec.decode_free(np.tile(self.task.initial_planner_q,(30,1))+step)
        np.testing.assert_array_equal(changed[:2],codec.fixed_controls)
        self.assertLessEqual(np.max(abs(step)),.25+1e-15)

    def test_native_geometry_gradient_is_coupled_private_and_has_fixed_budget(self):
        import mujoco
        planner=TeacherPlanner(self.spec)
        pair=next(p for p in self.verifier.pairs if p.pair_class=='arm_obstacle' and p.body_a_name=='end_link')
        data=self.data();before=data.qpos.tobytes()
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('physics forbidden')):
            gradient,evidence=planner._geometry_gradient(self.model,data.qpos,pair)
        self.assertEqual(data.qpos.tobytes(),before)
        self.assertEqual(evidence['native_distance_queries'],34)
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(np.linalg.norm(gradient[:10]),1e-7)
        self.assertGreater(np.linalg.norm(gradient[10:]),1e-7)

    def test_geometry_budget_closed_and_legacy_disable_explicit(self):
        self.assertEqual(TeacherPlanner(self.spec,geometry_optimization_rounds=0).geometry_optimization_rounds,0)
        self.assertEqual(TeacherPlanner(self.spec,geometry_optimization_rounds=32).geometry_optimization_rounds,32)
        for kwargs in [{'geometry_optimization_rounds':33},{'geometry_optimization_rounds':-1},
                       {'geometry_line_search_steps':5},{'geometry_optimization_rounds':True}]:
            with self.assertRaises(ValueError):TeacherPlanner(self.spec,**kwargs)

    def test_joint_witness_step_preserves_required_anchors_and_time_budget(self):
        codec=CubicBSplineCodec(self.task.initial_planner_q,self.task.initial_planner_dq)
        continuum=codec.basis(np.array([9.,22.,27.]))[:,2:]
        rigid=codec.basis(np.array([27.]))[:,2:]
        times=np.array([1.,5.,13.,23.]);gradients=np.tile(np.linspace(.1,1.,17),(4,1));deficits=np.full(4,.005)
        step=TeacherPlanner._geometry_joint_free_direction(codec,[continuum]*10+[rigid]*7,times,gradients,deficits)
        self.assertIsNotNone(step)
        np.testing.assert_allclose(continuum@step[:,:10],0.,atol=1e-13,rtol=0.)
        np.testing.assert_allclose(rigid@step[:,10:],0.,atol=1e-13,rtol=0.)
        increases=np.sum((codec.basis(times)[:,2:]@step)*gradients,axis=1)
        np.testing.assert_allclose(increases,deficits,atol=1e-12,rtol=0.)
        witnesses=[{'state_index':i,'time_s':i*.02,'minimum_m':-.01+i*1e-6} for i in range(1351)]
        selected=TeacherPlanner._geometry_active_witnesses({'state_witnesses':witnesses})
        self.assertEqual(len(selected),8);self.assertEqual(selected[0]['state_index'],0)
        self.assertEqual(len({min(7,int(w['time_s']/27*8)) for w in selected}),8)
        self.assertEqual(TeacherPlanner._geometry_active_witnesses({'state_witnesses':[dict(witnesses[0],minimum_m=.008)]}),[])


if __name__=='__main__':unittest.main()
