"""Preflight interface and numerical contract tests; no native integration."""
import ast,inspect,unittest
import numpy as np
from ff_prior.storage import read_json
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
from ff_prior_dynamic_reference.execution import integration
from ff_prior.coupled_predictor import pose_errors
from .protocol import PACKAGE,DYNAMIC,P
from .environment import Environment

class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=read_json(PACKAGE/'evidence/contract.json');cls.t=dict(np.load(PACKAGE/'evidence/terminal.npz'))
        cls.r=RobotAdapter(scene=SceneSpec.from_dict(cls.c['scene']))
        cls.env=Environment(cls.r,cls.t,cls.c,None,record_initial=False)

    def test_full_state_exact_and_time_nonzero(self):
        np.testing.assert_array_equal(integration(self.r,self.env.data),self.t['integration'])
        np.testing.assert_array_equal(self.env.data.qpos,self.t['qpos'])
        np.testing.assert_array_equal(self.env.data.qvel,self.t['qvel'])
        self.assertAlmostEqual(self.env.data.time,27.5,places=8)

    def test_original_seven_points_and_timing(self):
        old=read_json(DYNAMIC)['waypoints'];w=self.c['waypoints']
        for k in ('waypoint_points_m','segment_durations_s','target_rotation_world'):
            np.testing.assert_array_equal(w[k],old[k])
        self.assertEqual(len(w['waypoint_points_m']),7)
        self.assertEqual((w['transition_duration_s'],w['path_duration_s']),(4.5,21.))

    def test_reference_start_is_commanded_goal_not_actual(self):
        ref=self.env.desired_poses(self.env.origin_time)
        np.testing.assert_allclose(ref,np.asarray(self.c['static_goals']),atol=1e-14)
        pos,rot=pose_errors(self.r.ee(self.env.data),ref)
        self.assertTrue(np.all(pos<=.005) and np.all(rot<=np.pi/180))

    def test_reference_clock_and_bridge_continuity(self):
        end=self.env.reference(self.env.origin_time+25.5)
        np.testing.assert_allclose(end[0],self.c['waypoints']['waypoint_points_m'][-1],atol=1e-14)
        for t in (4.5,25.5):
            a=self.env.reference(self.env.origin_time+t-1e-6);b=self.env.reference(self.env.origin_time+t+1e-6)
            self.assertLess(np.linalg.norm(a[0]-b[0]),1e-5)
            self.assertLess(np.linalg.norm(a[2]-b[2]),1e-4)

    def test_command_and_target_velocity_preserved(self):
        np.testing.assert_array_equal(self.env.command,self.t['reference_qd'])
        np.testing.assert_array_equal(self.env.reference_q,self.t['reference_q'])
        np.testing.assert_array_equal(self.env.data.qvel[self.env.tv],self.c['target_velocity'])

    def test_no_actual_state_writes_in_controller(self):
        src=inspect.getsource(Environment.advance_native)
        for forbidden in ('qpos[:]','qvel[:]','mj_resetData','mj_setState','data.time =','d.time ='):
            self.assertNotIn(forbidden,src)
        self.assertEqual(src.count('mujoco.mj_step('),1)

    def test_budget_and_scope(self):
        self.assertEqual(P['max_native_replay_steps'],2*P['steps_each'])
        self.assertFalse(P['TEST'] or P['training'] or P['expansion'])
        self.assertEqual(P['executions'],1)

    def test_full_replay_participates_in_gate(self):
        source=(PACKAGE/'evaluation.py').read_text(encoding='utf-8')
        self.assertIn('replay_complete and fullerr<=1e-10',source)
        self.assertIn('all(gates.values())',source)
        self.assertIn("h['time']-env.origin_time>=25.5",source)

if __name__=='__main__':unittest.main()
