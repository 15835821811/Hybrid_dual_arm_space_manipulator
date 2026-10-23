"""Algebra, scheduling, and saved-pose rendering checks; no simulation steps."""
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
import mujoco
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.spline import Reference
from ff_prior.storage import read_json
from .protocol import P,ROOT,OUT
from .motion import matrices,Motion
from .solver import select_active,merit,solve_step
from .budget import Budget,SliceBudget,BudgetExhausted
from .media_geometry import frame_segments,add_saved_overlays,JS_OVERLAYS


class Algebra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Actual compiled robot metadata only. No MjData, rollout, geometry, or network.
        cls.robot=RobotAdapter();cls.q0=cls.robot.spec.planner_zero
        cls.motion=Motion(cls.robot,cls.q0)

    def test_derivative_control_matrices(self):
        z=np.tile(self.q0,(19,1))+.01*np.sin(np.arange(323)).reshape(19,17)
        maps,offsets=matrices(self.q0,27);ref=Reference(z,self.q0,27)
        for d in range(3):
            c=ref.curves[d];exact=c.c[:len(c.t)-c.k-1]/27**d
            np.testing.assert_allclose(maps[d]@z+offsets[d],exact,atol=3e-14,rtol=0)

    def test_initial_and_terminal_derivatives(self):
        z=np.tile(self.q0,(19,1))+.03*np.cos(np.arange(323)).reshape(19,17)
        ref=Reference(z,self.q0,27)
        np.testing.assert_array_equal(ref.evaluate(0),self.q0)
        for d in (1,2):np.testing.assert_allclose(ref.evaluate([0,27],d),0,atol=1e-13)

    def test_constant_motion_full_native_mapping(self):
        self.assertTrue(self.motion.valid(np.tile(self.q0,(19,1))))
        self.assertEqual(self.robot.B.shape,(67,17))
        self.assertEqual(self.motion.A.shape[1],323)

    def test_reject_rapid_motion(self):
        z=np.tile(self.robot.lower+1e-8,(19,1));z[1::2]=self.robot.upper-1e-8
        self.assertFalse(self.motion.valid(z))

    def test_constrained_fit_and_free_endpoint(self):
        t=np.linspace(0,27,65);desired=self.q0+.03*(t[:,None]/27)**2
        z,report=self.motion.fit(t,desired)
        self.assertTrue(report['returned_motion_valid'])
        self.assertLess(report['returned_loss'],report['initial_loss'])
        self.assertFalse(report['endpoint_fixed'])

    def test_solver_preserves_motion_and_can_change_endpoint(self):
        z=np.tile(self.q0,(19,1));goal=np.zeros(12);goal[0]=3
        jg=np.zeros((12,323));jg[0,-7]=100.;jd=np.empty((0,323))
        step,report=solve_step(z,dict(goal=goal,distances=np.empty(0)),jg,jd,self.motion,.0375,self.robot)
        self.assertTrue(report['returned_motion_valid'])
        self.assertLess(report['returned_objective'],report['initial_objective'])
        self.assertNotEqual(float(step[-7]),0.)

    def test_active_selection_coverage(self):
        near=np.asarray([[i/100,1+i%3,8+i%4,-.0001*i] for i in range(100)])
        ids,why=select_active(near,80)
        self.assertEqual(len(ids),80);self.assertEqual(len(set(ids)),80)
        self.assertEqual(len(why),80)
        self.assertEqual(set(map(tuple,near[ids,1:3])),set(map(tuple,near[:,1:3])))
        self.assertEqual(len(np.setdiff1d(np.arange(100),ids)),20)
        np.testing.assert_array_equal(ids,select_active(near,80)[0])

    def test_merit_uses_world_and_collision(self):
        goal=np.ones(12)
        self.assertGreater(merit(goal,[-.001]),merit(goal,[.001]))
        self.assertGreater(merit(goal,[.001]),merit(np.zeros(12),[.001]))

    def test_worst_cost_and_cycle_limits(self):
        calls=323+3+4+6*6
        intervals=327*2288+sum(n+240 for n in P['sparse_grids'])+6*sum(P['dense_grids'])
        self.assertEqual((calls,intervals),(P['iteration_query_reserve'],P['iteration_interval_reserve']))
        self.assertLessEqual(P['iterations']*calls+6,P['restore_coupled_cap'])
        self.assertLessEqual(P['iterations']*intervals+sum(P['dense_grids']),P['restore_interval_cap'])
        self.assertLessEqual(P['prior_D3_coupled']+P['coupled_cap']+P['future_dynamic_coupled_reserve'],60000)
        self.assertLessEqual(P['prior_D3_worker_s']+P['wall_s']+P['future_dynamic_wall_reserve_s'],P['D3_wall_s'])
        self.assertLessEqual(P['prior_D3_bytes']+P['output_bytes_cap']+P['evidence_bytes_reserve']+P['future_dynamic_output_bytes'],20*1024**3)
        self.assertLessEqual(P['native_steps_cap']+P['future_dynamic_native_reserve'],165000)

    def test_budget_persists_before_call(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            b=Budget(Path(d),time.time()+60);b.begin_unit();b.consume(512)
            saved=read_json(Path(d)/'budget.json');self.assertEqual(saved['reserved_coupled'],1)
            self.assertEqual(saved['integration_intervals'],512);self.assertEqual(saved['actual_coupled'],0)

    def test_budget_hard_stop(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            b=Budget(Path(d),time.time()+60);b.begin_unit();s=SliceBudget(b,1,512,time.time()+60)
            s.consume(512)
            with self.assertRaises(BudgetExhausted):s.consume(1)

    def test_saved_frame_axis_columns(self):
        T=np.eye(4);T[:3,3]=[1,2,3];T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        for j,(a,b) in enumerate(frame_segments(T,.1)):
            np.testing.assert_array_equal(a,T[:3,3]);np.testing.assert_allclose(b-a,.1*T[:3,j],atol=2e-16)

    def test_overlay_real_mujoco_geometries_no_steps(self):
        scene=mujoco.MjvScene(self.robot.model,maxgeom=1000)
        ee=np.tile(np.eye(4),(3,2,1,1));ee[:,0,0,3]=[0,.01,.02];ee[:,1,1,3]=[0,.01,.02]
        add_saved_overlays(scene,ee,ee[-1],2,np.arange(3))
        self.assertEqual(scene.ngeom,16)
        self.assertIn('Goal1 Z',[scene.geoms[i].label for i in range(scene.ngeom)])

    def test_no_physics_or_neural_work(self):
        for key in ('physics_steps','coupled','distance','forward','fk','mass_matrix'):
            self.assertEqual(self.robot.calls[key],0)

    def test_windows_path_envelope(self):
        paths=['units/u00/restore/iter/35/jacobian.npz.4294967295.tmp',
            'units/u00/validated/grid_16384_geometry_scans.json.4294967295.tmp',
            'scenes/capture_5_obstacles/initial_audit.json.4294967295.tmp']
        self.assertLess(max(len(str(OUT/p)) for p in paths),260)


if __name__=='__main__':unittest.main()
