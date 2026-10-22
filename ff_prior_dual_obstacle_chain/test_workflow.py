import unittest,inspect
import numpy as np
from ff_prior.storage import read_json
from v6_lite.irregular_waypoints import IrregularWaypointTarget
from ff_prior_terminal_tracking.environment import Environment as Controller
from .path import BypassPath
from .protocol import P,PACKAGE,DYNAMIC

class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=read_json(PACKAGE/'evidence/contract.json');w=cls.c['waypoints']
        cls.w=w;cls.old=IrregularWaypointTarget(np.asarray(w['initial_position_w']),np.asarray(w['waypoint_points_m']),
            np.asarray(w['target_rotation_world']),w['transition_duration_s'],w['path_duration_s'],np.asarray(w['segment_durations_s']),w)
        cls.path=BypassPath(cls.old,P['bypass_offset_m'])

    def test_exact_waypoints_and_timing(self):
        old=read_json(DYNAMIC)['waypoints']
        for key in ('waypoint_points_m','segment_durations_s','target_rotation_world'):
            np.testing.assert_array_equal(self.w[key],old[key])
        np.testing.assert_allclose([self.path.sample(t)[0] for t in self.path.boundaries],self.w['waypoint_points_m'],rtol=0,atol=1e-14)
        self.assertAlmostEqual(self.path.boundaries[-1],25.5)

    def test_all_new_obstacles_really_block_old_reference(self):
        centers=np.array([o['position'] for o in self.c['scene']['objects'][3:]])
        mid=(self.path.boundaries[:-1]+self.path.boundaries[1:])/2
        np.testing.assert_allclose([self.old.sample(t)[0] for t in mid],centers,rtol=0,atol=1e-14)
        self.assertEqual(len(centers),6)

    def test_derivative_and_C2_boundaries(self):
        eps=1e-5
        for t in np.linspace(4.501,25.499,30):
            p,v=self.path.sample(t);fd=(self.path.sample(t+eps)[0]-self.path.sample(t-eps)[0])/(2*eps)
            np.testing.assert_allclose(v,fd,rtol=1e-6,atol=1e-8)
        for t in self.path.boundaries:
            a=self.path.sample(t-eps);b=self.path.sample(t+eps)
            np.testing.assert_allclose(a[0],b[0],atol=1e-10,rtol=0)
            self.assertLess(np.linalg.norm(a[1]-b[1]),1e-8)

    def test_reference_does_not_intersect_new_balls(self):
        a=np.load(PACKAGE/'evidence/reference.npz')
        self.assertGreater(np.min(np.linalg.norm(a['position'][:,None]-a['centers'][None],axis=-1))-P['obstacle_radius_m'],.01)

    def test_initial_fullbody_transfer(self):
        p=read_json(PACKAGE/'evidence/preflight.json')
        self.assertTrue(p['initial_geometry']['passed'])
        self.assertTrue(p['initial_transfer']['complete_state_fields_preserved'])
        self.assertEqual(p['new_native_steps'],0)

    def test_same_rigid_obstacles_and_same_static_targets(self):
        from .protocol import ROOT
        a=read_json(ROOT/'ff_prior_diffusion_demo/evidence/selection.json')['entries'][0]
        b=read_json(PACKAGE/'evidence/selection.json')['entries'][0]
        self.assertEqual(b['scene']['objects'][:3],a['scene']['objects'])
        self.assertEqual(b['task']['condition'],a['task']['condition'])
        self.assertEqual(b['task']['contract']['acceptance'],a['task']['contract']['acceptance'])

    def test_no_actual_state_writes_in_controller(self):
        src=inspect.getsource(Controller.advance_native)
        for forbidden in ('qpos[:]','qvel[:]','mj_resetData','mj_setState','data.time =','d.time ='):
            self.assertNotIn(forbidden,src)

    def test_budget_and_original_gates(self):
        self.assertEqual(P['native_steps_cap'],27500+27000)
        self.assertLessEqual(P['coupled_cap'],8000)
        self.assertLessEqual(27000+P['native_steps_cap'],120000)
        self.assertFalse(P['TEST'] or P['training'] or P['expansion'])
        s=(PACKAGE/'evaluation.py').read_text(encoding='utf-8')
        self.assertIn('full_integration_replay=fullerr<=1e-10',s)
        self.assertIn('np.all(pos<=.005)',s)
        self.assertIn('env.max_geom_step<=.002',s)

if __name__=='__main__':unittest.main()
