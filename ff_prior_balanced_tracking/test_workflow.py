"""Pure software checks: public arithmetic, budget sequencing, and source contracts."""
import ast
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json
from .protocol import PACKAGE,DYNAMIC,P
from .reference import bypass
from .budget import Budget


class Software(unittest.TestCase):
    def test_source_parses(self):
        for path in PACKAGE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))

    def test_budget_exact_scope(self):
        self.assertEqual(P['executions'],1)
        self.assertEqual(P['steps_each'],13500)
        self.assertEqual(P['max_native_replay_steps'],27000)
        self.assertEqual(P['max_coupled'],2000)
        self.assertEqual(P['planned_coupled'],0)
        self.assertEqual(P['wall_s'],1200)
        self.assertEqual(P['max_bytes'],2*1024**3)
        self.assertFalse(P['training'] or P['expansion'] or P['TEST'] or P['advantage_comparison'])

    def test_original_waypoint_endpoints_and_times(self):
        w=read_json(DYNAMIC)['waypoints'];points=np.asarray(w['waypoint_points_m'])
        arrivals=4.5+np.r_[0.,np.cumsum(w['segment_durations_s'])]
        self.assertEqual(points.shape,(7,3));self.assertAlmostEqual(arrivals[-1],25.5)
        for point,t in zip(points,arrivals):
            p,v=bypass(point,np.zeros(3),t,w)
            np.testing.assert_allclose(p,point,atol=1e-14)
            np.testing.assert_allclose(v,0.,atol=1e-14)

    def test_bypass_midpoint_and_unselected_segments(self):
        w=read_json(DYNAMIC)['waypoints'];d=np.asarray(w['segment_durations_s'])
        starts=4.5+np.r_[0.,np.cumsum(d)[:-1]]
        for segment in range(6):
            p,v=bypass(np.zeros(3),np.zeros(3),starts[segment]+d[segment]/2,w)
            np.testing.assert_allclose(p,[-.07,0,0] if segment in (0,2,4) else [0,0,0],atol=1e-14)
            np.testing.assert_allclose(v,0.,atol=1e-14)
        for t in (0.,4.,25.5,27.):
            p,v=bypass(np.zeros(3),np.zeros(3),t,w)
            np.testing.assert_array_equal(p,np.zeros(3));np.testing.assert_array_equal(v,np.zeros(3))

    def test_c2_endpoint_formula_and_analytic_velocity(self):
        # p(u)=64*u^3(1-u)^3 has p,p',p'' zero at u=0,1.
        polynomial=np.polynomial.Polynomial([0,0,0,64,-192,192,-64])
        for order in range(3):
            np.testing.assert_allclose(polynomial.deriv(order)(np.asarray([0.,1.])),0.,atol=1e-12)
        w=read_json(DYNAMIC)['waypoints'];duration=w['segment_durations_s'][0]
        for u in (.1,.3,.7,.9):
            t=4.5+u*duration;epsilon=1e-6
            p,v=bypass(np.zeros(3),np.zeros(3),t,w)
            before,_=bypass(np.zeros(3),np.zeros(3),t-epsilon,w)
            after,_=bypass(np.zeros(3),np.zeros(3),t+epsilon,w)
            np.testing.assert_allclose(v,(after-before)/(2*epsilon),atol=1e-10)

    def test_budget_refusal_preserves_bookkeeping(self):
        with tempfile.TemporaryDirectory() as folder:
            b=Budget(Path(folder));b.physics_deadline=time.time()-1.
            with self.assertRaisesRegex(RuntimeError,'WALL_BUDGET'):b.reserve('native')
            self.assertEqual(b.counts,{'native':0,'replay':0})
            self.assertEqual(b.reserved,{'native':0,'replay':0});self.assertIsNone(b.inflight)

    def test_unknown_reservation_blocks_next(self):
        with tempfile.TemporaryDirectory() as folder:
            b=Budget(Path(folder));b.reserve('native')
            self.assertEqual(b.reserved['native'],1);self.assertEqual(b.counts['native'],0)
            with self.assertRaisesRegex(RuntimeError,'UNRESOLVED'):b.reserve('replay')
            b.complete('native');b.reserve('replay');b.complete('replay')
            self.assertEqual(b.counts,{'native':1,'replay':1})

    def test_pre_step_budget_precedes_command_mutation(self):
        source=(PACKAGE/'environment.py').read_text(encoding='utf-8-sig')
        native=source[source.index('    def advance_native'):]
        reservation=native.index("self.budget.reserve('native')")
        for target in ('self.reference_q=next_reference','d.ctrl[r.actuator_ids]=','self.controls.append','mujoco.mj_step('):
            self.assertLess(reservation,native.index(target))
        self.assertGreater(native.index("self.budget.complete('native')"),native.index('mujoco.mj_step('))
        for forbidden in ('qpos[:]=','qvel[:]=','mj_resetData','mj_setState','d.time='):
            self.assertNotIn(forbidden,native)

    def test_full_replay_and_actual_handoff_are_required(self):
        source=(PACKAGE/'evaluation.py').read_text(encoding='utf-8-sig')
        self.assertIn('replay_complete and fullerr<=1e-10',source)
        self.assertIn('replay_complete and err<=1e-10',source)
        self.assertIn('actual_terminal_handoff=',source)
        self.assertIn('all(gates.values())',source)
        self.assertIn("h['time']-env.origin_time>=25.5",source)
        freeze=(PACKAGE/'inputs.py').read_text(encoding='utf-8-sig')
        for required in ('balanced_restore_closeout.json','STATIC_REVIEW_NOT_CONFIRMED',
                         'SEVEN_SCENE_IDENTITY_MISMATCH','STATIC_FULL_REPLAY_MISMATCH',
                         'STATIC_FROZEN_SOURCE_CHANGED','DYNAMIC_SOURCE_NOT_IN_STATIC_BUNDLE',
                         'DYNAMIC_SOURCE_DIFFERS_FROM_STATIC_BUNDLE',
                         'D3_AUTHORIZED_BALANCED_TERMINAL_TRACKING_FREEZE'):
            self.assertIn(required,freeze)


if __name__=='__main__':unittest.main()
