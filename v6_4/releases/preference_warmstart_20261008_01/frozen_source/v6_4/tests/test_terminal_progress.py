from types import SimpleNamespace
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from v6_4.terminal_progress import terminal_progress,register_rigid_reference,TerminalProgressReferenceProvider
from v6_4.trajectory_codec import CubicBSplineCodec


class TerminalProgressTests(unittest.TestCase):
    def test_C2_endpoints_monotonic_identity_and_terminal_hold(self):
        times=np.linspace(0,27,2701);s,ds,dds=terminal_progress(times)
        self.assertTrue(np.all(np.diff(s)>=0));self.assertTrue(np.all(ds>=0))
        np.testing.assert_array_equal(s[times<=24],times[times<=24])
        for t,expected in [(24.,(24.,1.,0.)),(26.5,(27.,0.,0.)),(27.,(27.,0.,0.))]:
            np.testing.assert_allclose(terminal_progress(t),expected,atol=1e-12,rtol=0.)
        self.assertAlmostEqual(terminal_progress(25.07142857142857)[1],640/343,places=12)
        self.assertGreater(ds.max(),1.)
        for t in [-1.1e-9,27.+1.1e-9,float('nan')]:
            with self.assertRaises(ValueError):terminal_progress(t)
        self.assertEqual(terminal_progress(27.+5e-10),(27.,0.,0.))

    def test_posture_chain_derivatives_match_numerical_composition(self):
        codec=CubicBSplineCodec(np.zeros(17),np.zeros(17))
        controls=codec.decode_free(np.tile(np.linspace(0.,.2,30)[:,None],(1,17)))
        base=SimpleNamespace(task=SimpleNamespace(sha256=lambda:'same'),prediction={},codec=codec,controls=controls)
        provider=TerminalProgressReferenceProvider(base.task,base)
        h=1e-4;t=25.2
        v=provider.posture_at_physical_time(np.array([t-h,t,t+h]))
        np.testing.assert_allclose((v['q'][2]-v['q'][0])/(2*h),v['dq'][1],atol=1e-9,rtol=0.)
        np.testing.assert_allclose((v['dq'][2]-v['dq'][0])/(2*h),v['ddq'][1],atol=1e-8,rtol=0.)
        end=provider.posture_at_physical_time(np.array([26.5,26.75,27.]))
        np.testing.assert_array_equal(end['dq'],np.zeros((3,17)));np.testing.assert_array_equal(end['ddq'],np.zeros((3,17)))

    def test_actual_clock_near_terminal_knot_has_only_endpoint_roundoff(self):
        codec=CubicBSplineCodec(np.zeros(17),np.zeros(17))
        controls=codec.decode_free(np.zeros((30,17)))
        base=SimpleNamespace(task=SimpleNamespace(sha256=lambda:'same'),prediction={},codec=codec,controls=controls)
        provider=TerminalProgressReferenceProvider(base.task,base)
        times=np.array([26.5-1e-11,26.5-1e-9,27.+5e-10])
        s,_,_=terminal_progress(times)
        self.assertTrue(np.all(s<=27.));self.assertTrue(np.all(s>=0.))
        sample=provider.posture_at_physical_time(times)
        self.assertTrue(all(np.all(np.isfinite(v)) for v in sample.values()))

    def test_registered_rigid_velocity_and_angular_transport_finite_difference(self):
        def nominal(s):
            r=Rotation.from_rotvec([0.,0.,.02*s]).as_matrix()
            return np.array([.01*s,0.,0.]),np.array([.01,0.,0.]),r,np.array([0.,0.,.02])
        def actual(t):
            r=Rotation.from_rotvec([0.,0.,-.03*t]).as_matrix()
            return np.array([0.,.02*t,0.]),np.array([0.,.02,0.]),r,np.array([0.,0.,-.03])
        def base(s):
            return {'rigid_target_position':np.array([1.+.005*s*s,.2,0.]),'rigid_target_velocity':np.array([.01*s,0.,0.]),
                'rigid_target_rotation':Rotation.from_rotvec([0.,0.,.04*s]).as_matrix(),'rigid_target_angular_velocity':np.array([0.,0.,.04])}
        def registered(t):
            s,ds,_=terminal_progress(t)
            return register_rigid_reference(base(s),nominal(s),actual(t),ds)
        t=25.2;h=1e-5;before,center,after=registered(t-h),registered(t),registered(t+h)
        np.testing.assert_allclose((after['rigid_target_position']-before['rigid_target_position'])/(2*h),center['rigid_target_velocity'],atol=1e-9,rtol=0.)
        w=Rotation.from_matrix(after['rigid_target_rotation']@before['rigid_target_rotation'].T).as_rotvec()/(2*h)
        np.testing.assert_allclose(w,center['rigid_target_angular_velocity'],atol=1e-9,rtol=0.)
        held=registered(26.75)
        np.testing.assert_allclose(held['rigid_target_angular_velocity'],actual(26.75)[3],atol=1e-15,rtol=0.)
        self.assertGreater(np.linalg.norm(held['rigid_target_velocity']),0.)

    def test_current_target_physical_clock_private_forward_preserves_actual_cache(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="target"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        live=mujoco.MjData(model);live.qpos[0]=2.;live.qvel[:]=[.1,.2,0.,0.,0.,.3];live.time=26.75
        # Deliberately stale actual cache must not be refreshed by the provider.
        codec=CubicBSplineCodec(np.zeros(17),np.zeros(17));controls=codec.decode_free(np.zeros((30,17)))
        task=SimpleNamespace(sha256=lambda:'same')
        def base_sample(s):
            return {'rigid_target_position':np.array([s+1.,0.,0.]),'rigid_target_velocity':np.array([1.,0.,0.]),
                'rigid_target_rotation':np.eye(3),'rigid_target_angular_velocity':np.zeros(3),
                'continuum_target_position':np.array([s,1.,0.]),'continuum_target_velocity':np.array([1.,0.,0.]),
                'continuum_target_rotation':np.eye(3),'continuum_target_angular_velocity':np.zeros(3)}
        base=SimpleNamespace(task=task,prediction={},codec=codec,controls=controls,sample=base_sample)
        provider=TerminalProgressReferenceProvider(task,base)
        provider._model=model;provider._live_data=live;provider._private_live_data=mujoco.MjData(model);provider._target_body=1
        provider._nominal_target=lambda s:(np.array([s,0.,0.]),np.array([1.,0.,0.]),np.eye(3),np.zeros(3))
        snapshots={k:getattr(live,k).copy() for k in ['qpos','qvel','ctrl','qacc','qM','xpos','xmat','qfrc_bias']}
        with patch('mujoco.mj_step',side_effect=AssertionError('no physics')):
            sample=provider.sample(26.75)
        for k,old in snapshots.items():np.testing.assert_array_equal(getattr(live,k),old)
        np.testing.assert_allclose(sample['rigid_target_position'],[3.,0.,0.],atol=1e-15,rtol=0.)
        np.testing.assert_allclose(sample['rigid_target_velocity'],[.1,.5,0.],atol=1e-15,rtol=0.)
        np.testing.assert_array_equal(sample['posture_reference_dq'],np.zeros(17))
        np.testing.assert_array_equal(sample['continuum_target_velocity'],np.zeros(3))
        with self.assertRaises(ValueError):provider.sample(26.5)


if __name__=='__main__':unittest.main()
