import unittest
import numpy as np

from v6_4.trajectory_codec import CubicBSplineCodec


class CodecTests(unittest.TestCase):
    def setUp(self):
        self.q0=np.linspace(-.2,.2,17)
        self.dq0=np.linspace(-.01,.01,17)
        self.codec=CubicBSplineCodec(self.q0,self.dq0)
        self.controls=self.codec.decode_free(np.tile(self.q0,(30,1)))

    def test_eliminated_start_position_and_derivative(self):
        value=self.codec.sample(self.controls,np.array([0.,27.]))
        np.testing.assert_array_equal(value['q'][0],self.q0)
        np.testing.assert_allclose(value['dq'][0],self.dq0,atol=1e-15,rtol=0.)
        np.testing.assert_array_equal(self.controls[1],self.q0+(27/29)/3*self.dq0)

    def test_free_roundtrip_does_not_clip(self):
        free=np.full((30,17),12.)
        np.testing.assert_array_equal(self.codec.encode_free(self.codec.decode_free(free)),free)
        self.assertEqual(self.codec.sample(self.codec.decode_free(free),np.array([27.]))['q'][0,0],12.)

    def test_invalid_boundary_cannot_be_encoded(self):
        changed=self.controls.copy();changed[0,0]+=1e-12
        with self.assertRaises(ValueError):self.codec.encode_free(changed)

    def test_exact_identifiable_fit(self):
        controls=self.codec.decode_free(np.random.default_rng(13).normal(0,.08,(30,17)))
        times=np.linspace(0,27,200);q=self.codec.sample(controls,times)['q']
        fitted=self.codec.fit(times,q,regularization=0.)
        np.testing.assert_allclose(fitted,controls,atol=1e-14,rtol=0.)

    def test_analytic_velocity_matches_position_derivative(self):
        t=np.array([.13,8.42,26.73]);h=1e-5
        sampled=self.codec.sample(self.controls,t)
        difference=(self.codec.sample(self.controls,t+h)['q']-self.codec.sample(self.controls,t-h)['q'])/(2*h)
        np.testing.assert_allclose(sampled['dq'],difference,atol=1e-10,rtol=0.)

    def test_horizon_shape_and_nonfinite_fail_closed(self):
        for times in [np.array([-1e-6]),np.array([27.000001]),np.array([np.nan])]:
            with self.assertRaises(ValueError):self.codec.sample(self.controls,times)
        for free in [np.zeros((32,17)),np.full((30,17),np.inf)]:
            with self.assertRaises(ValueError):self.codec.decode_free(free)
        with self.assertRaises(ValueError):CubicBSplineCodec(self.q0,self.dq0,duration_s=20.)
        with self.assertRaises(ValueError):CubicBSplineCodec(self.q0,self.dq0,n_control_points=33)


if __name__=='__main__':unittest.main()
