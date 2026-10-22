import time
import unittest
import numpy as np
from scipy.interpolate import BSpline
from ff_prior_dynamic_reference.representation import design,fit,polynomial_bounds

class RepresentationTests(unittest.TestCase):
    def test_derivative_coefficients_and_bounds(self):
        knots,basis,d1,d2=design(27.)
        rng=np.random.default_rng(11);c=rng.normal(0,.01,(len(knots)-4,2))
        s=BSpline(knots,c,3);x=np.linspace(0,27,2001)
        np.testing.assert_allclose(s(x,1),BSpline(knots[1:-1],d1@c,2)(x),atol=1e-13)
        np.testing.assert_allclose(s(x,2),BSpline(knots[2:-2],d2@c,1)(x),atol=1e-12)
        bounds=polynomial_bounds(s)
        for k in range(3):
            self.assertTrue(np.all(s(x,k)>=bounds[k][0]-1e-12))
            self.assertTrue(np.all(s(x,k)<=bounds[k][1]+1e-12))

    def test_exact_initial_state_and_original_duration(self):
        t=np.linspace(0,27,1351)
        t[-1]-=4.1e-12
        q=(.02*(1-np.cos(2*np.pi*t/27)))[:,None]
        dq=(.02*2*np.pi/27*np.sin(2*np.pi*t/27))[:,None]
        s,r=fit(t,q,dq,np.array([-.5]),np.array([.5]),np.array([.1]),np.array([.2]),
                lambda *a,**k:None,time.monotonic()+120)
        self.assertTrue(all(r['gates'].values()),r)
        self.assertLess(r['maximum_sample_fit_error_rad'],1e-6)
        np.testing.assert_array_equal(s(0),q[0])
        self.assertAlmostEqual(s.t[-1],27.)
        self.assertTrue(np.isfinite(s(27.)).all())

    def test_acceleration_constraint_is_enforced(self):
        t=np.linspace(0,27,1351)
        q=(.025*(1-np.cos(2*np.pi*t/3)))[:,None]
        dq=(.025*2*np.pi/3*np.sin(2*np.pi*t/3))[:,None]
        s,r=fit(t,q,dq,np.array([-.5]),np.array([.5]),np.array([.1]),np.array([.08]),
                lambda *a,**k:None,time.monotonic()+180)
        self.assertTrue(all(r['gates'].values()),r)
        self.assertLessEqual(r['maximum_acceleration_ratio'],1.)
        self.assertEqual(r['solver'][0]['method'],'CONSTRAINED_QUADRATIC_SLSQP')
        self.assertGreater(r['maximum_sample_fit_error_rad'],1e-5)

if __name__=='__main__':unittest.main()
