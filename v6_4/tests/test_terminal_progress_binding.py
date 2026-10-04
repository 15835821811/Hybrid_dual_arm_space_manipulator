import unittest
import numpy as np

from v6_4.evaluate_planning import _selected_posture_reference
from v6_4.trajectory_codec import CubicBSplineCodec


class TerminalProgressBindingTests(unittest.TestCase):
    def setUp(self):
        self.codec=CubicBSplineCodec(np.zeros(17),np.zeros(17))
        free=np.linspace(0.,.3,30)[:,None]*np.linspace(.1,1.,17)[None,:]
        self.controls=self.codec.decode_free(free)
        self.reference={'reference_mode':np.asarray('terminal_progress_v1'),
            'progress_start_s':np.asarray(24.),'progress_end_s':np.asarray(26.5)}

    def test_physical_time_chain_matches_finite_difference_and_terminal_hold(self):
        times=np.array([24.2,25.,26.])
        h=1e-4
        mid=_selected_posture_reference(self.codec,self.controls,times,self.reference)
        left=_selected_posture_reference(self.codec,self.controls,times-h,self.reference)
        right=_selected_posture_reference(self.codec,self.controls,times+h,self.reference)
        np.testing.assert_allclose(mid['dq'],(right['q']-left['q'])/(2*h),atol=2e-9,rtol=0.)
        np.testing.assert_allclose(mid['ddq'],(right['dq']-left['dq'])/(2*h),atol=2e-9,rtol=0.)
        held=_selected_posture_reference(self.codec,self.controls,np.array([26.5,26.8,27.]),self.reference)
        endpoint=self.codec.sample(self.controls,np.array([27.]))['q'][0]
        np.testing.assert_array_equal(held['q'],np.tile(endpoint,(3,1)))
        np.testing.assert_array_equal(held['dq'],np.zeros((3,17)))
        np.testing.assert_array_equal(held['ddq'],np.zeros((3,17)))

    def test_undeclared_progress_policy_and_outside_physical_horizon_rejected(self):
        wrong={**self.reference,'progress_end_s':np.asarray(26.)}
        with self.assertRaises(ValueError):
            _selected_posture_reference(self.codec,self.controls,np.array([25.]),wrong)
        for time in (-.001,27.001):
            with self.assertRaises(ValueError):
                _selected_posture_reference(self.codec,self.controls,np.array([time]),self.reference)

    def test_disabled_progress_keeps_original_frozen_path(self):
        times=np.array([0.,24.,25.,27.])
        original=self.codec.sample(self.controls,times)
        actual=_selected_posture_reference(self.codec,self.controls,times,{})
        for key in original:np.testing.assert_array_equal(actual[key],original[key])


if __name__=='__main__':unittest.main()
