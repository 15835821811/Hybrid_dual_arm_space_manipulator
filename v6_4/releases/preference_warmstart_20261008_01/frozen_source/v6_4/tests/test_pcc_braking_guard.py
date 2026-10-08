"""Regression cases from original D0-D3 before their frozen domain conflict."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v6_lite.b2_interval_online import BoundedIntervalVelocityQP, ramp_endpoint_velocity_box
from v6_lite.execution_ramp import ramp_velocity
from v6_lite.hierarchical_qp import HierarchicalQPConfig, HierarchicalVelocityQP
from v6_lite.pcc_braking_guard import (
    peak_outward_displacement, ramp_braking_endpoint_velocity_box,
)


class PCCBrakingGuardTests(unittest.TestCase):
    def box(self, q, start):
        return ramp_braking_endpoint_velocity_box(
            q, start, -np.ones(10), np.ones(10), np.full(10, 2.5),
            np.full(10, .56), .02)

    def test_exact_peak_matches_shared_native_ramp_and_braking(self):
        for start, endpoint in ((.12, .15), (.4, .2), (.07, -.1),
                                (-.2, .13), (.1, 0.), (-.1, -.2)):
            position, peak = 0., 0.
            old, new = start, endpoint
            for cycle in range(100):
                for step in range(1, 11):
                    position += float(ramp_velocity(np.asarray(old), np.asarray(new), step)) * .002
                    peak = max(peak, position)
                if new <= 0.:
                    break
                old, new = new, max(new - .05, 0.)
            self.assertAlmostEqual(peak_outward_displacement(start, endpoint, 2.5, .02), peak, places=14)

    def test_four_historical_pre_failure_commands_need_earlier_braking(self):
        # D0/D3: prior cycle; D1/D2: two cycles before failure. These are
        # saved actual q and old/selected commands, never original q_ref.
        cases = (
            ('D0', 0, .9990574531526356, .06135182434380059, .035466840781072186),
            ('D1', 0, .996443143343593, .09164115369824431, .09605714446699408),
            ('D2', 6, .9967264086465221, .08857498555773327, .08785298514516485),
            ('D3', 6, .9972224846602161, .09464991550169712, .0945334462669616),
        )
        for name, index, actual_q, old_command, old_selected in cases:
            with self.subTest(name=name):
                q, start = np.zeros(10), np.zeros(10)
                q[index], start[index] = actual_q, old_command
                lower, upper = self.box(q, start)
                self.assertLess(upper[index], old_selected - 1e-4)
                self.assertGreaterEqual(upper[index], old_command - .05)
                self.assertLessEqual(lower[index], upper[index])
                repaired_peak = peak_outward_displacement(old_command, upper[index], 2.5, .02)
                self.assertLessEqual(actual_q + repaired_peak, 1. + 1e-14)
                original_peak = peak_outward_displacement(old_command, old_selected, 2.5, .02)
                self.assertGreater(actual_q + original_peak, 1.)

    def test_guard_is_symmetric_for_lower_domain(self):
        q, start = np.linspace(-.995, .995, 10), np.linspace(-.2, .2, 10)
        lower, upper = self.box(q, start)
        reflected_lower, reflected_upper = self.box(-q, -start)
        np.testing.assert_array_equal(lower, -reflected_upper)
        np.testing.assert_array_equal(upper, -reflected_lower)

    def test_disabled_strategy_keeps_original_endpoint_box_exactly(self):
        self.assertFalse(HierarchicalQPConfig().enable_pcc_braking_guard)
        qp = BoundedIntervalVelocityQP.__new__(BoundedIntervalVelocityQP)
        qp.spec = SimpleNamespace(low_level_to_planner=np.eye(17),
                                  planner_acceleration_limits=np.full(17, 2.5),
                                  planner_velocity_limits=np.full(17, .8))
        qp.qpos_ids = np.arange(17)
        qp.previous_velocity = np.linspace(-.1, .1, 17)
        qp.interval_evaluator = SimpleNamespace(shape_spec=SimpleNamespace(
            work_domain_lower_rad=-np.ones(10), work_domain_upper_rad=np.ones(10)))
        qp.config = HierarchicalQPConfig()
        data = SimpleNamespace(qpos=np.linspace(-.6, .6, 17))
        expected = ramp_endpoint_velocity_box(data.qpos[:10], qp.previous_velocity[:10],
                                              -np.ones(10), np.ones(10), .02)
        sentinel = object()
        with patch.object(HierarchicalVelocityQP, 'solve', return_value=sentinel):
            self.assertIs(qp.solve(data), sentinel)
        np.testing.assert_array_equal(qp._domain_rate_lower, expected[0])
        np.testing.assert_array_equal(qp._domain_rate_upper, expected[1])
        self.assertIsNone(qp._domain_braking_rate_lower)

    def test_enabled_strategy_only_tightens_existing_domain_rows(self):
        qp = BoundedIntervalVelocityQP.__new__(BoundedIntervalVelocityQP)
        qp.spec = SimpleNamespace(low_level_to_planner=np.eye(17),
                                  planner_acceleration_limits=np.full(17, 2.5),
                                  planner_velocity_limits=np.full(17, .8))
        qp.qpos_ids = np.arange(17)
        qp.previous_velocity = np.zeros(17)
        qp.previous_velocity[0] = .06135182434380059
        qp.interval_evaluator = SimpleNamespace(shape_spec=SimpleNamespace(
            work_domain_lower_rad=-np.ones(10), work_domain_upper_rad=np.ones(10)))
        qp.config = HierarchicalQPConfig(enable_pcc_braking_guard=True)
        data = SimpleNamespace(qpos=np.r_[.9990574531526356, np.zeros(16)])
        with patch.object(HierarchicalVelocityQP, 'solve', return_value=None):
            qp.solve(data)
        self.assertTrue(np.all(qp._domain_rate_lower >= qp._domain_endpoint_rate_lower))
        self.assertTrue(np.all(qp._domain_rate_upper <= qp._domain_endpoint_rate_upper))
        self.assertLess(qp._domain_rate_upper[0], .035466840781072186)


if __name__ == '__main__':
    unittest.main()
