"""Read-only frozen-row feasibility uses the existing ramp and tolerances."""

from __future__ import annotations

import unittest

import numpy as np

from v6_lite.b2_shadow_feasibility import (
    solve_frozen_linear_feasibility, velocity_box,
)
from v6_lite.hierarchical_qp import HierarchicalQPConfig, HierarchicalVelocityQP
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


class FrozenFeasibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HierarchicalQPConfig()
        self.lower = np.full(17, -1.0)
        self.upper = np.full(17, 1.0)

    def assess(self, matrix: np.ndarray, lower: np.ndarray,
               start: np.ndarray):
        count = len(lower)
        return solve_frozen_linear_feasibility(
            matrix, lower, np.zeros(count), np.full(count, 8.0),
            tuple(f"interval:{i}" for i in range(count)),
            self.lower, self.upper, start, self.config,
            interval_row_count=count, historical_endpoint=start,
        )

    def test_feasible_endpoint_and_historical_endpoint_are_separate(self) -> None:
        row = np.zeros((1, 17))
        row[0, 0] = 1.0
        start = np.zeros(17)
        start[0] = 0.5
        result = self.assess(row, np.array([0.2]), start)
        self.assertEqual(result.status, "FROZEN_ROWS_FEASIBLE")
        self.assertTrue(result.executable_candidate_exists)
        self.assertTrue(result.historical_endpoint_accepted_by_frozen_rows)

    def test_start_violation_blocks_execution_even_if_endpoint_exists(self) -> None:
        row = np.zeros((1, 17))
        row[0, 0] = 1.0
        result = self.assess(row, np.array([0.2]), np.zeros(17))
        self.assertEqual(result.status, "START_CLEARANCE_VIOLATION")
        self.assertTrue(result.candidate_feasible)
        self.assertFalse(result.executable_candidate_exists)

    def test_frozen_lookahead_can_make_an_instantaneous_row_infeasible(self) -> None:
        row = np.zeros((1, 17))
        result = self.assess(row, np.array([0.0]), np.zeros(17))
        self.assertEqual(result.status, "NO_FEASIBLE_ENDPOINT")
        self.assertFalse(result.candidate_feasible)
        self.assertFalse(result.executable_candidate_exists)
        self.assertEqual(result.start_clearance_min_slack_m_s, 0.0)

    def test_offline_velocity_box_matches_existing_qp(self) -> None:
        spec = default_v6_lite_robot_spec()
        qp = object.__new__(HierarchicalVelocityQP)
        qp.spec = spec
        qp.config = self.config
        rng = np.random.default_rng(1806)
        for _ in range(5):
            q = rng.uniform(spec.planner_lower * 0.5, spec.planner_upper * 0.5)
            previous = rng.uniform(-0.1, 0.1, size=17)
            qp.previous_velocity = previous
            expected = qp._velocity_bounds(q)
            actual = velocity_box(spec, self.config, q, previous)
            for left, right in zip(actual, expected):
                np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
