"""Frozen counterexample and exact reference-ramp regressions."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import numpy as np

from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.execution_ramp import advance_reference, ramp_mean_weights, ramp_velocity
from v6_lite.safety_contract import FailureReason, validate_action


FROZEN = (
    Path(__file__).resolve().parent / "output" / "v6_2_a1" / "frozen_failure"
    / "traces" / "v6_lite_scenario_00_counterexample.json"
)


class RampAwareTests(unittest.TestCase):
    def test_ten_reference_steps_match_legacy_interpolation_and_clips(self):
        rng = np.random.default_rng(824)
        start = rng.normal(0, 0.2, 17)
        endpoint = rng.normal(0, 0.2, 17)
        lower, upper = np.full(17, -0.4), np.full(17, 0.4)
        measured = np.full(17, 0.05)
        position = np.full(17, 0.01)
        velocities = []
        for index in range(1, 11):
            step = advance_reference(position, start, endpoint, index, measured, lower, upper)
            alpha = index / 10.0
            legacy_velocity = start + alpha * (endpoint - start)
            legacy_unclipped = position + 0.002 * legacy_velocity
            legacy_joint = np.clip(legacy_unclipped, lower, upper)
            legacy_position = np.clip(legacy_joint, measured - 0.012, measured + 0.012)
            np.testing.assert_allclose(step.velocity, legacy_velocity, rtol=0, atol=1e-15)
            np.testing.assert_allclose(step.position, legacy_position, rtol=0, atol=1e-15)
            np.testing.assert_allclose(step.feedforward_acceleration,
                                       (endpoint - start) / 0.02, rtol=0, atol=1e-15)
            position = step.position
            velocities.append(step.velocity)
        old_weight, new_weight = ramp_mean_weights()
        np.testing.assert_allclose(np.mean(velocities, axis=0),
                                   old_weight * start + new_weight * endpoint,
                                   rtol=0, atol=1e-15)
        self.assertAlmostEqual(old_weight, 0.45)
        self.assertAlmostEqual(new_weight, 0.55)
        np.testing.assert_allclose(ramp_velocity(start, endpoint, 0), start)

    def test_frozen_failure_state_still_rejected(self):
        evidence = json.loads(FROZEN.read_text(encoding="utf-8"))
        self.assertEqual(len(evidence["planning_snapshots"]), 3)
        self.assertEqual(
            [round(item["time_s"], 2) for item in evidence["planning_snapshots"]],
            [0.42, 0.44, 0.46],
        )
        partial = FROZEN.with_name("v6_lite_scenario_00_partial_trace.npz")
        self.assertEqual(
            hashlib.sha256(partial.read_bytes()).hexdigest(),
            evidence["partial_trace"]["sha256"],
        )
        with np.load(partial, allow_pickle=False) as trace:
            self.assertEqual(len(trace["time"]), 230)
            np.testing.assert_allclose(
                trace["task_qpos"][-1], evidence["planning_snapshots"][-1]["qpos"]
            )
        current = evidence["planning_snapshots"][-1]
        matrix = np.asarray([row["gradient_m_per_rad"] for row in current["rows"]])
        lower = np.asarray([row["lower_m_s"] for row in current["rows"]])
        result = validate_action(
            candidate=np.asarray(current["solver_candidate"]),
            solver_feasible=True,
            solver_status="solved",
            start_velocity=np.asarray(current["old_command"]),
            clearance_matrix=matrix,
            clearance_lower=lower,
            velocity_lower=np.asarray(current["velocity_lower"]),
            velocity_upper=np.asarray(current["velocity_upper"]),
            now_s=current["time_s"],
            state_timestamp_s=current["time_s"],
            target_timestamp_s=current["time_s"],
            max_input_age_s=0.02,
            command_period_s=0.02,
            clearance_rate_tolerance_m_s=1e-4,
            velocity_tolerance_rad_s=1e-4,
        )
        self.assertEqual(result.failure_reason, FailureReason.RAMP_VIOLATION)
        self.assertIsNone(result.selected_command)
        self.assertLess(result.ramp_clearance_min_slack_m_s, -0.01)

    def test_previous_cycle_lookahead_detects_frozen_failure(self):
        evidence = json.loads(FROZEN.read_text(encoding="utf-8"))
        prior = evidence["planning_snapshots"][-2]
        source = evidence["cross_cycle_report"]["dominant_source"]
        row = next(row for row in prior["rows"] if row["source"] == source)
        gradient = np.asarray(row["gradient_m_per_rad"])
        old = np.asarray(prior["old_command"])
        endpoint = np.asarray(prior["solver_candidate"])
        old_weight, new_weight = ramp_mean_weights()
        gain_dt = 8.0 * 0.02  # MuJoCo clearance barrier gain and task period.
        predicted_next_residual = (
            (1 + gain_dt * new_weight) * gradient @ endpoint
            - (row["lower_m_s"] - gain_dt * old_weight * gradient @ old
               + HierarchicalQPConfig().lookahead_model_margin_m_s)
        )
        self.assertLess(predicted_next_residual, 0.0)


if __name__ == "__main__":
    unittest.main()
