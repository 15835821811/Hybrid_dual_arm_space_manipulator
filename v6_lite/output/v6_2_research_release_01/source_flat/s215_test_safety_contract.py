"""Counterexamples for the local execution contract (no physics proof)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import mujoco

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.hierarchical_qp import HierarchicalQPConfig, HierarchicalVelocityQP, joint_addresses
from v6_lite.run_v6_lite import V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec

from v6_lite.safety_contract import (
    ExecutionMode, FailureReason, command_is_current, validate_action,
)


class SafetyContractTests(unittest.TestCase):
    def evaluate(self, **changes):
        row = np.zeros((1, 17), dtype=np.float64)
        row[0, 0] = 1.0
        inputs = dict(
            candidate=np.full(17, 0.01), solver_feasible=True,
            solver_status="solved", start_velocity=np.full(17, 0.01),
            clearance_matrix=row, clearance_lower=np.array([0.0]),
            velocity_lower=np.full(17, -1.0),
            velocity_upper=np.full(17, 1.0), now_s=1.0,
            state_timestamp_s=1.0, target_timestamp_s=1.0,
            max_input_age_s=0.02, command_period_s=0.02,
            clearance_rate_tolerance_m_s=1e-4,
            velocity_tolerance_rad_s=1e-4,
        )
        inputs.update(changes)
        return validate_action(**inputs)

    def test_moving_target_at_boundary_rejects_zero_velocity(self):
        # h=0 and target drift=-0.1 m/s gives qdot >= 0.1 m/s.
        result = self.evaluate(candidate=np.zeros(17),
                               clearance_lower=np.array([0.1]))
        self.assertEqual(result.mode, ExecutionMode.UNCERTIFIED)
        self.assertEqual(result.failure_reason, FailureReason.CANDIDATE_VIOLATION)
        self.assertIsNone(result.selected_command)
        self.assertAlmostEqual(result.candidate_clearance_min_slack_m_s, -0.1)

    def test_infeasible_qp_does_not_turn_candidate_into_safe_stop(self):
        result = self.evaluate(candidate=np.zeros(17), solver_feasible=False)
        self.assertEqual(result.failure_reason, FailureReason.QP_INFEASIBLE)
        self.assertIsNone(result.selected_command)
        self.assertTrue(np.isfinite(result.candidate_clearance_min_slack_m_s))

    def test_iteration_exhaustion_rejects_even_feasible_candidate(self):
        result = self.evaluate(solver_status="maximum_iterations")
        self.assertTrue(result.candidate_valid)
        self.assertEqual(result.failure_reason, FailureReason.ITERATION_LIMIT)
        self.assertIsNone(result.selected_command)

    def test_stale_state_target_and_expired_command(self):
        state = self.evaluate(state_timestamp_s=0.95)
        target = self.evaluate(target_timestamp_s=0.95)
        self.assertEqual(state.failure_reason, FailureReason.STALE_STATE)
        self.assertEqual(target.failure_reason, FailureReason.STALE_TARGET)
        valid = self.evaluate()
        self.assertEqual(valid.mode, ExecutionMode.TRACK)
        self.assertTrue(command_is_current(valid, 1.018))
        self.assertFalse(command_is_current(valid, 1.023))

    def test_high_initial_speed_makes_stop_ramp_invalid(self):
        # The existing acceleration box cannot jump from 0.5 to zero.
        start = np.zeros(17)
        start[0] = 0.5
        lower = np.full(17, -1.0)
        lower[0] = 0.4
        stop = self.evaluate(candidate=np.zeros(17), start_velocity=start,
                             velocity_lower=lower)
        self.assertEqual(stop.failure_reason, FailureReason.CANDIDATE_VIOLATION)
        self.assertLess(stop.candidate_velocity_min_slack_rad_s, 0)

        result = self.evaluate(candidate=np.zeros(17),
                               start_velocity=start,
                               clearance_lower=np.array([0.2]))
        self.assertEqual(result.failure_reason, FailureReason.CANDIDATE_VIOLATION)
        # A separate example has a valid endpoint but invalid ramp start.
        start[0] = -0.5
        candidate = np.zeros(17)
        candidate[0] = 0.25
        result = self.evaluate(candidate=candidate, start_velocity=start,
                               clearance_lower=np.array([0.2]))
        self.assertTrue(result.candidate_valid)
        self.assertFalse(result.ramp_valid)
        self.assertEqual(result.failure_reason, FailureReason.RAMP_VIOLATION)
        self.assertIsNone(result.selected_command)

    def test_candidate_selected_residuals_are_distinct(self):
        rejected = self.evaluate(candidate=np.full(17, -0.2))
        self.assertLess(rejected.candidate_clearance_min_slack_m_s, 0)
        self.assertTrue(np.isnan(rejected.selected_clearance_min_slack_m_s))
        accepted = self.evaluate()
        self.assertAlmostEqual(accepted.candidate_clearance_min_slack_m_s,
                               accepted.selected_clearance_min_slack_m_s)

    def test_real_qp_failure_does_not_return_zero_as_command(self):
        spec = default_v6_lite_robot_spec()
        scenario = build_scenarios(spec, V6LiteRunConfig())[0]
        verifier = WholeBodyCollisionVerifier(
            spec, scenario.obstacles,
            WholeBodyVerificationConfig(include_target_satellite_pairs=True),
        )
        model = verifier.model
        data = mujoco.MjData(model)
        qpos_ids, _ = joint_addresses(model, spec)
        data.qpos[qpos_ids] = spec.encode_position(spec.planner_zero)
        mujoco.mj_forward(model, data)
        qp = HierarchicalVelocityQP(spec, model, verifier.pairs, HierarchicalQPConfig())
        rigid = np.asarray(data.xpos[qp.rigid_body_id]).copy()
        continuum = np.asarray(data.xpos[qp.continuum_body_id]).copy()
        rigid_rotation = np.asarray(data.xmat[qp.rigid_body_id]).reshape(3, 3).copy()
        continuum_rotation = np.asarray(data.xmat[qp.continuum_body_id]).reshape(3, 3).copy()

        def failed_solver(_h, _g, matrix, _lower, _upper, _initial, _dual):
            return np.full(17, 0.013), False, "maximum_iterations", 1, np.zeros(matrix.shape[0])

        with patch.object(qp, "_solve_qp_admm", side_effect=failed_solver):
            result = qp.solve(
                data,
                rigid_target_position=rigid,
                rigid_target_velocity=np.zeros(3),
                rigid_target_rotation=rigid_rotation,
                rigid_target_angular_velocity=np.zeros(3),
                continuum_target_position=continuum,
                continuum_target_velocity=np.zeros(3),
                continuum_target_rotation=continuum_rotation,
                continuum_target_angular_velocity=np.zeros(3),
            )
        self.assertIsNone(result.planner_velocity)
        np.testing.assert_allclose(result.solver_candidate, 0.013)
        self.assertEqual(result.action_validation.mode, ExecutionMode.UNCERTIFIED)
        self.assertEqual(result.action_validation.failure_reason, FailureReason.ITERATION_LIMIT)
        np.testing.assert_array_equal(qp.previous_velocity, np.zeros(17))


if __name__ == "__main__":
    unittest.main()
