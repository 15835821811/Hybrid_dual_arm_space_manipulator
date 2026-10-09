"""Same-input original QP parity and version consumption; no physics/query work."""
import io
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from v6_4.conditional_execution import ExecutionCostLedger
from v6_4.residual_execution import consumed_reference_identity_binding
from v6_4.task_protocol import TaskSpec
from v6_4.task_anchored_reference import (
    TaskAnchoredResidualPlan, TaskAnchoredResidualReferenceProvider, build_reference_definition,
    REPRESENTATION_VERSION, PLATEAU_REPRESENTATION_VERSION,
)
from v6_lite.b2_interval_online_optimized import OptimizedBoundedIntervalVelocityQP
from v6_lite.hierarchical_qp import HierarchicalQPConfig, HierarchicalVelocityQP, _ClearanceConstraintSet
from v6_lite.run_v6_lite import (
    append_qp_execution_diagnostics, qp_diagnostic_trace_keys, reference_diagnostic_identity,
)

OBSTACLE = "v5_workspace_sphere_001_unit_sphere"


def fixed_qp_fixture():
    """Frozen numeric input exercises the actual optimized solver and validator.

    Geometry assembly is replaced by a declared matrix, rather than querying
    geometry or mocking the QP answer. Both runs receive identical inputs.
    """
    qp = object.__new__(OptimizedBoundedIntervalVelocityQP)
    qp.config = HierarchicalQPConfig()
    qp.spec = SimpleNamespace(low_level_to_planner=np.eye(17), planner_zero=np.zeros(17))
    qp.qpos_ids = np.arange(17)
    qp.base_dof_slice = slice(0, 6)
    qp.rigid_body_id, qp.continuum_body_id = 0, 1
    qp.previous_velocity = np.zeros(17)
    qp._previous_constraint_dual = {}
    qp.solve_count = 0
    qp._domain_rate_lower, qp._domain_rate_upper = np.full(10, -1.), np.full(10, 1.)
    qp.collision_pairs = (SimpleNamespace(pair_class="continuum_obstacle", geom_a_name="continuum_test", geom_b_name=OBSTACLE),)
    qp.reaction_velocity_map = lambda data: (np.eye(17), 0.)
    qp._velocity_bounds = lambda value: (np.full(17, -.4), np.full(17, .4))
    def jacobians(self, data, body, generalized_map, local_offset=None, diagnostic_velocity_out=None):
        matrix = np.zeros((3, 17))
        if body == 1:
            matrix[:, :3] = np.eye(3)
        else:
            matrix[:, 10:13] = np.eye(3)
        if diagnostic_velocity_out is not None:
            diagnostic_velocity_out.append(matrix @ data.qvel)
        return matrix, np.zeros((3, 17))
    qp._task_jacobians = MethodType(jacobians, qp)
    row = np.zeros((1, 17)); row[0, 0] = 1.
    qp._build_all_clearance_constraints = lambda data, mapping: _ClearanceConstraintSet(
        matrix=row.copy(), lower=np.asarray([-.01]),
        sources=(f"mujoco:continuum_obstacle:continuum_test:{OBSTACLE}",),
        distances_m=np.asarray([.02625]), target_drifts_m_s=np.zeros(1),
        barrier_gains_s_inv=np.asarray([8.]), minimum_clearance_m=.02625,
        degenerate_gradient_count=0, mujoco_continuum_target_distance_m=.5,
        mujoco_continuum_target_gradient=np.zeros(17),
        mujoco_continuum_target_gradient_valid=False, shape_clearance_latency_s=0.)
    qp._unconstrained_solve = Mock(wraps=qp._unconstrained_solve)
    qp._solve_qp_admm = Mock(wraps=qp._solve_qp_admm)
    data = SimpleNamespace(time=0., xpos=np.zeros((2, 3)),
        xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)), qpos=np.zeros(17), qvel=np.arange(17)*.001)
    arguments = dict(rigid_target_position=np.zeros(3), rigid_target_velocity=np.zeros(3),
        rigid_target_rotation=np.eye(3), rigid_target_angular_velocity=np.zeros(3),
        continuum_target_position=np.asarray([-.02, 0., 0.]), continuum_target_velocity=np.zeros(3),
        continuum_target_rotation=np.eye(3), continuum_target_angular_velocity=np.zeros(3),
        prepared_state=True, ramp_start_velocity=np.zeros(17))
    return qp, data, arguments


def provider_fixture(version="task_anchored_cartesian_residual_v2"):
    z = np.zeros((6, 2)); z[2, 0] = .012
    plan = SimpleNamespace(definition={"representation_version": version, "definition_sha256": "definition"},
                           z_m=z, sha256=lambda: "plan")
    provider = SimpleNamespace(plan=plan, metadata={"reference_mode": version,
        "plan_sha256": "plan", "definition_sha256": "definition"},
        prediction={"reference_mode": np.asarray(version)})
    return plan, provider


class ExecutionDiagnosticTests(unittest.TestCase):
    def test_real_v1_v2_provider_versions_bind_without_physics_or_geometry(self):
        path = Path(__file__).resolve().parents[1]/"releases/conditional_route_value_20261007_01/snapshot/tasks/b3_mother_00_c_plus/task.json"
        task = TaskSpec.from_dict(json.loads(path.read_text(encoding="utf8")))
        with patch.object(mujoco, "MjData", side_effect=AssertionError("state creation forbidden")), \
             patch.object(mujoco, "mj_geomDistance", side_effect=AssertionError("geometry forbidden")):
            for version in (REPRESENTATION_VERSION, PLATEAU_REPRESENTATION_VERSION):
                definition = build_reference_definition(task, version=version)
                z = np.zeros((6, 2)); z[2, 0] = .012
                plan = TaskAnchoredResidualPlan.from_definition(definition, z)
                provider = TaskAnchoredResidualReferenceProvider(task, plan)
                identity = reference_diagnostic_identity(provider)
                trace = {"task_"+key: np.asarray([value]) for key, value in identity.items()}
                binding = consumed_reference_identity_binding(plan, trace, 1)
                self.assertEqual(binding["representation_version"], version)
                self.assertEqual(binding["plan_sha256"], plan.sha256())
                self.assertTrue(binding["passed"])

    def test_logging_switch_keeps_real_optimized_qp_action_identical(self):
        ledger = ExecutionCostLedger()
        results = []
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("physics forbidden")), \
             patch.object(mujoco, "mj_step2", side_effect=AssertionError("physics forbidden")), \
             patch.object(mujoco, "mj_geomDistance", side_effect=AssertionError("geometry forbidden")), \
             ledger.installed():
            for enabled in (False, True):
                qp, data, arguments = fixed_qp_fixture()
                with ledger.scope("diagnostic_parity_unit"):
                    result = HierarchicalVelocityQP.solve(qp, data, **arguments,
                        execution_diagnostics=enabled, diagnostic_obstacle_name=OBSTACLE)
                self.assertEqual(qp._unconstrained_solve.call_count, 1)
                self.assertEqual(qp._solve_qp_admm.call_count, 1)
                self.assertEqual(qp.solve_count, 1)
                results.append(result)
        plain, logged = results
        self.assertIsNone(plain.execution_diagnostics)
        self.assertIsNotNone(logged.planner_velocity)
        for field in ("planner_velocity", "solver_candidate", "clearance_matrix", "clearance_lower", "lookahead_matrix", "lookahead_lower"):
            np.testing.assert_array_equal(getattr(plain, field), getattr(logged, field))
        for field in ("success", "solver_status", "solver_iterations", "objective", "unconstrained_to_command_norm"):
            self.assertEqual(getattr(plain, field), getattr(logged, field))
        self.assertEqual(plain.action_validation.mode, logged.action_validation.mode)
        self.assertEqual(plain.action_validation.failure_reason, logged.action_validation.failure_reason)
        diagnostics = logged.execution_diagnostics
        delta = diagnostics["selected_velocity"] - diagnostics["box_nominal_velocity"]
        self.assertEqual(float(np.linalg.norm(delta)), logged.unconstrained_to_command_norm)
        np.testing.assert_array_equal(diagnostics["box_nominal_velocity"],
            np.clip(diagnostics["raw_unconstrained_velocity"], logged.velocity_lower, logged.velocity_upper))
        self.assertAlmostEqual(float(delta @ delta), float(delta[:10] @ delta[:10] + delta[10:] @ delta[10:]))
        row = diagnostics["obstacle_rows"][0]
        self.assertTrue(row["active"])
        self.assertEqual(row["source"], logged.clearance_sources[0])
        self.assertEqual(row["selected_residual_m_s"], float(logged.clearance_matrix[0] @ logged.planner_velocity - logged.clearance_lower[0]))
        self.assertEqual(ledger.to_dict()["qp_solve_calls"], 2)
        self.assertEqual(ledger.to_dict()["native_geometry_query_calls"], 0)
        self.assertEqual(sum(ledger.physics_steps(phase) for phase in ("actual", "private_preview", "independent_torque_replay", "diagnostic_parity_unit")), 0)

    def test_actual_provider_identity_logs_arrays_and_rejects_wrong_version_or_z(self):
        qp, data, arguments = fixed_qp_fixture()
        result = HierarchicalVelocityQP.solve(qp, data, **arguments,
            execution_diagnostics=True, diagnostic_obstacle_name="unit_sphere")
        plan, provider = provider_fixture()
        identity = reference_diagnostic_identity(provider)
        log = {key: [] for key in qp_diagnostic_trace_keys()}
        append_qp_execution_diagnostics(log, result, identity)
        trace = {"task_"+key: np.asarray(value) for key, value in log.items()}
        buffer = io.BytesIO(); np.savez_compressed(buffer, **trace); buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            trace = {key: saved[key].copy() for key in saved.files}
        self.assertTrue(consumed_reference_identity_binding(plan, trace, 1)["passed"])
        self.assertEqual(trace["task_qp_selected_velocity_continuum"].shape, (1, 10))
        self.assertEqual(trace["task_qp_selected_velocity_rigid"].shape, (1, 7))
        self.assertEqual(json.loads(trace["task_qp_obstacle_rows_json"][0])[0]["source"], result.clearance_sources[0])
        trace["task_consumed_reference_z_m"][0, 2, 0] += .001
        with self.assertRaisesRegex(ValueError, "coefficients"):
            consumed_reference_identity_binding(plan, trace, 1)
        provider.metadata["reference_mode"] = "task_anchored_cartesian_residual_v1"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            reference_diagnostic_identity(provider)

    def test_legacy_v1_remains_valid_but_v2_requires_consumption_fields(self):
        plan, _ = provider_fixture("task_anchored_cartesian_residual_v1")
        self.assertFalse(consumed_reference_identity_binding(plan, {}, 1)["available"])
        plan, _ = provider_fixture()
        with self.assertRaisesRegex(ValueError, "lacks actual consumed"):
            consumed_reference_identity_binding(plan, {}, 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            consumed_reference_identity_binding(plan, {"task_consumed_reference_version": np.asarray([plan.definition["representation_version"]])}, 1)


if __name__ == "__main__":
    unittest.main()
