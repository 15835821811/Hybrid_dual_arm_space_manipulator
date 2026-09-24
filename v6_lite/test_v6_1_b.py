"""Acceptance tests for the V6.1-B PCC/capsule CBF integration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
)
from v6_lite.continuum_jacobian import compute_position_jacobian
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.continuum_shape_model import (
    ContinuumShapeModel,
    transform_from_free_qpos,
)
from v6_lite.hierarchical_qp import (
    CONTINUUM_EE_OFFSET_M,
    HierarchicalQPConfig,
    HierarchicalVelocityQP,
    free_joint_slices,
    joint_addresses,
)
from v6_lite.pcc_clearance import PCCClearanceEvaluator
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import OrientedBox, target_box_from_mujoco

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "v6_lite" / "output" / "v6_1_b"


def _load_json(name: str) -> dict:
    with (OUTPUT_ROOT / name).open("r", encoding="utf-8") as stream:
        return json.load(stream)


class V61BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot = default_v6_lite_robot_spec()
        cls.shape_spec = default_continuum_model_spec(cls.robot)
        cls.shape_model = ContinuumShapeModel(cls.shape_spec)
        cls.verifier = WholeBodyCollisionVerifier(
            cls.robot,
            (),
            WholeBodyVerificationConfig(
                include_target_satellite_pairs=True,
                adaptive_subdivisions=2,
            ),
        )
        cls.model = cls.verifier.model
        cls.qpos_ids, _ = joint_addresses(cls.model, cls.robot)
        cls.base_qpos, _ = free_joint_slices(
            cls.model, cls.robot.base_joint_name
        )
        cls.target_qpos, cls.target_dof = free_joint_slices(
            cls.model, cls.robot.target_free_joint_name
        )
        cls.target_geom_id = int(
            mujoco.mj_name2id(
                cls.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "target_satellite_collision",
            )
        )

    def _data(self, *, target_y_m: float = 0.326) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        data.qpos[self.qpos_ids] = self.robot.encode_position(
            self.robot.planner_zero
        )
        data.qpos[self.target_qpos.start : self.target_qpos.start + 3] = [
            1.2,
            target_y_m,
            0.05,
        ]
        data.qpos[self.target_qpos.start + 3 : self.target_qpos.stop] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        return data

    def _task_arguments(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        rigid_body = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.robot.rigid_tip_body_name,
            )
        )
        continuum_body = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.robot.continuum_tip_body_name,
            )
        )
        rigid_rotation = np.asarray(data.xmat[rigid_body]).reshape(3, 3).copy()
        continuum_rotation = (
            np.asarray(data.xmat[continuum_body]).reshape(3, 3).copy()
        )
        continuum_position = (
            np.asarray(data.xpos[continuum_body]).copy()
            + continuum_rotation @ CONTINUUM_EE_OFFSET_M
        )
        return {
            "rigid_target_position": np.asarray(data.xpos[rigid_body]).copy(),
            "rigid_target_velocity": np.zeros(3),
            "rigid_target_rotation": rigid_rotation,
            "rigid_target_angular_velocity": np.zeros(3),
            "continuum_target_position": continuum_position
            + np.asarray([0.0, -0.15, 0.0]),
            "continuum_target_velocity": np.zeros(3),
            "continuum_target_rotation": continuum_rotation,
            "continuum_target_angular_velocity": np.zeros(3),
        }

    def test_pcc_zero_state_is_straight_and_batch_query_matches_scalar(self) -> None:
        arclengths = np.linspace(0.0, self.shape_spec.total_length_m, 61)
        points = self.shape_model.batch_query(
            np.zeros(10), np.eye(4), arclengths
        )
        positions = np.asarray([item.position_world for item in points])
        expected = np.column_stack(
            [
                self.shape_spec.base_to_shape_start[0, 3] + arclengths,
                np.full(arclengths.shape, self.shape_spec.base_to_shape_start[1, 3]),
                np.full(arclengths.shape, self.shape_spec.base_to_shape_start[2, 3]),
            ]
        )
        np.testing.assert_allclose(positions, expected, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(
            np.asarray([item.tangent_world for item in points]),
            np.tile(self.shape_spec.base_to_shape_start[:3, 0], (61, 1)),
            atol=1e-10,
            rtol=0.0,
        )
        for arclength, point in zip(arclengths[::10], points[::10]):
            scalar = self.shape_model.evaluate(
                np.zeros(10), np.eye(4), float(arclength), with_jacobians=False
            )
            np.testing.assert_allclose(point.position_world, scalar.position_world)

    def test_position_jacobian_matches_finite_difference_for_1000_states(self) -> None:
        rng = np.random.default_rng(20260924)
        step = 1e-6
        maximum_relative_error = 0.0
        for _ in range(1_000):
            q = rng.uniform(-1.0, 1.0, size=10)
            arclength = float(rng.uniform(0.0, self.shape_spec.total_length_m))
            analytic = compute_position_jacobian(
                q, np.eye(4), arclength, model=self.shape_model
            )
            numeric = np.zeros((3, 10), dtype=np.float64)
            for coordinate in range(10):
                plus = q.copy()
                minus = q.copy()
                plus[coordinate] += step
                minus[coordinate] -= step
                p_plus = self.shape_model.evaluate(
                    plus, np.eye(4), arclength, with_jacobians=False
                ).position_world
                p_minus = self.shape_model.evaluate(
                    minus, np.eye(4), arclength, with_jacobians=False
                ).position_world
                numeric[:, coordinate] = (p_plus - p_minus) / (2.0 * step)
            relative = float(
                np.linalg.norm(analytic - numeric)
                / max(float(np.linalg.norm(numeric)), 1e-12)
            )
            maximum_relative_error = max(maximum_relative_error, relative)
        self.assertLess(maximum_relative_error, 0.05)

    def test_pcc_distance_gradient_matches_finite_difference(self) -> None:
        evaluator = PCCClearanceEvaluator(self.shape_model)
        box = OrientedBox(
            center=np.asarray([1.2, 0.326, 0.05]),
            rotation=np.eye(3),
            half_extents=np.asarray([0.2, 0.2, 0.2]),
        )
        rng = np.random.default_rng(20260925)
        step = 1e-6
        accepted = 0
        maximum_relative_error = 0.0
        for _ in range(1_000):
            q = rng.uniform(-0.7, 0.7, size=10)
            center = evaluator.evaluate(q, np.eye(4), box)
            # A minimum-distance function is intentionally nonsmooth at a
            # collision, a box medial plane, or when two arclength features
            # exchange ownership.  The analytic gradient contract applies to
            # the smooth exterior feature used by the CBF.
            if center.distance <= 0.005:
                continue
            numeric = np.zeros(10, dtype=np.float64)
            stable = True
            for coordinate in range(10):
                plus = q.copy()
                minus = q.copy()
                plus[coordinate] += step
                minus[coordinate] -= step
                value_plus = evaluator.evaluate(plus, np.eye(4), box)
                value_minus = evaluator.evaluate(minus, np.eye(4), box)
                stable &= (
                    value_plus.segment_id == center.segment_id
                    and value_minus.segment_id == center.segment_id
                    and abs(value_plus.arc_length - center.arc_length) < 1e-3
                    and abs(value_minus.arc_length - center.arc_length) < 1e-3
                    and np.dot(value_plus.normal, center.normal) > 0.999
                    and np.dot(value_minus.normal, center.normal) > 0.999
                )
                numeric[coordinate] = (
                    value_plus.distance - value_minus.distance
                ) / (2.0 * step)
            if not stable:
                continue
            relative = float(
                np.linalg.norm(center.gradient - numeric)
                / max(float(np.linalg.norm(numeric)), 1e-10)
            )
            maximum_relative_error = max(maximum_relative_error, relative)
            accepted += 1
            if accepted >= 32:
                break
        self.assertGreaterEqual(accepted, 32)
        self.assertLess(maximum_relative_error, 0.05)

    def test_free_floating_pcc_gradient_includes_reaction_motion(self) -> None:
        data = self._data()
        planner = self.robot.planner_zero.copy()
        planner[:10] = np.asarray(
            [
                -0.04444606239170623,
                -0.4606650285974338,
                0.12720213765869592,
                0.39009939189381615,
                -0.09298650555520172,
                0.14360581344515333,
                0.40358451742342305,
                0.3888946356152655,
                -0.2237593477125373,
                0.07888092129926527,
            ],
            dtype=np.float64,
        )
        data.qpos[self.qpos_ids] = self.robot.encode_position(planner)
        mujoco.mj_forward(self.model, data)
        qp = HierarchicalVelocityQP(
            self.robot,
            self.model,
            self.verifier.pairs,
            HierarchicalQPConfig(enable_pcc_cbf=True),
        )
        generalized_map, _ = qp.reaction_velocity_map(data)
        center = qp._pcc_clearance_kinematics(data, generalized_map)
        direction = np.asarray(
            [0.4, -0.2, 0.1, 0.3, -0.1, 0.2, -0.3, 0.2, 0.1, -0.2,
             0.1, -0.1, 0.05, 0.02, -0.03, 0.01, -0.02],
            dtype=np.float64,
        )
        direction /= np.linalg.norm(direction)
        qvel = generalized_map @ direction
        step = 1e-6
        distances = []
        for sign in (1.0, -1.0):
            perturbed = self._data()
            perturbed.qpos[:] = data.qpos
            mujoco.mj_integratePos(
                self.model, perturbed.qpos, qvel, sign * step
            )
            mujoco.mj_forward(self.model, perturbed)
            low = perturbed.qpos[self.qpos_ids[:60]]
            projection = self.shape_spec.project_actual_configuration(low)
            base = transform_from_free_qpos(perturbed.qpos[self.base_qpos])
            box = target_box_from_mujoco(
                self.model, perturbed, self.target_geom_id
            )
            distances.append(
                qp._pcc_clearance_evaluator.evaluate(
                    projection.planner_configuration, base, box
                ).distance
            )
        numeric = (distances[0] - distances[1]) / (2.0 * step)
        analytic = float(center.gradient @ direction)
        self.assertAlmostEqual(analytic, numeric, delta=2e-4)

    def test_shape_cbf_changes_command_and_retains_exact_clearance(self) -> None:
        data = self._data()
        task = self._task_arguments(data)
        baseline = HierarchicalVelocityQP(
            self.robot,
            self.model,
            self.verifier.pairs,
            HierarchicalQPConfig(),
        ).solve(data, **task)
        enabled = HierarchicalVelocityQP(
            self.robot,
            self.model,
            self.verifier.pairs,
            HierarchicalQPConfig(
                enable_pcc_cbf=True,
                enable_capsule_cbf=True,
            ),
        ).solve(data, **task)
        self.assertTrue(baseline.success)
        self.assertTrue(enabled.success)
        self.assertGreater(enabled.pcc_constraint_active_count, 0)
        self.assertGreater(enabled.pcc_binding_constraint_count, 0)
        self.assertGreater(enabled.pcc_avoidance_intervention, 1e-3)
        self.assertGreater(
            np.linalg.norm(enabled.planner_velocity - baseline.planner_velocity),
            1e-2,
        )
        self.assertGreater(enabled.mujoco_continuum_target_clearance_m, 0.005)

    def test_pcc_target_translation_and_rotation_enter_external_drift(self) -> None:
        data = self._data()
        planner = self.robot.planner_zero.copy()
        planner[:10] = np.asarray(
            [
                -0.04444606239170623,
                -0.4606650285974338,
                0.12720213765869592,
                0.39009939189381615,
                -0.09298650555520172,
                0.14360581344515333,
                0.40358451742342305,
                0.3888946356152655,
                -0.2237593477125373,
                0.07888092129926527,
            ],
            dtype=np.float64,
        )
        data.qpos[self.qpos_ids] = self.robot.encode_position(planner)
        target_twist = np.asarray(
            [0.03, -0.02, 0.01, 0.02, -0.03, 0.015], dtype=np.float64
        )
        data.qvel[self.target_dof] = target_twist
        mujoco.mj_forward(self.model, data)
        qp = HierarchicalVelocityQP(
            self.robot,
            self.model,
            self.verifier.pairs,
            HierarchicalQPConfig(enable_pcc_cbf=True),
        )
        generalized_map, _ = qp.reaction_velocity_map(data)
        center = qp._pcc_clearance_kinematics(data, generalized_map)
        step = 1e-6
        distances = []
        for sign in (1.0, -1.0):
            perturbed = self._data()
            perturbed.qpos[:] = data.qpos
            perturbed.qvel[:] = 0.0
            perturbed.qvel[self.target_dof] = target_twist
            mujoco.mj_integratePos(
                self.model, perturbed.qpos, perturbed.qvel, sign * step
            )
            mujoco.mj_forward(self.model, perturbed)
            projection = self.shape_spec.project_actual_configuration(
                perturbed.qpos[self.qpos_ids[:60]]
            )
            distances.append(
                qp._pcc_clearance_evaluator.evaluate(
                    projection.planner_configuration,
                    transform_from_free_qpos(perturbed.qpos[self.base_qpos]),
                    target_box_from_mujoco(
                        self.model, perturbed, self.target_geom_id
                    ),
                ).distance
            )
        numeric = (distances[0] - distances[1]) / (2.0 * step)
        self.assertAlmostEqual(
            center.target_distance_rate_m_s, numeric, delta=2e-5
        )

    def test_formal_v61b_artifacts_pass(self) -> None:
        audit = _load_json("pcc_audit.json")
        comparison = _load_json("clearance_compare.json")
        regression = _load_json("regression_report.json")
        self.assertTrue(audit["passed"])
        self.assertGreaterEqual(audit["configuration_count"], 10_000)
        self.assertGreaterEqual(audit["jacobian_case_count"], 1_000)
        self.assertEqual(audit["false_safe"]["pcc_count"], 0)
        self.assertEqual(audit["false_safe"]["capsule_count"], 0)
        self.assertTrue(regression["passed"])
        self.assertEqual(regression["baseline"]["validation_checks"], "26/26")
        self.assertEqual(regression["baseline"]["qp_failure_count"], 0)
        self.assertLess(
            regression["enabled_control"]["task_full_latency_p95_ms"], 20.0
        )
        self.assertGreater(
            regression["enabled_control"]["pcc_constraint_active_count"], 0
        )
        self.assertGreater(
            regression["enabled_control"]["pcc_avoidance_intervention_max"],
            1e-3,
        )
        self.assertGreater(comparison["sample_count"], 0)
        for name in (
            "distance_comparison.png",
            "gradient_comparison.png",
            "minimum_clearance_comparison.png",
        ):
            self.assertTrue((OUTPUT_ROOT / "plots" / name).is_file())


if __name__ == "__main__":
    unittest.main()
