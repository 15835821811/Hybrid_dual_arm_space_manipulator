"""Focused regressions for the standalone V6-lite controller."""

from __future__ import annotations

import ast
import json
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
)
from v6_lite.hierarchical_qp import (
    CONTINUUM_EE_OFFSET_M,
    HierarchicalQPConfig,
    HierarchicalVelocityQP,
    free_joint_slices,
    joint_addresses,
)
from v6_lite.irregular_waypoints import build_original_irregular_target
from v6_lite.run_v6_lite import (
    CONTRACT_VERSION,
    EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M,
    TARGET_SATELLITE_COLLISION_POLICY,
    V6LiteRunConfig,
    build_scenarios,
    default_v6_lite_robot_spec,
    quaternion_geodesic_angle_rad,
)


ROOT = Path(__file__).resolve().parent


class V6LiteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = default_v6_lite_robot_spec()
        cls.run_config = V6LiteRunConfig(scenario_count=5)
        cls.scenario = build_scenarios(cls.spec, cls.run_config)[0]
        cls.verifier = WholeBodyCollisionVerifier(
            cls.spec,
            cls.scenario.obstacles,
            WholeBodyVerificationConfig(
                minimum_clearance=cls.run_config.whole_body_minimum_clearance_m,
                adaptive_subdivisions=2,
                include_target_satellite_pairs=True,
            ),
        )
        cls.model = cls.verifier.model
        cls.data = mujoco.MjData(cls.model)
        qpos_ids, _dof_ids = joint_addresses(cls.model, cls.spec)
        cls.data.qpos[qpos_ids] = cls.spec.encode_position(cls.spec.planner_zero)
        target_qpos, _target_dof = free_joint_slices(
            cls.model, cls.spec.target_free_joint_name
        )
        cls.data.qpos[target_qpos.start : target_qpos.start + 3] += (
            cls.scenario.target_satellite_position_shift_m
        )
        cls.data.qvel[:] = 0.0
        mujoco.mj_forward(cls.model, cls.data)
        cls.qp = HierarchicalVelocityQP(
            cls.spec, cls.model, cls.verifier.pairs, HierarchicalQPConfig()
        )

    def _continuum_target_clearance_fixture(
        self, target_qvel: np.ndarray
    ) -> tuple[
        mujoco.MjData,
        HierarchicalVelocityQP,
        object,
        np.ndarray,
    ]:
        """Place one formal mesh--box pair at a smooth active clearance."""

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.data.qpos
        target_qpos, target_dof = free_joint_slices(
            self.model, self.spec.target_free_joint_name
        )
        data.qpos[target_qpos.start : target_qpos.start + 3] = np.asarray(
            [1.2, 0.35, 0.05], dtype=np.float64
        )
        data.qpos[target_qpos.start + 3 : target_qpos.stop] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        data.qvel[:] = 0.0
        data.qvel[target_dof] = np.asarray(target_qvel, dtype=np.float64)
        mujoco.mj_forward(self.model, data)
        pair = next(
            pair
            for pair in self.verifier.pairs
            if pair.pair_class == "continuum_target"
            and pair.geom_a_name == "collision_0033"
            and pair.geom_b_name == "target_satellite_collision"
        )
        qp = HierarchicalVelocityQP(
            self.spec,
            self.model,
            (pair,),
            HierarchicalQPConfig(),
        )
        generalized_map, _residual = qp.reaction_velocity_map(data)
        return data, qp, pair, generalized_map

    def _distance_finite_difference(
        self,
        data: mujoco.MjData,
        pair: object,
        generalized_velocity: np.ndarray,
        epsilon: float = 1e-6,
    ) -> float:
        distances = []
        for sign in (-1.0, 1.0):
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = data.qpos
            mujoco.mj_integratePos(
                self.model,
                scratch.qpos,
                generalized_velocity,
                sign * epsilon,
            )
            mujoco.mj_forward(self.model, scratch)
            distances.append(
                float(
                    mujoco.mj_geomDistance(
                        self.model,
                        scratch,
                        pair.geom_a,
                        pair.geom_b,
                        0.09,
                        np.empty(6),
                    )
                )
            )
        return (distances[1] - distances[0]) / (2.0 * epsilon)

    def test_rates_are_exact_and_single_stage(self) -> None:
        self.run_config.validate()
        qp_config = HierarchicalQPConfig()
        qp_config.validate()
        self.assertEqual(self.run_config.physics_period_s, 0.002)
        self.assertEqual(self.run_config.task_period_s, 0.02)
        self.assertEqual(
            int(self.run_config.task_period_s / self.run_config.physics_period_s), 10
        )
        self.assertGreater(
            qp_config.rigid_priority_weight, qp_config.continuum_priority_weight
        )
        self.assertLessEqual(self.run_config.rigid_final_error_threshold_m, 0.00010)
        self.assertLessEqual(self.run_config.rigid_steady_rmse_threshold_m, 0.00015)
        self.assertLessEqual(
            self.run_config.continuum_irregular_path_rmse_threshold_m, 0.00018
        )
        self.assertGreaterEqual(
            self.run_config.whole_body_minimum_clearance_m, 0.005
        )
        self.assertLessEqual(self.run_config.orientation_error_threshold_deg, 0.25)

    def test_reaction_map_enforces_zero_base_momentum(self) -> None:
        mapping, residual = self.qp.reaction_velocity_map(self.data)
        _target_qpos, target_dof = free_joint_slices(
            self.model, self.spec.target_free_joint_name
        )
        self.assertEqual(mapping.shape, (self.model.nv, 17))
        self.assertLessEqual(residual, 1e-10)
        self.assertGreater(np.linalg.norm(mapping[:6]), 1e-6)
        np.testing.assert_array_equal(
            mapping[target_dof, :], np.zeros((6, 17), dtype=np.float64)
        )

    def test_continuum_end_effector_initial_position_contract(self) -> None:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.spec.continuum_tip_body_name,
        )
        rotation = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        position = (
            np.asarray(self.data.xpos[body_id])
            + rotation @ CONTINUUM_EE_OFFSET_M
        )
        np.testing.assert_allclose(
            position,
            EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M,
            rtol=0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            self.scenario.continuum_target.initial_position_w,
            EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M,
            rtol=0.0,
            atol=1e-8,
        )
        rebuilt = build_original_irregular_target(
            EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M,
            self.scenario.continuum_target_rotation_world,
            self.scenario.seed,
        )
        np.testing.assert_allclose(
            self.scenario.continuum_target.waypoint_points_w,
            rebuilt.waypoint_points_w,
            rtol=0.0,
            atol=1e-12,
        )

    def test_continuum_offset_point_jacobian_matches_finite_difference(self) -> None:
        mapping, _residual = self.qp.reaction_velocity_map(self.data)
        jacobian, _rotation_jacobian = self.qp._task_jacobians(
            self.data,
            self.qp.continuum_body_id,
            mapping,
            CONTINUUM_EE_OFFSET_M,
        )
        direction = np.linspace(-0.12, 0.12, 17)
        generalized_velocity = mapping @ direction
        epsilon = 1e-6
        points = []
        for sign in (-1.0, 1.0):
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.data.qpos
            mujoco.mj_integratePos(
                self.model,
                scratch.qpos,
                generalized_velocity,
                sign * epsilon,
            )
            mujoco.mj_forward(self.model, scratch)
            rotation = np.asarray(
                scratch.xmat[self.qp.continuum_body_id]
            ).reshape(3, 3)
            points.append(
                np.asarray(scratch.xpos[self.qp.continuum_body_id]).copy()
                + rotation @ CONTINUUM_EE_OFFSET_M
            )
        numerical = (points[1] - points[0]) / (2.0 * epsilon)
        np.testing.assert_allclose(
            jacobian @ direction, numerical, rtol=0.0, atol=2e-6
        )

    def test_base_attitude_drift_uses_shortest_quaternion_angle(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        quarter_turn_z = np.asarray(
            [np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)]
        )
        self.assertAlmostEqual(
            float(quaternion_geodesic_angle_rad(identity, identity)), 0.0
        )
        self.assertAlmostEqual(
            float(quaternion_geodesic_angle_rad(identity, -identity)), 0.0
        )
        self.assertAlmostEqual(
            float(quaternion_geodesic_angle_rad(identity, quarter_turn_z)),
            np.pi / 2.0,
        )

    def test_mujoco_clearance_gradient_matches_finite_difference(self) -> None:
        target_qvel = np.asarray(
            [0.013, 0.041, -0.017, 0.021, -0.016, 0.012],
            dtype=np.float64,
        )
        data, qp, pair, mapping = self._continuum_target_clearance_fixture(
            target_qvel
        )
        distance, gradient, target_distance_rate = (
            qp.clearance_kinematics_for_pair(data, pair, mapping)
        )
        self.assertGreater(distance, qp.config.clearance_safe_m)
        self.assertLess(distance, qp.config.clearance_activation_m)
        self.assertGreater(float(np.linalg.norm(gradient)), 0.1)
        self.assertGreater(abs(target_distance_rate), 0.01)
        self.assertLess(target_distance_rate, 0.0)

        direction = np.linspace(-0.12, 0.12, 17)
        generalized_velocity = mapping @ direction
        generalized_velocity[qp.target_dof_slice] = target_qvel
        numerical = self._distance_finite_difference(
            data, pair, generalized_velocity
        )
        analytic = float(gradient @ direction) + target_distance_rate
        # Native-CCD mesh witnesses are approximate; the nonzero sign and the
        # first-order rate still agree well within 1 mm/s.
        self.assertAlmostEqual(analytic, numerical, delta=1.1e-3)

    def test_target_rotation_enters_witness_point_distance_rate(self) -> None:
        angular_target_qvel = np.asarray(
            [0.0, 0.0, 0.0, 0.21, -0.16, 0.12], dtype=np.float64
        )
        data, qp, pair, mapping = self._continuum_target_clearance_fixture(
            angular_target_qvel
        )
        _distance, _gradient, target_distance_rate = (
            qp.clearance_kinematics_for_pair(data, pair, mapping)
        )
        generalized_velocity = np.zeros(self.model.nv, dtype=np.float64)
        generalized_velocity[qp.target_dof_slice] = angular_target_qvel
        numerical = self._distance_finite_difference(
            data, pair, generalized_velocity
        )
        self.assertGreater(abs(target_distance_rate), 0.001)
        self.assertAlmostEqual(target_distance_rate, numerical, delta=5e-4)

    def test_target_drift_and_pair_specific_barrier_rhs(self) -> None:
        target_qvel = np.asarray(
            [0.0, 0.04, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        data, continuum_qp, pair, mapping = (
            self._continuum_target_clearance_fixture(target_qvel)
        )
        distance, _gradient, target_distance_rate = (
            continuum_qp.clearance_kinematics_for_pair(data, pair, mapping)
        )
        _matrix, continuum_lower, _minimum, _degenerate = (
            continuum_qp._clearance_constraints(data, mapping)
        )
        expected_continuum = (
            -continuum_qp.config.clearance_barrier_gain
            * (distance - continuum_qp.config.clearance_safe_m)
            - target_distance_rate
        )
        self.assertAlmostEqual(float(continuum_lower[0]), expected_continuum)

        stationary_data = mujoco.MjData(self.model)
        stationary_data.qpos[:] = data.qpos
        stationary_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, stationary_data)
        stationary_mapping, _residual = continuum_qp.reaction_velocity_map(
            stationary_data
        )
        _matrix, stationary_lower, _minimum, _degenerate = (
            continuum_qp._clearance_constraints(
                stationary_data, stationary_mapping
            )
        )
        self.assertAlmostEqual(
            float(continuum_lower[0] - stationary_lower[0]), 0.04
        )

        rigid_pair = replace(pair, pair_class="rigid_target")
        rigid_qp = HierarchicalVelocityQP(
            self.spec,
            self.model,
            (rigid_pair,),
            continuum_qp.config,
        )
        rigid_mapping, _residual = rigid_qp.reaction_velocity_map(data)
        _matrix, rigid_lower, _minimum, _degenerate = (
            rigid_qp._clearance_constraints(data, rigid_mapping)
        )
        expected_rigid = (
            -rigid_qp.config.clearance_barrier_gain
            * (distance - rigid_qp.config.rigid_target_clearance_safe_m)
            - target_distance_rate
        )
        self.assertAlmostEqual(float(rigid_lower[0]), expected_rigid)
        self.assertLess(float(rigid_lower[0]), float(continuum_lower[0]))
        self.assertAlmostEqual(
            float(continuum_lower[0] - rigid_lower[0]),
            continuum_qp.config.clearance_barrier_gain
            * (
                continuum_qp.config.clearance_safe_m
                - continuum_qp.config.rigid_target_clearance_safe_m
            ),
        )

    def test_rigid_target_clearance_margin_validation(self) -> None:
        with self.assertRaises(ValueError):
            HierarchicalQPConfig(rigid_target_clearance_safe_m=0.0).validate()
        with self.assertRaises(ValueError):
            HierarchicalQPConfig(rigid_target_clearance_safe_m=0.08).validate()

    def test_online_sources_do_not_import_learning_stack(self) -> None:
        roots = set()
        for name in ("hierarchical_qp.py", "run_v6_lite.py"):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".")[0])
        self.assertTrue({"torch", "diffusion_net", "diffusion_transformer"}.isdisjoint(roots))

    def test_original_irregular_waypoint_geometry_is_preserved(self) -> None:
        target = build_original_irregular_target(
            np.asarray([1.8, 0.626, 0.0], dtype=np.float64),
            np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                dtype=np.float64,
            ),
            20260801,
        )
        expected = np.asarray(
            [
                [1.805604671930219, 0.6953333733726723, 0.0],
                [1.750646649706560, 0.6143450699831265, 0.1235378135048494],
                [1.799146017322979, 0.4442572736423159, 0.1543857042120509],
                [1.808415747236327, 0.3664428568989503, -0.0036348471094757],
                [1.777551941989712, 0.4356292892923614, -0.1675586279685442],
                [1.805470540004460, 0.6046155793930722, -0.1682125273418716],
                [1.802334396463371, 0.7103013493499483, -0.0149496730390984],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            target.waypoint_points_w, expected, rtol=0.0, atol=1e-12
        )
        self.assertAlmostEqual(float(np.sum(target.segment_durations_s)), 21.0)

    def test_fixed_delivery_passes_all_declared_metrics(self) -> None:
        metrics = json.loads(
            (ROOT / "output" / "v6_lite_metrics.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (ROOT / "output" / "validation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["contract_version"], CONTRACT_VERSION)
        self.assertEqual(validation["contract_version"], CONTRACT_VERSION)
        self.assertTrue(metrics["passed"])
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["passed_count"], validation["total_count"])
        self.assertTrue(validation["checks"]["moving_target_continuum_clearance"])
        self.assertTrue(
            validation["checks"]["continuum_target_all_500hz_replay_states_clear"]
        )
        for scenario in metrics["scenarios"]:
            self.assertEqual(
                scenario["scenario"]["target_satellite_collision_policy"],
                TARGET_SATELLITE_COLLISION_POLICY,
            )
            self.assertTrue(
                scenario["execution_contract"]["moving_target_clearance_drift_in_qp"]
            )
            self.assertIn(
                "continuum_target",
                scenario["metrics"]["whole_body_clearance"]["minimum_by_class"],
            )
        aggregate = validation["recomputed_aggregate"]
        self.assertLessEqual(
            aggregate["continuum_initial_position_contract_error_m_max"], 1e-8
        )
        self.assertLessEqual(
            aggregate["rigid_final_error_m_max"],
            metrics["run_config"]["rigid_final_error_threshold_m"],
        )
        self.assertLessEqual(
            aggregate["rigid_steady_rmse_m_max"],
            metrics["run_config"]["rigid_steady_rmse_threshold_m"],
        )
        self.assertLessEqual(
            aggregate["continuum_irregular_path_rmse_m_max"],
            metrics["run_config"][
                "continuum_irregular_path_rmse_threshold_m"
            ],
        )
        self.assertLess(
            aggregate["rigid_orientation_error_deg_max"],
            metrics["run_config"]["orientation_error_threshold_deg"],
        )
        self.assertLess(
            aggregate["continuum_orientation_error_deg_max"],
            metrics["run_config"]["orientation_error_threshold_deg"],
        )
        self.assertGreaterEqual(
            aggregate["whole_body_minimum_clearance_m"],
            metrics["run_config"]["whole_body_minimum_clearance_m"],
        )
        self.assertGreaterEqual(
            aggregate["continuum_target_minimum_clearance_m"],
            metrics["run_config"]["whole_body_minimum_clearance_m"],
        )
        self.assertGreaterEqual(
            aggregate["continuum_target_500hz_minimum_clearance_m"],
            metrics["run_config"]["whole_body_minimum_clearance_m"],
        )
        self.assertTrue(np.isfinite(aggregate["base_translation_drift_m_max"]))
        self.assertTrue(np.isfinite(aggregate["base_orientation_drift_deg_max"]))
        self.assertGreaterEqual(aggregate["base_translation_drift_m_max"], 0.0)
        self.assertGreaterEqual(aggregate["base_orientation_drift_deg_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
