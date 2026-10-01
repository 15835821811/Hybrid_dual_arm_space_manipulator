"""B.2-1 fixed-interval safety function, coverage and differential tests."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.pcc_interval_cbf import (
    FixedIntervalCBFEvaluator, IntervalPartition, MaterialInterval,
    _obb_derivative_status,
)
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import OrientedBox, point_obb_signed_distance


class IntervalCBFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot = default_v6_lite_robot_spec()
        cls.model = cls.robot.compile_dynamic_model()
        cls.evaluator = FixedIntervalCBFEvaluator(cls.robot, cls.model)
        cls.qpos_ids, _ = joint_addresses(cls.model, cls.robot)
        cls.target_qpos, cls.target_dof = free_joint_slices(
            cls.model, cls.robot.target_free_joint_name
        )

    def _state(self) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        planner = self.robot.planner_zero.copy()
        planner[:10] = [0.13, -0.16, 0.08, 0.11, -0.12,
                        0.07, -0.09, 0.06, 0.04, -0.08]
        data.qpos[self.qpos_ids] = self.robot.encode_position(planner)
        data.qpos[self.target_qpos.start:self.target_qpos.start + 3] = [
            1.2, 0.41, 0.16,
        ]
        data.qpos[self.target_qpos.start + 3:self.target_qpos.stop] = [
            1.0, 0.0, 0.0, 0.0,
        ]
        data.qvel[:] = 0.0
        data.qvel[self.target_dof] = [0.03, -0.02, 0.01,
                                      0.02, -0.03, 0.01]
        mujoco.mj_forward(self.model, data)
        return data

    def _h_after_perturbation(self, data: mujoco.MjData, tangent: np.ndarray,
                              step: float, interval_id: str) -> float:
        copied = mujoco.MjData(self.model)
        copied.qpos[:] = data.qpos
        mujoco.mj_integratePos(self.model, copied.qpos, tangent, step)
        mujoco.mj_forward(self.model, copied)
        result = self.evaluator.evaluate_state(
            copied, IntervalPartition.uniform(1), with_derivatives=False
        )
        return next(row.h_m for row in result.rows if row.interval_id == interval_id)

    def test_partition_coverage_stable_ids_refine_and_merge(self) -> None:
        lengths = self.evaluator.shape_spec.segment_lengths_m
        base = IntervalPartition.uniform(1)
        self.assertTrue(base.coverage(lengths).coverage_complete)
        leaf = MaterialInterval(2, "0")
        refined = base.refine([leaf.interval_id])
        self.assertTrue(refined.coverage(lengths).coverage_complete)
        self.assertNotIn(leaf, refined.leaves)
        self.assertIn(MaterialInterval(2, "00"), refined.leaves)
        self.assertIn(MaterialInterval(2, "01"), refined.leaves)
        self.assertEqual(refined.merge(leaf), base)
        missing = IntervalPartition(tuple(x for x in base.leaves if x != leaf))
        self.assertFalse(missing.coverage(lengths).coverage_complete)
        self.assertTrue(missing.coverage(lengths).interval_well_formed)
        duplicate = IntervalPartition(base.leaves + (base.leaves[0],))
        self.assertFalse(duplicate.coverage(lengths).interval_well_formed)

    def test_batched_jacobians_match_scalar_at_all_five_sections(self) -> None:
        shape = self.evaluator.shape_model
        rng = np.random.default_rng(20260930)
        lengths = shape.spec.segment_lengths_m
        boundaries = shape.spec.segment_boundaries_m
        for _ in range(4):
            q = rng.uniform(-0.7, 0.7, 10)
            arclengths = [float(boundaries[i] + fraction * lengths[i])
                          for i in range(5) for fraction in (0.2, 0.5, 0.8)]
            arclengths += [float(value) for value in boundaries]
            arclengths += [-5e-13, float(boundaries[-1] + 5e-13)]
            arclengths.reverse()
            batch = shape.batch_query(q, np.eye(4), arclengths,
                                      with_jacobians=True)
            position_only = shape.batch_query(
                q, np.eye(4), arclengths, with_jacobians=True,
                with_rotation_jacobians=False,
            )
            for s, result, limited in zip(arclengths, batch, position_only):
                scalar = shape.evaluate(q, np.eye(4), s, with_jacobians=True)
                np.testing.assert_allclose(result.position_world, scalar.position_world,
                                           atol=1e-12, rtol=0.0)
                np.testing.assert_allclose(result.position_jacobian,
                                           scalar.position_jacobian,
                                           atol=1e-10, rtol=0.0)
                np.testing.assert_allclose(result.rotation_jacobian,
                                           scalar.rotation_jacobian,
                                           atol=1e-10, rtol=0.0)
                np.testing.assert_allclose(limited.position_world,
                                           result.position_world, atol=1e-12, rtol=0.0)
                np.testing.assert_allclose(limited.position_jacobian,
                                           result.position_jacobian, atol=1e-10, rtol=0.0)
                np.testing.assert_array_equal(limited.rotation_jacobian,
                                              np.zeros((3, 10)))
        for invalid in (-2e-12, float(boundaries[-1] + 2e-12), float("nan")):
            with self.assertRaises(ValueError):
                shape.batch_query(np.zeros(10), np.eye(4), [invalid])

    def test_fixed_function_and_shape_gradient_match_same_midpoint(self) -> None:
        data = self._state()
        batch = self.evaluator.evaluate_state(data, IntervalPartition.uniform(1))
        self.assertTrue(batch.coverage_complete)
        self.assertEqual(batch.floating_point_certification, "NOT_FORMALLY_CERTIFIED")
        self.assertEqual(batch.point_evaluation_count, 10)
        self.assertEqual(batch.shape_jacobian_evaluation_count, 10)
        self.assertEqual(batch.jacobian_evaluation_count,
                         batch.shape_jacobian_evaluation_count
                         + batch.mujoco_point_jacobian_count)
        row = next(x for x in batch.rows if x.derivative_status == "SUPPORTED")
        expected = (row.point_signed_distance_m - row.tube_radius_m
                    - row.coverage_term_m - row.numerical_pad_m
                    - row.safe_distance_m)
        self.assertAlmostEqual(row.h_m, expected, places=13)
        self.assertAlmostEqual(row.coverage_term_m,
                               (row.local_end_m - row.local_start_m) / 2.0)
        self.assertEqual(row.interval_id,
                         f"pcc_interval:{row.segment_id}:"
                         + ("0" if row.local_start_m == 0 else "1"))
        q = self.evaluator.shape_spec.project_actual_configuration(
            data.qpos[self.qpos_ids[:60]]
        ).planner_configuration
        base = np.eye(4)
        box = OrientedBox(
            center=np.asarray(data.geom_xpos[self.evaluator.target_geom_id]).copy(),
            rotation=np.asarray(data.geom_xmat[self.evaluator.target_geom_id]).reshape(3, 3).copy(),
            half_extents=np.asarray(self.model.geom_size[self.evaluator.target_geom_id]).copy(),
        )
        numerical = np.zeros(10)
        step = 1e-6
        for coordinate in range(10):
            altered = []
            for sign in (+1.0, -1.0):
                trial = q.copy()
                trial[coordinate] += sign * step
                point = self.evaluator.shape_model.evaluate(
                    trial, base, row.midpoint_arclength_m,
                    with_jacobians=False,
                ).position_world
                altered.append(point_obb_signed_distance(point, box).signed_distance_m)
            numerical[coordinate] = (altered[0] - altered[1]) / (2 * step)
        np.testing.assert_allclose(row.shape_gradient, numerical,
                                   atol=2e-5, rtol=2e-3)

    def test_full_reaction_gradient_and_target_6d_drift(self) -> None:
        data = self._state()
        partition = IntervalPartition.uniform(1)
        reaction = self.evaluator.reaction_map(data)
        batch = self.evaluator.evaluate_state(
            data, partition, generalized_map=reaction
        )
        row = next(x for x in batch.rows if x.derivative_status == "SUPPORTED"
                   and np.linalg.norm(x.generalized_gradient) > 1e-3)
        rng = np.random.default_rng(20260929)
        step = 1e-6
        for _ in range(4):
            direction = rng.normal(size=17)
            direction /= np.linalg.norm(direction)
            tangent = reaction @ direction
            plus = self._h_after_perturbation(data, tangent, step, row.interval_id)
            minus = self._h_after_perturbation(data, tangent, -step, row.interval_id)
            numeric = (plus - minus) / (2 * step)
            analytic = float(row.generalized_gradient @ direction)
            self.assertLessEqual(abs(numeric - analytic),
                                 2e-5 + 2e-3 * abs(numeric))
        target_only = np.zeros(self.model.nv)
        target_only[self.target_dof] = data.qvel[self.target_dof]
        plus = self._h_after_perturbation(data, target_only, step, row.interval_id)
        minus = self._h_after_perturbation(data, target_only, -step, row.interval_id)
        numeric_drift = (plus - minus) / (2 * step)
        self.assertLessEqual(abs(numeric_drift - row.target_drift_m_s),
                             2e-5 + 2e-3 * abs(numeric_drift))

    def test_body_origin_transport_matches_mujoco_point_jacobians(self) -> None:
        data = self._state()
        tangent = np.zeros(self.model.nv)
        tangent[self.evaluator.base_dof_slice] = [0.02, -0.03, 0.04,
                                                  0.13, -0.09, 0.07]
        tangent[self.target_dof] = [-0.04, 0.02, -0.01,
                                    -0.10, 0.08, 0.05]
        mujoco.mj_integratePos(self.model, data.qpos, tangent, 1.0)
        mujoco.mj_forward(self.model, data)
        reaction = self.evaluator.reaction_map(data)
        batch = self.evaluator.evaluate_state(
            data, IntervalPartition.uniform(2), generalized_map=reaction,
        )
        supported = [row for row in batch.rows
                     if row.derivative_status == "SUPPORTED"]
        self.assertTrue(supported)
        self.assertEqual(batch.mujoco_point_jacobian_count, 2)
        self.assertEqual(batch.jacobian_evaluation_count,
                         batch.shape_jacobian_evaluation_count + 2)
        distance_only = self.evaluator.evaluate_state(
            data, IntervalPartition.uniform(2), generalized_map=reaction,
            derivative_interval_ids=set(),
        )
        self.assertEqual(distance_only.shape_jacobian_evaluation_count, 0)
        self.assertEqual(distance_only.mujoco_point_jacobian_count, 0)
        np.testing.assert_allclose(
            [row.h_m for row in distance_only.rows],
            [row.h_m for row in batch.rows], atol=1e-12, rtol=0.0,
        )
        target_velocity = np.zeros(self.model.nv)
        target_velocity[self.target_dof] = data.qvel[self.target_dof]
        for row in supported:
            base_point = self.evaluator._point_jacobian(
                data, self.evaluator.base_body_id, row.centerline_point_world,
            )
            target_point = self.evaluator._point_jacobian(
                data, self.evaluator.target_body_id, row.obb_witness_world,
            )
            expected_gradient = (row.normal_box_to_point @ base_point @ reaction)
            expected_gradient[:10] += row.shape_gradient
            expected_drift = float(
                -row.normal_box_to_point @ target_point @ target_velocity
            )
            np.testing.assert_allclose(row.generalized_gradient, expected_gradient,
                                       atol=2e-10, rtol=2e-10)
            self.assertAlmostEqual(row.target_drift_m_s, expected_drift, places=10)

    def test_multiple_intervals_and_nonsmooth_interior_are_explicit(self) -> None:
        data = self._state()
        partition = IntervalPartition.uniform(1)
        batch = self.evaluator.evaluate_state(data, partition)
        self.assertEqual(len({row.interval_id for row in batch.rows}), 10)
        self.assertEqual({row.segment_id for row in batch.rows}, set(range(5)))
        selected = {batch.rows[0].interval_id}
        selective = self.evaluator.evaluate_state(
            data, partition, derivative_interval_ids=selected,
        )
        self.assertEqual(selective.point_evaluation_count, 10)
        self.assertEqual(selective.shape_jacobian_evaluation_count, 1)
        self.assertLessEqual(selective.mujoco_point_jacobian_count, 2)
        np.testing.assert_allclose(
            [x.h_m for x in selective.rows], [x.h_m for x in batch.rows],
            atol=1e-12, rtol=0.0,
        )
        self.assertEqual(selective.rows[1].derivative_status, "NOT_REQUESTED")
        cube = OrientedBox(np.zeros(3), np.eye(3), np.ones(3) * 0.1)
        self.assertEqual(_obb_derivative_status(np.zeros(3), cube),
                         "NONSMOOTH_OBB_INTERIOR_FEATURE")
        self.assertEqual(_obb_derivative_status(np.array([0.3, 0.0, 0.0]), cube),
                         "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
