from __future__ import annotations

import unittest

import mujoco
import numpy as np

from v6_lite.continuum_model_spec import (
    CONTINUUM_ACTUATED_DOF,
    default_continuum_model_spec,
)
from v6_lite.continuum_shape_model import (
    ContinuumShapeModel,
    DiscreteContinuumKinematics,
    rotation_angle,
    transform_from_free_qpos,
    vee,
)
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


class ContinuumShapeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot = default_v6_lite_robot_spec()
        cls.spec = default_continuum_model_spec(cls.robot)
        cls.shape = ContinuumShapeModel(cls.spec)
        cls.discrete = DiscreteContinuumKinematics(cls.spec)
        cls.model = cls.robot.compile_dynamic_model()
        cls.qpos_ids, _ = joint_addresses(cls.model, cls.robot)
        cls.base_qpos, _ = free_joint_slices(
            cls.model, cls.robot.base_joint_name
        )

    def test_versioned_contract_freezes_mount_lengths_and_mapping(self) -> None:
        np.testing.assert_allclose(
            self.spec.segment_lengths_m,
            [0.3, 0.3, 0.3, 0.3, 0.2975],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.spec.base_to_shape_start[:3, 3],
            [0.4325, 0.626, 5.38577421e-09],
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            self.spec.actuated_to_planner @ self.spec.planner_to_actuated,
            np.eye(10),
            rtol=0.0,
            atol=1e-12,
        )

    def test_zero_curvature_is_straight_and_finite(self) -> None:
        values = self.shape.sample(
            np.zeros(10),
            np.eye(4),
            np.linspace(0.0, self.spec.total_length_m, 101),
            with_jacobians=True,
        )
        points = np.asarray([item.position for item in values])
        expected = (
            self.spec.base_to_shape_start[:3, 3]
            + np.linspace(0.0, self.spec.total_length_m, 101)[:, None]
            * self.spec.base_to_shape_start[:3, 0]
        )
        np.testing.assert_allclose(points, expected, rtol=0.0, atol=1e-11)
        self.assertTrue(
            all(
                np.all(np.isfinite(item.position_jacobian))
                and np.all(np.isfinite(item.rotation_jacobian))
                for item in values
            )
        )

    def test_pcc_jacobians_match_central_finite_difference(self) -> None:
        rng = np.random.default_rng(731)
        step = 1e-6
        for _ in range(12):
            configuration = rng.uniform(-1.0, 1.0, size=10)
            arclength = float(rng.uniform(0.0, self.spec.total_length_m))
            center = self.shape.evaluate(configuration, np.eye(4), arclength)
            for coordinate in range(10):
                plus = configuration.copy()
                minus = configuration.copy()
                plus[coordinate] += step
                minus[coordinate] -= step
                p = self.shape.evaluate(
                    plus, np.eye(4), arclength, with_jacobians=False
                )
                m = self.shape.evaluate(
                    minus, np.eye(4), arclength, with_jacobians=False
                )
                numeric_position = (p.position - m.position) / (2.0 * step)
                numeric_rotation = vee(
                    ((p.rotation - m.rotation) / (2.0 * step))
                    @ center.rotation.T
                )
                np.testing.assert_allclose(
                    numeric_position,
                    center.position_jacobian[:, coordinate],
                    rtol=0.0,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    numeric_rotation,
                    center.rotation_jacobian[:, coordinate],
                    rtol=0.0,
                    atol=1e-7,
                )

    def test_independent_urdf_fk_matches_mujoco(self) -> None:
        rng = np.random.default_rng(1907)
        data = mujoco.MjData(self.model)
        for _ in range(20):
            configuration = rng.uniform(-1.0, 1.0, size=10)
            actual = self.spec.planner_to_actuated @ configuration
            data.qpos[:] = self.model.qpos0
            data.qpos[self.qpos_ids[:CONTINUUM_ACTUATED_DOF]] = actual
            data.qpos[self.base_qpos] = [
                0.1,
                -0.2,
                0.3,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            mujoco.mj_forward(self.model, data)
            transforms = self.discrete.body_transforms(
                actual, transform_from_free_qpos(data.qpos[self.base_qpos])
            )
            for body_name, expected in transforms.items():
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                np.testing.assert_allclose(
                    expected[:3, 3], data.xpos[body_id], rtol=0.0, atol=1e-12
                )
                self.assertLessEqual(
                    rotation_angle(
                        expected[:3, :3].T
                        @ np.asarray(data.xmat[body_id]).reshape(3, 3)
                    ),
                    1e-6,
                )

    def test_low_dimensional_residual_detects_off_subspace_motion(self) -> None:
        configuration = np.linspace(-0.8, 0.8, 10)
        ideal = self.spec.planner_to_actuated @ configuration
        exact = self.spec.project_actual_configuration(ideal)
        self.assertLessEqual(exact.residual_l2_rad, 1e-12)
        perturbed = ideal.copy()
        perturbed[0] += 0.01
        result = self.spec.project_actual_configuration(perturbed)
        self.assertGreater(result.residual_l2_rad, 1e-3)
        np.testing.assert_allclose(
            result.orthogonal_residual,
            perturbed
            - self.spec.planner_to_actuated @ result.planner_configuration,
            rtol=0.0,
            atol=1e-15,
        )


if __name__ == "__main__":
    unittest.main()
