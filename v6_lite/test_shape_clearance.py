from __future__ import annotations

import unittest

import mujoco
import numpy as np

from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.continuum_shape_model import transform_from_free_qpos
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.shape_clearance import (
    OrientedBox,
    ShapeClearanceShadow,
    build_continuum_capsule_envelopes,
    minimum_mujoco_geom_clearance,
    point_obb_signed_distance,
    segment_obb_signed_distance,
)


class ShapeClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot = default_v6_lite_robot_spec()
        cls.spec = default_continuum_model_spec(cls.robot)
        cls.model = cls.robot.compile_dynamic_model()
        cls.data = mujoco.MjData(cls.model)
        cls.qpos_ids, _ = joint_addresses(cls.model, cls.robot)
        cls.base_qpos, _ = free_joint_slices(
            cls.model, cls.robot.base_joint_name
        )
        cls.target_geom = int(
            mujoco.mj_name2id(
                cls.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "target_satellite_collision",
            )
        )

    def test_point_box_face_edge_corner_and_interior(self) -> None:
        box = OrientedBox(np.zeros(3), np.eye(3), np.ones(3))
        self.assertAlmostEqual(
            point_obb_signed_distance([2.0, 0.0, 0.0], box).signed_distance_m,
            1.0,
        )
        self.assertAlmostEqual(
            point_obb_signed_distance([2.0, 2.0, 0.0], box).signed_distance_m,
            np.sqrt(2.0),
        )
        self.assertAlmostEqual(
            point_obb_signed_distance([2.0, 2.0, 2.0], box).signed_distance_m,
            np.sqrt(3.0),
        )
        self.assertAlmostEqual(
            point_obb_signed_distance([0.0, 0.0, 0.0], box).signed_distance_m,
            -1.0,
        )

    def test_segment_box_exterior_and_penetration(self) -> None:
        box = OrientedBox(np.zeros(3), np.eye(3), np.ones(3))
        face = segment_obb_signed_distance([2, -0.5, 0], [2, 0.5, 0], box)
        self.assertAlmostEqual(face.signed_distance_m, 1.0)
        crossing = segment_obb_signed_distance([-2, 0, 0], [2, 0, 0], box)
        self.assertLess(crossing.signed_distance_m, 0.0)

    def test_capsules_cover_every_distal_geom_and_mount_has_fallback(self) -> None:
        envelopes = build_continuum_capsule_envelopes(self.model, self.spec)
        self.assertEqual(len(envelopes.continuum_geom_names), 62)
        self.assertEqual(len(envelopes.capsules), 61)
        self.assertEqual(envelopes.fallback_geom_names, ("collision_0003",))
        self.assertLessEqual(
            max(item.maximum_vertex_excess_m for item in envelopes.capsules),
            1e-10,
        )

    def test_shadow_is_read_only_and_returns_three_distances(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qpos_ids] = self.robot.encode_position(
            self.robot.planner_zero
        )
        mujoco.mj_forward(self.model, self.data)
        shadow = ShapeClearanceShadow(
            self.model,
            self.target_geom,
            np.full(5, 0.10),
            spec=self.spec,
        )
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ctrl = self.data.ctrl.copy()
        result = shadow.evaluate(
            self.data,
            np.zeros(10),
            transform_from_free_qpos(self.data.qpos[self.base_qpos]),
        )
        self.assertTrue(np.isfinite(result.capsule.signed_distance_m))
        self.assertTrue(np.isfinite(result.pcc_tube.signed_distance_m))
        self.assertTrue(np.isfinite(result.mujoco_geometry.signed_distance_m))
        np.testing.assert_array_equal(self.data.qpos, qpos)
        np.testing.assert_array_equal(self.data.qvel, qvel)
        np.testing.assert_array_equal(self.data.ctrl, ctrl)

    def test_mujoco_query_reports_distmax_truncation(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        target_qpos, _ = free_joint_slices(
            self.model, self.robot.target_free_joint_name
        )
        self.data.qpos[target_qpos] = [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        envelopes = build_continuum_capsule_envelopes(self.model, self.spec)
        result = minimum_mujoco_geom_clearance(
            self.model,
            self.data,
            [item.geom_id for item in envelopes.capsules],
            self.target_geom,
            query_distance_max_m=0.05,
        )
        self.assertTrue(result.query_truncated)
        self.assertAlmostEqual(result.signed_distance_m, 0.05)


if __name__ == "__main__":
    unittest.main()
