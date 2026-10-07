"""Initial native-FK and condition identity tests; no dynamics or safety query."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier
from v6_4.task_protocol import task_from_scenario
from v6_4.typed_condition_encoder import (CONDITION_SCHEMA, FIELD_SLICES, FLAT_DIM,
    GLOBAL_DIM, TOKEN_COUNT, TOKEN_DIM, TOKEN_TYPES, encode_typed_condition,
    flatten_typed_condition, schema_config)
from v6_lite.hierarchical_qp import CONTINUUM_EE_OFFSET_M, free_joint_slices
from v6_lite.run_v6_lite import V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec


class TypedConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = default_v6_lite_robot_spec()
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("no dynamics")), \
                patch.object(mujoco, "mj_geomDistance", side_effect=AssertionError("no distance query")):
            scene = build_scenarios(cls.spec, V6LiteRunConfig(scenario_count=3))[0]
            cls.task = task_from_scenario(cls.spec, scene, task_id="typed_unit", group_id="typed_unit",
                                          family="multiple_routes", split="train")
            cls.native_model = WholeBodyCollisionVerifier(cls.spec, scene.obstacles).model
            cls.condition = encode_typed_condition(cls.task, cls.spec)

    def encode(self, task):
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("no dynamics")), \
                patch.object(mujoco, "mj_geomDistance", side_effect=AssertionError("no distance query")):
            return encode_typed_condition(task, self.spec)

    def test_schema_and_declared_conditions_change_shared_inputs(self):
        c = self.condition
        self.assertEqual(c["schema"], CONDITION_SCHEMA)
        self.assertEqual(c["global"].shape, (GLOBAL_DIM,))
        self.assertEqual(c["token_features"].shape, (TOKEN_COUNT, TOKEN_DIM))
        self.assertEqual(flatten_typed_condition(c).shape, (FLAT_DIM,))
        self.assertEqual(schema_config()["flat_dim"], 2781)
        target_changed = self.task.to_dict()
        tp, _ = free_joint_slices(self.native_model, self.spec.target_free_joint_name)
        target_changed["target_pose"][0] += .017
        target_changed["initial_qpos"][tp.start] += .017
        changed = self.encode(target_changed)
        self.assertFalse(np.array_equal(c["global"], changed["global"]))
        self.assertFalse(np.array_equal(c["token_features"][24], changed["token_features"][24]))
        obstacle_changed = self.task.to_dict()
        obstacle_changed["scenario"]["workspace_obstacles"][0]["center_w"][1] += .019
        changed = self.encode(obstacle_changed)
        np.testing.assert_array_equal(c["global"], changed["global"])
        self.assertFalse(np.array_equal(flatten_typed_condition(c), flatten_typed_condition(changed)))
        task_changed = self.task.to_dict()
        task_changed["requirements"][0]["position_m"][2] += .013
        self.assertFalse(np.array_equal(c["token_features"], self.encode(task_changed)["token_features"]))

    def test_obstacle_permutation_and_joint_token_permutation_are_invariant(self):
        task = self.task.to_dict()
        task["scenario"]["workspace_obstacles"] = list(reversed(task["scenario"]["workspace_obstacles"]))
        other = self.encode(task)
        for field in ("global", "token_features", "token_types", "token_mask"):
            np.testing.assert_array_equal(self.condition[field], other[field])
        permutation = np.random.default_rng(174).permutation(TOKEN_COUNT)
        permuted = deepcopy(self.condition)
        for field in ("token_features", "token_types", "token_mask"):
            permuted[field] = permuted[field][permutation]
        np.testing.assert_array_equal(flatten_typed_condition(self.condition), flatten_typed_condition(permuted))
        self.assertTrue(np.all(self.condition["token_features"][16:24, 32] == 0.))

    def test_padding_is_explicit_and_never_becomes_a_real_obstacle(self):
        c = deepcopy(self.condition)
        invalid = ~c["token_mask"]
        self.assertTrue(np.any(invalid))
        self.assertTrue(np.all(c["token_types"][invalid] == 0))
        c["token_features"][invalid] = np.nan
        c["token_types"][invalid] = 997
        np.testing.assert_array_equal(flatten_typed_condition(c), flatten_typed_condition(self.condition))
        c["token_features"][c["token_mask"]][0, 0] = np.nan  # Advanced indexing is a copy.
        active = np.flatnonzero(c["token_mask"])[0]
        c["token_features"][active, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "active typed"):
            flatten_typed_condition(c)

    def test_native_initial_keypoints_preserve_arm_segment_and_tip_identity(self):
        c = self.condition
        points = c["token_features"][25:]
        types = c["token_types"][25:]
        np.testing.assert_array_equal(types[:31], np.full(31, TOKEN_TYPES["continuum_keypoint"]))
        np.testing.assert_array_equal(types[31:], np.full(8, TOKEN_TYPES["rigid_keypoint"]))
        np.testing.assert_array_equal(points[:30, 36], np.repeat(np.arange(1, 6), 6))
        np.testing.assert_array_equal(points[:30, 37], np.arange(1, 31))
        np.testing.assert_array_equal(points[:31, 21:23], np.tile([0., 1.], (31, 1)))
        np.testing.assert_array_equal(points[31:, 21:23], np.tile([1., 0.], (8, 1)))
        self.assertEqual(np.count_nonzero(points[:, 39]), 2)
        data = mujoco.MjData(self.native_model)
        data.qpos[:] = self.task.initial_qpos
        data.qvel[:] = self.task.initial_qvel
        mujoco.mj_forward(self.native_model, data)
        body = mujoco.mj_name2id(self.native_model, mujoco.mjtObj.mjOBJ_BODY, "link_30")
        world_tip = data.xpos[body] + data.xmat[body].reshape(3, 3) @ CONTINUUM_EE_OFFSET_M
        base = np.asarray(self.task.base_pose)
        rb = Rotation.from_quat(base[[4, 5, 6, 3]]).as_matrix()
        np.testing.assert_allclose(points[30, :3], rb.T @ (world_tip-base[:3]), atol=1e-14, rtol=0.)
        self.assertTrue(np.all(c["token_mask"][25:]))
        self.assertEqual(c["token_types"][24], TOKEN_TYPES["operative_target"])
        self.assertEqual(c["token_features"][24, 38], 1.)
        np.testing.assert_array_equal(c["token_features"][24, 18:21], [.4, .4, .4])

    def test_frozen_base_transform_and_world_twist_conversion(self):
        source = self.task.to_dict()
        transformed = deepcopy(source)
        rotation = Rotation.from_rotvec([.0, .0, .63]).as_matrix()
        translation = np.array([2.1, -.7, .3])
        for label, joint in (("base", self.spec.base_joint_name), ("target", self.spec.target_free_joint_name)):
            pose = np.array(source[label+"_pose"])
            original_rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            quat_xyzw = Rotation.from_matrix(rotation @ original_rotation).as_quat()
            pose = np.r_[rotation @ pose[:3]+translation, quat_xyzw[[3, 0, 1, 2]]]
            transformed[label+"_pose"] = pose.tolist()
            pos, vel = free_joint_slices(self.native_model, joint)
            transformed["initial_qpos"][pos] = pose.tolist()
            # MuJoCo free linear qvel is world; angular qvel is body local.
            transformed["initial_qvel"][vel.start:vel.start+3] = (rotation @ np.array(source["initial_qvel"])[vel.start:vel.start+3]).tolist()
            twist = np.array(source[label+"_twist"])
            transformed[label+"_twist"] = np.r_[rotation @ twist[:3], rotation @ twist[3:]].tolist()
        for point in transformed["requirements"]:
            if point["frame"] == "world":
                point["position_m"] = (rotation @ point["position_m"]+translation).tolist()
                point["rotation"] = (rotation @ np.array(point["rotation"]).reshape(3, 3)).ravel().tolist()
        for obstacle in transformed["scenario"]["workspace_obstacles"]:
            obstacle["center_w"] = (rotation @ obstacle["center_w"]+translation).tolist()
        transformed["scenario"]["continuum_target_rotation_world"] = (rotation @ np.array(source["scenario"]["continuum_target_rotation_world"]).reshape(3, 3)).tolist()
        actual = self.encode(transformed)
        np.testing.assert_allclose(actual["global"], self.condition["global"], atol=1e-14, rtol=0.)
        np.testing.assert_allclose(actual["token_features"], self.condition["token_features"], atol=1e-13, rtol=0.)

    def test_model_identity_and_initial_state_mismatch_reject_no_ids_or_future_features(self):
        unknown = self.task.to_dict()
        unknown["model_contract_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "matching nominal model"):
            self.encode(unknown)
        wrong = self.task.to_dict()
        wrong["initial_planner_q"][0] += .01
        with self.assertRaisesRegex(ValueError, "states disagree"):
            self.encode(wrong)
        enriched = self.task.to_dict()
        enriched["task_id"], enriched["group_id"], enriched["seed"] = "other_id", "other_group", 921
        enriched["future_actual_trace"] = {"qpos": [1e10], "target_pose": [999.]}
        other = self.encode(enriched)
        np.testing.assert_array_equal(flatten_typed_condition(other), flatten_typed_condition(self.condition))


if __name__ == "__main__":
    unittest.main()
