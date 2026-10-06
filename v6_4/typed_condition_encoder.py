"""One shared, initial-state-only condition for the B.1 architecture comparison.

Positions and rotations are expressed in the *fixed* initial base frame B0.
World linear/angular velocities are rotated into B0, not subtracted from a
moving base velocity.  Target-frame task points retain their frame flag after
being transported through the declared initial target pose.  Task/model IDs,
seeds, labels and future traces never become numerical network features.

M1 consumes the typed arrays as an unordered condition set (task order and
arm/module identity are explicit features).  M0 flattens the very same arrays,
including types and masks, after canonical ordering; padding contributes zero.
Native FK below performs only mj_forward on a private initial-state MjData.
There are no physics steps, collision queries or reference/teacher rollouts.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import mujoco
import numpy as np

from v6_4.dataset import (_matrix, _rotation, _vector, encode_condition,
                         task_dict)
from v6_lite.hierarchical_qp import (CONTINUUM_EE_OFFSET_M, free_joint_slices,
                                   joint_addresses)

CONDITION_SCHEMA = "v6_4_b1_initial_base_typed_condition_v1"
MAX_REQUIREMENTS = 16
MAX_OBSTACLES = 8
KEYPOINT_COUNT = 39  # 30 continuum module origins + tip; 7 rigid joints + tip.
GLOBAL_DIM = 93
TOKEN_COUNT = MAX_REQUIREMENTS + MAX_OBSTACLES + 1 + KEYPOINT_COUNT
TOKEN_DIM = 40
TYPE_COUNT = 8
FLAT_DIM = GLOBAL_DIM + TOKEN_COUNT * (TOKEN_DIM + 2)

TOKEN_TYPES = {"padding": 0, "rigid_requirement": 1,
               "continuum_requirement": 2, "sphere_obstacle": 3,
               "box_obstacle": 4, "operative_target": 5,
               "continuum_keypoint": 6, "rigid_keypoint": 7}
FIELD_SLICES = {
    "position_B0_m": (0, 3), "rotation_B0": (3, 12),
    "linear_velocity_B0_m_s": (12, 15),
    "angular_velocity_B0_rad_s": (15, 18), "size_m": (18, 21),
    "arm_one_hot_rigid_continuum": (21, 23),
    "frame_one_hot_world_target": (23, 25),
    "kind_one_hot_waypoint_terminal": (25, 27),
    "time_and_window_s": (27, 30), "position_orientation_tolerance": (30, 32),
    "requirement_order": (32, 33),
    "primitive_one_hot_sphere_box_keypoint": (33, 36),
    "segment_id": (36, 37), "module_or_joint_id": (37, 38),
    "operative_target_flag": (38, 39), "end_effector_flag": (39, 40)}
GLOBAL_FIELDS = (("planner_q", 17), ("planner_dq", 17),
                 ("base_pose_in_B0_wxyz", 7), ("base_twist_B0", 6),
                 ("target_twist_B0", 6), ("target_position_B0", 3),
                 ("target_rotation_B0", 9), ("time_configuration_s", 3),
                 ("grasp_position_target_frame", 3),
                 ("grasp_rotation_target_frame", 9),
                 ("continuum_rotation_B0", 9), ("task_family_one_hot", 3),
                 ("free_intermediate_path_flag", 1))


def schema_config() -> dict[str, Any]:
    """JSON-serializable input contract, suitable for the frozen run config."""
    return {"schema": CONDITION_SCHEMA, "global_dim": GLOBAL_DIM,
            "token_count": TOKEN_COUNT, "token_dim": TOKEN_DIM,
            "type_count": TYPE_COUNT, "flat_dim": FLAT_DIM,
            "global_fields": [list(x) for x in GLOBAL_FIELDS],
            "field_slices": {k: list(v) for k, v in FIELD_SLICES.items()},
            "token_types": dict(TOKEN_TYPES),
            "token_layout": {"requirements": [0, 16], "obstacles": [16, 24],
                             "operative_target": [24, 25], "initial_keypoints": [25, 64]},
            "frame": "fixed_initial_base_B0",
            "twist_rule": "rotate_world_linear_and_angular_by_R_B0_transpose",
            "model_identity_rule": "validate_matching_runtime_contract_not_a_feature",
            "padding_rule": "valid_true_mask_invalid_features_and_type_zero",
            "obstacle_order_rule": "canonical_physical_features_no_name_or_index",
            "keypoint_rule": "private_initial_state_native_FK_no_future_or_mj_step"}


@lru_cache(maxsize=1)
def _default_spec():
    from v6_lite.run_v6_lite import default_v6_lite_robot_spec
    return default_v6_lite_robot_spec()


_MODEL_CACHE = {}


def _initial_fk(task: dict, spec):
    identity = spec.runtime_contract_sha256()
    if task.get("model_contract_sha256") != identity:
        raise ValueError("typed condition requires the matching nominal model contract")
    spec.validate()
    if identity not in _MODEL_CACHE:
        model = spec.compile_dynamic_model()
        if (model.nq, model.nv, model.nu) != (81, 79, 67):
            raise ValueError("typed initial FK requires the frozen 17-to-67 model")
        _MODEL_CACHE[identity] = model
    model = _MODEL_CACHE[identity]
    data = mujoco.MjData(model)
    data.qpos[:] = _vector(task["initial_qpos"], 81, "initial qpos")
    data.qvel[:] = _vector(task["initial_qvel"], 79, "initial qvel")
    qp, qv = joint_addresses(model, spec)
    if (not np.allclose(data.qpos[qp], spec.encode_position(task["initial_planner_q"]),
                        atol=1e-9, rtol=0.)
            or not np.allclose(data.qvel[qv], spec.encode_velocity(task["initial_planner_dq"]),
                               atol=1e-9, rtol=0.)):
        raise ValueError("TaskSpec initial 17-coordinate and 67-coordinate states disagree")
    for label, joint in (("base", spec.base_joint_name),
                         ("target", spec.target_free_joint_name)):
        pos_slice, _ = free_joint_slices(model, joint)
        if not np.allclose(data.qpos[pos_slice], task[label + "_pose"], atol=1e-12, rtol=0.):
            raise ValueError("TaskSpec initial " + label + " pose differs from qpos")
    mujoco.mj_forward(model, data)
    # Declared twists are body-origin world velocities, including the native
    # free-joint angular convention conversion performed by mj_objectVelocity.
    for label, joint in (("base", spec.base_joint_name),
                         ("target", spec.target_free_joint_name)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                int(model.jnt_bodyid[jid]), velocity, 0)
        if not np.allclose(np.r_[velocity[3:], velocity[:3]], task[label + "_twist"],
                           atol=1e-9, rtol=0.):
            raise ValueError("TaskSpec initial " + label + " twist differs from qvel")
    return model, data


def _set(row, field, value):
    start, end = FIELD_SLICES[field]
    vector = np.asarray(value, dtype=np.float64).ravel()
    if vector.shape != (end-start,) or not np.all(np.isfinite(vector)):
        raise ValueError("invalid typed condition field " + field)
    row[start:end] = vector


def encode_typed_condition(task, spec=None) -> dict[str, Any]:
    """Encode TaskSpec declarations, validating the matching physical model.

    The output is raw physical data.  Shared TRAIN-only normalization is applied
    afterwards to global/features; categorical types and validity masks remain
    unscaled.  Only the first two spline controls can additionally be recovered
    from the global initial q/dq; those contain no extra raw information.
    """
    task = task_dict(task)
    encode_condition(task)  # Existing TaskSpec/frame/time/capacity validation.
    spec = _default_spec() if spec is None else spec
    model, data = _initial_fk(task, spec)
    base = _vector(task["base_pose"], 7, "base pose")
    target = _vector(task["target_pose"], 7, "target pose")
    rb, rt = _rotation(base[3:]), _rotation(target[3:])
    inverse = rb.T

    def position(p):
        return inverse @ (np.asarray(p, dtype=float)-base[:3])

    scenario = task["scenario"]
    families = ("end_effector_detour", "mid_arm_detour", "multiple_routes")
    values = [task["initial_planner_q"], task["initial_planner_dq"],
              [0., 0., 0., 1., 0., 0., 0.]]
    for body in ("base", "target"):
        twist = _vector(task[body+"_twist"], 6, body+" twist")
        values.append(np.r_[inverse @ twist[:3], inverse @ twist[3:]])
    values.extend([position(target[:3]), (inverse @ rt).ravel(),
                   [task["duration_s"], task["task_period_s"], task["physics_period_s"]],
                   scenario["grasp_point_target_frame_m"],
                   _matrix(scenario["grasp_rotation_target_frame"], "grasp rotation").ravel(),
                   (inverse @ _matrix(scenario["continuum_target_rotation_world"],
                                       "continuum rotation")).ravel(),
                   [float(task["family"] == f) for f in families], [1.]])
    global_state = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values])
    if global_state.shape != (GLOBAL_DIM,) or not np.all(np.isfinite(global_state)):
        raise ValueError("typed global state does not match schema")
    features = np.zeros((TOKEN_COUNT, TOKEN_DIM), dtype=np.float64)
    types = np.zeros(TOKEN_COUNT, dtype=np.int64)
    mask = np.zeros(TOKEN_COUNT, dtype=bool)
    for order, point in enumerate(task["requirements"]):
        row = features[order]
        p = _vector(point["position_m"], 3, "task point")
        rotation = _matrix(point["rotation"], "task rotation")
        if point["frame"] == "target":
            p, rotation = target[:3] + rt @ p, rt @ rotation
        _set(row, "position_B0_m", position(p))
        _set(row, "rotation_B0", inverse @ rotation)
        _set(row, "arm_one_hot_rigid_continuum", [point["arm"] == "rigid", point["arm"] == "continuum"])
        _set(row, "frame_one_hot_world_target", [point["frame"] == "world", point["frame"] == "target"])
        _set(row, "kind_one_hot_waypoint_terminal", [point["kind"] == "waypoint", point["kind"] == "terminal"])
        _set(row, "time_and_window_s", [point["time_s"], *point["time_window_s"]])
        _set(row, "position_orientation_tolerance", [point["position_tolerance_m"], point["orientation_tolerance_rad"]])
        _set(row, "requirement_order", [order])
        types[order] = TOKEN_TYPES[point["arm"] + "_requirement"]
        mask[order] = True

    obstacle_rows = []
    for obstacle in scenario["workspace_obstacles"]:
        row = np.zeros(TOKEN_DIM)
        kind = obstacle.get("type", "sphere")
        if kind not in ("sphere", "box"):
            raise ValueError("unsupported declared obstacle primitive")
        _set(row, "position_B0_m", position(_vector(obstacle["center_w"], 3, "obstacle center")))
        if kind == "sphere":
            radius = float(obstacle["radius_m"])
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("invalid obstacle radius")
            # A sphere has no orientation; identity is defined directly in B0.
            _set(row, "rotation_B0", np.eye(3))
            _set(row, "size_m", [radius, radius, radius])
        else:
            size = _vector(obstacle["size_m"], 3, "box full size")
            if np.any(size <= 0):
                raise ValueError("invalid box dimensions")
            _set(row, "size_m", size)
            _set(row, "rotation_B0", inverse @ _matrix(obstacle["rotation_world"], "box rotation"))
        _set(row, "linear_velocity_B0_m_s", inverse @ _vector(obstacle.get("linear_velocity_w_m_s", [0., 0., 0.]), 3, "obstacle velocity"))
        _set(row, "angular_velocity_B0_rad_s", inverse @ _vector(obstacle.get("angular_velocity_w_rad_s", [0., 0., 0.]), 3, "obstacle angular velocity"))
        _set(row, "primitive_one_hot_sphere_box_keypoint", [kind == "sphere", kind == "box", False])
        token_type = TOKEN_TYPES[kind + "_obstacle"]
        obstacle_rows.append((token_type, row))
    obstacle_rows.sort(key=lambda entry: (entry[0], *entry[1].tolist()))
    for index, (token_type, row) in enumerate(obstacle_rows):
        slot = MAX_REQUIREMENTS + index
        features[slot], types[slot], mask[slot] = row, token_type, True

    target_slot = MAX_REQUIREMENTS + MAX_OBSTACLES
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision")
    if target_geom < 0 or int(model.geom_type[target_geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("frozen operative target box is missing")
    row = features[target_slot]
    _set(row, "position_B0_m", position(data.geom_xpos[target_geom]))
    _set(row, "rotation_B0", inverse @ data.geom_xmat[target_geom].reshape(3, 3))
    _set(row, "size_m", 2.*model.geom_size[target_geom])  # Native box stores half extents.
    target_twist = _vector(task["target_twist"], 6, "target twist")
    geom_offset = data.geom_xpos[target_geom] - target[:3]
    _set(row, "linear_velocity_B0_m_s", inverse @ (target_twist[:3] + np.cross(target_twist[3:], geom_offset)))
    _set(row, "angular_velocity_B0_rad_s", inverse @ target_twist[3:])
    _set(row, "primitive_one_hot_sphere_box_keypoint", [0., 1., 0.])
    _set(row, "operative_target_flag", [1.])
    types[target_slot], mask[target_slot] = TOKEN_TYPES["operative_target"], True

    keypoints = [("continuum", i//6+1, i+1, f"link_{i+1}", np.zeros(3), False)
                 for i in range(30)]
    keypoints.append(("continuum", 5, 31, spec.continuum_tip_body_name,
                      CONTINUUM_EE_OFFSET_M, True))
    for i, name in enumerate(spec.low_level_joint_names[60:]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid]))
        keypoints.append(("rigid", i+1, i+1, body_name, np.zeros(3), False))
    keypoints.append(("rigid", 8, 8, spec.rigid_tip_body_name, np.zeros(3), True))
    for index, (arm, segment, module, body_name, local_offset, is_tip) in enumerate(keypoints):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body < 0:
            raise ValueError("initial keypoint body is missing: " + str(body_name))
        slot = target_slot + 1 + index
        row = features[slot]
        rotation = data.xmat[body].reshape(3, 3)
        offset = rotation @ np.asarray(local_offset)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        _set(row, "position_B0_m", position(data.xpos[body] + offset))
        _set(row, "rotation_B0", inverse @ rotation)
        _set(row, "linear_velocity_B0_m_s", inverse @ (velocity[3:] + np.cross(velocity[:3], offset)))
        _set(row, "angular_velocity_B0_rad_s", inverse @ velocity[:3])
        _set(row, "arm_one_hot_rigid_continuum", [arm == "rigid", arm == "continuum"])
        _set(row, "primitive_one_hot_sphere_box_keypoint", [0., 0., 1.])
        _set(row, "segment_id", [segment])
        _set(row, "module_or_joint_id", [module])
        _set(row, "end_effector_flag", [is_tip])
        types[slot], mask[slot] = TOKEN_TYPES[arm + "_keypoint"], True
    return {"schema": CONDITION_SCHEMA, "global": global_state,
            "token_features": features, "token_types": types, "token_mask": mask}


def _validated_arrays(encoded):
    global_state = np.asarray(encoded["global"], dtype=np.float64)
    features = np.asarray(encoded["token_features"], dtype=np.float64)
    types = np.asarray(encoded["token_types"])
    mask = np.asarray(encoded["token_mask"])
    if (global_state.shape != (GLOBAL_DIM,) or features.shape != (TOKEN_COUNT, TOKEN_DIM)
            or types.shape != (TOKEN_COUNT,) or mask.shape != (TOKEN_COUNT,)):
        raise ValueError("typed condition arrays differ from the frozen schema")
    if mask.dtype != np.dtype(bool) or types.dtype.kind not in "iu":
        raise ValueError("typed mask must be boolean and types must be integer")
    if (not np.all(np.isfinite(global_state)) or not np.all(np.isfinite(features[mask]))
            or np.any(types[mask] <= 0) or np.any(types[mask] >= TYPE_COUNT)):
        raise ValueError("invalid active typed condition data")
    # Invalid rows can contain arbitrary caller padding.  They never reach M0.
    features, types = features.copy(), types.astype(np.int64, copy=True)
    features[~mask], types[~mask] = 0., 0
    return global_state, features, types, mask


def flatten_typed_condition(encoded) -> np.ndarray:
    """Canonical M0 vector containing exactly M1's global/features/types/masks.

    Sorting uses physical feature values and explicit type/identity fields, not
    token positions or obstacle names.  It remains invariant if condition
    arrays are jointly permuted, including after shared normalization.
    """
    global_state, features, types, mask = _validated_arrays(encoded)
    active = [i for i in range(TOKEN_COUNT) if mask[i]]
    active.sort(key=lambda i: (int(types[i]), *features[i].tolist()))
    order = active + [i for i in range(TOKEN_COUNT) if not mask[i]]
    rows = np.c_[features[order], types[order].astype(np.float64), mask[order].astype(np.float64)]
    result = np.r_[global_state, rows.ravel()]
    if result.shape != (FLAT_DIM,):
        raise AssertionError("typed flattened dimension mismatch")
    return result
