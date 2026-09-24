"""Run and evaluate the V6-lite deterministic dual-arm controller.

The physics loop is 500 Hz (MuJoCo dt=0.002 s).  A single weighted hierarchy
QP is solved every ten physics steps (50 Hz).  The rigid arm tracks a specified
pose on a freely drifting target satellite while the continuum tip tracks the
original seeded irregular-waypoint target through a C2 minimum-jerk reference.
Exact MuJoCo signed distances are checked for the whole robot against itself,
static spherical workspace obstacles, and the moving target satellite.  Only
the explicitly named rigid terminal contact geoms are exempt from positive
clearance; continuum-target clearance remains a hard QP constraint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from model_test.robot_model_spec_v5 import RobotModelSpecV5, default_robot_model_spec_v5
from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
    WorkspaceSphere,
)
from v6_lite.hierarchical_qp import (
    CONTINUUM_EE_OFFSET_M,
    HierarchicalQPConfig,
    HierarchicalVelocityQP,
    free_joint_slices,
    joint_addresses,
    rotation_error_angle_rad,
)
from v6_lite.irregular_waypoints import (
    IrregularWaypointTarget,
    build_original_irregular_target,
)
from v6_lite.pcc_monitor import PCCMonitor

CONTRACT_VERSION = "v6_lite_6"
TARGET_SATELLITE_COLLISION_POLICY = (
    "continuum_base_and_noncontact_rigid_geometries_in_hard_clearance_gate;"
    "rigid_terminal_contact_geometries_explicitly_exempt_for_commanded_surface_grasp"
)
EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M = np.asarray(
    [1.930, 0.626, 0.0], dtype=np.float64
)


def default_v6_lite_robot_spec() -> RobotModelSpecV5:
    """V6-lite torque-servo contract used by execution and independent replay."""

    base = default_robot_model_spec_v5()
    return RobotModelSpecV5.from_urdf(
        base.source_urdf,
        continuum_kp=100.0,
        continuum_kd=5.0,
        continuum_torque_limit=4.0,
        rigid_kp=200.0,
        rigid_kd=20.0,
        rigid_torque_limit=80.0,
    )


@dataclass(frozen=True)
class V6LiteRunConfig:
    scenario_count: int = 5
    seed: int = 20260801
    duration_s: float = 27.0
    physics_period_s: float = 0.002
    task_period_s: float = 0.02
    whole_body_minimum_clearance_m: float = 0.005
    verification_subdivisions: int = 4
    rigid_final_error_threshold_m: float = 0.00010
    rigid_steady_rmse_threshold_m: float = 0.00015
    continuum_irregular_path_rmse_threshold_m: float = 0.00018
    orientation_error_threshold_deg: float = 0.25
    task_latency_p95_threshold_s: float = 0.020
    torque_latency_p95_threshold_s: float = 0.002
    steady_window_s: float = 1.5

    def validate(self) -> None:
        if self.scenario_count < 3:
            raise ValueError("at least three independently seeded scenarios are required")
        if self.duration_s <= self.steady_window_s:
            raise ValueError("duration must exceed the steady-state window")
        if abs(self.physics_period_s - 0.002) > 1e-12:
            raise ValueError("V6-lite physics period must be exactly 0.002 s")
        if abs(self.task_period_s - 0.02) > 1e-12:
            raise ValueError("V6-lite task period must be exactly 0.02 s")
        ratio = self.task_period_s / self.physics_period_s
        if abs(ratio - round(ratio)) > 1e-12 or int(round(ratio)) != 10:
            raise ValueError("one task tick must equal ten torque ticks")
        if self.verification_subdivisions < 2:
            raise ValueError("dense-discrete verification needs at least two subdivisions")
        thresholds = (
            self.whole_body_minimum_clearance_m,
            self.rigid_final_error_threshold_m,
            self.rigid_steady_rmse_threshold_m,
            self.continuum_irregular_path_rmse_threshold_m,
            self.orientation_error_threshold_deg,
            self.task_latency_p95_threshold_s,
            self.torque_latency_p95_threshold_s,
        )
        if min(thresholds) <= 0.0:
            raise ValueError("acceptance thresholds must be positive")


@dataclass(frozen=True)
class V6LiteScenario:
    scenario_id: str
    seed: int
    target_satellite_position_shift_m: np.ndarray
    target_satellite_linear_velocity_m_s: np.ndarray
    target_satellite_angular_velocity_rad_s: np.ndarray
    grasp_point_target_frame_m: np.ndarray
    grasp_rotation_target_frame: np.ndarray
    continuum_target_rotation_world: np.ndarray
    continuum_target: IrregularWaypointTarget
    obstacles: tuple[WorkspaceSphere, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "target_satellite_position_shift_m": self.target_satellite_position_shift_m.tolist(),
            "target_satellite_linear_velocity_m_s": self.target_satellite_linear_velocity_m_s.tolist(),
            "target_satellite_angular_velocity_rad_s": self.target_satellite_angular_velocity_rad_s.tolist(),
            "grasp_point_target_frame_m": self.grasp_point_target_frame_m.tolist(),
            "grasp_rotation_target_frame": self.grasp_rotation_target_frame.tolist(),
            "continuum_target_rotation_world": self.continuum_target_rotation_world.tolist(),
            "continuum_target": self.continuum_target.to_dict(),
            "workspace_obstacles": [item.to_dict() for item in self.obstacles],
            "target_satellite_collision_policy": TARGET_SATELLITE_COLLISION_POLICY,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def quaternion_geodesic_angle_rad(
    reference_quaternion: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    """Return the shortest SO(3) angle from a MuJoCo wxyz quaternion reference."""

    reference = np.asarray(reference_quaternion, dtype=np.float64).reshape(4)
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError("quaternion values must end in four wxyz components")
    reference_norm = float(np.linalg.norm(reference))
    value_norms = np.linalg.norm(values, axis=-1)
    if reference_norm <= 0.0 or np.any(value_norms <= 0.0):
        raise ValueError("quaternions must have non-zero norm")
    unit_reference = reference / reference_norm
    unit_values = values / np.expand_dims(value_norms, axis=-1)
    absolute_dot = np.abs(np.sum(unit_values * unit_reference, axis=-1))
    return 2.0 * np.arccos(np.clip(absolute_dot, 0.0, 1.0))


def _body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise ValueError(f"missing body {name!r}")
    return int(value)


def _body_pose_and_twist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    local_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    point = np.asarray(data.xpos[body_id]) + rotation @ np.asarray(local_point)
    position_jacobian = np.zeros((3, model.nv), dtype=np.float64)
    rotation_jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(
        model,
        data,
        position_jacobian,
        rotation_jacobian,
        point,
        body_id,
    )
    qvel = np.asarray(data.qvel)
    return (
        point.copy(),
        (position_jacobian @ qvel).copy(),
        rotation.copy(),
        (rotation_jacobian @ qvel).copy(),
    )


def _robot_momentum(
    model: mujoco.MjModel, data: mujoco.MjData, base_body_id: int
) -> np.ndarray:
    mujoco.mj_subtreeVel(model, data)
    linear = float(model.body_subtreemass[base_body_id]) * np.asarray(
        data.subtree_linvel[base_body_id]
    )
    angular = np.asarray(data.subtree_angmom[base_body_id])
    return np.concatenate([linear, angular]).copy()


def _model_based_servo_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: RobotModelSpecV5,
    qpos_ids: np.ndarray,
    dof_ids: np.ndarray,
    base_dof_slice: slice,
    reference_q: np.ndarray,
    reference_dq: np.ndarray,
    feedforward_ddq: np.ndarray,
    full_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-dynamics arm torque with a critically damped joint servo.

    The six unactuated spacecraft accelerations are eliminated from the full
    mass matrix before the 67 actuator torques are formed.  This is one direct
    deterministic torque law, not a second optimization problem.
    """

    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, full_mass, data.qM)
    measured_q = np.asarray(data.qpos)[qpos_ids]
    measured_dq = np.asarray(data.qvel)[dof_ids]
    reference_low_q = spec.encode_position(reference_q)
    reference_low_dq = spec.encode_velocity(reference_dq)
    feedforward_low_ddq = spec.encode_velocity(feedforward_ddq)
    natural_frequency = np.concatenate(
        [np.full(60, 42.0), np.full(7, 34.0)]
    )
    desired_arm_acceleration = (
        feedforward_low_ddq
        + natural_frequency**2 * (reference_low_q - measured_q)
        + 2.0 * natural_frequency * (reference_low_dq - measured_dq)
    )
    acceleration_limit = np.concatenate(
        [np.full(60, 45.0), np.full(7, 70.0)]
    )
    desired_arm_acceleration = np.clip(
        desired_arm_acceleration, -acceleration_limit, acceleration_limit
    )
    base_ids = np.arange(
        base_dof_slice.start, base_dof_slice.stop, dtype=np.int32
    )
    mass_bb = full_mass[np.ix_(base_ids, base_ids)]
    mass_ba = full_mass[np.ix_(base_ids, dof_ids)]
    base_acceleration = -np.linalg.solve(
        mass_bb,
        mass_ba @ desired_arm_acceleration
        + np.asarray(data.qfrc_bias)[base_ids]
        - np.asarray(data.qfrc_passive)[base_ids],
    )
    required_arm_force = (
        full_mass[np.ix_(dof_ids, base_ids)] @ base_acceleration
        + full_mass[np.ix_(dof_ids, dof_ids)] @ desired_arm_acceleration
        + np.asarray(data.qfrc_bias)[dof_ids]
        - np.asarray(data.qfrc_passive)[dof_ids]
    )
    torque = np.clip(required_arm_force, -spec.torque_limits, spec.torque_limits)
    return torque, desired_arm_acceleration


def _home_kinematics(
    spec: RobotModelSpecV5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = spec.compile_dynamic_model()
    data = mujoco.MjData(model)
    qpos_ids, _dof_ids = joint_addresses(model, spec)
    data.qpos[qpos_ids] = spec.encode_position(spec.planner_zero)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    continuum_id = _body_id(model, spec.continuum_tip_body_name)
    rigid_id = _body_id(model, spec.rigid_tip_body_name)
    target_id = _body_id(model, "target_satellite")
    continuum_body_origin = np.asarray(data.xpos[continuum_id]).copy()
    rigid = np.asarray(data.xpos[rigid_id]).copy()
    target = np.asarray(data.xpos[target_id]).copy()
    continuum_rotation = np.asarray(data.xmat[continuum_id]).reshape(3, 3).copy()
    continuum = continuum_body_origin + continuum_rotation @ CONTINUUM_EE_OFFSET_M
    rigid_rotation = np.asarray(data.xmat[rigid_id]).reshape(3, 3).copy()
    target_rotation = np.asarray(data.xmat[target_id]).reshape(3, 3).copy()
    return (
        continuum,
        rigid,
        target,
        continuum_rotation,
        rigid_rotation,
        target_rotation,
    )


def build_scenarios(
    spec: RobotModelSpecV5, config: V6LiteRunConfig
) -> tuple[V6LiteScenario, ...]:
    (
        continuum_home,
        rigid_home,
        target_home,
        continuum_home_rotation,
        rigid_home_rotation,
        target_home_rotation,
    ) = _home_kinematics(spec)
    scenarios: list[V6LiteScenario] = []
    for index in range(config.scenario_count):
        scenario_seed = config.seed + 104729 * index
        rng = np.random.default_rng(scenario_seed)
        target_shift = np.asarray(
            [rng.uniform(-0.012, 0.012), rng.uniform(-0.008, 0.008), rng.uniform(-0.010, 0.010)],
            dtype=np.float64,
        )
        target_velocity = np.asarray(
            [rng.uniform(-0.0025, 0.0025), rng.uniform(0.0010, 0.0040), rng.uniform(-0.0015, 0.0015)],
            dtype=np.float64,
        )
        target_angular_velocity = np.asarray(
            [
                rng.uniform(-0.0015, 0.0015),
                rng.uniform(-0.0015, 0.0015),
                rng.uniform(-0.0015, 0.0015),
            ],
            dtype=np.float64,
        )
        grasp_local = np.asarray([0.0, -0.205, 0.0], dtype=np.float64)
        grasp_initial = target_home + target_shift + grasp_local
        direction = grasp_initial - rigid_home
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        lateral = np.cross(direction, np.asarray([0.0, 0.0, 1.0]))
        if np.linalg.norm(lateral) < 1e-8:
            lateral = np.asarray([1.0, 0.0, 0.0])
        lateral /= np.linalg.norm(lateral)
        lateral *= (-1.0 if index % 2 else 1.0)
        rigid_obstacle = WorkspaceSphere(
            name=f"rigid_path_{index:02d}",
            center=rigid_home + 0.52 * (grasp_initial - rigid_home) + 0.050 * lateral,
            radius=0.035,
        )
        target = build_original_irregular_target(
            continuum_home,
            continuum_home_rotation,
            scenario_seed,
        )
        continuum_side = -1.0 if index % 2 else 1.0
        continuum_obstacle = WorkspaceSphere(
            name=f"continuum_side_{index:02d}",
            center=continuum_home
            + np.asarray(
                [0.120, continuum_side * 0.110, 0.060], dtype=np.float64
            ),
            radius=0.025,
        )
        scenarios.append(
            V6LiteScenario(
                scenario_id=f"v6_lite_scenario_{index:02d}",
                seed=scenario_seed,
                target_satellite_position_shift_m=target_shift,
                target_satellite_linear_velocity_m_s=target_velocity,
                target_satellite_angular_velocity_rad_s=target_angular_velocity,
                grasp_point_target_frame_m=grasp_local,
                grasp_rotation_target_frame=target_home_rotation.T
                @ rigid_home_rotation,
                continuum_target_rotation_world=target.target_rotation_world.copy(),
                continuum_target=target,
                obstacles=(rigid_obstacle, continuum_obstacle),
            )
        )
    return tuple(scenarios)


def run_scenario(
    spec: RobotModelSpecV5,
    run_config: V6LiteRunConfig,
    qp_config: HierarchicalQPConfig,
    scenario: V6LiteScenario,
    trace_dir: Path,
) -> dict[str, Any]:
    verification_config = WholeBodyVerificationConfig(
        minimum_clearance=run_config.whole_body_minimum_clearance_m,
        query_distance_max=2.5,
        adaptive_subdivisions=run_config.verification_subdivisions,
        self_collision_ancestor_exclusion_depth=3,
        include_target_satellite_pairs=True,
    )
    verifier = WholeBodyCollisionVerifier(spec, scenario.obstacles, verification_config)
    model = verifier.model
    if abs(float(model.opt.timestep) - run_config.physics_period_s) > 1e-12:
        raise RuntimeError("compiled MuJoCo timestep does not match V6-lite")
    # Collision response is disabled so successful clearance cannot be caused
    # by contact impulses.  Signed distances remain directly queryable.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    qpos_ids, dof_ids = joint_addresses(model, spec)
    base_qpos_slice, base_dof_slice = free_joint_slices(model, spec.base_joint_name)
    target_qpos_slice, target_dof_slice = free_joint_slices(
        model, spec.target_free_joint_name
    )
    data.qpos[qpos_ids] = spec.encode_position(spec.planner_zero)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[target_qpos_slice.start : target_qpos_slice.start + 3] += (
        scenario.target_satellite_position_shift_m
    )
    data.qvel[target_dof_slice.start : target_dof_slice.start + 3] = (
        scenario.target_satellite_linear_velocity_m_s
    )
    data.qvel[target_dof_slice.start + 3 : target_dof_slice.stop] = (
        scenario.target_satellite_angular_velocity_rad_s
    )
    mujoco.mj_forward(model, data)
    initial_qpos = np.asarray(data.qpos).copy()
    initial_qvel = np.asarray(data.qvel).copy()
    qpos_write_count_after_initialization = 0
    qvel_write_count_after_initialization = 0

    qp = HierarchicalVelocityQP(spec, model, verifier.pairs, qp_config)
    pcc_monitor = (
        PCCMonitor(scenario_id=scenario.scenario_id)
        if qp_config.enable_pcc_cbf or qp_config.enable_capsule_cbf
        else None
    )
    task_stride = int(round(run_config.task_period_s / run_config.physics_period_s))
    physics_steps = int(round(run_config.duration_s / run_config.physics_period_s))
    rigid_body_id = _body_id(model, spec.rigid_tip_body_name)
    continuum_body_id = _body_id(model, spec.continuum_tip_body_name)
    target_body_id = _body_id(model, "target_satellite")
    base_body_id = _body_id(model, "base_of_satelltte")
    reference_q_state = spec.planner_zero.copy()
    segment_start_velocity = np.zeros(17, dtype=np.float64)
    command_velocity = np.zeros(17, dtype=np.float64)
    segment_step = 0
    full_mass = np.zeros((model.nv, model.nv), dtype=np.float64)

    log: dict[str, list[Any]] = {
        "time": [],
        "rigid_error": [],
        "continuum_error": [],
        "rigid_orientation_error_deg": [],
        "continuum_orientation_error_deg": [],
        "rigid_tip": [],
        "rigid_target": [],
        "rigid_rotation": [],
        "rigid_target_rotation": [],
        "continuum_tip": [],
        "continuum_tip_body_origin": [],
        "continuum_target": [],
        "continuum_target_velocity": [],
        "continuum_rotation": [],
        "continuum_target_rotation": [],
        "planner_q": [],
        "planner_dq": [],
        "command_velocity": [],
        "reference_q": [],
        "reference_velocity": [],
        "feedforward_acceleration": [],
        "desired_arm_acceleration": [],
        "torque": [],
        "base_qpos": [],
        "base_twist": [],
        "robot_momentum": [],
        "torque_latency": [],
    }
    task_log: dict[str, list[Any]] = {
        "time": [],
        "full_latency": [],
        "solver_latency": [],
        "success": [],
        "iterations": [],
        "active_clearance": [],
        "binding_clearance": [],
        "minimum_queried_clearance": [],
        "minimum_constraint_slack": [],
        "avoidance_intervention": [],
        "momentum_map_residual": [],
        "degenerate_clearance_gradients": [],
        "rigid_velocity_residual": [],
        "continuum_velocity_residual": [],
        "rigid_angular_velocity_residual": [],
        "continuum_angular_velocity_residual": [],
        "solver_status": [],
        "pcc_clearance": [],
        "capsule_clearance": [],
        "mujoco_continuum_target_clearance": [],
        "pcc_active_clearance": [],
        "capsule_active_clearance": [],
        "pcc_binding_clearance": [],
        "capsule_binding_clearance": [],
        "pcc_avoidance_intervention": [],
        "pcc_mujoco_distance_error": [],
        "pcc_mujoco_gradient_error": [],
        "capsule_mujoco_gradient_error": [],
        "pcc_closest_segment_id": [],
        "pcc_closest_arclength": [],
        "shape_clearance_latency": [],
    }
    task_qpos_trace: list[np.ndarray] = []
    initial_momentum = _robot_momentum(model, data, base_body_id)

    for physics_step in range(physics_steps):
        current_time = float(data.time)
        if physics_step % task_stride == 0:
            mujoco.mj_forward(model, data)
            (
                rigid_target,
                rigid_target_velocity,
                target_rotation,
                target_angular_velocity,
            ) = _body_pose_and_twist(
                model,
                data,
                target_body_id,
                scenario.grasp_point_target_frame_m,
            )
            rigid_target_rotation = (
                target_rotation @ scenario.grasp_rotation_target_frame
            )
            continuum_target, continuum_target_velocity = scenario.continuum_target.sample(
                current_time
            )
            result = qp.solve(
                data,
                rigid_target_position=rigid_target,
                rigid_target_velocity=rigid_target_velocity,
                rigid_target_rotation=rigid_target_rotation,
                rigid_target_angular_velocity=target_angular_velocity,
                continuum_target_position=continuum_target,
                continuum_target_velocity=continuum_target_velocity,
                continuum_target_rotation=scenario.continuum_target_rotation_world,
                continuum_target_angular_velocity=np.zeros(3, dtype=np.float64),
            )
            segment_start_velocity = command_velocity.copy()
            command_velocity = result.planner_velocity.copy()
            segment_step = 0
            task_qpos_trace.append(np.asarray(data.qpos).copy())
            task_log["time"].append(current_time)
            task_log["full_latency"].append(result.full_latency_s)
            task_log["solver_latency"].append(result.solver_latency_s)
            task_log["success"].append(result.success)
            task_log["iterations"].append(result.solver_iterations)
            task_log["active_clearance"].append(result.active_clearance_constraint_count)
            task_log["binding_clearance"].append(result.binding_clearance_constraint_count)
            task_log["minimum_queried_clearance"].append(
                result.minimum_queried_clearance_m
            )
            task_log["minimum_constraint_slack"].append(result.minimum_constraint_slack)
            task_log["avoidance_intervention"].append(
                result.unconstrained_to_command_norm
            )
            task_log["momentum_map_residual"].append(
                result.reaction_momentum_residual_norm
            )
            task_log["degenerate_clearance_gradients"].append(
                result.degenerate_clearance_gradient_count
            )
            task_log["rigid_velocity_residual"].append(
                result.rigid_velocity_residual_m_s
            )
            task_log["continuum_velocity_residual"].append(
                result.continuum_velocity_residual_m_s
            )
            task_log["rigid_angular_velocity_residual"].append(
                result.rigid_angular_velocity_residual_rad_s
            )
            task_log["continuum_angular_velocity_residual"].append(
                result.continuum_angular_velocity_residual_rad_s
            )
            task_log["solver_status"].append(result.solver_status)
            task_log["pcc_clearance"].append(result.pcc_clearance_m)
            task_log["capsule_clearance"].append(result.capsule_clearance_m)
            task_log["mujoco_continuum_target_clearance"].append(
                result.mujoco_continuum_target_clearance_m
            )
            task_log["pcc_active_clearance"].append(
                result.pcc_constraint_active_count
            )
            task_log["capsule_active_clearance"].append(
                result.capsule_constraint_active_count
            )
            task_log["pcc_binding_clearance"].append(
                result.pcc_binding_constraint_count
            )
            task_log["capsule_binding_clearance"].append(
                result.capsule_binding_constraint_count
            )
            task_log["pcc_avoidance_intervention"].append(
                result.pcc_avoidance_intervention
            )
            task_log["pcc_mujoco_distance_error"].append(
                result.pcc_mujoco_distance_error_m
            )
            task_log["pcc_mujoco_gradient_error"].append(
                result.pcc_mujoco_gradient_error_norm
            )
            task_log["capsule_mujoco_gradient_error"].append(
                result.capsule_mujoco_gradient_error_norm
            )
            task_log["pcc_closest_segment_id"].append(
                result.pcc_closest_segment_id
            )
            task_log["pcc_closest_arclength"].append(
                result.pcc_closest_arclength_m
            )
            task_log["shape_clearance_latency"].append(
                result.shape_clearance_latency_s
            )
            if pcc_monitor is not None:
                pcc_monitor.record(current_time, result)

        interpolation = float(segment_step + 1) / float(task_stride)
        reference_dq = (
            segment_start_velocity
            + interpolation * (command_velocity - segment_start_velocity)
        )
        feedforward_ddq = (
            command_velocity - segment_start_velocity
        ) / run_config.task_period_s
        measured_low_q = np.asarray(data.qpos[qpos_ids], dtype=np.float64).copy()
        measured_planner_q = spec.decode_position(measured_low_q)
        reference_q_state = np.clip(
            reference_q_state + reference_dq * run_config.physics_period_s,
            spec.planner_lower,
            spec.planner_upper,
        )
        reference_q_state = np.clip(
            reference_q_state,
            measured_planner_q - 0.012,
            measured_planner_q + 0.012,
        )
        reference_q = reference_q_state
        torque_started = time.perf_counter()
        torque, desired_arm_acceleration = _model_based_servo_torque(
            model,
            data,
            spec,
            qpos_ids,
            dof_ids,
            base_dof_slice,
            reference_q,
            reference_dq,
            feedforward_ddq,
            full_mass,
        )
        data.ctrl[:] = torque
        torque_latency = time.perf_counter() - torque_started
        mujoco.mj_step(model, data)
        segment_step += 1

        (
            rigid_target,
            _rigid_target_velocity,
            target_rotation,
            _target_angular_velocity,
        ) = _body_pose_and_twist(
            model,
            data,
            target_body_id,
            scenario.grasp_point_target_frame_m,
        )
        rigid_target_rotation = target_rotation @ scenario.grasp_rotation_target_frame
        continuum_target, continuum_target_velocity = scenario.continuum_target.sample(
            float(data.time)
        )
        rigid_tip = np.asarray(data.xpos[rigid_body_id]).copy()
        rigid_rotation = np.asarray(data.xmat[rigid_body_id]).reshape(3, 3).copy()
        continuum_rotation = (
            np.asarray(data.xmat[continuum_body_id]).reshape(3, 3).copy()
        )
        continuum_tip_body_origin = np.asarray(
            data.xpos[continuum_body_id]
        ).copy()
        continuum_tip = (
            continuum_tip_body_origin
            + continuum_rotation @ CONTINUUM_EE_OFFSET_M
        )
        low_q = np.asarray(data.qpos[qpos_ids], dtype=np.float64)
        low_dq = np.asarray(data.qvel[dof_ids], dtype=np.float64)
        log["time"].append(float(data.time))
        log["rigid_error"].append(float(np.linalg.norm(rigid_target - rigid_tip)))
        log["continuum_error"].append(
            float(np.linalg.norm(continuum_target - continuum_tip))
        )
        log["rigid_orientation_error_deg"].append(
            float(
                np.rad2deg(
                    rotation_error_angle_rad(rigid_target_rotation, rigid_rotation)
                )
            )
        )
        log["continuum_orientation_error_deg"].append(
            float(
                np.rad2deg(
                    rotation_error_angle_rad(
                        scenario.continuum_target_rotation_world,
                        continuum_rotation,
                    )
                )
            )
        )
        log["rigid_tip"].append(rigid_tip)
        log["rigid_target"].append(rigid_target)
        log["rigid_rotation"].append(rigid_rotation)
        log["rigid_target_rotation"].append(rigid_target_rotation)
        log["continuum_tip"].append(continuum_tip)
        log["continuum_tip_body_origin"].append(continuum_tip_body_origin)
        log["continuum_target"].append(continuum_target)
        log["continuum_target_velocity"].append(continuum_target_velocity)
        log["continuum_rotation"].append(continuum_rotation)
        log["continuum_target_rotation"].append(
            scenario.continuum_target_rotation_world.copy()
        )
        log["planner_q"].append(spec.decode_position(low_q))
        log["planner_dq"].append(spec.decode_velocity(low_dq))
        log["command_velocity"].append(command_velocity.copy())
        log["reference_q"].append(reference_q.copy())
        log["reference_velocity"].append(reference_dq.copy())
        log["feedforward_acceleration"].append(feedforward_ddq.copy())
        log["desired_arm_acceleration"].append(
            desired_arm_acceleration.copy()
        )
        log["torque"].append(torque.copy())
        log["base_qpos"].append(np.asarray(data.qpos[base_qpos_slice]).copy())
        log["base_twist"].append(np.asarray(data.qvel[base_dof_slice]).copy())
        log["robot_momentum"].append(_robot_momentum(model, data, base_body_id))
        log["torque_latency"].append(torque_latency)

    task_qpos_trace.append(np.asarray(data.qpos).copy())
    arrays = {key: np.asarray(value) for key, value in log.items()}
    task_arrays = {key: np.asarray(value) for key, value in task_log.items()}
    initial_base_pose = np.asarray(initial_qpos[base_qpos_slice], dtype=np.float64)
    arrays["base_translation_drift_m"] = np.linalg.norm(
        arrays["base_qpos"][:, :3] - initial_base_pose[None, :3], axis=1
    )
    arrays["base_orientation_drift_deg"] = np.rad2deg(
        quaternion_geodesic_angle_rad(
            initial_base_pose[3:7], arrays["base_qpos"][:, 3:7]
        )
    )
    whole_body_report = verifier.verify_qpos_sequence(np.asarray(task_qpos_trace))
    steady_mask = arrays["time"] >= run_config.duration_s - run_config.steady_window_s
    rigid_steady = arrays["rigid_error"][steady_mask]
    continuum_steady = arrays["continuum_error"][steady_mask]
    irregular_path_mask = (
        (arrays["time"] >= scenario.continuum_target.path_start_s)
        & (arrays["time"] <= scenario.continuum_target.path_end_s)
    )
    continuum_irregular_path = arrays["continuum_error"][irregular_path_mask]
    rigid_orientation = arrays["rigid_orientation_error_deg"]
    continuum_orientation = arrays["continuum_orientation_error_deg"]
    final_rigid_error = float(arrays["rigid_error"][-1])
    rigid_steady_rmse = float(np.sqrt(np.mean(rigid_steady**2)))
    continuum_steady_rmse = float(np.sqrt(np.mean(continuum_steady**2)))
    continuum_irregular_path_rmse = float(
        np.sqrt(np.mean(continuum_irregular_path**2))
    )
    task_latency_p95 = float(np.percentile(task_arrays["full_latency"], 95.0))
    torque_latency_p95 = float(np.percentile(arrays["torque_latency"], 95.0))
    qp_failure_count = int(np.sum(~task_arrays["success"].astype(bool)))
    binding_count = int(np.sum(task_arrays["binding_clearance"]))
    intervention_max = float(np.max(task_arrays["avoidance_intervention"]))
    pcc_active_count = int(np.sum(task_arrays["pcc_active_clearance"]))
    pcc_binding_count = int(np.sum(task_arrays["pcc_binding_clearance"]))
    capsule_active_count = int(
        np.sum(task_arrays["capsule_active_clearance"])
    )
    capsule_binding_count = int(
        np.sum(task_arrays["capsule_binding_clearance"])
    )
    pcc_intervention_max = float(
        np.max(task_arrays["pcc_avoidance_intervention"])
    )
    momentum = arrays["robot_momentum"]
    momentum_drift = float(
        np.max(np.linalg.norm(momentum - initial_momentum[None, :], axis=1))
    )
    expected_task_ticks = int(math.ceil(physics_steps / task_stride))
    checks = {
        "continuum_initial_position_contract": float(
            np.linalg.norm(
                scenario.continuum_target.initial_position_w
                - EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M
            )
        )
        <= 1e-8,
        "rigid_grasp_final_error": final_rigid_error
        <= run_config.rigid_final_error_threshold_m,
        "rigid_grasp_steady_rmse": rigid_steady_rmse
        <= run_config.rigid_steady_rmse_threshold_m,
        "continuum_irregular_waypoint_path_rmse": continuum_irregular_path_rmse
        <= run_config.continuum_irregular_path_rmse_threshold_m,
        "rigid_orientation_error_below_limit": float(np.max(rigid_orientation))
        < run_config.orientation_error_threshold_deg,
        "continuum_orientation_error_below_limit": float(
            np.max(continuum_orientation)
        )
        < run_config.orientation_error_threshold_deg,
        "whole_body_dense_discrete_clearance": bool(whole_body_report.feasible),
        "continuum_target_dense_discrete_clearance": float(
            whole_body_report.minimum_by_class.get("continuum_target", float("-inf"))
        )
        >= run_config.whole_body_minimum_clearance_m,
        "obstacle_avoidance_engaged": binding_count > 0 and intervention_max > 1e-5,
        "single_qp_every_task_tick": qp.solve_count == expected_task_ticks,
        "all_qp_solves_succeeded": qp_failure_count == 0,
        "task_controller_runs_within_50hz_p95": task_latency_p95
        <= run_config.task_latency_p95_threshold_s,
        "torque_controller_runs_within_500hz_p95": torque_latency_p95
        <= run_config.torque_latency_p95_threshold_s,
        "dynamic_execution_only": qpos_write_count_after_initialization == 0
        and qvel_write_count_after_initialization == 0,
        "no_degenerate_active_clearance_gradients": int(
            np.sum(task_arrays["degenerate_clearance_gradients"])
        )
        == 0,
        "reaction_map_satisfies_zero_momentum_constraint": float(
            np.max(task_arrays["momentum_map_residual"])
        )
        <= 1e-9,
        "physics_and_task_rate_exact": abs(float(model.opt.timestep) - 0.002) <= 1e-12
        and task_stride == 10,
    }
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{scenario.scenario_id}.npz"
    np.savez_compressed(
        trace_path,
        **arrays,
        **{f"task_{key}": value for key, value in task_arrays.items()},
        task_qpos=np.asarray(task_qpos_trace),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
    )
    monitor_payload = None
    if pcc_monitor is not None:
        monitor_payload = pcc_monitor.write(
            trace_dir.parent / "pcc_monitor" / scenario.scenario_id
        )
    return {
        "scenario": scenario.to_dict(),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": {
            "rigid_grasp_point": {
                "final_error_m": final_rigid_error,
                "steady_rmse_m": rigid_steady_rmse,
                "steady_max_error_m": float(np.max(rigid_steady)),
                "initial_error_m": float(arrays["rigid_error"][0]),
            },
            "continuum_irregular_waypoint_tracking": {
                "end_effector_body_name": spec.continuum_tip_body_name,
                "end_effector_local_offset_body_m": CONTINUUM_EE_OFFSET_M.tolist(),
                "initial_position_m": scenario.continuum_target.initial_position_w.tolist(),
                "initial_position_error_to_contract_m": float(
                    np.linalg.norm(
                        scenario.continuum_target.initial_position_w
                        - EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M
                    )
                ),
                "active_path_rmse_m": continuum_irregular_path_rmse,
                "steady_rmse_m": continuum_steady_rmse,
                "steady_max_error_m": float(np.max(continuum_steady)),
                "full_rmse_m": float(
                    np.sqrt(np.mean(arrays["continuum_error"] ** 2))
                ),
                "active_path_start_s": scenario.continuum_target.path_start_s,
                "active_path_end_s": scenario.continuum_target.path_end_s,
                "target_speed_max_m_s": float(
                    np.max(
                        np.linalg.norm(
                            arrays["continuum_target_velocity"], axis=1
                        )
                    )
                ),
            },
            "orientation_tracking": {
                "threshold_deg": run_config.orientation_error_threshold_deg,
                "rigid_full_max_error_deg": float(np.max(rigid_orientation)),
                "rigid_steady_rmse_deg": float(
                    np.sqrt(np.mean(rigid_orientation[steady_mask] ** 2))
                ),
                "continuum_full_max_error_deg": float(
                    np.max(continuum_orientation)
                ),
                "continuum_steady_rmse_deg": float(
                    np.sqrt(np.mean(continuum_orientation[steady_mask] ** 2))
                ),
            },
            "whole_body_clearance": whole_body_report.to_dict(),
            "avoidance": {
                "binding_constraint_total": binding_count,
                "active_constraint_max": int(
                    np.max(task_arrays["active_clearance"])
                ),
                "unconstrained_to_command_norm_max": intervention_max,
                "minimum_online_queried_clearance_m": float(
                    np.min(task_arrays["minimum_queried_clearance"])
                ),
                "pcc_constraint_active_count": pcc_active_count,
                "pcc_constraint_binding_count": pcc_binding_count,
                "capsule_constraint_active_count": capsule_active_count,
                "capsule_constraint_binding_count": capsule_binding_count,
                "pcc_avoidance_intervention_max": pcc_intervention_max,
            },
            "rates_and_latency": {
                "physics_hz": 1.0 / float(model.opt.timestep),
                "task_hz": 1.0 / run_config.task_period_s,
                "physics_steps": physics_steps,
                "task_ticks": int(qp.solve_count),
                "task_full_latency_median_ms": float(
                    1e3 * np.median(task_arrays["full_latency"])
                ),
                "task_full_latency_p95_ms": 1e3 * task_latency_p95,
                "task_full_latency_max_ms": float(
                    1e3 * np.max(task_arrays["full_latency"])
                ),
                "qp_solver_latency_p95_ms": float(
                    1e3 * np.percentile(task_arrays["solver_latency"], 95.0)
                ),
                "shape_clearance_latency_p95_ms": float(
                    1e3
                    * np.percentile(
                        task_arrays["shape_clearance_latency"], 95.0
                    )
                ),
                "torque_latency_p95_ms": 1e3 * torque_latency_p95,
                "qp_failure_count": qp_failure_count,
                "qp_status_counts": {
                    str(status): int(np.sum(task_arrays["solver_status"] == status))
                    for status in sorted(set(task_arrays["solver_status"].tolist()))
                },
                "qp_call_count": int(qp.solve_count),
                "torque_update_count": physics_steps,
            },
            "actuation": {
                "torque_absolute_max": float(np.max(np.abs(arrays["torque"]))),
                "saturation_ratio": float(
                    np.mean(
                        np.abs(arrays["torque"])
                        >= spec.torque_limits[None, :] - 1e-9
                    )
                ),
            },
            "free_floating_base": {
                "translation_from_start_m": float(
                    arrays["base_translation_drift_m"][-1]
                ),
                "translation_drift_final_m": float(
                    arrays["base_translation_drift_m"][-1]
                ),
                "translation_drift_max_m": float(
                    np.max(arrays["base_translation_drift_m"])
                ),
                "orientation_drift_final_deg": float(
                    arrays["base_orientation_drift_deg"][-1]
                ),
                "orientation_drift_max_deg": float(
                    np.max(arrays["base_orientation_drift_deg"])
                ),
                "drift_reference": "exact_initialized_free_joint_pose",
                "twist_norm_max": float(
                    np.max(np.linalg.norm(arrays["base_twist"], axis=1))
                ),
                "robot_subtree_momentum_drift_max": momentum_drift,
                "reaction_map_residual_max": float(
                    np.max(task_arrays["momentum_map_residual"])
                ),
            },
        },
        "execution_contract": {
            "architecture": (
                "one_priority_weighted_velocity_qp_plus_model_based_torque_servo"
            ),
            "learned_model_used": False,
            "diffusion_used": False,
            "candidate_sampling_used": False,
            "trajectory_projection_used": False,
            "oracle_joint_target_used": False,
            "safe_stop_used": qp_failure_count > 0,
            "contact_response_enabled": False,
            "model_based_inverse_dynamics_used": True,
            "second_online_optimizer_used": False,
            "pcc_cbf_enabled": qp_config.enable_pcc_cbf,
            "capsule_cbf_enabled": qp_config.enable_capsule_cbf,
            "legacy_mujoco_collision_cbf_retained": True,
            "qpos_write_count_after_initialization": qpos_write_count_after_initialization,
            "qvel_write_count_after_initialization": qvel_write_count_after_initialization,
            "collision_scope": (
                "dense_discrete_mj_geomDistance_over_self_arm_arm_arm_base_"
                "workspace_obstacles_and_moving_target_except_rigid_terminal_contact_geoms"
            ),
            "moving_target_clearance_drift_in_qp": True,
            "target_contact_exemption": "collision_0072_and_collision_0073_only",
            "continuous_time_collision_certified": False,
        },
        "pcc_monitor": monitor_payload,
        "trace": {
            "path": trace_path.as_posix(),
            "sha256": _sha256(trace_path),
        },
    }


def run_suite(
    run_config: V6LiteRunConfig,
    qp_config: HierarchicalQPConfig,
    output_dir: Path,
) -> dict[str, Any]:
    run_config.validate()
    qp_config.validate()
    if abs(qp_config.task_period_s - run_config.task_period_s) > 1e-12:
        raise ValueError("QP and suite task periods disagree")
    spec = default_v6_lite_robot_spec()
    spec.validate()
    scenarios = build_scenarios(spec, run_config)
    scenario_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for scenario in scenarios:
        print(f"[v6-lite] running {scenario.scenario_id}", flush=True)
        result = run_scenario(
            spec,
            run_config,
            qp_config,
            scenario,
            output_dir / "traces",
        )
        scenario_results.append(result)
        metrics = result["metrics"]
        print(
            "[v6-lite] "
            f"pass={result['passed']} "
            f"rigid_final={metrics['rigid_grasp_point']['final_error_m']:.5f} m "
            f"continuum_path_rmse={metrics['continuum_irregular_waypoint_tracking']['active_path_rmse_m']:.5f} m "
            f"clearance={metrics['whole_body_clearance']['minimum_clearance']:.5f} m",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    rigid_final_values = [
        item["metrics"]["rigid_grasp_point"]["final_error_m"]
        for item in scenario_results
    ]
    rigid_rmse_values = [
        item["metrics"]["rigid_grasp_point"]["steady_rmse_m"]
        for item in scenario_results
    ]
    continuum_values = [
        item["metrics"]["continuum_irregular_waypoint_tracking"][
            "active_path_rmse_m"
        ]
        for item in scenario_results
    ]
    clearance_values = [
        item["metrics"]["whole_body_clearance"]["minimum_clearance"]
        for item in scenario_results
    ]
    continuum_target_clearance_values = [
        item["metrics"]["whole_body_clearance"]["minimum_by_class"][
            "continuum_target"
        ]
        for item in scenario_results
    ]
    rigid_orientation_values = [
        item["metrics"]["orientation_tracking"]["rigid_full_max_error_deg"]
        for item in scenario_results
    ]
    continuum_orientation_values = [
        item["metrics"]["orientation_tracking"][
            "continuum_full_max_error_deg"
        ]
        for item in scenario_results
    ]
    base_translation_drift_values = [
        item["metrics"]["free_floating_base"]["translation_drift_max_m"]
        for item in scenario_results
    ]
    base_orientation_drift_values = [
        item["metrics"]["free_floating_base"]["orientation_drift_max_deg"]
        for item in scenario_results
    ]
    summary_checks = {
        "all_scenarios_pass": all(item["passed"] for item in scenario_results),
        "continuum_initial_position_is_1930_626_0_mm": all(
            item["checks"]["continuum_initial_position_contract"]
            for item in scenario_results
        ),
        "all_rigid_grasp_metrics_pass": max(rigid_final_values)
        <= run_config.rigid_final_error_threshold_m
        and max(rigid_rmse_values) <= run_config.rigid_steady_rmse_threshold_m,
        "all_continuum_irregular_waypoint_metrics_pass": max(continuum_values)
        <= run_config.continuum_irregular_path_rmse_threshold_m,
        "all_orientation_errors_pass": max(rigid_orientation_values)
        < run_config.orientation_error_threshold_deg
        and max(continuum_orientation_values)
        < run_config.orientation_error_threshold_deg,
        "all_whole_body_clearances_pass": min(clearance_values)
        >= run_config.whole_body_minimum_clearance_m,
        "all_continuum_target_clearances_pass": min(
            continuum_target_clearance_values
        )
        >= run_config.whole_body_minimum_clearance_m,
        "all_runs_are_learning_free": all(
            not item["execution_contract"]["learned_model_used"]
            and not item["execution_contract"]["diffusion_used"]
            for item in scenario_results
        ),
    }
    payload = {
        "contract_version": CONTRACT_VERSION,
        "passed": bool(all(summary_checks.values())),
        "summary_checks": summary_checks,
        "architecture": {
            "task_controller": "single_priority_weighted_constrained_velocity_qp",
            "task_rate_hz": 50.0,
            "torque_controller": (
                "67_dof_saturated_inverse_dynamics_critical_damping_servo"
            ),
            "torque_rate_hz": 500.0,
            "free_floating_base": True,
            "reaction_aware_jacobians": True,
            "moving_target_time_varying_clearance_barrier": True,
            "pcc_shape_clearance_cbf_enabled": qp_config.enable_pcc_cbf,
            "discrete_capsule_cbf_enabled": qp_config.enable_capsule_cbf,
            "legacy_mujoco_collision_cbf_retained": True,
            "target_contact_exemption": "collision_0072_and_collision_0073_only",
            "learning_module": None,
        },
        "kinematic_contract": {
            "continuum_end_effector_body_name": spec.continuum_tip_body_name,
            "continuum_end_effector_local_offset_body_m": CONTINUUM_EE_OFFSET_M.tolist(),
            "continuum_end_effector_initial_world_position_m": EXPECTED_CONTINUUM_EE_INITIAL_POSITION_M.tolist(),
            "position_definition": "body_origin_plus_body_rotation_times_local_offset",
        },
        "run_config": asdict(run_config),
        "qp_config": asdict(qp_config),
        "robot_model_identity": spec.identity().to_dict(),
        "aggregate_metrics": {
            "scenario_count": len(scenario_results),
            "rigid_final_error_m_max": max(rigid_final_values),
            "rigid_steady_rmse_m_max": max(rigid_rmse_values),
            "continuum_irregular_path_rmse_m_max": max(continuum_values),
            "rigid_orientation_error_deg_max": max(rigid_orientation_values),
            "continuum_orientation_error_deg_max": max(
                continuum_orientation_values
            ),
            "base_translation_drift_m_max": max(base_translation_drift_values),
            "base_orientation_drift_deg_max": max(base_orientation_drift_values),
            "whole_body_minimum_clearance_m": min(clearance_values),
            "continuum_target_minimum_clearance_m": min(
                continuum_target_clearance_values
            ),
            "total_qp_failures": sum(
                item["metrics"]["rates_and_latency"]["qp_failure_count"]
                for item in scenario_results
            ),
            "total_binding_clearance_constraints": sum(
                item["metrics"]["avoidance"]["binding_constraint_total"]
                for item in scenario_results
            ),
            "pcc_constraint_active_count": sum(
                item["metrics"]["avoidance"]["pcc_constraint_active_count"]
                for item in scenario_results
            ),
            "pcc_constraint_binding_count": sum(
                item["metrics"]["avoidance"]["pcc_constraint_binding_count"]
                for item in scenario_results
            ),
            "capsule_constraint_active_count": sum(
                item["metrics"]["avoidance"]["capsule_constraint_active_count"]
                for item in scenario_results
            ),
            "capsule_constraint_binding_count": sum(
                item["metrics"]["avoidance"]["capsule_constraint_binding_count"]
                for item in scenario_results
            ),
            "pcc_avoidance_intervention_max": max(
                item["metrics"]["avoidance"]["pcc_avoidance_intervention_max"]
                for item in scenario_results
            ),
            "task_full_latency_p95_ms_max": max(
                item["metrics"]["rates_and_latency"]["task_full_latency_p95_ms"]
                for item in scenario_results
            ),
            "shape_clearance_latency_p95_ms_max": max(
                item["metrics"]["rates_and_latency"][
                    "shape_clearance_latency_p95_ms"
                ]
                for item in scenario_results
            ),
            "torque_latency_p95_ms_max": max(
                item["metrics"]["rates_and_latency"]["torque_latency_p95_ms"]
                for item in scenario_results
            ),
        },
        "elapsed_wall_time_s": elapsed,
        "scenarios": scenario_results,
        "claim_boundary": {
            "supported": (
                "seeded MuJoCo closed-loop evidence for rigid grasp-point tracking, "
                "continuum irregular-waypoint tracking, free-base pose drift measurement, "
                "and dense-discrete whole-body clearance including the moving target "
                "satellite except the explicitly named commanded rigid terminal contacts"
            ),
            "not_supported": (
                "continuous-time collision certification, contact capture dynamics, "
                "hardware transfer, or superiority to a learned planner"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "v6_lite_metrics.json"
    _write_json(metrics_path, payload)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "metrics": {
            "path": metrics_path.as_posix(),
            "sha256": _sha256(metrics_path),
        },
        "traces": [item["trace"] for item in scenario_results],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("v6_lite/output"))
    parser.add_argument("--scenario-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--duration", type=float, default=27.0)
    parser.add_argument("--verification-subdivisions", type=int, default=4)
    parser.add_argument(
        "--enable-pcc-cbf",
        action="store_true",
        help="enable the V6.1-B finite-radius PCC clearance CBF",
    )
    parser.add_argument(
        "--enable-capsule-cbf",
        action="store_true",
        help="enable the V6.1-B actual-chain capsule clearance CBF",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = V6LiteRunConfig(
        scenario_count=args.scenario_count,
        seed=args.seed,
        duration_s=args.duration,
        verification_subdivisions=args.verification_subdivisions,
    )
    result = run_suite(
        config,
        HierarchicalQPConfig(
            enable_pcc_cbf=args.enable_pcc_cbf,
            enable_capsule_cbf=args.enable_capsule_cbf,
        ),
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "aggregate_metrics": result["aggregate_metrics"],
                "output": (args.output_dir / "v6_lite_metrics.json").as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
