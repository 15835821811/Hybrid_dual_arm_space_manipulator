"""One priority-weighted velocity QP for V6-lite.

The online planner deliberately has no learned proposal, sampling loop,
trajectory projection, or oracle target.  Every 20 ms it solves one convex
quadratic program over the audited 17 planner velocities.  Rigid-arm grasp
tracking has higher weight than continuum-tip tracking; joint and signed
distance barriers are hard linear constraints.

The free spacecraft base is not frozen.  The task and clearance Jacobians use
the instantaneous zero-momentum reaction map

    v_base = -M_bb^-1 M_ba B qdot_plan,

where ``B`` is the V5 17-to-67 arm-coordinate map.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from model_test.whole_body_verifier_v5 import CollisionPair
from v6_lite.continuum_model_spec import (
    CONTINUUM_ACTUATED_DOF,
    ContinuumModelSpec,
    default_continuum_model_spec,
)
from v6_lite.continuum_shape_model import (
    ContinuumShapeModel,
    transform_from_free_qpos,
)
from v6_lite.pcc_clearance import PCCClearanceEvaluator
from v6_lite.shape_clearance import (
    CapsuleEnvelopeSet,
    build_continuum_capsule_envelopes,
    minimum_capsule_clearance,
    target_box_from_mujoco,
)

CONTINUUM_EE_OFFSET_M = np.asarray([0.0475, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class HierarchicalQPConfig:
    task_period_s: float = 0.02
    rigid_position_gain: float = 5.0
    continuum_position_gain: float = 12.0
    rigid_orientation_gain: float = 8.0
    continuum_orientation_gain: float = 10.0
    rigid_priority_weight: float = 500.0
    continuum_priority_weight: float = 260.0
    rigid_orientation_weight: float = 4000.0
    continuum_orientation_weight: float = 2500.0
    posture_weight: float = 0.02
    base_reaction_weight: float = 0.05
    posture_gain: float = 0.08
    rigid_speed_limit_m_s: float = 0.24
    continuum_speed_limit_m_s: float = 0.24
    rigid_angular_speed_limit_rad_s: float = 0.45
    continuum_angular_speed_limit_rad_s: float = 0.45
    velocity_limit_scale: float = 0.70
    joint_position_margin_rad: float = 0.04
    joint_barrier_gain: float = 4.0
    clearance_safe_m: float = 0.025
    rigid_target_clearance_safe_m: float = 0.005
    clearance_activation_m: float = 0.080
    clearance_barrier_gain: float = 8.0
    clearance_query_max_m: float = 0.090
    enable_pcc_cbf: bool = False
    enable_capsule_cbf: bool = False
    pcc_clearance_safe_m: float = 0.005
    pcc_clearance_activation_m: float = 0.100
    pcc_clearance_barrier_gain: float = 8.0
    pcc_coarse_samples_per_segment: int = 5
    pcc_active_segment_count: int = 1
    pcc_refinement_max_iterations: int = 5
    capsule_clearance_safe_m: float = 0.005
    capsule_clearance_activation_m: float = 0.080
    capsule_clearance_barrier_gain: float = 8.0
    qp_max_iterations: int = 1200
    qp_ftol: float = 5e-5
    feasibility_tolerance: float = 1e-4
    admm_rho: float = 50.0
    admm_sigma: float = 1e-6
    admm_relaxation: float = 1.8

    def validate(self) -> None:
        if abs(self.task_period_s - 0.02) > 1e-12:
            raise ValueError("V6-lite task period must be exactly 0.02 s (50 Hz)")
        positive = (
            self.rigid_position_gain,
            self.continuum_position_gain,
            self.rigid_orientation_gain,
            self.continuum_orientation_gain,
            self.rigid_priority_weight,
            self.continuum_priority_weight,
            self.rigid_orientation_weight,
            self.continuum_orientation_weight,
            self.rigid_speed_limit_m_s,
            self.continuum_speed_limit_m_s,
            self.rigid_angular_speed_limit_rad_s,
            self.continuum_angular_speed_limit_rad_s,
            self.velocity_limit_scale,
            self.joint_barrier_gain,
            self.clearance_safe_m,
            self.rigid_target_clearance_safe_m,
            self.clearance_activation_m,
            self.clearance_barrier_gain,
            self.clearance_query_max_m,
            self.pcc_clearance_safe_m,
            self.pcc_clearance_activation_m,
            self.pcc_clearance_barrier_gain,
            self.capsule_clearance_safe_m,
            self.capsule_clearance_activation_m,
            self.capsule_clearance_barrier_gain,
            self.qp_ftol,
            self.feasibility_tolerance,
            self.admm_rho,
            self.admm_sigma,
            self.admm_relaxation,
        )
        if min(positive) <= 0.0:
            raise ValueError("positive V6-lite QP parameters must be positive")
        if self.rigid_priority_weight <= self.continuum_priority_weight:
            raise ValueError("rigid grasp tracking must have higher priority")
        if self.clearance_safe_m >= self.clearance_activation_m:
            raise ValueError("clearance activation must exceed the hard safety margin")
        if self.rigid_target_clearance_safe_m >= self.clearance_activation_m:
            raise ValueError(
                "clearance activation must exceed the rigid-target safety margin"
            )
        if self.clearance_activation_m >= self.clearance_query_max_m:
            raise ValueError("clearance query range must exceed activation distance")
        if self.pcc_clearance_safe_m >= self.pcc_clearance_activation_m:
            raise ValueError("PCC activation must exceed its hard safety margin")
        if self.capsule_clearance_safe_m >= self.capsule_clearance_activation_m:
            raise ValueError("capsule activation must exceed its hard safety margin")
        if self.pcc_coarse_samples_per_segment < 3:
            raise ValueError("PCC broad phase needs at least three samples per segment")
        if self.pcc_active_segment_count not in range(1, 6):
            raise ValueError("PCC active segment count must lie in [1,5]")
        if self.pcc_refinement_max_iterations < 1:
            raise ValueError("PCC refinement iteration count must be positive")
        if not isinstance(self.enable_pcc_cbf, bool) or not isinstance(
            self.enable_capsule_cbf, bool
        ):
            raise TypeError("shape-CBF enable flags must be boolean")
        if self.qp_max_iterations < 1:
            raise ValueError("qp_max_iterations must be positive")
        if self.admm_relaxation >= 2.0:
            raise ValueError("ADMM relaxation must be below two")


@dataclass(frozen=True)
class HierarchicalQPResult:
    planner_velocity: np.ndarray
    success: bool
    solver_status: str
    solver_iterations: int
    objective: float
    full_latency_s: float
    solver_latency_s: float
    rigid_position_error_m: float
    continuum_position_error_m: float
    rigid_velocity_residual_m_s: float
    continuum_velocity_residual_m_s: float
    rigid_orientation_error_rad: float
    continuum_orientation_error_rad: float
    rigid_angular_velocity_residual_rad_s: float
    continuum_angular_velocity_residual_rad_s: float
    minimum_queried_clearance_m: float
    active_clearance_constraint_count: int
    binding_clearance_constraint_count: int
    minimum_constraint_slack: float
    unconstrained_to_command_norm: float
    reaction_momentum_residual_norm: float
    degenerate_clearance_gradient_count: int
    pcc_clearance_m: float
    capsule_clearance_m: float
    mujoco_continuum_target_clearance_m: float
    pcc_constraint_active_count: int
    capsule_constraint_active_count: int
    pcc_binding_constraint_count: int
    capsule_binding_constraint_count: int
    pcc_avoidance_intervention: float
    pcc_mujoco_distance_error_m: float
    pcc_mujoco_gradient_error_norm: float
    capsule_mujoco_gradient_error_norm: float
    pcc_closest_segment_id: int
    pcc_closest_arclength_m: float
    shape_clearance_latency_s: float


@dataclass(frozen=True)
class _ShapeClearanceKinematics:
    distance_m: float
    gradient: np.ndarray
    target_distance_rate_m_s: float
    source_name: str
    closest_segment_id: int = -1
    closest_arclength_m: float = float("nan")


@dataclass(frozen=True)
class _ClearanceConstraintSet:
    matrix: np.ndarray
    lower: np.ndarray
    sources: tuple[str, ...]
    minimum_clearance_m: float
    degenerate_gradient_count: int
    mujoco_continuum_target_distance_m: float
    mujoco_continuum_target_gradient: np.ndarray
    mujoco_continuum_target_gradient_valid: bool
    shape_clearance_latency_s: float
    pcc: _ShapeClearanceKinematics | None = None
    capsule: _ShapeClearanceKinematics | None = None


def joint_addresses(
    model: mujoco.MjModel, spec: RobotModelSpecV5
) -> tuple[np.ndarray, np.ndarray]:
    qpos_ids: list[int] = []
    dof_ids: list[int] = []
    for name in spec.low_level_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing low-level joint {name!r}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_ids, dtype=np.int32), np.asarray(dof_ids, dtype=np.int32)


def free_joint_slices(
    model: mujoco.MjModel, joint_name: str
) -> tuple[slice, slice]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"{joint_name!r} is not a free joint")
    qpos_start = int(model.jnt_qposadr[joint_id])
    dof_start = int(model.jnt_dofadr[joint_id])
    return slice(qpos_start, qpos_start + 7), slice(dof_start, dof_start + 6)


def _clip_vector_norm(value: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= limit:
        return value
    return value * (limit / max(norm, 1e-12))


def rotation_error_vector_world(
    target_rotation: np.ndarray, current_rotation: np.ndarray
) -> np.ndarray:
    """SO(3) logarithm taking current orientation toward target in world axes."""

    target = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
    current = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    delta = target @ current.T
    cosine = float(np.clip(0.5 * (np.trace(delta) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=np.float64,
    )
    if angle <= 1e-8:
        return 0.5 * skew
    if np.pi - angle <= 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(delta)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return angle * axis
    return (0.5 * angle / np.sin(angle)) * skew


def rotation_error_angle_rad(
    target_rotation: np.ndarray, current_rotation: np.ndarray
) -> float:
    return float(np.linalg.norm(rotation_error_vector_world(target_rotation, current_rotation)))


class HierarchicalVelocityQP:
    """Solve exactly one deterministic task/clearance QP per task tick."""

    def __init__(
        self,
        spec: RobotModelSpecV5,
        model: mujoco.MjModel,
        collision_pairs: Sequence[CollisionPair],
        config: HierarchicalQPConfig = HierarchicalQPConfig(),
    ) -> None:
        spec.validate()
        config.validate()
        if abs(float(model.opt.timestep) - 0.002) > 1e-12:
            raise ValueError("V6-lite physics timestep must be 0.002 s (500 Hz)")
        self.spec = spec
        self.model = model
        self.collision_pairs = tuple(collision_pairs)
        self.config = config
        self.qpos_ids, self.dof_ids = joint_addresses(model, spec)
        self.base_qpos_slice, self.base_dof_slice = free_joint_slices(
            model, spec.base_joint_name
        )
        self.target_qpos_slice, self.target_dof_slice = free_joint_slices(
            model, spec.target_free_joint_name
        )
        self.continuum_body_id = self._body_id(spec.continuum_tip_body_name)
        self.rigid_body_id = self._body_id(spec.rigid_tip_body_name)
        self.previous_velocity = np.zeros(17, dtype=np.float64)
        self._previous_constraint_dual: dict[str, float] = {}
        self.solve_count = 0
        self._full_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
        self._jacobian_a = np.zeros((3, model.nv), dtype=np.float64)
        self._jacobian_b = np.zeros((3, model.nv), dtype=np.float64)
        self._target_exogenous_qvel = np.zeros(model.nv, dtype=np.float64)
        self._fromto = np.zeros(6, dtype=np.float64)
        self._shape_spec: ContinuumModelSpec | None = None
        self._shape_model: ContinuumShapeModel | None = None
        self._pcc_clearance_evaluator: PCCClearanceEvaluator | None = None
        self._capsule_envelopes: CapsuleEnvelopeSet | None = None
        self._shape_target_geom_id = -1
        self._shape_target_body_id = -1
        self._shape_base_body_id = -1
        if config.enable_pcc_cbf or config.enable_capsule_cbf:
            self._shape_spec = default_continuum_model_spec(spec)
            self._shape_model = ContinuumShapeModel(self._shape_spec)
            self._shape_target_geom_id = int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "target_satellite_collision",
                )
            )
            if self._shape_target_geom_id < 0:
                raise ValueError("moving target satellite collision OBB is missing")
            self._shape_target_body_id = int(
                model.geom_bodyid[self._shape_target_geom_id]
            )
            self._shape_base_body_id = self._body_id(
                self._shape_spec.base_body_name
            )
            if config.enable_pcc_cbf:
                self._pcc_clearance_evaluator = PCCClearanceEvaluator(
                    self._shape_model,
                    coarse_samples_per_segment=(
                        config.pcc_coarse_samples_per_segment
                    ),
                    active_segment_count=config.pcc_active_segment_count,
                    refinement_max_iterations=(
                        config.pcc_refinement_max_iterations
                    ),
                )
            if config.enable_capsule_cbf:
                self._capsule_envelopes = build_continuum_capsule_envelopes(
                    model, self._shape_spec
                )

    def _body_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if value < 0:
            raise ValueError(f"missing body {name!r}")
        return int(value)

    def reset(self) -> None:
        self.previous_velocity[:] = 0.0
        self._previous_constraint_dual.clear()
        self.solve_count = 0

    def reaction_velocity_map(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """Map 17 arm rates to all generalized rates at zero base momentum."""

        mujoco.mj_fullM(self.model, self._full_mass, data.qM)
        base_ids = np.arange(
            self.base_dof_slice.start, self.base_dof_slice.stop, dtype=np.int32
        )
        mass_bb = self._full_mass[np.ix_(base_ids, base_ids)]
        mass_ba = self._full_mass[np.ix_(base_ids, self.dof_ids)]
        arm_map = self.spec.planner_to_low_level
        base_map = -np.linalg.solve(mass_bb, mass_ba @ arm_map)
        generalized_map = np.zeros((self.model.nv, 17), dtype=np.float64)
        generalized_map[base_ids, :] = base_map
        generalized_map[self.dof_ids, :] = arm_map
        residual = mass_bb @ base_map + mass_ba @ arm_map
        return generalized_map, float(np.linalg.norm(residual))

    def point_jacobian(
        self, data: mujoco.MjData, body_id: int, point_world: np.ndarray
    ) -> np.ndarray:
        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            data,
            jacobian,
            None,
            np.asarray(point_world, dtype=np.float64),
            int(body_id),
        )
        return jacobian

    def target_exogenous_qvel(self, data: mujoco.MjData) -> np.ndarray:
        """Return the prescribed target twist omitted from the 17-DoF map.

        ``reaction_velocity_map`` deliberately maps only the arm command and
        its zero-momentum base reaction.  The target satellite is a separate
        free body, so its six measured generalized rates must enter a
        time-varying distance barrier as an affine drift rather than as QP
        decision variables.  Keeping only the named target free-joint slice
        also prevents current arm/base tracking velocity from being counted
        twice.
        """

        target_velocity = np.asarray(
            data.qvel[self.target_dof_slice], dtype=np.float64
        )
        if np.any(~np.isfinite(target_velocity)):
            raise ValueError("target satellite generalized velocity must be finite")
        self._target_exogenous_qvel.fill(0.0)
        self._target_exogenous_qvel[self.target_dof_slice] = target_velocity
        return self._target_exogenous_qvel

    def _pair_witness_points(
        self, pair: CollisionPair
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map MuJoCo's collision-dispatch witnesses back to pair order.

        MuJoCo's narrow phase canonicalizes mixed geometry pairs by geometry
        type.  Consequently ``fromto`` follows ascending ``mjtGeom`` type for
        a mixed pair even when the ids were supplied in the opposite order
        (notably mesh--box under native CCD).  Equal-type pairs retain call
        order.  The returned tuple is always ``(point_on_a, point_on_b)``.
        """

        geom_a = int(pair.geom_a)
        geom_b = int(pair.geom_b)
        first = self._fromto[:3].copy()
        second = self._fromto[3:].copy()
        if int(self.model.geom_type[geom_a]) <= int(self.model.geom_type[geom_b]):
            return first, second
        return second, first

    def _pair_clearance_safe_m(self, pair: CollisionPair) -> float:
        if pair.pair_class == "rigid_target":
            return self.config.rigid_target_clearance_safe_m
        return self.config.clearance_safe_m

    def _task_jacobians(
        self,
        data: mujoco.MjData,
        body_id: int,
        generalized_map: np.ndarray,
        local_offset: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        offset = (
            np.zeros(3, dtype=np.float64)
            if local_offset is None
            else np.asarray(local_offset, dtype=np.float64).reshape(3)
        )
        position = (
            np.asarray(data.xpos[body_id], dtype=np.float64) + rotation @ offset
        )
        position_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        rotation_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            data,
            position_jacobian,
            rotation_jacobian,
            position,
            body_id,
        )
        return (
            position_jacobian @ generalized_map,
            rotation_jacobian @ generalized_map,
        )

    def _mujoco_clearance_constraint_set(
        self, data: mujoco.MjData, generalized_map: np.ndarray
    ) -> _ClearanceConstraintSet:
        """Build the unchanged V6 MuJoCo-geometry CBF constraint block."""

        rows: list[np.ndarray] = []
        lowers: list[float] = []
        sources: list[str] = []
        minimum = float("inf")
        degenerate = 0
        target_minimum = float("inf")
        target_gradient = np.zeros(17, dtype=np.float64)
        target_gradient_valid = False
        cfg = self.config
        target_exogenous_qvel = self.target_exogenous_qvel(data)
        for pair in self.collision_pairs:
            distance = float(
                mujoco.mj_geomDistance(
                    self.model,
                    data,
                    int(pair.geom_a),
                    int(pair.geom_b),
                    cfg.clearance_query_max_m,
                    self._fromto,
                )
            )
            minimum = min(minimum, distance)
            is_continuum_target = pair.pair_class == "continuum_target"
            target_reported_distance = min(
                distance, cfg.clearance_query_max_m
            )
            if is_continuum_target and target_reported_distance < target_minimum:
                target_minimum = target_reported_distance
                target_gradient.fill(0.0)
                target_gradient_valid = False
            if distance > cfg.clearance_activation_m:
                continue
            point_a, point_b = self._pair_witness_points(pair)
            line = point_a - point_b
            line_norm = float(np.linalg.norm(line))
            if line_norm <= 1e-10:
                degenerate += 1
                continue
            normal_b_to_a = line / line_norm
            body_a = int(self.model.geom_bodyid[int(pair.geom_a)])
            body_b = int(self.model.geom_bodyid[int(pair.geom_b)])
            self._jacobian_a.fill(0.0)
            self._jacobian_b.fill(0.0)
            if body_a != 0:
                mujoco.mj_jac(
                    self.model,
                    data,
                    self._jacobian_a,
                    None,
                    point_a,
                    body_a,
                )
            if body_b != 0:
                mujoco.mj_jac(
                    self.model,
                    data,
                    self._jacobian_b,
                    None,
                    point_b,
                    body_b,
                )
            # A nonzero finite-difference regression locks both the witness
            # order and the target-drift sign convention.
            relative_point_jacobian = self._jacobian_a - self._jacobian_b
            distance_gradient = normal_b_to_a @ (
                relative_point_jacobian
            ) @ generalized_map
            target_distance_rate = float(
                normal_b_to_a
                @ relative_point_jacobian
                @ target_exogenous_qvel
            )
            rows.append(np.asarray(distance_gradient, dtype=np.float64))
            sources.append(
                f"mujoco:{pair.pair_class}:{pair.geom_a_name}:{pair.geom_b_name}"
            )
            pair_safe_m = self._pair_clearance_safe_m(pair)
            lowers.append(
                -cfg.clearance_barrier_gain * (distance - pair_safe_m)
                - target_distance_rate
            )
            if (
                is_continuum_target
                and distance < cfg.clearance_query_max_m - 1e-12
                and distance <= target_minimum + 1e-15
            ):
                target_minimum = distance
                target_gradient = np.asarray(
                    distance_gradient, dtype=np.float64
                ).copy()
                target_gradient_valid = True
        matrix = (
            np.vstack(rows)
            if rows
            else np.zeros((0, 17), dtype=np.float64)
        )
        return _ClearanceConstraintSet(
            matrix=matrix,
            lower=np.asarray(lowers, dtype=np.float64),
            sources=tuple(sources),
            minimum_clearance_m=float(minimum),
            degenerate_gradient_count=int(degenerate),
            mujoco_continuum_target_distance_m=float(target_minimum),
            mujoco_continuum_target_gradient=target_gradient,
            mujoco_continuum_target_gradient_valid=target_gradient_valid,
            shape_clearance_latency_s=0.0,
        )

    def _clearance_constraints(
        self, data: mujoco.MjData, generalized_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Compatibility facade for the frozen V6 tests and diagnostics."""

        block = self._mujoco_clearance_constraint_set(data, generalized_map)
        return (
            block.matrix,
            block.lower,
            block.minimum_clearance_m,
            block.degenerate_gradient_count,
        )

    def _target_point_velocity_jacobian(
        self, data: mujoco.MjData, point_world: np.ndarray
    ) -> np.ndarray:
        if self._shape_target_body_id < 0:
            raise RuntimeError("shape-CBF target geometry is not initialized")
        return self.point_jacobian(
            data, self._shape_target_body_id, np.asarray(point_world)
        )

    def _pcc_clearance_kinematics(
        self, data: mujoco.MjData, generalized_map: np.ndarray
    ) -> _ShapeClearanceKinematics:
        """Return PCC distance dynamics including free-base reaction and target drift.

        The controlled centerline velocity is the sum of the internal PCC
        differential motion and the velocity of the same world point rigidly
        attached to the floating base.  The latter is evaluated through the
        generalized reaction map, so the CBF row acts on all 17 planner rates.
        """

        if (
            self._shape_spec is None
            or self._pcc_clearance_evaluator is None
            or self._shape_target_geom_id < 0
        ):
            raise RuntimeError("PCC CBF was requested without initialized geometry")
        low_level_continuum = np.asarray(
            data.qpos[self.qpos_ids[:CONTINUUM_ACTUATED_DOF]], dtype=np.float64
        )
        projection = self._shape_spec.project_actual_configuration(
            low_level_continuum
        )
        base_transform = transform_from_free_qpos(
            np.asarray(data.qpos[self.base_qpos_slice], dtype=np.float64)
        )
        target_box = target_box_from_mujoco(
            self.model, data, self._shape_target_geom_id
        )
        result = self._pcc_clearance_evaluator.evaluate(
            projection.planner_configuration,
            base_transform,
            target_box,
        )
        base_point_jacobian = self.point_jacobian(
            data,
            self._shape_base_body_id,
            result.centerline_point,
        )
        controlled_gradient = (
            result.normal @ base_point_jacobian @ generalized_map
        )
        controlled_gradient = np.asarray(
            controlled_gradient, dtype=np.float64
        )
        controlled_gradient[:10] += result.gradient
        target_jacobian = self._target_point_velocity_jacobian(
            data, result.point_on_obb
        )
        target_rate = float(
            -result.normal
            @ target_jacobian
            @ self.target_exogenous_qvel(data)
        )
        return _ShapeClearanceKinematics(
            distance_m=float(result.distance),
            gradient=controlled_gradient,
            target_distance_rate_m_s=target_rate,
            source_name=f"pcc:segment_{result.segment_id + 1}",
            closest_segment_id=int(result.segment_id),
            closest_arclength_m=float(result.arc_length),
        )

    def _capsule_clearance_kinematics(
        self, data: mujoco.MjData, generalized_map: np.ndarray
    ) -> _ShapeClearanceKinematics:
        """Return the actual-chain capsule clearance and generalized gradient."""

        if (
            self._shape_spec is None
            or self._capsule_envelopes is None
            or self._shape_target_geom_id < 0
        ):
            raise RuntimeError("capsule CBF was requested without initialized geometry")
        target_box = target_box_from_mujoco(
            self.model, data, self._shape_target_geom_id
        )
        result = minimum_capsule_clearance(
            self.model,
            data,
            self._capsule_envelopes,
            target_box,
            spec=self._shape_spec,
            compute_planner_gradient=False,
        )
        arm_body_id = int(self.model.geom_bodyid[result.source_index])
        arm_jacobian = self.point_jacobian(
            data, arm_body_id, result.point_on_arm
        )
        controlled_gradient = (
            result.normal_box_to_arm @ arm_jacobian @ generalized_map
        )
        target_jacobian = self._target_point_velocity_jacobian(
            data, result.point_on_box
        )
        target_rate = float(
            -result.normal_box_to_arm
            @ target_jacobian
            @ self.target_exogenous_qvel(data)
        )
        return _ShapeClearanceKinematics(
            distance_m=float(result.signed_distance_m),
            gradient=np.asarray(controlled_gradient, dtype=np.float64),
            target_distance_rate_m_s=target_rate,
            source_name=f"capsule:{result.source_name}",
        )

    def _build_all_clearance_constraints(
        self, data: mujoco.MjData, generalized_map: np.ndarray
    ) -> _ClearanceConstraintSet:
        """Combine legacy MuJoCo, PCC-tube, and capsule CBF rows.

        MuJoCo rows are always retained.  Each shape row is opt-in and is
        appended only inside its own activation distance.  No existing row is
        replaced or weakened.
        """

        mujoco_block = self._mujoco_clearance_constraint_set(
            data, generalized_map
        )
        rows = [row.copy() for row in mujoco_block.matrix]
        lowers = mujoco_block.lower.tolist()
        sources = list(mujoco_block.sources)
        minimum = mujoco_block.minimum_clearance_m
        pcc: _ShapeClearanceKinematics | None = None
        capsule: _ShapeClearanceKinematics | None = None
        shape_started = time.perf_counter()
        if self.config.enable_pcc_cbf:
            pcc = self._pcc_clearance_kinematics(data, generalized_map)
            minimum = min(minimum, pcc.distance_m)
            if pcc.distance_m <= self.config.pcc_clearance_activation_m:
                rows.append(pcc.gradient)
                lowers.append(
                    -self.config.pcc_clearance_barrier_gain
                    * (pcc.distance_m - self.config.pcc_clearance_safe_m)
                    - pcc.target_distance_rate_m_s
                )
                sources.append(pcc.source_name)
        if self.config.enable_capsule_cbf:
            capsule = self._capsule_clearance_kinematics(data, generalized_map)
            minimum = min(minimum, capsule.distance_m)
            if capsule.distance_m <= self.config.capsule_clearance_activation_m:
                rows.append(capsule.gradient)
                lowers.append(
                    -self.config.capsule_clearance_barrier_gain
                    * (
                        capsule.distance_m
                        - self.config.capsule_clearance_safe_m
                    )
                    - capsule.target_distance_rate_m_s
                )
                sources.append(capsule.source_name)
        shape_latency = time.perf_counter() - shape_started
        matrix = (
            np.vstack(rows)
            if rows
            else np.zeros((0, 17), dtype=np.float64)
        )
        return _ClearanceConstraintSet(
            matrix=matrix,
            lower=np.asarray(lowers, dtype=np.float64),
            sources=tuple(sources),
            minimum_clearance_m=float(minimum),
            degenerate_gradient_count=(
                mujoco_block.degenerate_gradient_count
            ),
            mujoco_continuum_target_distance_m=(
                mujoco_block.mujoco_continuum_target_distance_m
            ),
            mujoco_continuum_target_gradient=(
                mujoco_block.mujoco_continuum_target_gradient
            ),
            mujoco_continuum_target_gradient_valid=(
                mujoco_block.mujoco_continuum_target_gradient_valid
            ),
            shape_clearance_latency_s=float(shape_latency),
            pcc=pcc,
            capsule=capsule,
        )

    def clearance_gradient_for_pair(
        self,
        data: mujoco.MjData,
        pair: CollisionPair,
        generalized_map: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Expose one analytic signed-distance gradient for regression tests."""

        distance, gradient, _target_distance_rate = (
            self.clearance_kinematics_for_pair(data, pair, generalized_map)
        )
        return distance, gradient

    def clearance_kinematics_for_pair(
        self,
        data: mujoco.MjData,
        pair: CollisionPair,
        generalized_map: np.ndarray,
    ) -> tuple[float, np.ndarray, float]:
        """Return distance, controlled gradient, and target-induced rate.

        The complete first-order distance dynamics are

        ``d_dot = gradient @ planner_velocity + target_distance_rate``.

        Translation and rotation of the target are both included naturally by
        its witness-point Jacobian multiplied by the target free-joint qvel.
        """

        distance = float(
            mujoco.mj_geomDistance(
                self.model,
                data,
                int(pair.geom_a),
                int(pair.geom_b),
                self.config.clearance_query_max_m,
                self._fromto,
            )
        )
        point_a, point_b = self._pair_witness_points(pair)
        line = point_a - point_b
        normal = line / max(float(np.linalg.norm(line)), 1e-12)
        body_a = int(self.model.geom_bodyid[int(pair.geom_a)])
        body_b = int(self.model.geom_bodyid[int(pair.geom_b)])
        jacobian_a = self.point_jacobian(data, body_a, point_a) if body_a else np.zeros((3, self.model.nv))
        jacobian_b = self.point_jacobian(data, body_b, point_b) if body_b else np.zeros((3, self.model.nv))
        relative_point_jacobian = jacobian_a - jacobian_b
        gradient = normal @ relative_point_jacobian @ generalized_map
        target_distance_rate = float(
            normal
            @ relative_point_jacobian
            @ self.target_exogenous_qvel(data)
        )
        return (
            distance,
            np.asarray(gradient, dtype=np.float64),
            target_distance_rate,
        )

    def _velocity_bounds(self, planner_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        dt = cfg.task_period_s
        speed = cfg.velocity_limit_scale * self.spec.planner_velocity_limits
        lower = np.maximum(-speed, self.previous_velocity - self.spec.planner_acceleration_limits * dt)
        upper = np.minimum(speed, self.previous_velocity + self.spec.planner_acceleration_limits * dt)
        lower = np.maximum(
            lower,
            -cfg.joint_barrier_gain
            * (planner_q - (self.spec.planner_lower + cfg.joint_position_margin_rad)),
        )
        upper = np.minimum(
            upper,
            cfg.joint_barrier_gain
            * ((self.spec.planner_upper - cfg.joint_position_margin_rad) - planner_q),
        )
        return lower, upper

    def _solve_qp_admm(
        self,
        hessian: np.ndarray,
        linear: np.ndarray,
        matrix: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        initial: np.ndarray,
        initial_dual: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, str, int, np.ndarray]:
        """Solve one strictly convex QP with OSQP-style ADMM iterations.

        The factorization is only 17-by-17.  Box limits and clearance
        barriers share the same constraint operator, so there is no secondary
        projection optimizer or candidate loop.
        """

        cfg = self.config
        rho = cfg.admm_rho
        sigma = cfg.admm_sigma
        system = hessian + sigma * np.eye(17) + rho * (matrix.T @ matrix)
        factor = cho_factor(system, lower=True, check_finite=False)
        value = np.asarray(initial, dtype=np.float64).copy()
        product = matrix @ value
        dual = (
            np.zeros(matrix.shape[0], dtype=np.float64)
            if initial_dual is None
            else np.asarray(initial_dual, dtype=np.float64).copy()
        )
        if dual.shape != (matrix.shape[0],) or np.any(~np.isfinite(dual)):
            raise ValueError("ADMM dual warm start has the wrong shape or is non-finite")
        auxiliary = np.minimum(
            np.maximum(product + dual / rho, lower), upper
        )
        status = "maximum_iterations"
        for iteration in range(1, cfg.qp_max_iterations + 1):
            rhs = sigma * value - linear + matrix.T @ (rho * auxiliary - dual)
            value = cho_solve(factor, rhs, check_finite=False)
            product = matrix @ value
            old_auxiliary = auxiliary.copy()
            relaxed = (
                cfg.admm_relaxation * product
                + (1.0 - cfg.admm_relaxation) * old_auxiliary
            )
            auxiliary = np.minimum(
                np.maximum(relaxed + dual / rho, lower), upper
            )
            dual += rho * (relaxed - auxiliary)
            primal_residual = float(np.max(np.abs(product - auxiliary)))
            dual_residual = float(
                np.max(np.abs(hessian @ value + linear + matrix.T @ dual))
            )
            primal_scale = max(
                1.0,
                float(np.max(np.abs(product))),
                float(np.max(np.abs(auxiliary))),
            )
            dual_scale = max(
                1.0,
                float(np.max(np.abs(hessian @ value))),
                float(np.max(np.abs(matrix.T @ dual))),
                float(np.max(np.abs(linear))),
            )
            if (
                primal_residual <= cfg.qp_ftol * primal_scale
                and dual_residual <= 5.0 * cfg.qp_ftol * dual_scale
            ):
                status = "solved"
                break
        feasibility = np.minimum(product - lower, upper - product)
        feasible = bool(float(np.min(feasibility)) >= -cfg.feasibility_tolerance)
        if feasible and status != "solved":
            status = "solved_inaccurate"
        return value, feasible, status, iteration, dual

    def solve(
        self,
        data: mujoco.MjData,
        *,
        rigid_target_position: np.ndarray,
        rigid_target_velocity: np.ndarray,
        rigid_target_rotation: np.ndarray,
        rigid_target_angular_velocity: np.ndarray,
        continuum_target_position: np.ndarray,
        continuum_target_velocity: np.ndarray,
        continuum_target_rotation: np.ndarray,
        continuum_target_angular_velocity: np.ndarray,
    ) -> HierarchicalQPResult:
        started = time.perf_counter()
        mujoco.mj_forward(self.model, data)
        cfg = self.config
        generalized_map, momentum_residual = self.reaction_velocity_map(data)
        rigid_position = np.asarray(data.xpos[self.rigid_body_id]).copy()
        rigid_rotation = np.asarray(data.xmat[self.rigid_body_id]).reshape(3, 3).copy()
        continuum_rotation = (
            np.asarray(data.xmat[self.continuum_body_id]).reshape(3, 3).copy()
        )
        continuum_position = (
            np.asarray(data.xpos[self.continuum_body_id]).copy()
            + continuum_rotation @ CONTINUUM_EE_OFFSET_M
        )
        rigid_error_vector = np.asarray(rigid_target_position) - rigid_position
        continuum_error_vector = np.asarray(continuum_target_position) - continuum_position
        rigid_orientation_error = rotation_error_vector_world(
            rigid_target_rotation, rigid_rotation
        )
        continuum_orientation_error = rotation_error_vector_world(
            continuum_target_rotation, continuum_rotation
        )
        rigid_command = _clip_vector_norm(
            np.asarray(rigid_target_velocity)
            + cfg.rigid_position_gain * rigid_error_vector,
            cfg.rigid_speed_limit_m_s,
        )
        continuum_command = _clip_vector_norm(
            np.asarray(continuum_target_velocity)
            + cfg.continuum_position_gain * continuum_error_vector,
            cfg.continuum_speed_limit_m_s,
        )
        rigid_angular_command = _clip_vector_norm(
            np.asarray(rigid_target_angular_velocity)
            + cfg.rigid_orientation_gain * rigid_orientation_error,
            cfg.rigid_angular_speed_limit_rad_s,
        )
        continuum_angular_command = _clip_vector_norm(
            np.asarray(continuum_target_angular_velocity)
            + cfg.continuum_orientation_gain * continuum_orientation_error,
            cfg.continuum_angular_speed_limit_rad_s,
        )
        rigid_jacobian, rigid_rotation_jacobian = self._task_jacobians(
            data, self.rigid_body_id, generalized_map
        )
        continuum_jacobian, continuum_rotation_jacobian = self._task_jacobians(
            data,
            self.continuum_body_id,
            generalized_map,
            CONTINUUM_EE_OFFSET_M,
        )
        low_level_q = np.asarray(data.qpos[self.qpos_ids], dtype=np.float64)
        planner_q = self.spec.low_level_to_planner @ low_level_q
        posture_velocity = -cfg.posture_gain * (planner_q - self.spec.planner_zero)

        hessian = (
            cfg.rigid_priority_weight * (rigid_jacobian.T @ rigid_jacobian)
            + cfg.continuum_priority_weight
            * (continuum_jacobian.T @ continuum_jacobian)
            + cfg.rigid_orientation_weight
            * (rigid_rotation_jacobian.T @ rigid_rotation_jacobian)
            + cfg.continuum_orientation_weight
            * (continuum_rotation_jacobian.T @ continuum_rotation_jacobian)
            + cfg.posture_weight * np.eye(17)
            + cfg.base_reaction_weight
            * (generalized_map[self.base_dof_slice, :].T @ generalized_map[self.base_dof_slice, :])
        )
        linear = -(
            cfg.rigid_priority_weight * rigid_jacobian.T @ rigid_command
            + cfg.continuum_priority_weight
            * continuum_jacobian.T
            @ continuum_command
            + cfg.rigid_orientation_weight
            * rigid_rotation_jacobian.T
            @ rigid_angular_command
            + cfg.continuum_orientation_weight
            * continuum_rotation_jacobian.T
            @ continuum_angular_command
            + cfg.posture_weight * posture_velocity
        )
        # Symmetrize and add a tiny numerical ridge; the objective remains the
        # declared weighted task hierarchy.
        hessian = 0.5 * (hessian + hessian.T) + 1e-10 * np.eye(17)
        clearance_block = self._build_all_clearance_constraints(
            data, generalized_map
        )
        clearance_matrix = clearance_block.matrix
        clearance_lower = clearance_block.lower
        minimum_clearance = clearance_block.minimum_clearance_m
        degenerate = clearance_block.degenerate_gradient_count
        clearance_sources = np.asarray(clearance_block.sources, dtype=object)
        lower, upper = self._velocity_bounds(planner_q)
        unconstrained = np.clip(-np.linalg.solve(hessian, linear), lower, upper)
        initial = np.clip(self.previous_velocity, lower, upper)

        pcc_row_mask = np.asarray(
            [str(source).startswith("pcc:") for source in clearance_sources],
            dtype=bool,
        )
        capsule_row_mask = np.asarray(
            [str(source).startswith("capsule:") for source in clearance_sources],
            dtype=bool,
        )
        if np.any(pcc_row_mask):
            pcc_deficit = np.maximum(
                clearance_lower[pcc_row_mask]
                - clearance_matrix[pcc_row_mask] @ unconstrained,
                0.0,
            )
            pcc_norm = np.maximum(
                np.linalg.norm(clearance_matrix[pcc_row_mask], axis=1),
                1e-12,
            )
            pcc_intervention = float(np.max(pcc_deficit / pcc_norm))
        else:
            pcc_intervention = 0.0

        def objective(value: np.ndarray) -> float:
            return 0.5 * float(value @ hessian @ value) + float(linear @ value)

        def gradient(value: np.ndarray) -> np.ndarray:
            return hessian @ value + linear

        matrix = np.vstack([clearance_matrix, np.eye(17)])
        constraint_lower = np.concatenate([clearance_lower, lower])
        constraint_upper = np.concatenate(
            [np.full(clearance_lower.shape, np.inf), upper]
        )
        constraint_keys = [str(source) for source in clearance_sources]
        constraint_keys.extend(f"planner_velocity_bound:{index}" for index in range(17))
        initial_dual = np.asarray(
            [self._previous_constraint_dual.get(key, 0.0) for key in constraint_keys],
            dtype=np.float64,
        )
        solver_started = time.perf_counter()
        candidate, success, solver_status, solver_iterations, dual = self._solve_qp_admm(
            hessian,
            linear,
            matrix,
            constraint_lower,
            constraint_upper,
            initial,
            initial_dual,
        )
        solver_latency = time.perf_counter() - solver_started
        if clearance_matrix.shape[0]:
            slacks = clearance_matrix @ candidate - clearance_lower
            minimum_slack = float(np.min(slacks))
            binding = int(np.sum(slacks <= 2e-5))
            pcc_binding = int(np.sum((slacks <= 2e-5) & pcc_row_mask))
            capsule_binding = int(
                np.sum((slacks <= 2e-5) & capsule_row_mask)
            )
        else:
            minimum_slack = float("inf")
            binding = 0
            pcc_binding = 0
            capsule_binding = 0
        bound_slack = float(min(np.min(candidate - lower), np.min(upper - candidate)))
        minimum_slack = min(minimum_slack, bound_slack)
        success = bool(success and minimum_slack >= -cfg.feasibility_tolerance)
        if not success:
            # Safe stop is not another optimizer or an oracle plan.  The full
            # delivery gate requires this branch never to be used.
            candidate = np.zeros(17, dtype=np.float64)
            self._previous_constraint_dual.clear()
        else:
            self._previous_constraint_dual = {
                key: float(value) for key, value in zip(constraint_keys, dual)
            }
        self.previous_velocity = candidate.copy()
        self.solve_count += 1
        full_latency = time.perf_counter() - started
        pcc_distance = (
            float(clearance_block.pcc.distance_m)
            if clearance_block.pcc is not None
            else float("inf")
        )
        capsule_distance = (
            float(clearance_block.capsule.distance_m)
            if clearance_block.capsule is not None
            else float("inf")
        )
        mujoco_target_distance = float(
            clearance_block.mujoco_continuum_target_distance_m
        )
        pcc_distance_error = (
            pcc_distance - mujoco_target_distance
            if np.isfinite(pcc_distance) and np.isfinite(mujoco_target_distance)
            else float("nan")
        )
        if (
            clearance_block.pcc is not None
            and clearance_block.mujoco_continuum_target_gradient_valid
        ):
            pcc_gradient_error = float(
                np.linalg.norm(
                    clearance_block.pcc.gradient
                    - clearance_block.mujoco_continuum_target_gradient
                )
            )
        else:
            pcc_gradient_error = float("nan")
        if (
            clearance_block.capsule is not None
            and clearance_block.mujoco_continuum_target_gradient_valid
        ):
            capsule_gradient_error = float(
                np.linalg.norm(
                    clearance_block.capsule.gradient
                    - clearance_block.mujoco_continuum_target_gradient
                )
            )
        else:
            capsule_gradient_error = float("nan")
        return HierarchicalQPResult(
            planner_velocity=candidate,
            success=success,
            solver_status=solver_status,
            solver_iterations=int(solver_iterations),
            objective=float(objective(candidate)),
            full_latency_s=float(full_latency),
            solver_latency_s=float(solver_latency),
            rigid_position_error_m=float(np.linalg.norm(rigid_error_vector)),
            continuum_position_error_m=float(np.linalg.norm(continuum_error_vector)),
            rigid_velocity_residual_m_s=float(
                np.linalg.norm(rigid_jacobian @ candidate - rigid_command)
            ),
            continuum_velocity_residual_m_s=float(
                np.linalg.norm(continuum_jacobian @ candidate - continuum_command)
            ),
            rigid_orientation_error_rad=float(np.linalg.norm(rigid_orientation_error)),
            continuum_orientation_error_rad=float(
                np.linalg.norm(continuum_orientation_error)
            ),
            rigid_angular_velocity_residual_rad_s=float(
                np.linalg.norm(
                    rigid_rotation_jacobian @ candidate - rigid_angular_command
                )
            ),
            continuum_angular_velocity_residual_rad_s=float(
                np.linalg.norm(
                    continuum_rotation_jacobian @ candidate
                    - continuum_angular_command
                )
            ),
            minimum_queried_clearance_m=float(minimum_clearance),
            active_clearance_constraint_count=int(clearance_matrix.shape[0]),
            binding_clearance_constraint_count=binding,
            minimum_constraint_slack=minimum_slack,
            unconstrained_to_command_norm=float(np.linalg.norm(candidate - unconstrained)),
            reaction_momentum_residual_norm=momentum_residual,
            degenerate_clearance_gradient_count=int(degenerate),
            pcc_clearance_m=pcc_distance,
            capsule_clearance_m=capsule_distance,
            mujoco_continuum_target_clearance_m=mujoco_target_distance,
            pcc_constraint_active_count=int(np.sum(pcc_row_mask)),
            capsule_constraint_active_count=int(np.sum(capsule_row_mask)),
            pcc_binding_constraint_count=pcc_binding,
            capsule_binding_constraint_count=capsule_binding,
            pcc_avoidance_intervention=pcc_intervention,
            pcc_mujoco_distance_error_m=pcc_distance_error,
            pcc_mujoco_gradient_error_norm=pcc_gradient_error,
            capsule_mujoco_gradient_error_norm=capsule_gradient_error,
            pcc_closest_segment_id=(
                clearance_block.pcc.closest_segment_id
                if clearance_block.pcc is not None
                else -1
            ),
            pcc_closest_arclength_m=(
                clearance_block.pcc.closest_arclength_m
                if clearance_block.pcc is not None
                else float("nan")
            ),
            shape_clearance_latency_s=(
                clearance_block.shape_clearance_latency_s
            ),
        )
