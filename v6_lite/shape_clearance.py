"""Finite-volume shadow geometry for V6.1-A.

The module builds conservative body-fixed capsules around every distal
continuum collision geometry, evaluates capsule/OBB and PCC-tube/OBB
clearance, and keeps MuJoCo geometry distance as the reference.  Nothing in
this module is connected to the V6 control command path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import mujoco
import numpy as np

from v6_lite.continuum_model_spec import (
    CONTINUUM_ACTUATED_DOF,
    ContinuumModelSpec,
    default_continuum_model_spec,
)
from v6_lite.continuum_shape_model import ContinuumShapeModel


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, np.asarray(quaternion, dtype=np.float64))
    return flat.reshape(3, 3)


def _point_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    squared = float(direction @ direction)
    if squared <= 1e-24:
        return np.linalg.norm(values - start, axis=-1)
    parameter = np.clip(((values - start) @ direction) / squared, 0.0, 1.0)
    closest = start + parameter[..., None] * direction
    return np.linalg.norm(values - closest, axis=-1)


@dataclass(frozen=True)
class OrientedBox:
    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _readonly(self.center))
        object.__setattr__(self, "rotation", _readonly(self.rotation))
        object.__setattr__(self, "half_extents", _readonly(self.half_extents))
        if self.center.shape != (3,):
            raise ValueError("OBB center must have shape (3,)")
        if self.rotation.shape != (3, 3):
            raise ValueError("OBB rotation must have shape (3,3)")
        if self.half_extents.shape != (3,) or np.any(self.half_extents <= 0.0):
            raise ValueError("OBB half extents must be positive with shape (3,)")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-10):
            raise ValueError("OBB rotation must be orthonormal")


@dataclass(frozen=True)
class PointOBBResult:
    signed_distance_m: float
    point: np.ndarray
    box_point: np.ndarray
    normal_box_to_point: np.ndarray


@dataclass(frozen=True)
class SegmentOBBResult:
    signed_distance_m: float
    segment_parameter: float
    segment_point: np.ndarray
    box_point: np.ndarray
    normal_box_to_segment: np.ndarray


@dataclass(frozen=True)
class CapsuleEnvelope:
    name: str
    geom_name: str
    body_name: str
    body_id: int
    geom_id: int
    segment_index: int
    local_start: np.ndarray
    local_end: np.ndarray
    radius_m: float
    vertex_count: int
    maximum_vertex_excess_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_start", _readonly(self.local_start))
        object.__setattr__(self, "local_end", _readonly(self.local_end))
        if self.local_start.shape != (3,) or self.local_end.shape != (3,):
            raise ValueError("capsule endpoints must have shape (3,)")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("capsule radius must be positive")
        if self.segment_index not in range(5):
            raise ValueError("capsule segment index must lie in [0,4]")
        if self.maximum_vertex_excess_m > 1e-10:
            raise ValueError("capsule does not cover all source vertices")

    @property
    def axis_length_m(self) -> float:
        return float(np.linalg.norm(self.local_end - self.local_start))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["local_start"] = self.local_start.tolist()
        payload["local_end"] = self.local_end.tolist()
        payload["axis_length_m"] = self.axis_length_m
        return payload


@dataclass(frozen=True)
class CapsuleEnvelopeSet:
    capsules: tuple[CapsuleEnvelope, ...]
    fallback_geom_names: tuple[str, ...]
    continuum_geom_names: tuple[str, ...]

    def validate(self) -> None:
        assigned = {item.geom_name for item in self.capsules}
        fallback = set(self.fallback_geom_names)
        expected = set(self.continuum_geom_names)
        if assigned & fallback:
            raise ValueError("a geometry cannot be both capsule and fallback")
        if assigned | fallback != expected:
            missing = expected - (assigned | fallback)
            extra = (assigned | fallback) - expected
            raise ValueError(f"geometry ownership mismatch: missing={missing}, extra={extra}")
        if len(assigned) != len(self.capsules):
            raise ValueError("each geometry must have exactly one capsule")


@dataclass(frozen=True)
class ClearanceResult:
    signed_distance_m: float
    point_on_arm: np.ndarray
    point_on_box: np.ndarray
    normal_box_to_arm: np.ndarray
    planner_gradient: np.ndarray
    source_name: str
    source_index: int
    query_truncated: bool = False


@dataclass(frozen=True)
class ShadowClearanceComparison:
    capsule: ClearanceResult
    pcc_tube: ClearanceResult
    mujoco_geometry: ClearanceResult


def point_obb_signed_distance(point: np.ndarray, box: OrientedBox) -> PointOBBResult:
    world_point = np.asarray(point, dtype=np.float64).reshape(3)
    local = box.rotation.T @ (world_point - box.center)
    closest_local = np.clip(local, -box.half_extents, box.half_extents)
    delta_local = local - closest_local
    outside_distance = float(np.linalg.norm(delta_local))
    if outside_distance > 1e-14:
        local_normal = delta_local / outside_distance
        signed_distance = outside_distance
    else:
        margins = box.half_extents - np.abs(local)
        axis = int(np.argmin(margins))
        sign = 1.0 if local[axis] >= 0.0 else -1.0
        local_normal = np.zeros(3, dtype=np.float64)
        local_normal[axis] = sign
        closest_local = local.copy()
        closest_local[axis] = sign * box.half_extents[axis]
        signed_distance = -float(margins[axis])
    normal = box.rotation @ local_normal
    return PointOBBResult(
        signed_distance_m=signed_distance,
        point=world_point.copy(),
        box_point=box.center + box.rotation @ closest_local,
        normal_box_to_point=normal,
    )


def segment_obb_signed_distance(
    start: np.ndarray, end: np.ndarray, box: OrientedBox
) -> SegmentOBBResult:
    """Exact exterior distance from a segment to an OBB.

    The squared exterior point-to-box distance is piecewise quadratic in the
    segment parameter.  Testing every box-boundary breakpoint and each
    interval's stationary point therefore gives its global minimum.  For an
    intersecting segment, the returned signed value is a conservative unsafe
    indicator obtained from interior candidates.
    """

    world_start = np.asarray(start, dtype=np.float64).reshape(3)
    world_end = np.asarray(end, dtype=np.float64).reshape(3)
    local_start = box.rotation.T @ (world_start - box.center)
    local_direction = box.rotation.T @ (world_end - world_start)
    breaks = [0.0, 1.0]
    for axis in range(3):
        velocity = float(local_direction[axis])
        if abs(velocity) <= 1e-15:
            continue
        for bound in (-box.half_extents[axis], box.half_extents[axis]):
            value = float((bound - local_start[axis]) / velocity)
            if 0.0 < value < 1.0:
                breaks.append(value)
    boundaries = np.asarray(sorted(set(breaks)), dtype=np.float64)
    candidates = set(float(item) for item in boundaries)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        midpoint = 0.5 * (float(lower) + float(upper))
        candidates.add(midpoint)
        location = local_start + midpoint * local_direction
        active: list[tuple[int, float]] = []
        for axis in range(3):
            if location[axis] < -box.half_extents[axis]:
                active.append((axis, -float(box.half_extents[axis])))
            elif location[axis] > box.half_extents[axis]:
                active.append((axis, float(box.half_extents[axis])))
        denominator = sum(float(local_direction[axis] ** 2) for axis, _ in active)
        if denominator > 1e-24:
            numerator = sum(
                float(local_direction[axis] * (local_start[axis] - bound))
                for axis, bound in active
            )
            stationary = float(np.clip(-numerator / denominator, lower, upper))
            candidates.add(stationary)
    results = [
        (
            value,
            point_obb_signed_distance(
                world_start + value * (world_end - world_start), box
            ),
        )
        for value in sorted(candidates)
    ]
    exterior = [item for item in results if item[1].signed_distance_m > 0.0]
    interior = [item for item in results if item[1].signed_distance_m <= 0.0]
    if interior:
        parameter, result = min(interior, key=lambda item: item[1].signed_distance_m)
    elif exterior:
        parameter, result = min(exterior, key=lambda item: item[1].signed_distance_m)
    else:  # pragma: no cover - candidates always contain endpoints
        raise RuntimeError("segment/OBB minimization produced no candidate")
    return SegmentOBBResult(
        signed_distance_m=float(result.signed_distance_m),
        segment_parameter=float(parameter),
        segment_point=result.point,
        box_point=result.box_point,
        normal_box_to_segment=result.normal_box_to_point,
    )


def _geom_vertices_in_body(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    geom_position = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
    geom_rotation = _rotation_from_quaternion(model.geom_quat[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        vertices = np.asarray(
            [
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )
    else:
        raise ValueError(f"unsupported continuum collision geom type {geom_type}")
    return geom_position + vertices @ geom_rotation.T


def _fit_capsule(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    points = np.asarray(vertices, dtype=np.float64)
    center = np.mean(points, axis=0)
    covariance = (points - center).T @ (points - center)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    projections = (points - center) @ axis
    start = center + float(np.min(projections)) * axis
    end = center + float(np.max(projections)) * axis
    distances = _point_segment_distance(points, start, end)
    radius = float(np.max(distances)) + 1e-9
    excess = float(np.max(distances - radius))
    return start, end, radius, excess


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _segment_for_body(body_name: str) -> int:
    if body_name == "end_effector":
        return 4
    for prefix in ("joint_", "link_"):
        if body_name.startswith(prefix):
            index = int(body_name[len(prefix) :])
            return (index - 1) // 6
    raise ValueError(f"cannot assign continuum body {body_name!r} to a PCC section")


def build_continuum_capsule_envelopes(
    model: mujoco.MjModel,
    spec: ContinuumModelSpec | None = None,
) -> CapsuleEnvelopeSet:
    contract = default_continuum_model_spec() if spec is None else spec
    root = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, contract.mount_body_name)
    )
    tip = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, contract.end_effector_body_name
        )
    )
    if root < 0 or tip < 0:
        raise ValueError("continuum root or tip body is missing")
    body_ids: set[int] = set()
    current = tip
    while current != 0:
        body_ids.add(current)
        if current == root:
            break
        current = int(model.body_parentid[current])
    if root not in body_ids:
        raise ValueError("continuum tip is not below its mount body")
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
        and (
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        )
    ]
    capsules: list[CapsuleEnvelope] = []
    fallback: list[str] = []
    for geom_id in geom_ids:
        body_id = int(model.geom_bodyid[geom_id])
        body_name = _body_name(model, body_id)
        geom_name = _geom_name(model, geom_id)
        # The mounting block remains under the exact MuJoCo policy.  Every
        # distal joint, link and terminal box receives one explicit capsule.
        if body_name == contract.mount_body_name:
            fallback.append(geom_name)
            continue
        vertices = _geom_vertices_in_body(model, geom_id)
        start, end, radius, excess = _fit_capsule(vertices)
        capsules.append(
            CapsuleEnvelope(
                name=f"capsule_{geom_name}",
                geom_name=geom_name,
                body_name=body_name,
                body_id=body_id,
                geom_id=geom_id,
                segment_index=_segment_for_body(body_name),
                local_start=start,
                local_end=end,
                radius_m=radius,
                vertex_count=int(vertices.shape[0]),
                maximum_vertex_excess_m=excess,
            )
        )
    result = CapsuleEnvelopeSet(
        capsules=tuple(capsules),
        fallback_geom_names=tuple(sorted(fallback)),
        continuum_geom_names=tuple(sorted(_geom_name(model, item) for item in geom_ids)),
    )
    result.validate()
    return result


def target_box_from_mujoco(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> OrientedBox:
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("target collision geometry must be a box")
    return OrientedBox(
        center=np.asarray(data.geom_xpos[geom_id], dtype=np.float64),
        rotation=np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3),
        half_extents=np.asarray(model.geom_size[geom_id], dtype=np.float64),
    )


def capsule_clearance_to_obb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    capsule: CapsuleEnvelope,
    box: OrientedBox,
    *,
    continuum_dof_ids: np.ndarray | None = None,
    planner_to_actuated: np.ndarray | None = None,
) -> ClearanceResult:
    body_rotation = np.asarray(data.xmat[capsule.body_id], dtype=np.float64).reshape(3, 3)
    body_position = np.asarray(data.xpos[capsule.body_id], dtype=np.float64)
    start = body_position + body_rotation @ capsule.local_start
    end = body_position + body_rotation @ capsule.local_end
    centerline = segment_obb_signed_distance(start, end, box)
    point_on_capsule = centerline.segment_point - (
        capsule.radius_m * centerline.normal_box_to_segment
    )
    gradient = np.zeros(10, dtype=np.float64)
    if continuum_dof_ids is not None and planner_to_actuated is not None:
        local_point = (
            (1.0 - centerline.segment_parameter) * capsule.local_start
            + centerline.segment_parameter * capsule.local_end
        )
        world_point = body_position + body_rotation @ local_point
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model,
            data,
            jacobian,
            None,
            world_point,
            capsule.body_id,
        )
        gradient = (
            centerline.normal_box_to_segment
            @ jacobian[:, np.asarray(continuum_dof_ids, dtype=np.int32)]
            @ np.asarray(planner_to_actuated, dtype=np.float64)
        )
    return ClearanceResult(
        signed_distance_m=float(centerline.signed_distance_m - capsule.radius_m),
        point_on_arm=point_on_capsule,
        point_on_box=centerline.box_point,
        normal_box_to_arm=centerline.normal_box_to_segment,
        planner_gradient=np.asarray(gradient, dtype=np.float64),
        source_name=capsule.geom_name,
        source_index=capsule.geom_id,
    )


def minimum_capsule_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    envelopes: CapsuleEnvelopeSet,
    box: OrientedBox,
    *,
    spec: ContinuumModelSpec | None = None,
) -> ClearanceResult:
    contract = default_continuum_model_spec() if spec is None else spec
    dof_ids = []
    for joint_name in contract.low_level_joint_names:
        joint_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        )
        if joint_id < 0:
            raise ValueError(f"missing continuum joint {joint_name!r}")
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
    values = [
        capsule_clearance_to_obb(
            model,
            data,
            capsule,
            box,
        )
        for capsule in envelopes.capsules
    ]
    minimum_index = int(np.argmin([item.signed_distance_m for item in values]))
    return capsule_clearance_to_obb(
        model,
        data,
        envelopes.capsules[minimum_index],
        box,
        continuum_dof_ids=np.asarray(dof_ids, dtype=np.int32),
        planner_to_actuated=contract.planner_to_actuated,
    )


def pcc_tube_clearance(
    shape_model: ContinuumShapeModel,
    planner_configuration: np.ndarray,
    base_transform: np.ndarray,
    box: OrientedBox,
    tube_radii_m: np.ndarray,
    *,
    samples_per_segment: int = 25,
) -> ClearanceResult:
    if samples_per_segment < 3:
        raise ValueError("at least three PCC samples per segment are required")
    radii = np.asarray(tube_radii_m, dtype=np.float64)
    if radii.shape != (5,) or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("PCC tube radii must be finite and positive with shape (5,)")
    candidates: list[tuple[ClearanceResult, float]] = []
    boundaries = shape_model.spec.segment_boundaries_m
    sample_index = 0
    for segment_index in range(5):
        local_values = np.linspace(
            0.0,
            float(shape_model.spec.segment_lengths_m[segment_index]),
            samples_per_segment,
        )
        if segment_index > 0:
            local_values = local_values[1:]
        for local in local_values:
            arclength = float(boundaries[segment_index] + local)
            evaluation = shape_model.evaluate(
                planner_configuration,
                base_transform,
                arclength,
                with_jacobians=False,
            )
            point_result = point_obb_signed_distance(evaluation.position, box)
            candidates.append(
                (
                    ClearanceResult(
                        signed_distance_m=float(
                            point_result.signed_distance_m - radii[segment_index]
                        ),
                        point_on_arm=(
                            evaluation.position
                            - radii[segment_index]
                            * point_result.normal_box_to_point
                        ),
                        point_on_box=point_result.box_point,
                        normal_box_to_arm=point_result.normal_box_to_point,
                        planner_gradient=np.zeros(10, dtype=np.float64),
                        source_name=f"pcc_segment_{segment_index + 1}",
                        source_index=sample_index,
                    ),
                    arclength,
                )
            )
            sample_index += 1
    minimum, minimum_arclength = min(
        candidates, key=lambda item: item[0].signed_distance_m
    )
    evaluation = shape_model.evaluate(
        planner_configuration,
        base_transform,
        minimum_arclength,
        with_jacobians=True,
    )
    return ClearanceResult(
        signed_distance_m=minimum.signed_distance_m,
        point_on_arm=minimum.point_on_arm,
        point_on_box=minimum.point_on_box,
        normal_box_to_arm=minimum.normal_box_to_arm,
        planner_gradient=minimum.normal_box_to_arm @ evaluation.position_jacobian,
        source_name=minimum.source_name,
        source_index=minimum.source_index,
    )


def minimum_mujoco_geom_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: Sequence[int],
    target_geom_id: int,
    *,
    query_distance_max_m: float = 0.5,
) -> ClearanceResult:
    if query_distance_max_m <= 0.0:
        raise ValueError("query maximum must be positive")
    best_distance = float("inf")
    best_geom = -1
    best_fromto = np.zeros(6, dtype=np.float64)
    best_truncated = True
    for geom_id in geom_ids:
        fromto = np.zeros(6, dtype=np.float64)
        raw_distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                int(geom_id),
                int(target_geom_id),
                float(query_distance_max_m),
                fromto,
            )
        )
        truncated = raw_distance >= query_distance_max_m - 1e-12
        distance = query_distance_max_m if truncated else raw_distance
        if distance < best_distance:
            best_distance = distance
            best_geom = int(geom_id)
            best_fromto = fromto.copy()
            best_truncated = truncated
    if best_geom < 0:
        raise ValueError("no MuJoCo geometry IDs were supplied")
    point_arm = best_fromto[:3]
    point_box = best_fromto[3:]
    delta = point_arm - point_box
    norm = float(np.linalg.norm(delta))
    normal = delta / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
    return ClearanceResult(
        signed_distance_m=best_distance,
        point_on_arm=point_arm,
        point_on_box=point_box,
        normal_box_to_arm=normal,
        planner_gradient=np.zeros(10, dtype=np.float64),
        source_name=_geom_name(model, best_geom),
        source_index=best_geom,
        query_truncated=best_truncated,
    )


class ShapeClearanceShadow:
    """Read-only comparison of PCC, capsules, and exact MuJoCo geometry."""

    def __init__(
        self,
        model: mujoco.MjModel,
        target_geom_id: int,
        tube_radii_m: np.ndarray,
        *,
        spec: ContinuumModelSpec | None = None,
        samples_per_segment: int = 25,
        query_distance_max_m: float = 0.5,
    ) -> None:
        self.model = model
        self.spec = default_continuum_model_spec() if spec is None else spec
        self.shape_model = ContinuumShapeModel(self.spec)
        self.envelopes = build_continuum_capsule_envelopes(model, self.spec)
        self.target_geom_id = int(target_geom_id)
        self.tube_radii_m = np.asarray(tube_radii_m, dtype=np.float64).copy()
        self.samples_per_segment = int(samples_per_segment)
        self.query_distance_max_m = float(query_distance_max_m)
        fallback = set(self.envelopes.fallback_geom_names)
        self.enveloped_geom_ids = tuple(
            capsule.geom_id for capsule in self.envelopes.capsules
        )
        self.fallback_geom_ids = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if _geom_name(model, geom_id) in fallback
        )

    def evaluate(
        self,
        data: mujoco.MjData,
        planner_configuration: np.ndarray,
        base_transform: np.ndarray,
    ) -> ShadowClearanceComparison:
        qpos_before = data.qpos.copy()
        qvel_before = data.qvel.copy()
        ctrl_before = data.ctrl.copy()
        box = target_box_from_mujoco(self.model, data, self.target_geom_id)
        capsule = minimum_capsule_clearance(
            self.model, data, self.envelopes, box, spec=self.spec
        )
        pcc = pcc_tube_clearance(
            self.shape_model,
            planner_configuration,
            base_transform,
            box,
            self.tube_radii_m,
            samples_per_segment=self.samples_per_segment,
        )
        exact = minimum_mujoco_geom_clearance(
            self.model,
            data,
            self.enveloped_geom_ids,
            self.target_geom_id,
            query_distance_max_m=self.query_distance_max_m,
        )
        if not (
            np.array_equal(data.qpos, qpos_before)
            and np.array_equal(data.qvel, qvel_before)
            and np.array_equal(data.ctrl, ctrl_before)
        ):
            raise RuntimeError("shadow clearance evaluation mutated simulation state")
        return ShadowClearanceComparison(
            capsule=capsule,
            pcc_tube=pcc,
            mujoco_geometry=exact,
        )
