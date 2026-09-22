"""Versioned geometric contract for the V6.1 continuum shape stack.

This module is the single authority for the five-section PCC geometry, the
10-to-60 joint distribution, the floating-base installation transform, and
the declared audit domain.  The legacy V6 controller remains numerically
unchanged; this contract validates that its existing robot specification uses
the same joint map and tip offset.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model_test.robot_model_spec_v5 import (
    RobotModelSpecV5,
    default_robot_model_spec_v5,
)


CONTINUUM_MODEL_SPEC_VERSION = "v6.1-a.1"
CONTINUUM_SEGMENT_COUNT = 5
CONTINUUM_MODULE_COUNT = 30
CONTINUUM_ACTUATED_DOF = 60
CONTINUUM_PLANNER_DOF = 10
CONTINUUM_BASE_BODY_NAME = "base_of_satelltte"
CONTINUUM_MOUNT_BODY_NAME = "base_link"
CONTINUUM_TIP_BODY_NAME = "link_30"
CONTINUUM_END_EFFECTOR_BODY_NAME = "end_effector"
CONTINUUM_TIP_OFFSET_M = np.asarray([0.0475, 0.0, 0.0], dtype=np.float64)
CONTINUUM_TIP_OFFSET_M.setflags(write=False)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _vector(text: str | None, *, expected: int = 3) -> np.ndarray:
    if text is None:
        return np.zeros(expected, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (expected,) or np.any(~np.isfinite(value)):
        raise ValueError(f"invalid {expected}-vector {text!r}")
    return value


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def urdf_origin_transform(element: ET.Element | None) -> np.ndarray:
    """Return the URDF parent-to-joint transform (fixed-axis RPY)."""

    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform
    xyz = _vector(element.attrib.get("xyz"))
    roll, pitch, yaw = _vector(element.attrib.get("rpy"))
    transform[:3, :3] = (
        _rotation_z(float(yaw))
        @ _rotation_y(float(pitch))
        @ _rotation_x(float(roll))
    )
    transform[:3, 3] = xyz
    return transform


def _path_to_body(
    root: ET.Element, base_body: str, child_body: str
) -> tuple[ET.Element, ...]:
    child_to_joint: dict[str, ET.Element] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or "link" not in child.attrib:
            raise ValueError("URDF joint is missing its child link")
        child_to_joint[child.attrib["link"]] = joint
    path: list[ET.Element] = []
    current = child_body
    while current != base_body:
        if current not in child_to_joint:
            raise ValueError(f"{child_body!r} is not below {base_body!r}")
        joint = child_to_joint[current]
        path.append(joint)
        parent = joint.find("parent")
        if parent is None or "link" not in parent.attrib:
            raise ValueError("URDF joint is missing its parent link")
        current = parent.attrib["link"]
    path.reverse()
    return tuple(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LowDimensionalProjection:
    planner_configuration: np.ndarray
    orthogonal_residual: np.ndarray
    residual_l2_rad: float
    residual_linf_rad: float


@dataclass(frozen=True)
class ContinuumModelSpec:
    version: str
    source_urdf: Path
    source_urdf_sha256: str
    base_body_name: str
    mount_body_name: str
    tip_body_name: str
    end_effector_body_name: str
    base_to_shape_start: np.ndarray
    segment_lengths_m: np.ndarray
    module_lengths_m: np.ndarray
    pcc_bending_map: np.ndarray
    planner_to_actuated: np.ndarray
    actuated_to_planner: np.ndarray
    low_level_joint_names: tuple[str, ...]
    module_body_names: tuple[str, ...]
    work_domain_lower_rad: np.ndarray
    work_domain_upper_rad: np.ndarray
    tip_offset_m: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "base_to_shape_start",
            "segment_lengths_m",
            "module_lengths_m",
            "pcc_bending_map",
            "planner_to_actuated",
            "actuated_to_planner",
            "work_domain_lower_rad",
            "work_domain_upper_rad",
            "tip_offset_m",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "source_urdf", Path(self.source_urdf))
        self.validate()

    @property
    def total_length_m(self) -> float:
        return float(np.sum(self.segment_lengths_m))

    @property
    def segment_boundaries_m(self) -> np.ndarray:
        return np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(self.segment_lengths_m)]
        )

    @property
    def module_boundaries_m(self) -> np.ndarray:
        return np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(self.module_lengths_m)]
        )

    def validate(self) -> None:
        expected_shapes = {
            "base_to_shape_start": (4, 4),
            "segment_lengths_m": (CONTINUUM_SEGMENT_COUNT,),
            "module_lengths_m": (CONTINUUM_MODULE_COUNT,),
            "pcc_bending_map": (3, 2),
            "planner_to_actuated": (
                CONTINUUM_ACTUATED_DOF,
                CONTINUUM_PLANNER_DOF,
            ),
            "actuated_to_planner": (
                CONTINUUM_PLANNER_DOF,
                CONTINUUM_ACTUATED_DOF,
            ),
            "work_domain_lower_rad": (CONTINUUM_PLANNER_DOF,),
            "work_domain_upper_rad": (CONTINUUM_PLANNER_DOF,),
            "tip_offset_m": (3,),
        }
        for name, shape in expected_shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
        if self.version != CONTINUUM_MODEL_SPEC_VERSION:
            raise ValueError("unexpected continuum model contract version")
        if not self.source_urdf.is_file():
            raise FileNotFoundError(self.source_urdf)
        if _sha256(self.source_urdf) != self.source_urdf_sha256:
            raise ValueError("continuum source URDF hash changed")
        if len(self.low_level_joint_names) != CONTINUUM_ACTUATED_DOF:
            raise ValueError("continuum contract must name exactly 60 joints")
        if len(set(self.low_level_joint_names)) != CONTINUUM_ACTUATED_DOF:
            raise ValueError("continuum joint names must be unique")
        if len(self.module_body_names) != CONTINUUM_MODULE_COUNT:
            raise ValueError("continuum contract must name exactly 30 modules")
        if np.any(self.segment_lengths_m <= 0.0) or np.any(
            self.module_lengths_m <= 0.0
        ):
            raise ValueError("continuum lengths must be positive")
        grouped = self.module_lengths_m.reshape(CONTINUUM_SEGMENT_COUNT, 6).sum(1)
        if not np.allclose(grouped, self.segment_lengths_m, atol=1e-12, rtol=0.0):
            raise ValueError("segment lengths do not match their six modules")
        if not np.allclose(
            self.actuated_to_planner @ self.planner_to_actuated,
            np.eye(CONTINUUM_PLANNER_DOF),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("10-to-60 map does not have the declared left inverse")
        if np.any(self.work_domain_lower_rad >= self.work_domain_upper_rad):
            raise ValueError("invalid declared PCC work domain")
        rotation = self.base_to_shape_start[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
            raise ValueError("base-to-shape-start rotation is not orthonormal")
        if not np.allclose(self.base_to_shape_start[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("base-to-shape-start must be homogeneous")

    def project_actual_configuration(
        self, actuated_configuration: np.ndarray
    ) -> LowDimensionalProjection:
        actual = np.asarray(actuated_configuration, dtype=np.float64)
        if actual.shape != (CONTINUUM_ACTUATED_DOF,) or np.any(~np.isfinite(actual)):
            raise ValueError("actual continuum configuration must be finite (60,)")
        planner = self.actuated_to_planner @ actual
        residual = actual - self.planner_to_actuated @ planner
        return LowDimensionalProjection(
            planner_configuration=planner,
            orthogonal_residual=residual,
            residual_l2_rad=float(np.linalg.norm(residual)),
            residual_linf_rad=float(np.max(np.abs(residual))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_urdf": self.source_urdf.as_posix(),
            "source_urdf_sha256": self.source_urdf_sha256,
            "base_body_name": self.base_body_name,
            "mount_body_name": self.mount_body_name,
            "tip_body_name": self.tip_body_name,
            "end_effector_body_name": self.end_effector_body_name,
            "base_to_shape_start": self.base_to_shape_start.tolist(),
            "segment_lengths_m": self.segment_lengths_m.tolist(),
            "module_lengths_m": self.module_lengths_m.tolist(),
            "pcc_bending_map": self.pcc_bending_map.tolist(),
            "planner_to_actuated": self.planner_to_actuated.tolist(),
            "actuated_to_planner": self.actuated_to_planner.tolist(),
            "low_level_joint_names": list(self.low_level_joint_names),
            "module_body_names": list(self.module_body_names),
            "work_domain_lower_rad": self.work_domain_lower_rad.tolist(),
            "work_domain_upper_rad": self.work_domain_upper_rad.tolist(),
            "tip_offset_m": self.tip_offset_m.tolist(),
            "angle_unit": "radian",
            "pcc_tangent_axis": "+x",
        }

    def contract_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def default_continuum_model_spec(
    robot_spec: RobotModelSpecV5 | None = None,
) -> ContinuumModelSpec:
    """Build the V6.1-A contract from the audited URDF and V5 joint map."""

    robot = default_robot_model_spec_v5() if robot_spec is None else robot_spec
    robot.validate()
    source_urdf = Path(robot.source_urdf)
    root = ET.parse(source_urdf).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    base_to_mount = np.eye(4, dtype=np.float64)
    for joint in _path_to_body(
        root, CONTINUUM_BASE_BODY_NAME, CONTINUUM_MOUNT_BODY_NAME
    ):
        if joint.attrib.get("type") != "fixed":
            raise ValueError("continuum mounting chain must be fixed")
        base_to_mount = base_to_mount @ urdf_origin_transform(joint.find("origin"))
    first_joint = joints.get("base_link_2_joint_1")
    if first_joint is None:
        raise ValueError("missing first continuum joint")
    base_to_shape_start = base_to_mount @ urdf_origin_transform(
        first_joint.find("origin")
    )

    joint_by_edge = {
        (
            joint.find("parent").attrib["link"],
            joint.find("child").attrib["link"],
        ): joint
        for joint in root.findall("joint")
    }
    module_lengths: list[float] = []
    for index in range(1, CONTINUUM_MODULE_COUNT):
        joint = joint_by_edge.get((f"link_{index}", f"joint_{index + 1}"))
        if joint is None:
            raise ValueError(f"missing module spacing joint after link_{index}")
        translation = urdf_origin_transform(joint.find("origin"))[:3, 3]
        if not np.allclose(translation[1:], 0.0, atol=1e-12):
            raise ValueError("module spacing is not aligned with local +x")
        module_lengths.append(float(translation[0]))
    end_joint = joints.get("end_effector_joint")
    if end_joint is None:
        raise ValueError("missing continuum end-effector joint")
    end_translation = urdf_origin_transform(end_joint.find("origin"))[:3, 3]
    if not np.allclose(end_translation, CONTINUUM_TIP_OFFSET_M, atol=1e-12):
        raise ValueError("URDF end-effector offset changed")
    module_lengths.append(float(end_translation[0]))
    module_lengths_array = np.asarray(module_lengths, dtype=np.float64)
    segment_lengths = module_lengths_array.reshape(CONTINUUM_SEGMENT_COUNT, 6).sum(1)

    planner_to_actuated = np.asarray(
        robot.planner_to_low_level[:CONTINUUM_ACTUATED_DOF, :CONTINUUM_PLANNER_DOF],
        dtype=np.float64,
    )
    if np.any(
        np.abs(
            robot.planner_to_low_level[:CONTINUUM_ACTUATED_DOF, CONTINUUM_PLANNER_DOF:]
        )
        > 0.0
    ):
        raise ValueError("rigid planner coordinates leak into the continuum map")
    actuated_to_planner = np.linalg.pinv(planner_to_actuated)

    # q[2i] bends about local -Y; q[2i+1] bends about local +Z.  The map
    # describes total section bending, not curvature; division by section
    # length is performed by ContinuumShapeModel.
    bending_map = np.asarray(
        [[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    domain = np.ones(CONTINUUM_PLANNER_DOF, dtype=np.float64)
    return ContinuumModelSpec(
        version=CONTINUUM_MODEL_SPEC_VERSION,
        source_urdf=source_urdf,
        source_urdf_sha256=_sha256(source_urdf),
        base_body_name=CONTINUUM_BASE_BODY_NAME,
        mount_body_name=CONTINUUM_MOUNT_BODY_NAME,
        tip_body_name=CONTINUUM_TIP_BODY_NAME,
        end_effector_body_name=CONTINUUM_END_EFFECTOR_BODY_NAME,
        base_to_shape_start=base_to_shape_start,
        segment_lengths_m=segment_lengths,
        module_lengths_m=module_lengths_array,
        pcc_bending_map=bending_map,
        planner_to_actuated=planner_to_actuated,
        actuated_to_planner=actuated_to_planner,
        low_level_joint_names=tuple(
            robot.low_level_joint_names[:CONTINUUM_ACTUATED_DOF]
        ),
        module_body_names=tuple(
            f"link_{index}" for index in range(1, CONTINUUM_MODULE_COUNT + 1)
        ),
        work_domain_lower_rad=-domain,
        work_domain_upper_rad=domain,
        tip_offset_m=CONTINUUM_TIP_OFFSET_M,
    )
