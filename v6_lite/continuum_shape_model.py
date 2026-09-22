"""Analytic five-section PCC and independent URDF-chain kinematics.

The PCC model is a continuous proxy.  The discrete model exactly follows the
URDF joint origins and axes and is retained as the implementation-consistency
reference.  Their difference is reported as model discrepancy; it is never
silently treated as numerical error.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from scipy.linalg import expm_frechet

from v6_lite.continuum_model_spec import (
    CONTINUUM_ACTUATED_DOF,
    CONTINUUM_MODULE_COUNT,
    CONTINUUM_PLANNER_DOF,
    ContinuumModelSpec,
    default_continuum_model_spec,
    urdf_origin_transform,
)


def skew(value: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    return 0.5 * np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("rotation axis must be nonzero")
    direction /= norm
    generator = skew(direction)
    return (
        np.eye(3)
        + np.sin(angle) * generator
        + (1.0 - np.cos(angle)) * (generator @ generator)
    )


def rotation_angle(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip(0.5 * (np.trace(matrix) - 1.0), -1.0, 1.0))
    return float(np.arccos(cosine))


def validate_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or np.any(~np.isfinite(value)):
        raise ValueError("base transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("base transform is not homogeneous")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise ValueError("base rotation is not orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("base rotation must be proper")
    return value


def transform_from_free_qpos(free_qpos: np.ndarray) -> np.ndarray:
    """Convert MuJoCo free-joint ``[xyz, wxyz]`` to a homogeneous transform."""

    value = np.asarray(free_qpos, dtype=np.float64)
    if value.shape != (7,) or np.any(~np.isfinite(value)):
        raise ValueError("free-joint qpos must be finite with shape (7,)")
    quaternion = value[3:].copy()
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("free-joint quaternion is zero")
    quaternion /= norm
    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, quaternion)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = flat.reshape(3, 3)
    transform[:3, 3] = value[:3]
    return transform


@dataclass(frozen=True)
class ShapeEvaluation:
    arclength_m: float
    segment_index: int
    segment_arclength_m: float
    position: np.ndarray
    rotation: np.ndarray
    position_jacobian: np.ndarray
    rotation_jacobian: np.ndarray


@dataclass(frozen=True)
class DiscreteShapeEvaluation:
    arclength_m: float
    module_index: int
    module_arclength_m: float
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class _UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class ContinuumShapeModel:
    """Five-section PCC model with stable zero-curvature derivatives."""

    def __init__(self, spec: ContinuumModelSpec | None = None) -> None:
        self.spec = default_continuum_model_spec() if spec is None else spec
        self.spec.validate()

    @staticmethod
    def _section_transform(
        bending: np.ndarray,
        length: float,
        arclength: float,
        bending_map: np.ndarray,
    ) -> np.ndarray:
        if arclength < -1e-12 or arclength > length + 1e-12:
            raise ValueError("section arclength lies outside the section")
        u = float(np.clip(arclength, 0.0, length))
        angular_strain = bending_map @ np.asarray(bending, dtype=np.float64) / length
        generator = skew(angular_strain)
        generator2 = generator @ generator
        magnitude = float(np.linalg.norm(angular_strain))
        # The generic coefficients subtract nearly equal trigonometric terms.
        # Select the series by the dimensionless bend angle, not by curvature
        # alone, so finite differences through zero remain smooth.
        if magnitude * max(u, length) <= 1e-4:
            magnitude2 = magnitude * magnitude
            a = u - magnitude2 * u**3 / 6.0 + magnitude2**2 * u**5 / 120.0
            b = u**2 / 2.0 - magnitude2 * u**4 / 24.0 + magnitude2**2 * u**6 / 720.0
            c = u**3 / 6.0 - magnitude2 * u**5 / 120.0 + magnitude2**2 * u**7 / 5040.0
        else:
            angle = magnitude * u
            a = np.sin(angle) / magnitude
            b = (1.0 - np.cos(angle)) / (magnitude * magnitude)
            c = (angle - np.sin(angle)) / (magnitude**3)
        rotation = np.eye(3) + a * generator + b * generator2
        integration = u * np.eye(3) + b * generator + c * generator2
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = integration @ np.asarray([1.0, 0.0, 0.0])
        return transform

    @staticmethod
    def _section_derivatives(
        bending: np.ndarray,
        length: float,
        arclength: float,
        bending_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        u = float(np.clip(arclength, 0.0, length))
        angular_strain = bending_map @ np.asarray(bending, dtype=np.float64) / length
        algebra = np.zeros((4, 4), dtype=np.float64)
        algebra[:3, :3] = skew(angular_strain)
        algebra[:3, 3] = [1.0, 0.0, 0.0]
        argument = algebra * u
        derivatives = []
        for column in range(2):
            direction = np.zeros((4, 4), dtype=np.float64)
            direction[:3, :3] = skew(bending_map[:, column] / length)
            derivatives.append(
                expm_frechet(argument, direction * u, compute_expm=False)
            )
        return derivatives[0], derivatives[1]

    def _segment_location(self, arclength: float) -> tuple[int, float]:
        total = self.spec.total_length_m
        if not np.isfinite(arclength) or arclength < -1e-12 or arclength > total + 1e-12:
            raise ValueError(f"arclength must lie in [0, {total}]")
        value = float(np.clip(arclength, 0.0, total))
        boundaries = self.spec.segment_boundaries_m
        if value >= total:
            return len(self.spec.segment_lengths_m) - 1, float(
                self.spec.segment_lengths_m[-1]
            )
        index = int(np.searchsorted(boundaries[1:], value, side="right"))
        return index, value - float(boundaries[index])

    def evaluate(
        self,
        planner_configuration: np.ndarray,
        base_transform: np.ndarray,
        arclength: float,
        *,
        with_jacobians: bool = True,
    ) -> ShapeEvaluation:
        configuration = np.asarray(planner_configuration, dtype=np.float64)
        if configuration.shape != (CONTINUUM_PLANNER_DOF,) or np.any(
            ~np.isfinite(configuration)
        ):
            raise ValueError("PCC configuration must be finite with shape (10,)")
        base = validate_transform(base_transform)
        segment_index, local_arclength = self._segment_location(float(arclength))
        transform = base @ self.spec.base_to_shape_start
        derivatives = [np.zeros((4, 4), dtype=np.float64) for _ in range(10)]
        for index in range(segment_index + 1):
            length = float(self.spec.segment_lengths_m[index])
            current_arclength = length if index < segment_index else local_arclength
            local = self._section_transform(
                configuration[2 * index : 2 * index + 2],
                length,
                current_arclength,
                self.spec.pcc_bending_map,
            )
            previous = transform
            previous_derivatives = [item.copy() for item in derivatives]
            transform = previous @ local
            for coordinate in range(10):
                derivatives[coordinate] = previous_derivatives[coordinate] @ local
            if with_jacobians:
                local_derivatives = self._section_derivatives(
                    configuration[2 * index : 2 * index + 2],
                    length,
                    current_arclength,
                    self.spec.pcc_bending_map,
                )
                derivatives[2 * index] += previous @ local_derivatives[0]
                derivatives[2 * index + 1] += previous @ local_derivatives[1]
        rotation = transform[:3, :3]
        position_jacobian = np.zeros((3, 10), dtype=np.float64)
        rotation_jacobian = np.zeros((3, 10), dtype=np.float64)
        if with_jacobians:
            for coordinate, derivative in enumerate(derivatives):
                position_jacobian[:, coordinate] = derivative[:3, 3]
                rotation_jacobian[:, coordinate] = vee(
                    derivative[:3, :3] @ rotation.T
                )
        return ShapeEvaluation(
            arclength_m=float(arclength),
            segment_index=segment_index,
            segment_arclength_m=local_arclength,
            position=transform[:3, 3].copy(),
            rotation=rotation.copy(),
            position_jacobian=position_jacobian,
            rotation_jacobian=rotation_jacobian,
        )

    def sample(
        self,
        planner_configuration: np.ndarray,
        base_transform: np.ndarray,
        arclengths: Iterable[float],
        *,
        with_jacobians: bool = False,
    ) -> tuple[ShapeEvaluation, ...]:
        return tuple(
            self.evaluate(
                planner_configuration,
                base_transform,
                float(arclength),
                with_jacobians=with_jacobians,
            )
            for arclength in arclengths
        )


class DiscreteContinuumKinematics:
    """Independent URDF forward kinematics for the actual 60-joint chain."""

    def __init__(self, spec: ContinuumModelSpec | None = None) -> None:
        self.spec = default_continuum_model_spec() if spec is None else spec
        self.spec.validate()
        self._chain = self._load_chain(Path(self.spec.source_urdf))
        revolute_names = tuple(
            joint.name for joint in self._chain if joint.joint_type == "revolute"
        )
        if revolute_names != self.spec.low_level_joint_names:
            raise ValueError("URDF continuum joint order differs from the contract")
        zero_transforms = self.body_transforms(
            np.zeros(CONTINUUM_ACTUATED_DOF, dtype=np.float64), np.eye(4)
        )
        canonical_rotation = self.spec.base_to_shape_start[:3, :3]
        self._module_frame_corrections = tuple(
            zero_transforms[body_name][:3, :3].T @ canonical_rotation
            for body_name in self.spec.module_body_names
        )

    def _load_chain(self, path: Path) -> tuple[_UrdfJoint, ...]:
        root = ET.parse(path).getroot()
        child_to_joint: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                child_to_joint[child.attrib["link"]] = joint
        elements: list[ET.Element] = []
        current = self.spec.end_effector_body_name
        while current != self.spec.base_body_name:
            if current not in child_to_joint:
                raise ValueError("continuum tip is disconnected from the floating base")
            joint = child_to_joint[current]
            elements.append(joint)
            current = joint.find("parent").attrib["link"]
        elements.reverse()
        chain: list[_UrdfJoint] = []
        for joint in elements:
            axis_element = joint.find("axis")
            axis = (
                np.asarray([1.0, 0.0, 0.0])
                if axis_element is None
                else np.fromstring(axis_element.attrib["xyz"], sep=" ")
            )
            chain.append(
                _UrdfJoint(
                    name=joint.attrib["name"],
                    joint_type=joint.attrib["type"],
                    parent=joint.find("parent").attrib["link"],
                    child=joint.find("child").attrib["link"],
                    origin=urdf_origin_transform(joint.find("origin")),
                    axis=np.asarray(axis, dtype=np.float64),
                )
            )
        return tuple(chain)

    def body_transforms(
        self, actuated_configuration: np.ndarray, base_transform: np.ndarray
    ) -> dict[str, np.ndarray]:
        actual = np.asarray(actuated_configuration, dtype=np.float64)
        if actual.shape != (CONTINUUM_ACTUATED_DOF,) or np.any(~np.isfinite(actual)):
            raise ValueError("discrete configuration must be finite with shape (60,)")
        transform = validate_transform(base_transform).copy()
        result = {self.spec.base_body_name: transform.copy()}
        revolute_index = 0
        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.joint_type == "revolute":
                rotation = np.eye(4, dtype=np.float64)
                rotation[:3, :3] = axis_angle_rotation(
                    joint.axis, float(actual[revolute_index])
                )
                transform = transform @ rotation
                revolute_index += 1
            elif joint.joint_type != "fixed":
                raise ValueError(f"unsupported continuum joint type {joint.joint_type!r}")
            result[joint.child] = transform.copy()
        if revolute_index != CONTINUUM_ACTUATED_DOF:
            raise RuntimeError("discrete FK did not consume all continuum joints")
        return result

    def _module_location(self, arclength: float) -> tuple[int, float]:
        total = self.spec.total_length_m
        if not np.isfinite(arclength) or arclength < -1e-12 or arclength > total + 1e-12:
            raise ValueError(f"arclength must lie in [0, {total}]")
        value = float(np.clip(arclength, 0.0, total))
        if value >= total:
            return CONTINUUM_MODULE_COUNT - 1, float(self.spec.module_lengths_m[-1])
        ends = np.cumsum(self.spec.module_lengths_m)
        index = int(np.searchsorted(ends, value, side="left"))
        start = 0.0 if index == 0 else float(ends[index - 1])
        return index, value - start

    def evaluate(
        self,
        actuated_configuration: np.ndarray,
        base_transform: np.ndarray,
        arclength: float,
        *,
        transforms: dict[str, np.ndarray] | None = None,
    ) -> DiscreteShapeEvaluation:
        module_index, local_arclength = self._module_location(float(arclength))
        values = (
            self.body_transforms(actuated_configuration, base_transform)
            if transforms is None
            else transforms
        )
        body_name = self.spec.module_body_names[module_index]
        transform = values[body_name]
        position = transform[:3, 3] + transform[:3, :3] @ np.asarray(
            [local_arclength, 0.0, 0.0]
        )
        return DiscreteShapeEvaluation(
            arclength_m=float(arclength),
            module_index=module_index,
            module_arclength_m=local_arclength,
            position=position,
            # URDF universal-joint helper bodies alternate their local Y/Z
            # axes by 90 degrees.  Remove that coordinate-only alternation so
            # the reported backbone frame is continuous at zero curvature.
            rotation=(
                transform[:3, :3]
                @ self._module_frame_corrections[module_index]
            ),
        )
