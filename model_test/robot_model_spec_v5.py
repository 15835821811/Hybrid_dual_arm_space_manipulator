"""Versioned physical model contract for the V5.1 closed-loop baseline.

This file is the single production authority for the meanings of planner
coordinates, MuJoCo generalized coordinates/velocities, and control inputs.
Validation code intentionally cross-checks it through MuJoCo-native finite
differences and actuator responses instead of reusing encode/decode methods.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import mujoco
import numpy as np

try:
    from .angle_convention import (
        CONTINUUM_INPUT_ORDER_10,
        INPUT_ORDER_17,
        RIGID_INPUT_ORDER_7,
        RIGID_HOME_DEG,
        continuum_mujoco_map_60_to_10,
    )
except ImportError:  # pragma: no cover - direct script execution compatibility
    from angle_convention import (
        CONTINUUM_INPUT_ORDER_10,
        INPUT_ORDER_17,
        RIGID_INPUT_ORDER_7,
        RIGID_HOME_DEG,
        continuum_mujoco_map_60_to_10,
    )


ROBOT_MODEL_SPEC_VERSION = "v5.1.1"
PLANNER_CONFIGURATION_DIM = 17
MUJOCO_ACTUATED_DOF = 67
BASE_JOINT_NAME = "world_joint"
TARGET_FREE_JOINT_NAME = "target_world_joint"
CONTINUUM_TIP_BODY_NAME = "link_30"
RIGID_TIP_BODY_NAME = "end_effector_r"


@dataclass(frozen=True)
class RobotModelIdentity:
    spec_version: str
    model_name: str
    source_bundle_sha256: str
    runtime_contract_sha256: str
    mujoco_version: str
    planner_coordinate_names: tuple[str, ...]
    low_level_joint_names: tuple[str, ...]
    control_mode: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["planner_coordinate_names"] = list(self.planner_coordinate_names)
        payload["low_level_joint_names"] = list(self.low_level_joint_names)
        return payload


@dataclass(frozen=True)
class RobotModelSpecV5:
    source_urdf: Path
    low_level_joint_names: tuple[str, ...]
    planner_to_low_level: np.ndarray
    low_level_to_planner: np.ndarray
    position_kp: np.ndarray
    velocity_kd: np.ndarray
    torque_limits: np.ndarray
    joint_armature: np.ndarray
    joint_damping: np.ndarray
    planner_lower: np.ndarray
    planner_upper: np.ndarray
    planner_velocity_limits: np.ndarray
    planner_acceleration_limits: np.ndarray
    simulation_timestep: float = 0.002
    base_joint_name: str = BASE_JOINT_NAME
    target_free_joint_name: str = TARGET_FREE_JOINT_NAME
    continuum_tip_body_name: str = CONTINUUM_TIP_BODY_NAME
    rigid_tip_body_name: str = RIGID_TIP_BODY_NAME

    @classmethod
    def from_urdf(
        cls,
        source_urdf: Path,
        *,
        simulation_timestep: float = 0.002,
        continuum_kp: float = 60.0,
        continuum_kd: float = 3.0,
        continuum_torque_limit: float = 4.0,
        rigid_kp: float = 160.0,
        rigid_kd: float = 18.0,
        rigid_torque_limit: float = 80.0,
        continuum_armature: float = 0.01,
        continuum_damping: float = 0.05,
        rigid_armature: float = 0.02,
        rigid_damping: float = 0.1,
    ) -> "RobotModelSpecV5":
        source_urdf = Path(source_urdf)
        model = mujoco.MjModel.from_xml_path(str(source_urdf))
        low_level_joint_names = cls._scalar_actuated_joint_names(model)
        if len(low_level_joint_names) != MUJOCO_ACTUATED_DOF:
            raise ValueError(
                f"expected {MUJOCO_ACTUATED_DOF} scalar arm joints, got {len(low_level_joint_names)}"
            )
        configuration_map = np.zeros(
            (MUJOCO_ACTUATED_DOF, PLANNER_CONFIGURATION_DIM), dtype=np.float64
        )
        configuration_map[:60, :10] = continuum_mujoco_map_60_to_10()
        configuration_map[60:, 10:] = np.eye(7, dtype=np.float64)
        decoder = np.linalg.pinv(configuration_map)

        position_kp = np.concatenate(
            [np.full(60, continuum_kp), np.full(7, rigid_kp)]
        ).astype(np.float64)
        velocity_kd = np.concatenate(
            [np.full(60, continuum_kd), np.full(7, rigid_kd)]
        ).astype(np.float64)
        torque_limits = np.concatenate(
            [np.full(60, continuum_torque_limit), np.full(7, rigid_torque_limit)]
        ).astype(np.float64)
        # The imported URDF gives several short continuum links rotational
        # inertias near 1e-6 kg*m^2.  Explicit reflected motor inertia is part
        # of the V5 runtime contract; without it, even sub-newton-metre inputs
        # produce five-digit accelerations and the dynamic simulation is not a
        # meaningful actuator-level validation.
        joint_armature = np.concatenate(
            [np.full(60, continuum_armature), np.full(7, rigid_armature)]
        ).astype(np.float64)
        joint_damping = np.concatenate(
            [np.full(60, continuum_damping), np.full(7, rigid_damping)]
        ).astype(np.float64)
        planner_lower = np.concatenate(
            [np.full(10, -np.pi), np.full(7, -np.pi)]
        ).astype(np.float64)
        planner_upper = np.concatenate(
            [np.full(10, np.pi), np.full(7, np.pi)]
        ).astype(np.float64)
        planner_velocity_limits = np.concatenate(
            [np.full(10, 0.8), np.full(7, 1.2)]
        ).astype(np.float64)
        planner_acceleration_limits = np.concatenate(
            [np.full(10, 2.5), np.full(7, 4.0)]
        ).astype(np.float64)
        return cls(
            source_urdf=source_urdf,
            low_level_joint_names=low_level_joint_names,
            planner_to_low_level=configuration_map,
            low_level_to_planner=decoder,
            position_kp=position_kp,
            velocity_kd=velocity_kd,
            torque_limits=torque_limits,
            joint_armature=joint_armature,
            joint_damping=joint_damping,
            planner_lower=planner_lower,
            planner_upper=planner_upper,
            planner_velocity_limits=planner_velocity_limits,
            planner_acceleration_limits=planner_acceleration_limits,
            simulation_timestep=float(simulation_timestep),
        )

    @staticmethod
    def _scalar_actuated_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
        names = []
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if name in (BASE_JOINT_NAME, TARGET_FREE_JOINT_NAME):
                continue
            joint_type = int(model.jnt_type[joint_id])
            if joint_type in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                names.append(name)
        return tuple(names)

    @property
    def planner_coordinate_names(self) -> tuple[str, ...]:
        return tuple(INPUT_ORDER_17)

    @property
    def planner_zero(self) -> np.ndarray:
        """Physical home in radians; continuum zero plus rigid URDF home."""

        return np.concatenate([np.zeros(10), np.deg2rad(RIGID_HOME_DEG)]).astype(np.float64)

    def validate(self) -> None:
        if self.planner_coordinate_names != tuple(CONTINUUM_INPUT_ORDER_10 + RIGID_INPUT_ORDER_7):
            raise ValueError("planner coordinate order no longer matches the audited convention")
        expected_shapes = {
            "planner_to_low_level": (MUJOCO_ACTUATED_DOF, PLANNER_CONFIGURATION_DIM),
            "low_level_to_planner": (PLANNER_CONFIGURATION_DIM, MUJOCO_ACTUATED_DOF),
            "position_kp": (MUJOCO_ACTUATED_DOF,),
            "velocity_kd": (MUJOCO_ACTUATED_DOF,),
            "torque_limits": (MUJOCO_ACTUATED_DOF,),
            "joint_armature": (MUJOCO_ACTUATED_DOF,),
            "joint_damping": (MUJOCO_ACTUATED_DOF,),
            "planner_lower": (PLANNER_CONFIGURATION_DIM,),
            "planner_upper": (PLANNER_CONFIGURATION_DIM,),
            "planner_velocity_limits": (PLANNER_CONFIGURATION_DIM,),
            "planner_acceleration_limits": (PLANNER_CONFIGURATION_DIM,),
        }
        for name, shape in expected_shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")
        roundtrip = self.low_level_to_planner @ self.planner_to_low_level
        if not np.allclose(roundtrip, np.eye(PLANNER_CONFIGURATION_DIM), atol=1e-12):
            raise ValueError("planner/low-level maps are not a left-inverse pair")
        if np.any(self.position_kp <= 0.0) or np.any(self.velocity_kd < 0.0):
            raise ValueError("servo gains are invalid")
        if np.any(self.torque_limits <= 0.0):
            raise ValueError("torque limits must be positive")
        if np.any(self.joint_armature <= 0.0) or np.any(self.joint_damping < 0.0):
            raise ValueError("joint armature must be positive and damping non-negative")
        if np.any(self.planner_lower >= self.planner_upper):
            raise ValueError("planner position limits are invalid")
        if self.simulation_timestep <= 0.0:
            raise ValueError("simulation_timestep must be positive")

    def encode_position(self, planner_q: np.ndarray) -> np.ndarray:
        planner_q = np.asarray(planner_q, dtype=np.float64)
        if planner_q.shape[-1] != PLANNER_CONFIGURATION_DIM:
            raise ValueError("planner_q must end with 17 coordinates")
        return np.einsum("ad,...d->...a", self.planner_to_low_level, planner_q)

    def decode_position(self, low_level_q: np.ndarray) -> np.ndarray:
        low_level_q = np.asarray(low_level_q, dtype=np.float64)
        if low_level_q.shape[-1] != MUJOCO_ACTUATED_DOF:
            raise ValueError("low_level_q must end with 67 coordinates")
        return np.einsum("da,...a->...d", self.low_level_to_planner, low_level_q)

    def encode_velocity(self, planner_dq: np.ndarray) -> np.ndarray:
        return self.encode_position(planner_dq)

    def decode_velocity(self, low_level_dq: np.ndarray) -> np.ndarray:
        return self.decode_position(low_level_dq)

    def low_level_pd_torque(
        self,
        measured_q: np.ndarray,
        measured_dq: np.ndarray,
        reference_q: np.ndarray,
        reference_dq: np.ndarray,
    ) -> np.ndarray:
        values = [np.asarray(item, dtype=np.float64) for item in (
            measured_q, measured_dq, reference_q, reference_dq
        )]
        if any(value.shape != (MUJOCO_ACTUATED_DOF,) for value in values):
            raise ValueError("low-level PD inputs must all have shape (67,)")
        torque = self.position_kp * (values[2] - values[0]) + self.velocity_kd * (
            values[3] - values[1]
        )
        return np.clip(torque, -self.torque_limits, self.torque_limits)

    def compile_dynamic_model(self) -> mujoco.MjModel:
        """Compile zero-gravity dynamics with one direct torque motor per arm DoF."""

        self.validate()
        spec = mujoco.MjSpec.from_file(str(self.source_urdf))
        spec.option.gravity[:] = np.zeros(3)
        spec.option.timestep = self.simulation_timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for joint_name, armature, damping in zip(
            self.low_level_joint_names, self.joint_armature, self.joint_damping
        ):
            joint = spec.joint(joint_name)
            if joint is None:
                raise RuntimeError(f"cannot configure missing joint {joint_name!r}")
            joint.armature = float(armature)
            joint.damping = float(damping)
        for joint_name, torque_limit in zip(self.low_level_joint_names, self.torque_limits):
            actuator = spec.add_actuator(
                name=f"v5_torque_{joint_name}",
                target=joint_name,
            )
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = 1.0
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            actuator.gear[0] = 1.0
            actuator.ctrllimited = True
            actuator.ctrlrange[:] = (-float(torque_limit), float(torque_limit))
            actuator.forcelimited = True
            actuator.forcerange[:] = (-float(torque_limit), float(torque_limit))
        model = spec.compile()
        if model.nu != MUJOCO_ACTUATED_DOF:
            raise RuntimeError(f"compiled model has nu={model.nu}, expected {MUJOCO_ACTUATED_DOF}")
        return model

    def _source_assets(self) -> tuple[Path, ...]:
        root = ET.parse(self.source_urdf).getroot()
        mujoco_element = root.find("mujoco")
        compiler = None if mujoco_element is None else mujoco_element.find("compiler")
        mesh_directory = ""
        if compiler is not None:
            mesh_directory = str(compiler.attrib.get("meshdir", ""))
        assets = {self.source_urdf.resolve()}
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            assets.add((self.source_urdf.parent / mesh_directory / filename).resolve())
        missing = [path for path in assets if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing robot assets: {missing[:3]}")
        return tuple(sorted(assets, key=lambda path: str(path).lower()))

    def source_bundle_sha256(self) -> str:
        digest = hashlib.sha256()
        base = self.source_urdf.parent.resolve()
        for path in self._source_assets():
            try:
                relative = path.relative_to(base)
            except ValueError:
                relative = path.name
            digest.update(str(relative).replace("\\", "/").encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def runtime_contract_sha256(self) -> str:
        payload = {
            "spec_version": ROBOT_MODEL_SPEC_VERSION,
            "source_bundle_sha256": self.source_bundle_sha256(),
            "planner_coordinate_names": list(self.planner_coordinate_names),
            "low_level_joint_names": list(self.low_level_joint_names),
            "planner_to_low_level": self.planner_to_low_level.tolist(),
            "position_kp": self.position_kp.tolist(),
            "velocity_kd": self.velocity_kd.tolist(),
            "torque_limits": self.torque_limits.tolist(),
            "joint_armature": self.joint_armature.tolist(),
            "joint_damping": self.joint_damping.tolist(),
            "planner_lower": self.planner_lower.tolist(),
            "planner_upper": self.planner_upper.tolist(),
            "planner_velocity_limits": self.planner_velocity_limits.tolist(),
            "planner_acceleration_limits": self.planner_acceleration_limits.tolist(),
            "simulation_timestep": self.simulation_timestep,
            "gravity": [0.0, 0.0, 0.0],
            "control_mode": "67_direct_joint_torque_motors_with_external_pd",
            "mujoco_version": mujoco.__version__,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def identity(self) -> RobotModelIdentity:
        self.validate()
        return RobotModelIdentity(
            spec_version=ROBOT_MODEL_SPEC_VERSION,
            model_name="dual_arm_space_robot_2026",
            source_bundle_sha256=self.source_bundle_sha256(),
            runtime_contract_sha256=self.runtime_contract_sha256(),
            mujoco_version=mujoco.__version__,
            planner_coordinate_names=self.planner_coordinate_names,
            low_level_joint_names=self.low_level_joint_names,
            control_mode="67_direct_joint_torque_motors_with_external_pd",
        )

    def assert_compatible_identity(self, value: Mapping[str, Any]) -> None:
        expected = self.identity().to_dict()
        required = (
            "spec_version",
            "source_bundle_sha256",
            "runtime_contract_sha256",
            "mujoco_version",
            "planner_coordinate_names",
            "low_level_joint_names",
            "control_mode",
        )
        mismatch = {
            key: {"expected": expected[key], "actual": value.get(key)}
            for key in required
            if value.get(key) != expected[key]
        }
        if mismatch:
            raise ValueError(f"V5 robot model identity mismatch: {mismatch}")


def default_robot_model_spec_v5(
    source_urdf: Optional[Path] = None,
) -> RobotModelSpecV5:
    if source_urdf is None:
        source_urdf = (
            Path("dual_arm_space_robot_2026")
            / "urdf"
            / "dual_arm_space_robot_2026.urdf"
        )
    return RobotModelSpecV5.from_urdf(source_urdf)
