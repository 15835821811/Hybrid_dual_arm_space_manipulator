"""Analytic displacement bound between two URDF discrete-backbone poses.

This bound does not establish the separate PCC-to-discrete-chain tube envelope
or cover the physical robot's collision geometry. It is a read-only diagnostic
for an actual 60-joint state and its declared 10-D projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v6_lite.continuum_model_spec import CONTINUUM_ACTUATED_DOF, ContinuumModelSpec
from v6_lite.continuum_shape_model import DiscreteContinuumKinematics


@dataclass(frozen=True)
class DiscreteBackboneResidualBound:
    """Per-revolute-joint lever arms from the validated serial URDF chain.

    A downstream point is a sum of rotated URDF joint translations plus at
    most one module's local arclength. For joint j, its distance from the
    pivot cannot exceed the sum of all following origin-translation norms
    plus the longest module. Replacing joint angles one at a time and using
    ``2 sin(|delta|/2) <= |delta|`` gives ``sum_j R_j |delta_j|`` for every
    point on the discrete backbone, at any common base pose and joint angles.
    """

    joint_lever_upper_m: np.ndarray
    source_urdf_sha256: str
    numerical_certification: str = "NOT_FORMALLY_CERTIFIED"

    @classmethod
    def from_spec(cls, spec: ContinuumModelSpec) -> "DiscreteBackboneResidualBound":
        discrete = DiscreteContinuumKinematics(spec)
        chain = discrete._chain  # The constructor validates order against the 60-joint map.
        translations = np.asarray(
            [np.linalg.norm(joint.origin[:3, 3]) for joint in chain],
            dtype=np.float64,
        )
        suffix = np.cumsum(translations[::-1])[::-1]
        longest_module = float(np.max(spec.module_lengths_m))
        levers = np.asarray([
            float(suffix[index + 1] if index + 1 < len(chain) else 0.0)
            + longest_module
            for index, joint in enumerate(chain) if joint.joint_type == "revolute"
        ], dtype=np.float64)
        if (levers.shape != (CONTINUUM_ACTUATED_DOF,)
                or np.any(~np.isfinite(levers)) or np.any(levers < 0)):
            raise ValueError("validated URDF has invalid residual lever arms")
        levers.setflags(write=False)
        return cls(levers, spec.source_urdf_sha256)

    def position_difference_upper_m(self, joint_delta_rad: np.ndarray) -> float:
        delta = np.asarray(joint_delta_rad, dtype=np.float64)
        if delta.shape != (CONTINUUM_ACTUATED_DOF,) or np.any(~np.isfinite(delta)):
            raise ValueError("finite 60-D joint difference required")
        value = float(self.joint_lever_upper_m @ np.abs(delta))
        return float(np.nextafter(value, np.inf)) if value else 0.0
