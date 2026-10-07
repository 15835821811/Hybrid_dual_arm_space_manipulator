"""Read-only, state-local containment of MuJoCo continuum geoms in PCC tubes.

For each fitted collision capsule, sample its straight axis at equally spaced
parameters. A point between samples is at most half a sample spacing from a
sampled axis point. The distance from that sample to *one point on the PCC
curve in the same section* therefore bounds its distance to the continuous
PCC curve. Adding the capsule radius covers the whole convex capsule, and
the fitted capsule covers its source geom vertices. This is an exact-state
geometric argument under the declared model; it says nothing about the next
state, servo trajectory, or floating-point formal certification.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from v6_lite.continuum_model_spec import ContinuumModelSpec
from v6_lite.continuum_shape_model import ContinuumShapeModel
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.shape_clearance import build_continuum_capsule_envelopes


PCC_SAMPLES_PER_SEGMENT = 17
CAPSULE_AXIS_SAMPLES = 5
ARITHMETIC_PAD_M = 1e-6


@dataclass(frozen=True)
class CapsuleContainment:
    geom_name: str
    segment_id: int
    sampled_axis_to_pcc_max_m: float
    axis_sampling_allowance_m: float
    fitted_capsule_radius_m: float
    required_tube_radius_upper_m: float
    declared_tube_radius_m: float
    margin_m: float


@dataclass(frozen=True)
class StateLocalEnvelopeResult:
    status: str
    min_margin_m: float
    capsules: tuple[CapsuleContainment, ...]
    fallback_geom_names: tuple[str, ...]
    numerical_certification: str = "NOT_FORMALLY_CERTIFIED"


class StateLocalPCCEnvelopeAudit:
    """Check actual MuJoCo capsule axes against the unchanged V6.1-A tubes."""

    def __init__(self, model: mujoco.MjModel, spec: ContinuumModelSpec) -> None:
        self.model = model
        self.spec = spec
        self.shape = ContinuumShapeModel(spec)
        self.envelopes = build_continuum_capsule_envelopes(model, spec)
        if not self.envelopes.capsules:
            raise ValueError("no continuum capsules were assigned")
        boundaries = spec.segment_boundaries_m
        self.arclengths = np.concatenate([
            boundaries[index] + np.linspace(
                0.0, float(spec.segment_lengths_m[index]),
                PCC_SAMPLES_PER_SEGMENT,
            )
            for index in range(len(spec.segment_lengths_m))
        ])

    def evaluate(self, data: mujoco.MjData, planner_q: np.ndarray,
                 base_transform: np.ndarray) -> StateLocalEnvelopeResult:
        points = self.shape.batch_query(
            planner_q, base_transform, self.arclengths, with_jacobians=False,
        )
        pcc = np.asarray([item.position for item in points], dtype=np.float64)
        pcc = pcc.reshape(len(self.spec.segment_lengths_m),
                          PCC_SAMPLES_PER_SEGMENT, 3)
        parameters = np.linspace(0.0, 1.0, CAPSULE_AXIS_SAMPLES)
        rows = []
        for capsule in self.envelopes.capsules:
            local = ((1.0 - parameters[:, None]) * capsule.local_start
                     + parameters[:, None] * capsule.local_end)
            rotation = np.asarray(data.xmat[capsule.body_id], dtype=np.float64).reshape(3, 3)
            origin = np.asarray(data.xpos[capsule.body_id], dtype=np.float64)
            axis = origin + local @ rotation.T
            distances = np.linalg.norm(
                axis[:, None, :] - pcc[capsule.segment_index][None, :, :], axis=2,
            )
            sampled_max = float(np.max(np.min(distances, axis=1)))
            allowance = capsule.axis_length_m / (2.0 * (CAPSULE_AXIS_SAMPLES - 1))
            required = float(np.nextafter(
                sampled_max + allowance + capsule.radius_m
                + max(0.0, capsule.maximum_vertex_excess_m) + ARITHMETIC_PAD_M,
                np.inf,
            ))
            declared = float(V61A_PCC_TUBE_RADII_M[capsule.segment_index])
            rows.append(CapsuleContainment(
                geom_name=capsule.geom_name,
                segment_id=capsule.segment_index,
                sampled_axis_to_pcc_max_m=sampled_max,
                axis_sampling_allowance_m=allowance,
                fitted_capsule_radius_m=capsule.radius_m,
                required_tube_radius_upper_m=required,
                declared_tube_radius_m=declared,
                margin_m=declared - required,
            ))
        minimum = min(row.margin_m for row in rows)
        return StateLocalEnvelopeResult(
            status=("COVERED_AT_THIS_STATE" if minimum >= 0.0
                    else "NOT_COVERED_AT_THIS_STATE"),
            min_margin_m=minimum,
            capsules=tuple(rows),
            fallback_geom_names=self.envelopes.fallback_geom_names,
        )
