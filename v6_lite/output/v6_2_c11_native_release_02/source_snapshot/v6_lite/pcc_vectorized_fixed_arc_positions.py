"""Private five-section fixed-arc PCC positions with batched coefficients.

This evaluates the same Rodrigues and integrated-position expressions as the
reference fixed-arc implementation. It changes only arithmetic grouping and
is not selected by the production controller.
"""

from __future__ import annotations

import numpy as np

from v6_lite.continuum_model_spec import CONTINUUM_PLANNER_DOF
from v6_lite.continuum_shape_model import validate_transform
from v6_lite.pcc_fixed_arc_positions import FixedArcPCCPositions


class VectorizedFixedArcPCCPositions(FixedArcPCCPositions):
    def __init__(self, spec, arclengths: np.ndarray) -> None:
        super().__init__(spec, arclengths)
        self._lengths = np.asarray(spec.segment_lengths_m, dtype=np.float64)
        self._segment_ids = np.empty(self._count, dtype=np.int32)
        for segment_id, indexes, _length in self._sections:
            self._segment_ids[indexes] = segment_id

    @staticmethod
    def _coefficients(magnitude: np.ndarray, u: np.ndarray,
                      small: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
        a = np.empty_like(u)
        b = np.empty_like(u)
        c = np.empty_like(u)
        if np.any(small):
            m2 = magnitude[small] ** 2
            us = u[small]
            a[small] = us - m2 * us**3 / 6.0 + m2**2 * us**5 / 120.0
            b[small] = us**2 / 2.0 - m2 * us**4 / 24.0 + m2**2 * us**6 / 720.0
            c[small] = us**3 / 6.0 - m2 * us**5 / 120.0 + m2**2 * us**7 / 5040.0
        regular = ~small
        if np.any(regular):
            mr = magnitude[regular]
            angle = mr * u[regular]
            sine = np.sin(angle)
            a[regular] = sine / mr
            b[regular] = (1.0 - np.cos(angle)) / (mr * mr)
            c[regular] = (angle - sine) / (mr**3)
        return a, b, c

    def evaluate(self, planner_configuration: np.ndarray,
                 base_transform: np.ndarray) -> np.ndarray:
        q = np.asarray(planner_configuration, dtype=np.float64)
        if q.shape != (CONTINUUM_PLANNER_DOF,) or np.any(~np.isfinite(q)):
            raise ValueError("PCC configuration must be finite with shape (10,)")
        base = validate_transform(base_transform)
        omega = (q.reshape(-1, 2) @ self.spec.pcc_bending_map.T
                 / self._lengths[:, None])
        magnitude = np.linalg.norm(omega, axis=1)
        generator = np.zeros((len(self._lengths), 3, 3), dtype=np.float64)
        generator[:, 0, 1] = -omega[:, 2]
        generator[:, 0, 2] = omega[:, 1]
        generator[:, 1, 0] = omega[:, 2]
        generator[:, 1, 2] = -omega[:, 0]
        generator[:, 2, 0] = -omega[:, 1]
        generator[:, 2, 1] = omega[:, 0]
        generator2 = generator @ generator
        full_a, full_b, full_c = self._coefficients(
            magnitude, self._lengths,
            magnitude * self._lengths <= 1e-4)
        identity = np.eye(3, dtype=np.float64)
        full = np.zeros((len(self._lengths), 4, 4), dtype=np.float64)
        full[:, :3, :3] = (identity + full_a[:, None, None] * generator
                           + full_b[:, None, None] * generator2)
        full[:, :3, 3] = (
            self._lengths[:, None] * identity[:, 0]
            + full_b[:, None] * generator[:, :, 0]
            + full_c[:, None] * generator2[:, :, 0])
        full[:, 3, 3] = 1.0
        prefixes = [base @ self.spec.base_to_shape_start]
        for section in full:
            prefixes.append(prefixes[-1] @ section)
        point_magnitude = magnitude[self._segment_ids]
        point_u = self._local_s
        _point_a, point_b, point_c = self._coefficients(
            point_magnitude, point_u,
            magnitude[self._segment_ids] * self._lengths[self._segment_ids]
            <= 1e-4)
        first = generator[self._segment_ids, :, 0]
        second = generator2[self._segment_ids, :, 0]
        local = (point_u[:, None] * identity[:, 0]
                 + point_b[:, None] * first
                 + point_c[:, None] * second)
        positions = np.empty((self._count, 3), dtype=np.float64)
        for segment_id, indexes, _length in self._sections:
            if len(indexes):
                prefix = prefixes[segment_id]
                positions[indexes] = (local[indexes] @ prefix[:3, :3].T
                                      + prefix[:3, 3])
        return positions
