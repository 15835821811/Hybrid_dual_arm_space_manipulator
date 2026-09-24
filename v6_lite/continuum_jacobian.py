"""Differential-kinematics facade for the V6.1-B PCC safety layer.

The underlying analytic derivative is produced by the matrix-exponential
Frechet derivative in :mod:`v6_lite.continuum_shape_model`.  This module keeps
the online clearance code independent from the shape model's internal
representation and exposes the requested ``(3, 10)`` position Jacobian API.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from v6_lite.continuum_shape_model import ContinuumShapeModel


def compute_position_jacobian(
    q_continuum: np.ndarray,
    base_transform: np.ndarray,
    arc_length_m: float,
    *,
    model: ContinuumShapeModel | None = None,
) -> np.ndarray:
    """Return ``d p_world(s) / d q_continuum`` with shape ``(3, 10)``."""

    shape_model = ContinuumShapeModel() if model is None else model
    result = shape_model.evaluate(
        q_continuum,
        base_transform,
        float(arc_length_m),
        with_jacobians=True,
    )
    jacobian = np.asarray(result.position_jacobian, dtype=np.float64)
    if jacobian.shape != (3, 10) or np.any(~np.isfinite(jacobian)):
        raise RuntimeError("PCC position Jacobian is not finite with shape (3,10)")
    return jacobian.copy()


def compute_position_jacobians(
    q_continuum: np.ndarray,
    base_transform: np.ndarray,
    arc_lengths_m: Iterable[float],
    *,
    model: ContinuumShapeModel | None = None,
) -> np.ndarray:
    """Return a batch of PCC position Jacobians with shape ``(N, 3, 10)``."""

    shape_model = ContinuumShapeModel() if model is None else model
    values = [
        compute_position_jacobian(
            q_continuum,
            base_transform,
            float(arc_length),
            model=shape_model,
        )
        for arc_length in arc_lengths_m
    ]
    if not values:
        return np.zeros((0, 3, 10), dtype=np.float64)
    return np.stack(values, axis=0)
