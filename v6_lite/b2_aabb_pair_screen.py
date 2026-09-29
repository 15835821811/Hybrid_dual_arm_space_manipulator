"""Private conservative world-AABB lower bound for original MuJoCo pairs.

MuJoCo primitive sizes are radii/half-lengths/half-extents in each geom frame:
https://mujoco.readthedocs.io/en/3.2.6/XMLreference.html#body-geom
Unknown shapes retain their compiled geom_rbound sphere as a conservative
axis-aligned box. The 1 pm expansion guards floating-point AABB construction;
the caller retains its original 1 um activation screen pad.
"""

from __future__ import annotations

import mujoco
import numpy as np


AABB_EXPANSION_M = 1e-12


class ConservativePairAABBScreen:
    def __init__(self, model: mujoco.MjModel, geom_a: np.ndarray,
                 geom_b: np.ndarray) -> None:
        self.model = model
        a = np.asarray(geom_a, dtype=np.int32)
        b = np.asarray(geom_b, dtype=np.int32)
        if a.shape != b.shape or a.ndim != 1 or not len(a):
            raise ValueError("pair arrays must be nonempty and aligned")
        ids, inverse = np.unique(np.concatenate((a, b)), return_inverse=True)
        self._ids = ids
        self._a = a
        self._b = b
        self._a_local = inverse[:len(a)]
        self._b_local = inverse[len(a):]
        self._types = np.asarray(model.geom_type[ids], dtype=np.int32)
        self._sizes = np.asarray(model.geom_size[ids], dtype=np.float64)
        self._radii = np.asarray(model.geom_rbound[ids], dtype=np.float64)
        self._primitive_indexes = {
            name: np.flatnonzero(self._types == int(getattr(
                mujoco.mjtGeom, f"mjGEOM_{name}")))
            for name in ("SPHERE", "BOX", "CAPSULE", "CYLINDER", "ELLIPSOID")
        }

    def _half_extents(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.asarray(data.geom_xmat[self._ids],
                              dtype=np.float64).reshape(-1, 3, 3)
        half = np.repeat(self._radii[:, None], 3, axis=1)
        sizes = self._sizes
        for name, indexes in self._primitive_indexes.items():
            if not len(indexes):
                continue
            r = rotation[indexes]
            s = sizes[indexes]
            if name == "SPHERE":
                half[indexes] = s[:, :1]
            elif name == "BOX":
                half[indexes] = np.einsum("nij,nj->ni", np.abs(r), s)
            elif name == "CAPSULE":
                half[indexes] = (s[:, :1]
                                 + np.abs(r[:, :, 2]) * s[:, 1:2])
            elif name == "CYLINDER":
                half[indexes] = (
                    s[:, :1] * np.linalg.norm(r[:, :, :2], axis=2)
                    + np.abs(r[:, :, 2]) * s[:, 1:2])
            elif name == "ELLIPSOID":
                half[indexes] = np.linalg.norm(r * s[:, None, :], axis=2)
        valid = (np.all(np.isfinite(half), axis=1)
                 & np.all(half >= 0.0, axis=1)
                 & np.isfinite(self._radii)
                 & (self._radii > 0.0))
        return half + AABB_EXPANSION_M, valid

    def far_mask(self, data: mujoco.MjData, threshold_m: float) -> np.ndarray:
        half, geom_valid = self._half_extents(data)
        center_a = np.asarray(data.geom_xpos[self._a], dtype=np.float64)
        center_b = np.asarray(data.geom_xpos[self._b], dtype=np.float64)
        difference = center_a - center_b
        separation = np.maximum(
            np.abs(difference) - half[self._a_local] - half[self._b_local],
            0.0)
        lower = np.linalg.norm(separation, axis=1)
        pair_valid = (geom_valid[self._a_local] & geom_valid[self._b_local]
                      & np.all(np.isfinite(difference), axis=1))
        return pair_valid & (lower > threshold_m)
