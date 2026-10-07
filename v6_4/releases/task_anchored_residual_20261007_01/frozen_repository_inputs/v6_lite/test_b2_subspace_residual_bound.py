"""Check the serial-URDF residual bound on frozen material points."""

from __future__ import annotations

import unittest

import numpy as np

from v6_lite.continuum_shape_model import DiscreteContinuumKinematics
from v6_lite.pcc_subspace_residual_bound import DiscreteBackboneResidualBound


class ResidualBoundTests(unittest.TestCase):
    def test_joint_difference_bounds_every_discrete_backbone_sample(self) -> None:
        discrete = DiscreteContinuumKinematics()
        bound = DiscreteBackboneResidualBound.from_spec(discrete.spec)
        self.assertEqual(bound.joint_lever_upper_m.shape, (60,))
        self.assertGreater(bound.joint_lever_upper_m[0],
                           bound.joint_lever_upper_m[-1])
        self.assertFalse(bound.joint_lever_upper_m.flags.writeable)
        rng = np.random.default_rng(20261001)
        arclengths = np.linspace(0.0, discrete.spec.total_length_m, 61)
        for scale in (1e-7, 1e-3, 0.08):
            original = rng.uniform(-0.3, 0.3, 60)
            delta = rng.normal(0.0, scale, 60)
            second = original + delta
            first_frames = discrete.body_transforms(original, np.eye(4))
            second_frames = discrete.body_transforms(second, np.eye(4))
            upper = bound.position_difference_upper_m(delta)
            for s in arclengths:
                first = discrete.evaluate(original, np.eye(4), float(s),
                                          transforms=first_frames)
                later = discrete.evaluate(second, np.eye(4), float(s),
                                          transforms=second_frames)
                self.assertLessEqual(
                    float(np.linalg.norm(later.position - first.position)),
                    upper + 2e-12,
                )
        self.assertEqual(bound.position_difference_upper_m(np.zeros(60)), 0.0)
        for invalid in (np.zeros(59), np.full(60, np.nan)):
            with self.assertRaises(ValueError):
                bound.position_difference_upper_m(invalid)


if __name__ == "__main__":
    unittest.main()
