"""Check the exact frozen-ramp work-domain rate inequalities."""

from __future__ import annotations

import unittest

import numpy as np

from v6_lite.b2_work_domain_velocity_box import ramp_endpoint_velocity_box


class WorkDomainVelocityBoxTests(unittest.TestCase):
    def test_endpoint_box_keeps_all_ten_shape_coordinates_in_domain(self) -> None:
        q = np.array([.9998, -.9999] + [0.] * 8)
        previous = np.array([.03059, -.03] + [0.] * 8)
        lower, upper = ramp_endpoint_velocity_box(
            q, previous, np.full(10, -1.), np.full(10, 1.), .02)
        chosen = np.linspace(0., 1., 10) * upper + (1. - np.linspace(
            0., 1., 10)) * lower
        predicted = q + .02 * (.45 * previous + .55 * chosen)
        self.assertTrue(np.all(predicted >= -1. - 1e-14))
        self.assertTrue(np.all(predicted <= 1. + 1e-14))
        self.assertLess(upper[0], 0.)
        self.assertGreater(lower[1], 0.)

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ramp_endpoint_velocity_box(np.zeros(9), np.zeros(10),
                                       np.full(10, -1.), np.ones(10), .02)
        with self.assertRaises(ValueError):
            ramp_endpoint_velocity_box(np.zeros(10), np.zeros(10),
                                       np.full(10, -1.), np.ones(10), 0.)


if __name__ == "__main__":
    unittest.main()
