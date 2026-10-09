"""Parity checks for state-prepared offline PCC decision queries."""

from __future__ import annotations

import unittest

import numpy as np

from v6_lite.continuum_shape_model import ContinuumShapeModel
from v6_lite.pcc_batched_distance_query import (
    BatchedDistanceDecisionQuery, signed_distance_many,
)
from v6_lite.pcc_interval_cbf import IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.shape_clearance import OrientedBox, point_obb_signed_distance


class PreparedQueryTests(unittest.TestCase):
    def test_batched_obb_inside_outside_and_boundary_distance(self) -> None:
        box = OrientedBox(np.array([.1, -.2, .3]), np.eye(3),
                          np.array([.04, .05, .06]))
        points = np.array([
            [.1, -.2, .3], [.14, -.2, .3], [.2, -.2, .3],
            [.1, -.26, .3], [.1, -.2, .361], [.14, -.15, .36],
        ])
        rng = np.random.default_rng(260931)
        points = np.vstack((points, rng.uniform(-.3, .5, (100, 3))))
        actual = signed_distance_many(points, box)
        expected = np.array([point_obb_signed_distance(point, box).signed_distance_m
                             for point in points])
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)

    def test_frozen_decisions_partitions_and_bounds_match_reference(self) -> None:
        shape = ContinuumShapeModel()
        reference = PersistentIntervalDecisionQuery(shape)
        prepared = BatchedDistanceDecisionQuery(shape)
        rng = np.random.default_rng(260932)
        base = np.eye(4)
        q_cases = [np.zeros(10), np.full(10, 1e-9)]
        q_cases.extend(rng.uniform(-.5, .5, 10) for _ in range(16))
        for q in q_cases:
            point = shape.batch_query(q, base,
                [.6 * shape.spec.total_length_m])[0].position_world
            box = OrientedBox(point + rng.uniform(-.1, .1, 3),
                              np.eye(3), np.array([.05, .05, .05]))
            for budget in (31, 63, 127):
                partition = IntervalPartition.uniform()
                left = reference.evaluate(q, base, box, partition,
                                          max_point_evaluations=budget)
                right = prepared.evaluate(q, base, box, partition,
                                          max_point_evaluations=budget)
                self.assertEqual(left.proxy_clearance_status,
                                 right.proxy_clearance_status)
                self.assertEqual(left.partition, right.partition)
                self.assertEqual(left.point_evaluation_count,
                                 right.point_evaluation_count)
                self.assertEqual(left.failure_reason, right.failure_reason)
                self.assertEqual(set(left.lower_by_interval_id),
                                 set(right.lower_by_interval_id))
                for key in left.lower_by_interval_id:
                    self.assertLessEqual(abs(left.lower_by_interval_id[key]
                                             - right.lower_by_interval_id[key]),
                                         1e-12)
                    self.assertLessEqual(abs(left.midpoint_upper_by_interval_id[key]
                                             - right.midpoint_upper_by_interval_id[key]),
                                         1e-12)


if __name__ == "__main__":
    unittest.main()
