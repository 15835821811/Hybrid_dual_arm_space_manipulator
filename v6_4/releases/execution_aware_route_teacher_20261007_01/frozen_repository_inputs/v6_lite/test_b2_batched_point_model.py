"""Reference parity for the read-only vectorized PCC point path."""

from __future__ import annotations

import unittest

import numpy as np

from v6_lite.continuum_shape_model import ContinuumShapeModel
from v6_lite.pcc_batched_point_model import BatchedPointContinuumShapeModel
from v6_lite.pcc_interval_cbf import IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.shape_clearance import OrientedBox


class BatchedPointModelTests(unittest.TestCase):
    def test_all_sections_boundaries_zero_and_curved_states_match_reference(self) -> None:
        reference = ContinuumShapeModel()
        batched = BatchedPointContinuumShapeModel(reference.spec)
        total = reference.spec.total_length_m
        boundaries = reference.spec.segment_boundaries_m
        arclengths = np.concatenate((
            np.array([total, 0.0, *boundaries]),
            np.linspace(0.0, total, 71),
            np.array([.81 * total, .17 * total, .81 * total]),
        ))
        rng = np.random.default_rng(260929)
        configurations = [np.zeros(10), np.full(10, 1e-9)]
        configurations.extend(rng.uniform(-.5, .5, 10) for _ in range(10))
        base = np.eye(4)
        base[:3, 3] = [.2, -.1, .3]
        for q in configurations:
            expected = reference.batch_query(q, base, arclengths,
                                             with_jacobians=False)
            actual = batched.batch_query(q, base, arclengths,
                                         with_jacobians=False)
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assertEqual(left.segment_id, right.segment_id)
                self.assertAlmostEqual(left.segment_arclength_m,
                                       right.segment_arclength_m, places=14)
                np.testing.assert_allclose(left.position_world,
                                           right.position_world, rtol=0, atol=2e-14)
                np.testing.assert_allclose(left.rotation_world,
                                           right.rotation_world, rtol=0, atol=2e-14)
        with self.assertRaises(ValueError):
            batched.batch_query(np.zeros(10), base, [total + 1e-3])
        with self.assertRaises(ValueError):
            batched.batch_query(np.full(10, np.nan), base, [0.0])

    def test_persistent_decision_and_interval_values_match_reference(self) -> None:
        reference = ContinuumShapeModel()
        batched = BatchedPointContinuumShapeModel(reference.spec)
        rng = np.random.default_rng(260930)
        base = np.eye(4)
        for _ in range(8):
            q = rng.uniform(-.45, .45, 10)
            center = reference.batch_query(q, base,
                [.6 * reference.spec.total_length_m])[0].position_world
            target = OrientedBox(center + rng.uniform(-.08, .08, 3),
                                 np.eye(3), np.array([.05, .04, .04]))
            partition = IntervalPartition.uniform()
            left = PersistentIntervalDecisionQuery(reference).evaluate(
                q, base, target, partition, max_point_evaluations=63)
            right = PersistentIntervalDecisionQuery(batched).evaluate(
                q, base, target, partition, max_point_evaluations=63)
            self.assertEqual(left.proxy_clearance_status,
                             right.proxy_clearance_status)
            self.assertEqual(left.partition, right.partition)
            self.assertEqual(left.point_evaluation_count,
                             right.point_evaluation_count)
            self.assertEqual(left.split_count, right.split_count)
            self.assertEqual(set(left.lower_by_interval_id),
                             set(right.lower_by_interval_id))
            for key in left.lower_by_interval_id:
                self.assertLessEqual(abs(left.lower_by_interval_id[key]
                                         - right.lower_by_interval_id[key]), 2e-14)
                self.assertLessEqual(abs(left.midpoint_upper_by_interval_id[key]
                                         - right.midpoint_upper_by_interval_id[key]),
                                     2e-14)


if __name__ == "__main__":
    unittest.main()
