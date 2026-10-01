"""Decision query with persistent interval topology and explicit budgets."""

from __future__ import annotations

import unittest

import numpy as np

from v6_lite.pcc_interval_cbf import IntervalPartition
from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
from v6_lite.shape_clearance import OrientedBox


class PersistentQueryTests(unittest.TestCase):
    def test_complete_partition_persists_without_merge(self) -> None:
        query = PersistentIntervalDecisionQuery()
        box = OrientedBox(np.array([0.95, 0.25, 0.0]),
                          np.eye(3), np.full(3, 0.1))
        q = np.zeros(10)
        first = query.evaluate(q, np.eye(4), box,
                               IntervalPartition.uniform(),
                               max_point_evaluations=64)
        self.assertTrue(first.bounds_valid)
        self.assertTrue(first.partition.coverage(
            query.shape_model.spec.segment_lengths_m).coverage_complete)
        self.assertEqual(first.point_evaluation_count,
                         5 + 2 * first.split_count)
        second = query.evaluate(q + np.full(10, 1e-4), np.eye(4), box,
                                first.partition, max_point_evaluations=64)
        self.assertTrue(second.bounds_valid)
        self.assertTrue(set(first.partition.leaves) <= set(second.partition.leaves)
                        or second.split_count > 0)
        self.assertGreaterEqual(second.interval_count, first.interval_count)
        self.assertTrue(second.partition.coverage(
            query.shape_model.spec.segment_lengths_m).coverage_complete)

    def test_budget_insufficient_to_cover_persistent_leaves_is_explicit(self) -> None:
        query = PersistentIntervalDecisionQuery()
        box = OrientedBox(np.array([0.95, 0.25, 0.0]),
                          np.eye(3), np.full(3, 0.1))
        result = query.evaluate(np.zeros(10), np.eye(4), box,
                                IntervalPartition.uniform(2),
                                max_point_evaluations=5)
        self.assertFalse(result.bounds_valid)
        self.assertEqual(result.failure_reason,
                         "BUDGET_CANNOT_EVALUATE_ALL_EXISTING_LEAVES")
        self.assertEqual(result.proxy_clearance_status,
                         "UNKNOWN_BUDGET_INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
