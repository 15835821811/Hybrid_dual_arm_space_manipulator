"""Boundary checks for the shared point/Jacobian/refinement budget."""

import unittest

from v6_lite.audit_b2_private_unified_budget import assess_budget
from v6_lite.shadow_b2_interval_cbf import ShadowBudget


class UnifiedBudgetTest(unittest.TestCase):
    def test_exact_limits_are_accepted_and_every_source_is_charged(self):
        budget = ShadowBudget(branch_point_evaluations=31,
                              total_point_evaluations=64,
                              jacobian_evaluations=96,
                              local_refinement_evaluations=4,
                              total_query_time_ms=20.0)
        self.assertEqual(assess_budget(31, 29, 4, 96, 20.0, budget), [])
        self.assertEqual(assess_budget(32, 29, 4, 97, 20.1, budget), [
            "BRANCH_POINTS", "TOTAL_POINTS", "JACOBIANS", "QUERY_TIME"])
        self.assertEqual(assess_budget(31, 30, 5, 96, 20.0, budget), [
            "TOTAL_POINTS", "LOCAL_REFINEMENT"])

    def test_invalid_usage_is_rejected(self):
        budget = ShadowBudget()
        for args in ((-1, 5, 0, 2, 1.0),
                     (5, 5, 0, 2, float("nan")),
                     (5, 5, 0, 2, -0.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                assess_budget(*args, budget)


if __name__ == "__main__":
    unittest.main()
