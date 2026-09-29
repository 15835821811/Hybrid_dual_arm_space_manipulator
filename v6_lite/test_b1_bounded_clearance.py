"""Analytic and adverse offline bounds cases for B.1."""

from __future__ import annotations

import math
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from v6_lite.pcc_bounded_clearance import (
    CurveSegment, PCCBoundedClearanceEvaluator, bounded_curve_obb_clearance,
)
from v6_lite.shape_clearance import OrientedBox


BOX = OrientedBox(np.zeros(3), np.eye(3), np.full(3, 0.1))


class BoundedClearanceTests(unittest.TestCase):
    def test_formal_audit_artifact_hashes_and_counts(self) -> None:
        directory = Path(__file__).parent / "output" / "v6_2_b1" / "formal_audit_r02"
        manifest = json.loads((directory / "audit_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["passed"])
        for item in manifest["artifacts"]:
            data = (directory / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(len(data), item["bytes"])
        report = json.loads((directory / "bounded_clearance_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(report["cohorts"]["historical_regenerated"]["count"], 10_000)
        self.assertEqual(report["cohorts"]["independent_heldout"]["count"], 1024)
        self.assertEqual(report["cohorts"]["historical_regenerated"][
            "empirical_false_safe_against_mujoco"], 0)

    def test_line_face_edge_corner_and_inside(self) -> None:
        # For x traversing [-1,1], the closest point is x=0. A fixed
        # transverse offset supplies independently calculated box distances.
        for transverse, expected in (
            ((0.3, 0.0), 0.2),
            ((0.3, 0.4), math.hypot(0.2, 0.3)),
            ((0.0, 0.0), -0.1),
        ):
            y, z = transverse
            segment = CurveSegment(2.0, 0.01,
                                   lambda s, y=y, z=z: np.array([s-1, y, z]))
            result = bounded_curve_obb_clearance((segment,), BOX,
                                                  max_evaluations=1001,
                                                  tolerance_m=1e-3)
            truth = expected - 0.01
            self.assertTrue(result.bounds_valid)
            self.assertLessEqual(result.distance_lower_bound_m, truth + 1e-10)
            self.assertGreaterEqual(result.distance_upper_bound_m, truth - 1e-10)
            self.assertTrue(result.tolerance_met)
        corner = CurveSegment(2.0, 0.01,
                              lambda s: np.array([s+0.2, 0.3, 0.4]))
        result = bounded_curve_obb_clearance((corner,), BOX,
                                              max_evaluations=1001)
        truth = math.sqrt(0.1**2 + 0.2**2 + 0.3**2) - 0.01
        self.assertLessEqual(result.distance_lower_bound_m, truth)
        self.assertGreaterEqual(result.distance_upper_bound_m, truth)

    def test_arc_and_between_initial_samples(self) -> None:
        # Unit-speed arc: p(t)=(sin t, 1.3-cos t, 0), t in [-0.5,0.5].
        # Closest box-face distance is 0.2 at the interior midpoint.
        arc = CurveSegment(1.0, 0.02,
                           lambda s: np.array([math.sin(s-0.5),
                                               1.3-math.cos(s-0.5), 0.0]))
        result = bounded_curve_obb_clearance((arc,), BOX,
                                              max_evaluations=1001)
        self.assertLessEqual(result.distance_lower_bound_m, 0.18)
        self.assertGreaterEqual(result.distance_upper_bound_m, 0.18)
        self.assertTrue(result.tolerance_met)

        # A narrow unsafe interior zone lies between the initial midpoint
        # and endpoints. The global interval cannot report safe on 1 sample.
        line = CurveSegment(1.0, 0.01,
                            lambda s: np.array([s-0.23, 0.0, 0.0]))
        rough = bounded_curve_obb_clearance((line,), BOX,
                                             max_evaluations=1)
        self.assertTrue(rough.budget_exhausted)
        self.assertNotEqual(rough.proxy_clearance_status,
                            "PROXY_CLEARANCE_AT_LEAST_GATE")
        refined = bounded_curve_obb_clearance((line,), BOX,
                                               max_evaluations=1001)
        self.assertLessEqual(refined.distance_lower_bound_m, -0.11)
        self.assertGreaterEqual(refined.distance_upper_bound_m, -0.11)
        self.assertEqual(refined.proxy_clearance_status,
                         "PROXY_CLEARANCE_BELOW_GATE")

    def test_competing_sections_budget_and_local_upper_only(self) -> None:
        far = CurveSegment(1.0, 0.01, lambda s: np.array([s, 2.0, 0.0]))
        near = CurveSegment(1.0, 0.02, lambda s: np.array([s-0.4, 0.3, 0.0]))
        rough = bounded_curve_obb_clearance((far, near), BOX,
                                             max_evaluations=2)
        self.assertTrue(rough.budget_exhausted)
        self.assertEqual({x.segment_id for x in rough.candidate_intervals}, {0, 1})
        local = bounded_curve_obb_clearance((far, near), BOX,
                                             max_evaluations=2,
                                             local_refinement_count=2)
        self.assertEqual(local.evaluation_count, 2)
        self.assertGreater(local.local_refinement_evaluation_count, 0)
        self.assertLessEqual(local.distance_lower_bound_m,
                             rough.distance_lower_bound_m)
        self.assertLessEqual(local.distance_upper_bound_m,
                             rough.distance_upper_bound_m)
        self.assertEqual(local.best_segment_id, 1)

    def test_work_domain_boundary_outside_and_shape_subspace(self) -> None:
        evaluator = PCCBoundedClearanceEvaluator()
        spec = evaluator.shape_model.spec
        q = spec.work_domain_upper_rad.copy()
        box = OrientedBox(np.array([10.0, 0.0, 0.0]),
                          np.eye(3), np.full(3, 0.1))
        actual = spec.planner_to_actuated @ q
        inside = evaluator.evaluate(q, np.eye(4), box,
                                    actual_configuration=actual,
                                    max_evaluations=5)
        self.assertIn("INSIDE_DECLARED_WORK_DOMAIN",
                      inside.geometry_domain_status)
        self.assertIn("ON_DECLARED_SHAPE_SUBSPACE",
                      inside.geometry_domain_status)
        self.assertTrue(inside.bounds_valid)
        self.assertEqual(inside.proxy_clearance_status,
                         "PROXY_CLEARANCE_AT_LEAST_GATE")
        for segment_id, length in enumerate(spec.segment_lengths_m):
            intervals = sorted(
                (item for item in inside.candidate_intervals
                 if item.segment_id == segment_id),
                key=lambda item: item.local_start_m,
            )
            self.assertTrue(intervals)
            self.assertAlmostEqual(intervals[0].local_start_m, 0.0)
            self.assertAlmostEqual(intervals[-1].local_end_m, length)
            for first, second in zip(intervals, intervals[1:]):
                self.assertAlmostEqual(first.local_end_m, second.local_start_m)
        outside_q = q.copy()
        outside_q[0] += 1e-3
        outside = evaluator.evaluate(outside_q, np.eye(4), box,
                                     max_evaluations=5)
        self.assertIn("OUTSIDE_DECLARED_WORK_DOMAIN",
                      outside.geometry_domain_status)
        self.assertEqual(outside.envelope_status,
                         "NO_ENVELOPE_EVIDENCE_FOR_THIS_STATE")
        disturbed = actual.copy()
        disturbed[0] += 1e-3
        off_shape = evaluator.evaluate(q, np.eye(4), box,
                                       actual_configuration=disturbed,
                                       max_evaluations=5)
        self.assertIn("OUTSIDE_DECLARED_SHAPE_SUBSPACE",
                      off_shape.geometry_domain_status)
        self.assertEqual(off_shape.envelope_status,
                         "NO_ENVELOPE_EVIDENCE_FOR_THIS_STATE")


if __name__ == "__main__":
    unittest.main()
