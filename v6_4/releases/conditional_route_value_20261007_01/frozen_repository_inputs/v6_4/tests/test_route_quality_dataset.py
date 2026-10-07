"""Metric semantics and stopping decisions, without physics or sampling."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4.route_quality_dataset import (
    build_route_quality, command_intervention_metrics, data_qualification, intervals_array,
    path_and_base_metrics, pilot_discriminability, quality_filter,
    quality_replay_evidence, safety_eligibility, sha, window_mask,
)


def record(task, candidate, intervention, *, success=True, split="train", length=1.):
    return {"slot_id": task + "_" + candidate, "task_id": task,
            "candidate_name": candidate, "split": split,
            "quality_label_eligible": success,
            "safety": {"full_task_and_original_safety_passed": success},
            "full_metrics": {"I_route_rad_s": intervention,
                             "continuum_path_length_m": length} if success else None,
            "failed_prefix_metrics": None if success else {"I_route_rad_s": intervention}}


class RouteQualityTests(unittest.TestCase):
    def test_rms_is_vector_norm_squared_mean_and_consumed_prefix(self):
        trace = {"task_time": np.array([0., .02, .04, .06, .08]),
                 "task_avoidance_intervention": np.array([100., 3., 4., 100., np.nan])}
        result = command_intervention_metrics(trace, 4, [[.02, .04]])
        self.assertEqual(result["route_planning_sample_count"], 2)
        self.assertAlmostEqual(result["I_route_rad_s"], np.sqrt(12.5))
        self.assertNotAlmostEqual(result["I_route_rad_s"], 3.5)
        self.assertFalse(result["proposal_minus_selected_is_used"])
        self.assertFalse(result["source_vectors_available"])

    def test_no_window_samples_are_unknown_not_zero(self):
        trace = {"task_time": [0., .02], "task_avoidance_intervention": [1., 2.]}
        self.assertIsNone(command_intervention_metrics(trace, 2, [[1., 2.]])["I_route_rad_s"])

    def test_consumed_nonfinite_diagnostic_rejected(self):
        with self.assertRaises(ValueError):
            command_intervention_metrics({"task_time": [0.], "task_avoidance_intervention": [np.nan]}, 1, [[0., 1.]])

    def test_window_is_closed_and_union_not_convex_hull(self):
        mask = window_mask([0., 1., 2., 3., 4., 5.], [[0., 1.], [4., 5.]])
        np.testing.assert_array_equal(mask, [True, True, False, False, True, True])

    def test_invalid_windows_rejected(self):
        for intervals in ([], [[1., 1.]], [[1., 3.], [2., 4.]], [[0., np.nan]], [[-1., 2.]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                intervals_array(intervals)

    def test_path_does_not_bridge_disjoint_windows(self):
        state = {"time": np.arange(6.),
                 "continuum_position": np.column_stack((np.arange(6.), np.zeros((6, 2)))),
                 "base_pose": np.tile([0., 0., 0., 1., 0., 0., 0.], (6, 1))}
        result = path_and_base_metrics(state, [[0., 1.], [4., 5.]])
        self.assertEqual(result["continuum_path_length_m"], 5.)
        self.assertEqual(result["continuum_route_window_path_length_m"], 2.)
        self.assertEqual(result["base_rotation_peak_rad"], 0.)

    def test_original_five_independent_gates_required_and_old_curve_separate(self):
        attempt = {"actual_steps": 13500, "full_task_success": True, "fallback_used": False}
        evaluation = {"complete": True, "full_task_success": True, "evidence_valid": True,
                      "runtime_reported_passed": False}
        for key in ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding"):
            evaluation[key] = {"passed": True}
        self.assertTrue(safety_eligibility(attempt, evaluation)["path_quality_comparison_eligible"])
        for key in ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding"):
            changed = copy.deepcopy(evaluation)
            changed[key]["passed"] = False
            self.assertFalse(safety_eligibility(attempt, changed)["path_quality_comparison_eligible"])
        attempt["actual_steps"] = 8310
        self.assertFalse(safety_eligibility(attempt, evaluation)["path_quality_comparison_eligible"])

    def test_elite_floor_accepts_zero_and_retains_modes_without_average(self):
        rows = [record("t", "z0", .005), record("t", "z+", .006),
                record("t", "z-", .0060001), record("t", "z_perp", .0001, success=False)]
        elite = quality_filter(rows)
        self.assertEqual(elite["tasks"][0]["elite_slot_ids"], ["t_z0", "t_z+"])
        self.assertEqual(elite["tasks"][0]["failure_slot_ids"], ["t_z_perp"])
        self.assertFalse(elite["coordinate_averaging_used"])

    def test_relative_elite_threshold_and_task_local_comparison(self):
        rows = [record("a", "z0", .1), record("a", "z+", .109),
                record("a", "z-", .111), record("b", "z0", .5)]
        result = quality_filter(rows)
        self.assertAlmostEqual(result["tasks"][0]["elite_limit_rad_s"], .11)
        self.assertEqual(result["tasks"][1]["elite_slot_ids"], ["b_z0"])

    def test_path_length_tie_break_does_not_change_elite_membership(self):
        rows = [record("t", "z0", .01, length=2.), record("t", "z+", .01, length=1.)]
        result = quality_filter(rows)["tasks"][0]
        self.assertEqual(result["elite_slot_ids"], ["t_z+", "t_z0"])

    def test_failed_prefix_never_enters_elite_even_if_very_low_cost(self):
        result = quality_filter([record("t", "z0", .02), record("t", "z+", 0., success=False)])
        self.assertEqual(result["tasks"][0]["I_min_rad_s"], .02)

    def test_pilot_requires_both_absolute_and_relative_threshold_and_reversal(self):
        rows = [record("a", "z0", .02), record("a", "z+", .017), record("a", "z-", .021),
                record("b", "z0", .02), record("b", "z+", .021), record("b", "z-", .017)]
        self.assertTrue(pilot_discriminability(rows, ["a", "b"])["route_value_identifiable"])
        rows[5]["full_metrics"]["I_route_rad_s"] = .0199
        self.assertFalse(pilot_discriminability(rows, ["a", "b"])["route_value_identifiable"])

    def test_pilot_opposite_success_vs_failure_can_identify(self):
        rows = [record("a", "z0", .01, success=False), record("a", "z+", .03), record("a", "z-", .01, success=False),
                record("b", "z0", .01, success=False), record("b", "z+", .01, success=False), record("b", "z-", .03)]
        self.assertTrue(pilot_discriminability(rows, ["a", "b"])["route_value_identifiable"])

    def test_both_sides_same_preference_is_negative(self):
        rows = [record(t, c, v) for t in ("a", "b") for c, v in (("z0", .02), ("z+", .016), ("z-", .021))]
        self.assertFalse(pilot_discriminability(rows, ["a", "b"])["training_authorized_by_pilot"])

    def test_both_nonzero_improve_zero_better_direction_can_reverse(self):
        rows = [record("a", "z0", .05), record("a", "z+", .01), record("a", "z-", .02),
                record("b", "z0", .05), record("b", "z+", .02), record("b", "z-", .01)]
        result = pilot_discriminability(rows, ["a", "b"])
        self.assertTrue(result["route_value_identifiable"])
        self.assertEqual([r["preferred_nonzero_direction"] for r in result["sides"]], ["z+", "z-"])

    def test_nonzero_direction_tie_not_broken_by_path_length_or_slot(self):
        rows = [record(t, c, v, success=c != "z0", length=length)
                for t in ("a", "b") for c, v, length in (("z0", .05, 1.), ("z+", .01, 1.), ("z-", .0100000000001, 2.))]
        result = pilot_discriminability(rows, ["a", "b"])
        self.assertFalse(result["route_value_identifiable"])
        self.assertIsNone(result["sides"][0]["preferred_nonzero_direction"])

    def test_qualifying_zero_gain_does_not_overstate_direct_opposite_gain(self):
        rows = [record("a", "z0", .03), record("a", "z+", .01), record("a", "z-", .0101),
                record("b", "z0", .03), record("b", "z+", .0101), record("b", "z-", .01)]
        result = pilot_discriminability(rows, ["a", "b"])
        self.assertTrue(result["route_value_identifiable"])
        self.assertFalse(result["sides"][0]["preferred_direct_opposite_meets_A_or_B"])

    def test_training_data_minimum_counts_distinct_tasks_not_exposure(self):
        rows = [record("train" + str(i), "z0", .01) for i in range(4)] + [record("v1", "z0", .01, split="val"), record("v2", "z0", .01, split="val")]
        self.assertEqual(data_qualification(quality_filter(rows))["status"], "DATA_QUALIFIED_FOR_THIS_PILOT")
        rows = [record("only_one", str(i), .01) for i in range(20)] + [record("v", "z0", .01, split="val")]
        self.assertEqual(data_qualification(quality_filter(rows))["status"], "DATA_INSUFFICIENT_FOR_THIS_PILOT")

    def test_each_frozen_val_task_requires_an_evaluable_elite_set(self):
        rows = [record("train" + str(i), "z0", .01) for i in range(4)] + [record("v1", "z0", .01, split="val")]
        self.assertEqual(data_qualification(quality_filter(rows))["status"], "DATA_INSUFFICIENT_FOR_THIS_PILOT")
        self.assertEqual(data_qualification(quality_filter(rows), required_val_tasks=1)["status"], "DATA_QUALIFIED_FOR_THIS_PILOT")


class ReplayAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evaluation_dir = self.root / "evaluation"
        self.evaluation_dir.mkdir()
        self.evaluation_path = self.evaluation_dir / "report.json"
        self.evaluation = {"evidence_valid": False, "full_task_success": False,
                           "errors": [{"type": "ValueError", "message": "captured replay failure"}]}
        self.attempt = {"full_task_success": False, "full_27s_success": False}
        self.save_evaluation()

    def save_evaluation(self, *, replay=False, native=False):
        if native:
            self.evaluation["native_geometry"] = {"whole_body": {"pair_policy_sha256": "a" * 64}}
        self.evaluation_path.write_text(json.dumps(self.evaluation), encoding="utf8")
        files = {"report.json": sha(self.evaluation_path)}
        if replay:
            (self.evaluation_dir / "fresh_replay.npz").write_bytes(b"hashed saved states, never decoded by availability checks")
            files["fresh_replay.npz"] = sha(self.evaluation_dir / "fresh_replay.npz")
        (self.evaluation_dir / "manifest.json").write_text(json.dumps(files), encoding="utf8")

    def test_captured_failure_without_fresh_replay_is_unavailable(self):
        replay, sources, unavailable = quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)
        self.assertIsNone(replay)
        self.assertEqual([p.name for p in sources], ["manifest.json"])
        self.assertEqual(unavailable["reason"], "INDEPENDENT_REPLAY_UNAVAILABLE_AFTER_EVALUATION_FAILURE")

    def test_no_evaluation_path_is_unavailable_without_success_claim(self):
        replay, sources, unavailable = quality_replay_evidence(self.attempt, {}, None)
        self.assertIsNone(replay)
        self.assertFalse(sources)
        self.assertEqual(unavailable["reason"], "INDEPENDENT_EVALUATION_PATH_UNAVAILABLE")

    def test_successful_slot_without_evaluation_path_is_hard_failure(self):
        self.attempt["full_task_success"] = True
        with self.assertRaisesRegex(ValueError, "lacks independent evaluation path"):
            quality_replay_evidence(self.attempt, {}, None)

    def test_manifest_claimed_missing_replay_is_hard_failure_even_after_error(self):
        manifest = json.loads((self.evaluation_dir / "manifest.json").read_text())
        manifest["fresh_replay.npz"] = "a" * 64
        (self.evaluation_dir / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "declared.*artifact is missing"):
            quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)

    def test_present_unbound_replay_is_hard_failure(self):
        (self.evaluation_dir / "fresh_replay.npz").write_bytes(b"unbound")
        with self.assertRaisesRegex(ValueError, "lacks independent manifest digest"):
            quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)

    def test_present_replay_digest_mismatch_is_hard_failure(self):
        self.save_evaluation(replay=True)
        (self.evaluation_dir / "fresh_replay.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "artifact digest differs"):
            quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)

    def test_other_declared_present_evidence_digest_mismatch_is_hard_failure(self):
        self.evaluation_path.write_text("tampered report", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "artifact digest differs"):
            quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)

    def test_successful_evaluation_missing_unlisted_replay_is_hard_failure(self):
        self.evaluation["evidence_valid"] = True
        self.save_evaluation()
        with self.assertRaisesRegex(ValueError, "successful independent evaluation lacks fresh replay"):
            quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)

    def test_unexplained_absence_is_hard_failure(self):
        self.evaluation["errors"] = []
        self.save_evaluation()
        with self.assertRaisesRegex(ValueError, "not explained by a captured evaluation failure"):
            quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)

    def test_hashed_fresh_but_geometry_failure_is_unavailable(self):
        self.save_evaluation(replay=True)
        replay, sources, unavailable = quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)
        self.assertIsNone(replay)
        self.assertEqual({p.name for p in sources}, {"manifest.json", "fresh_replay.npz"})
        self.assertEqual(unavailable["reason"], "INDEPENDENT_GEOMETRY_BINDING_UNAVAILABLE_AFTER_EVALUATION_FAILURE")

    def test_missing_pair_policy_after_captured_failure_is_unavailable(self):
        self.evaluation["native_geometry"] = {"whole_body": {}}
        self.save_evaluation(replay=True)
        _, _, unavailable = quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)
        self.assertEqual(unavailable["reason"], "INDEPENDENT_GEOMETRY_BINDING_UNAVAILABLE_AFTER_EVALUATION_FAILURE")

    def test_successful_evaluation_without_geometry_is_hard_failure(self):
        self.evaluation["evidence_valid"] = True
        self.save_evaluation(replay=True)
        with self.assertRaisesRegex(ValueError, "lacks required native geometry/pair-policy evidence"):
            quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)

    def test_available_hashed_replay_preserves_prefix_metric_processing(self):
        self.save_evaluation(replay=True, native=True)
        replay, _, unavailable = quality_replay_evidence(self.attempt, self.evaluation, self.evaluation_path)
        self.assertEqual(replay.name, "fresh_replay.npz")
        self.assertIsNone(unavailable)

    def test_build_writes_negative_record_without_loading_or_querying_missing_evidence(self):
        for scenario in ("captured_missing_replay", "missing_evaluation_path", "captured_missing_geometry"):
            with self.subTest(scenario=scenario):
                attempt_dir = self.root / scenario
                attempt_dir.mkdir()
                (attempt_dir / "task.json").write_text("{}", encoding="utf8")
                trace = attempt_dir / "trace.npz"
                trace.write_bytes(b"actual evidence exists; no decoding needed for unavailable metrics")
                attempt = {**self.attempt, "slot_id": scenario, "task_id": "task", "task_sha256": "taskhash",
                           "status": "TASK_OR_EVIDENCE_FAILED", "actual_steps": 10,
                           "trace_path": str(trace), "trace_sha256": sha(trace), "fallback_used": False}
                if scenario != "missing_evaluation_path":
                    self.save_evaluation(replay=scenario == "captured_missing_geometry")
                    attempt.update(evaluation_path=str(self.evaluation_path), evaluation_sha256=sha(self.evaluation_path))
                (attempt_dir / "attempt_result.json").write_text(json.dumps(attempt), encoding="utf8")
                task = SimpleNamespace(task_id="task", split="pilot", sha256=lambda: "taskhash")
                output = self.root / (scenario + "_quality")
                with patch("v6_4.task_protocol.TaskSpec.from_dict", return_value=task), \
                        patch("v6_4.route_quality_dataset.np.load", side_effect=AssertionError("must not decode unavailable evidence")), \
                        patch("v6_4.route_quality_dataset.local_continuum_clearance", side_effect=AssertionError("must not query unavailable evidence")):
                    result = build_route_quality(attempt_dir, [[0., 1.]], "sphere", output)
                self.assertFalse(result["quality_label_eligible"])
                self.assertIsNone(result["full_metrics"])
                self.assertIsNone(result["failed_prefix_metrics"])
                self.assertTrue(result["metric_unavailable"]["reason"])
                self.assertEqual(result["costs"]["additional_route_quality_geometry_queries"], 0)
                self.assertTrue((output / "route_quality.json").is_file())
                self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
