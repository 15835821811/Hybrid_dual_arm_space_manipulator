"""Synthetic reporting checks; no live experiment, model or physics is invoked."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from v6_4.route_optimizer_protocol import sha, write
from v6_4.visualization.interpret_preference_warmstart_results import (
    ENDPOINTS, REVIEW_SCHEMA, amortization, cold_cost_bracket, complete_quality, first_hits,
    load_scientific_review, observed_mixture_amortization, paired_evidence, raw_initializer_evidence,
)


def successful_slot():
    return {"status": "COMPLETE", "full_task_success": True,
        "original_independent_gates_passed": True, "actual_steps": 13500,
        "quality": {"I_support": .002, "L_full": .4, "d_support": .031,
            "base_translation_peak_m": .001, "base_rotation_peak_rad": .002}}


class InterpretationTests(unittest.TestCase):
    def fixture(self):
        tids = [f"task{i}" for i in range(4)]
        slots = {(t, e, p): successful_slot() for t in tids for e in ENDPOINTS for p in ("A", "B")}
        costs = {(t, e): {"cold_s": 20. if e == "R12" else 10., "warm_s": 9.}
            for t in tids for e in ENDPOINTS}
        return tids, slots, costs

    def bracket_fixture(self, tids, costs, overhead=2.):
        brackets = {}
        for t in tids:
            for e in ENDPOINTS:
                internal = costs[(t, "R12" if e == "R8" else e)]["cold_s"]
                final = internal - 2.
                prefix = costs[(t, "R8")]["cold_s"] - 2. if e == "R8" else None
                brackets[(t, e)] = cold_cost_bracket(t, e, costs[(t, e)]["cold_s"], internal + overhead,
                    internal, final, prefix)
        return brackets

    def test_quality_requires_complete_actual_five_gates_and_finite_metrics(self):
        slot = successful_slot()
        self.assertIsNotNone(complete_quality(slot))
        for key, value in (("full_task_success", False), ("original_independent_gates_passed", False),
                ("actual_steps", 13499), ("quality", None)):
            invalid = copy.deepcopy(slot); invalid[key] = value
            self.assertIsNone(complete_quality(invalid))
        invalid = copy.deepcopy(slot); invalid["quality"]["d_support"] = float("nan")
        self.assertIsNone(complete_quality(invalid))

    def test_pairs_keep_all_tasks_and_separate_formal_and_diagnostic_bands(self):
        tids, slots, costs = self.fixture()
        slots[(tids[0], "D8", "A")].update(status="NO_PLAN", full_task_success=False, quality=None)
        comparisons = paired_evidence(slots, costs, tids, self.bracket_fixture(tids, costs))
        self.assertEqual(len(comparisons), 5)
        for comparison in comparisons:
            self.assertEqual((comparison["task_denominator"], comparison["logical_preference_denominator"]), (4, 8))
            self.assertEqual(len(comparison["rows"]), 8)
            self.assertEqual(comparison["near_quality_rule_scope"], "predeclared_R12_near_quality_band"
                if comparison["baseline"] == "R12" else "diagnostic_reuse_of_thresholds_not_formal_noninferiority")
            if comparison["method"] == "D8":
                self.assertEqual(comparison["complete_paired_preferences"], 7)
                self.assertIsNone(comparison["rows"][0]["quality_delta_method_minus_baseline"])
            if (comparison["method"], comparison["baseline"]) == ("D8", "R12"):
                self.assertEqual(comparison["rows"][0]["cold_planning_delta_s"], -10.)
                self.assertEqual(comparison["rows"][0]["conservative_cold_saving_lower_bound_s"], 8.)
                self.assertTrue(comparison["rows"][0]["guaranteed_cheaper_within_recorded_bounds"])

    def test_amortization_counts_shared_search_once_per_task_and_separates_service(self):
        tids, slots, costs = self.fixture()
        result = amortization(slots, costs, tids, 100., 350., 20., self.bracket_fixture(tids, costs))[0]
        self.assertEqual(result["mean_cold_saving_per_task_s"], 8.)
        self.assertEqual(result["break_even_tasks"], 15.)
        self.assertEqual(result["break_even_tasks_upper_bound"], 15.)
        self.assertEqual(result["cumulative_service_cost_per_observed_wall_saving_ratio"], 46.25)
        self.assertTrue(result["service_ratio_is_not_elapsed_makespan"])

    def test_amortization_fails_closed_for_any_failed_task_nonpositive_saving_or_missing_cost(self):
        for defect in ("failed", "no_saving", "missing_offline", "B_below30"):
            tids, slots, costs = self.fixture()
            teacher = 100.
            if defect == "failed":
                slots[(tids[0], "D8", "A")]["original_independent_gates_passed"] = False
            elif defect == "no_saving":
                costs[(tids[0], "D8")]["cold_s"] = 20.
            elif defect == "missing_offline":
                teacher = None
            else:
                slots[(tids[0], "D8", "B")]["quality"]["d_support"] = .029
            result = amortization(slots, costs, tids, teacher, 350., 20., self.bracket_fixture(tids, costs))[0]
            self.assertEqual(result["break_even_tasks"], "NOT_ESTABLISHED")
            self.assertIsNone(result["cumulative_service_cost_per_observed_wall_saving_ratio"])

    def test_cold_brackets_preserve_original_lower_and_r8_algebra(self):
        tids, _, costs = self.fixture()
        brackets = self.bracket_fixture(tids, costs)
        self.assertEqual(len(brackets), 16)
        r8 = brackets[(tids[0], "R8")]
        self.assertEqual((r8["cold_lower_bound_s"], r8["cold_upper_bound_s"]), (10., 12.))
        self.assertEqual(r8["cold_upper_bound_s"], 22. - 18. + 8.)
        self.assertEqual(r8["startup_and_tail_residual_s"], 2.)
        self.assertEqual(r8["bound_width_s"], 2.)
        self.assertTrue(r8["not_exact_process_start_to_seal_measurement"])

    def test_cold_brackets_reject_reversed_mismatched_and_nonfinite_intervals(self):
        examples = [("D8", 10., 9., 10., 8., None), ("D8", 10., 12., 10., 11., None),
            ("R8", 10., 22., 20., 18., 19.), ("R8", 9., 22., 20., 18., 8.),
            ("D8", 10., float("nan"), 10., 8., None)]
        for example in examples:
            with self.subTest(example=example), self.assertRaises(ValueError):
                cold_cost_bracket("task", *example)

    def test_positive_partial_timer_saving_does_not_establish_amortization(self):
        tids, slots, costs = self.fixture()
        for t in tids:
            costs[(t, "D8")]["cold_s"] = 19.
        result = amortization(slots, costs, tids, 100., 350., 20., self.bracket_fixture(tids, costs))[0]
        self.assertEqual(result["tasks"][0]["measured_cold_saving_s"], 1.)
        self.assertEqual(result["tasks"][0]["conservative_cold_saving_lower_bound_s"], -1.)
        self.assertFalse(result["tasks"][0]["guaranteed_cheaper_within_recorded_bounds"])
        self.assertEqual(result["break_even_tasks"], "NOT_ESTABLISHED")
        self.assertEqual(amortization(slots, costs, tids, 100., 350., 20.)[0]["break_even_tasks"], "NOT_ESTABLISHED")

    def test_observed_mixture_preserves_no_plan_na_without_calling_it_quality_success(self):
        tids, slots, costs = self.fixture()
        for t in tids[:2]:
            for e in ("R12", "D8"):
                slots[(t, e, "B")].update(status="NO_PLAN", full_task_success=False, quality=None)
        brackets = self.bracket_fixture(tids, costs)
        strict = amortization(slots, costs, tids, 100., 350., 20., brackets)[0]
        result = observed_mixture_amortization(slots, costs, tids, 100., 350., 20., brackets)[0]
        self.assertEqual(strict["break_even_tasks"], "NOT_ESTABLISHED")
        self.assertIn("not_predeclared_protocol_gate", strict["eligibility_convention"])
        self.assertEqual(result["coverage"]["R12_observed_complete_gated_preferences"], 6)
        self.assertEqual(result["coverage"]["preserved_observed_ability_and_quality_preferences"], 6)
        self.assertEqual(result["coverage"]["R12_NO_PLAN_N_A_preferences"], 2)
        self.assertEqual(result["coverage"]["unchanged_missing_capability_preferences"], 2)
        self.assertEqual(result["conditional_empirical_mixture_break_even_tasks_upper_bound"], 15.)
        self.assertIsNone(result["tasks"][0]["preferences"][1]["preserved_observed_ability_and_quality"])
        self.assertFalse(result["tasks"][0]["preferences"][1]["N_A_is_quality_success"])

    def test_observed_mixture_retains_negative_task_cost_in_four_task_mean_and_raw_rejection(self):
        tids, slots, costs = self.fixture()
        costs[(tids[0], "D8")]["cold_s"] = 21.
        raw = [{"task_id": tids[0], "endpoint": "D8", "raw_legal": False, "rejection_reason": "raw amplitude exceeds limit"}]
        result = observed_mixture_amortization(slots, costs, tids, 100., 350., 20., self.bracket_fixture(tids, costs), raw)[0]
        self.assertEqual(result["coverage"]["Task_denominator"], 4)
        self.assertEqual(result["tasks"][0]["conservative_cold_saving_lower_bound_s"], -3.)
        self.assertEqual(result["mean_conservative_cold_saving_per_task_s"], (-3. + 8. + 8. + 8.) / 4)
        self.assertAlmostEqual(result["conditional_empirical_mixture_break_even_tasks_upper_bound"], 120. / 5.25)
        self.assertEqual(result["tasks"][0]["raw_initializer_diagnostics"][0]["rejection_reason"], "raw amplitude exceeds limit")
        self.assertFalse(result["invalid_initializer_slot_credited_as_neural_benefit"])

    def test_observed_mixture_rejects_lost_observed_ability_or_near_quality(self):
        for defect in ("actual", "quality"):
            tids, slots, costs = self.fixture()
            if defect == "actual":
                slots[(tids[0], "D8", "A")]["full_task_success"] = False
            else:
                slots[(tids[0], "D8", "A")]["quality"]["L_full"] = .406
            result = observed_mixture_amortization(slots, costs, tids, 100., 350., 20., self.bracket_fixture(tids, costs))[0]
            self.assertEqual(result["conditional_empirical_mixture_break_even_tasks_upper_bound"], "NOT_ESTABLISHED")
            self.assertEqual(result["coverage"]["lost_observed_baseline_abilities"] if defect == "actual" else
                result["coverage"]["lost_predeclared_quality_preferences"], 1)

    def test_observed_mixture_added_full_method_capability_is_separate_from_B30_and_neural_seed(self):
        tids, slots, costs = self.fixture()
        slots[(tids[0], "R12", "B")].update(status="NO_PLAN", full_task_success=False, quality=None)
        slots[(tids[0], "D8", "B")]["quality"]["d_support"] = .029
        result = observed_mixture_amortization(slots, costs, tids, 100., 350., 20., self.bracket_fixture(tids, costs))[0]
        self.assertEqual(result["coverage"]["added_full_gated_method_capability_preferences"], 1)
        self.assertEqual(result["coverage"]["added_B30mm_method_capability_preferences"], 0)
        self.assertIn("not_raw_neural_seed", result["tasks"][0]["preferences"][1]["capability_attribution"])

    def test_observed_mixture_nonpositive_mean_and_incomplete_baseline_remain_explicit(self):
        tids, slots, costs = self.fixture()
        for t in tids:
            costs[(t, "D8")]["cold_s"] = 19.
        slots[(tids[0], "R12", "B")]["original_independent_gates_passed"] = False
        result = observed_mixture_amortization(slots, costs, tids, 100., 350., 20., self.bracket_fixture(tids, costs))[0]
        self.assertEqual(result["coverage"]["R12_incomplete_gated_N_A_preferences"], 1)
        self.assertEqual(result["mean_conservative_cold_saving_per_task_s"], -1.)
        self.assertEqual(result["conditional_empirical_mixture_break_even_tasks_upper_bound"], "NOT_ESTABLISHED")
        self.assertIn("NONPOSITIVE_FOUR_TASK_MEAN_SAVING_LOWER_BOUND", result["ineligibility_reasons"])

    def test_raw_initializer_rejection_reason_preserves_source_diagnostics(self):
        proposals = [{"source": "diffusion", "preference": "A", "family": "v1", "initializer_slot": 1,
            "raw_seed_diagnostics": {"raw_legal": False, "rejection_reason": "raw amplitude exceeds limit",
                "raw_repaired": False, "resampled": False}}, {"source": "rule"}]
        result = raw_initializer_evidence("task", "D8", proposals)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rejection_reason"], "raw amplitude exceeds limit")
        self.assertFalse(result[0]["raw_seed_acceptance_is_actual_success"])

    def test_first_hits_keep_censor_and_missing_oracle_distinct(self):
        metric = {"I_support": .002, "L_full": .4, "d_support": .029}
        rows = [{"prediction_admissible": False}, {"prediction_admissible": True, "prediction_metrics": metric}]
        oracle = {"preferences": {"A": {"prediction_metrics": {"I_support": 0., "L_full": .4}},
            "B": {"prediction_metrics": None}}}
        result = first_hits(rows, oracle, 8)
        self.assertEqual(result["N_first_admissible"]["slot"], 2)
        self.assertEqual(result["N_first_B"]["right_censored_budget"], 2)
        self.assertEqual(result["N_match_R12"]["A"]["status"], "RIGHT_CENSORED")
        self.assertEqual(result["N_match_R12"]["B"]["status"], "N/A_R12_NO_PLAN")

    def review_fixture(self, run):
        write(run / "actual_complete.json", {"slot_hashes": {}})
        write(run / "tables/raw.json", {"quality": "synthetic"})
        write(run / "summary.json", {"base": True})
        (run / "REPORT.md").write_text("base", encoding="utf8")
        write(run / "tables/report_interpretation/evidence.json", {"enriched": True})
        review = {"schema": REVIEW_SCHEMA, "reviewed": True, "model_freeze_sha256": "freeze",
            "source_sha256": {p: sha(run / p) for p in ("actual_complete.json", "tables/raw.json")},
            "learning_benefit_established_in_pilot": "NOT_ESTABLISHED",
            "default_initializer_decision": "retain_C1_rule",
            "scientific_interpretation_zh": ["Synthetic bounded review."]}
        write(run / "tables/pilot_scientific_review.json", review)
        return review

    def test_optional_review_accepts_bound_raw_sources_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory(prefix="c2_interpret_") as temporary:
            run = Path(temporary); self.review_fixture(run)
            self.assertIsNotNone(load_scientific_review(run, "freeze"))
            with self.assertRaises(ValueError):
                load_scientific_review(run, "another_freeze")
            (run / "tables/raw.json").write_text(json.dumps({"quality": "changed"}), encoding="utf8")
            with self.assertRaises(ValueError):
                load_scientific_review(run, "freeze")

    def test_optional_review_rejects_mutable_source_aliases_and_noncanonical_raw_paths(self):
        with tempfile.TemporaryDirectory(prefix="c2_interpret_") as temporary:
            run = Path(temporary); original = self.review_fixture(run)
            for relative in ("./summary.json", "tables/../summary.json", "./REPORT.md",
                    "tables/../tables/report_interpretation/evidence.json", "./tables/raw.json"):
                review = copy.deepcopy(original)
                review["source_sha256"][relative] = sha(run / relative)
                (run / "tables/pilot_scientific_review.json").write_text(json.dumps(review), encoding="utf8")
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    load_scientific_review(run, "freeze")


if __name__ == "__main__":
    unittest.main()
