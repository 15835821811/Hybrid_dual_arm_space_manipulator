"""Frozen generation/selection/accounting checks; no physical integration."""
import contextlib
import copy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from v6_4.route_optimizer_protocol import (HISTORY, TaskSpec, PreferenceSpec, SearchSpec,
    active_intervals, parameter_plan, project_disks, initial_candidates, read, digest, next_unused_mother_seeds)
from v6_4.continuous_route_optimizer import optimize, rank_candidates, non_dominated
from v6_4.route_candidate_evaluator import command_metrics, seal, verify_seal, NominalCandidateEvaluator
from v6_4.task_anchored_reference import build_reference_definition
from v6_4.evaluate_route_optimizer import selected_methods


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = TaskSpec.from_dict(read(HISTORY / "snapshot/tasks/b3_mother_00_c_plus/task.json"))
        cls.d = build_reference_definition(cls.task)
        cls.active = active_intervals(cls.d)
        cls.identity = {"source_identity_sha256": "source", "config_sha256": "qp",
                        "task_sha256": cls.task.sha256(), "model_contract_sha256": cls.task.model_contract_sha256}
        cls.preferences = [PreferenceSpec(name, cls.task.sha256(), (tuple(cls.d["intervals_s"][2]),),
            tuple(tuple(cls.d["intervals_s"][i]) for i in cls.active), "sphere", (("original_body", "original_sphere"),)) for name in ("A", "B")]

    def run_mock(self, evaluator=None, spec=None, task=None, prefs=None, identity=None):
        task = task or self.task
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(mujoco, "mj_step", side_effect=AssertionError("mock test cannot step physics")):
            result = optimize(task, prefs if prefs is not None else self.preferences,
                identity or self.identity, evaluator or self.flat, tmp, search_spec=spec)
            rows = read(Path(tmp) / "candidate_registry.json")
            proposals = read(Path(tmp) / "proposals.json")
        return result, rows, proposals

    @staticmethod
    def flat(plan, cid):
        return {"status": "PREDICTION_ADMISSIBLE", "prediction_rollout_started": True,
            "prediction_admissible": True, "prediction_task_passed": True, "online_guards_passed": True,
            "prediction_metrics": {"I_support": .1, "L_full": 1., "d_support": .029999}, "costs": {}}

    def test_active_interval_nearest_legal_predecessor_and_disabled_key(self):
        self.assertEqual(active_intervals({"interval_mask": [True, False, True, True, False, False]}), [0, 2])
        self.assertEqual(active_intervals({"interval_mask": [False, False, True, True, False, False]}), [2])
        self.assertEqual(active_intervals({"interval_mask": [True, True, False, True, False, False]}), [])

    def test_next_unused_seed_rule_uses_only_old_identity(self):
        used = {2026100701 + 104729 * i for i in range(12)} | {2026100731}
        self.assertEqual(next_unused_mother_seeds(used), [(12, 2027357449), (13, 2027462178)])

    def test_map_four_coordinates_inactive_exact_zero_separate_disks(self):
        self.assertEqual(len(self.active), 2)
        plan = parameter_plan(self.task, "v2", [.012, .016, -.012, .016])
        self.assertTrue(np.array_equal(plan.z_m[self.active], [[.012, .016], [-.012, .016]]))
        self.assertTrue(np.all(plan.z_m[[i for i in range(6) if i not in self.active]] == 0.))
        self.assertGreater(np.linalg.norm(plan.z_m), .020)
        with self.assertRaises(ValueError): parameter_plan(self.task, "v1", [.020, .001, 0., 0.])

    def test_disk_projection_not_coordinate_square(self):
        p = project_disks([.020, .020, .0, -.020])
        self.assertLessEqual(np.linalg.norm(p[:2]), .020)
        self.assertAlmostEqual(p[0], p[1]); self.assertEqual(p[3], -.020)

    def test_non_axis_initial_20mm_seed_cannot_round_outside_disk(self):
        direction = np.array([-.9999928938932474, -.0037699022545064796])
        self.assertGreater(np.linalg.norm(.020 * direction), .020)
        with patch("v6_4.route_optimizer_protocol.geometry_direction", return_value=(direction, False)):
            seeds = initial_candidates(self.task)
        for seed in seeds:
            plan = parameter_plan(self.task, seed["family"], seed["x_m"])
            self.assertTrue(np.all(np.linalg.norm(plan.z_m, axis=1) <= .020))
        self.assertAlmostEqual(np.linalg.norm(seeds[3]["x_m"][-2:]), .020, places=16)

    def test_zero_is_canonical_and_original_protected_cutoff(self):
        first = parameter_plan(self.task, "v1", [0.] * 4)
        second = parameter_plan(self.task, "v2", [0.] * 4)
        self.assertEqual(first.sha256(), second.sha256())
        self.assertEqual(first.definition["support_cutoff_s"], 23.98)
        self.assertTrue(np.array_equal(first.offset_kinematics(np.array([0., 23.98, 24., 27.]))[0], np.zeros((4, 3))))

    def test_all_four_coordinates_reachable_new_continuous_points(self):
        s, rows, _ = self.run_mock()
        adaptive = [r for r in rows if r["source"] == "adaptive_poll"]
        self.assertEqual({r["coordinate"] for r in adaptive}, {0, 1, 2, 3})
        self.assertEqual([r["preference_center"] for r in adaptive], ["A", "B"] * 4)
        self.assertTrue(any(r["non_historical_continuous_point"] for r in adaptive))
        self.assertEqual(s["budget"]["slots_consumed"], 12)
        self.assertTrue(s["budget"]["poll_truncated"])

    def test_shared_budget_no_hidden_retry_for_rejected_reference(self):
        calls = []
        def rejected(plan, cid):
            calls.append(cid)
            if cid == "C01":
                return {"status": "REFERENCE_PRECHECK_REJECTED", "prediction_rollout_started": False,
                        "prediction_admissible": False, "prediction_metrics": None}
            return self.flat(plan, cid)
        s, rows, _ = self.run_mock(rejected)
        self.assertEqual(len(calls), 12); self.assertEqual(rows[1]["status"], "REFERENCE_PRECHECK_REJECTED")
        self.assertEqual(s["budget"]["prediction_rollouts_started"], 11)

    def test_all_initials_fail_search_continues_without_infeasibility_claim(self):
        def evaluator(plan, cid):
            if int(cid[1:]) < 4:
                return {"status": "EXECUTION_REFUSED", "prediction_rollout_started": True,
                        "prediction_admissible": False, "prediction_metrics": {"I_support": .0, "L_full": .01, "d_support": .1}}
            return self.flat(plan, cid)
        s, rows, _ = self.run_mock(evaluator)
        self.assertEqual(len(rows), 12)
        self.assertIsNotNone(s["preferences"]["A"]["selected_plan"])
        self.assertNotIn(s["strict_winner"], ("C00", "C01", "C02", "C03"))

    def test_best_initial_is_allowed_and_B_cannot_dispatch_below_30mm(self):
        s, _, _ = self.run_mock()
        self.assertEqual(s["tie_selected_winner"], "C00")
        self.assertIsNone(s["preferences"]["B"]["selected_plan"])
        self.assertEqual(s["preferences"]["B"]["status"], "PREFERENCE_UNMET_WITHIN_BUDGET")
        self.assertIsNotNone(s["preferences"]["B"]["best_safe_diagnostic"])

    def test_minimum_tie_anchor_never_chain_expands(self):
        rows = [self.row("a", .1000, 1.2, .04), self.row("b", .1009, 1.1, .04), self.row("c", .1018, .5, .04)]
        rank = rank_candidates(rows)
        self.assertEqual(rank["tie_group_ids"], ["a", "b"])
        self.assertEqual(rank["strict"]["candidate_id"], "a")
        self.assertEqual(rank["tie_selected"]["candidate_id"], "b")

    @staticmethod
    def row(cid, I, L, d, admissible=True):
        return {"candidate_id": cid, "x_m": [0., 0., 0., 0.], "family": "v1",
                "prediction_admissible": admissible, "prediction_metrics": {"I_support": I, "L_full": L, "d_support": d}}

    def test_exact_B_threshold_missing_clearance_and_failed_prefix_excluded(self):
        rows = [self.row("below", .1, .1, .029999), self.row("missing", .1, .01, None),
                self.row("failed", 0., 0., .1, False), self.row("valid", .2, 1., .030)]
        self.assertEqual(rank_candidates(rows)["B"]["candidate_id"], "valid")
        self.assertNotIn("failed", non_dominated(rows))

    def test_finite_frontier_keeps_tradeoff_and_zero(self):
        rows = [self.row("zero", .1, 1., .026), self.row("clear", .11, 1.1, .04), self.row("dominated", .2, 1.2, .02)]
        self.assertEqual(non_dominated(rows), ["zero", "clear"])

    def test_identity_rejects_wrong_task_or_model(self):
        for field in ("task_sha256", "model_contract_sha256"):
            identity = {**self.identity, field: "wrong"}
            with self.assertRaises(ValueError): self.run_mock(identity=identity)

    def test_exact_cache_hits_no_new_calls_and_proposal_cap(self):
        # Repeated family/zero poll must not consume extra slots; the explicit
        # proposal cap terminates even if every construction is identical.
        seed = initial_candidates(self.task)[0]
        with patch("v6_4.continuous_route_optimizer.initial_candidates", return_value=[seed] * 4), patch("v6_4.continuous_route_optimizer.project_disks", return_value=np.zeros(4)):
            s, rows, _ = self.run_mock(spec=SearchSpec(proposal_limit=9))
        self.assertEqual(len(rows), 1)
        self.assertEqual(s["budget"]["proposal_attempts"], 9)
        self.assertEqual(s["budget"]["cache_hits"], 8)
        self.assertEqual(s["budget"]["stop_reason"], "PROPOSAL_LIMIT")

    def test_resume_verifies_source_config_and_does_not_evaluate_again(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            first = optimize(self.task, self.preferences, self.identity, self.flat, tmp)
            def forbidden(*args): raise AssertionError("completed stage must not integrate again")
            second = optimize(self.task, self.preferences, self.identity, forbidden, tmp)
            self.assertEqual(first, second)
            with self.assertRaises(ValueError):
                optimize(self.task, self.preferences, {**self.identity, "config_sha256": "changed"}, forbidden, tmp)

    def test_tool_error_is_retained_distinct_from_research_negative(self):
        def broken(plan, cid):
            return {"status": "TOOL_ERROR", "prediction_rollout_started": False, "prediction_admissible": False,
                    "tool_error": {"type": "IOError"}, "prediction_metrics": None}
        s, rows, _ = self.run_mock(broken)
        self.assertEqual(len(rows), 1)
        self.assertEqual(s["budget"]["stop_reason"], "TOOL_ERROR")
        self.assertEqual(s["preferences"]["A"]["status"], "NO_ADMISSIBLE_PLAN_WITHIN_BUDGET")

    def test_key_disabled_not_replaced(self):
        task = replace(self.task, path_freedom="strict_full_curve")
        identity = {**self.identity, "task_sha256": task.sha256()}
        s, rows, _ = self.run_mock(task=task, prefs=[], identity=identity)
        self.assertEqual(s["budget"]["stop_reason"], "NOT_APPLICABLE")
        self.assertEqual(rows, [])

    def test_selected_identical_plans_have_same_actual_alias_key(self):
        s, _, _ = self.run_mock()
        methods = selected_methods(self.task, s)
        self.assertEqual(methods["Z0"].sha256(), methods["OI"].sha256())
        self.assertIsNone(methods["OC"])

    def test_command_vectors_consumed_prefix_components_and_fixed_windows(self):
        nominal = np.zeros((4, 17)); nominal[0, :10] = 1.; nominal[1, 10:] = 2.; nominal[2, :] = 3.; nominal[3, :] = np.nan
        selected = np.zeros((4, 17)); selected[3] = np.nan
        trace = {"task_time": np.array([.0, .02, .04, .06]), "task_qp_box_nominal_velocity": nominal,
            "task_qp_selected_velocity": selected, "task_selected_command": selected,
            "task_avoidance_intervention": np.linalg.norm(nominal - selected, axis=1)}
        m, vectors = command_metrics(trace, 3, {"I_support": [[0., .02]], "I_key": [[.02, .02 + 1e-3]], "I_full": [[0., 27.]]})
        self.assertAlmostEqual(m["I_support"] ** 2, (10. + 28.) / 2.)
        self.assertAlmostEqual(m["I_key"], np.sqrt(28.))
        self.assertNotEqual(m["I_support"], m["I_full"])
        self.assertTrue(m["component_square_identity_checked"])
        self.assertEqual(len(vectors["time"]), 3)
        with self.assertRaises(ValueError): command_metrics(trace, 4, {"I": [[0., 27.]]})

    def test_quality_seal_detects_tamper_in_nested_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "nested").mkdir(); (root / "nested/manifest.json").write_text("{}")
            seal(root); verify_seal(root)
            (root / "nested/manifest.json").write_text('{"tamper": true}')
            with self.assertRaises(ValueError): verify_seal(root)

    def test_report_keeps_no_plan_in_denominators_without_false_success(self):
        from v6_4.evaluate_route_optimizer import report
        from v6_4.route_optimizer_protocol import write, METHODS
        selection, candidates, _ = self.run_mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tasks = []
            for i in range(4):
                tid = f"mock_{i}"
                tasks.append({"task_id": tid, "task_sha256": self.task.sha256(), "group_id": f"mock_mother_{i//2}",
                    "seed": i, "mother_source_sha256": "mock", "obstacle_name": "sphere",
                    "W_support": [[0., 1.]], "related_pair_ids": [["body", "sphere"]], "initial_geometry_query_count": 0})
                write(root / "planning" / tid / "selection.json", selection)
                write(root / "planning" / tid / "candidate_registry.json", candidates)
                for method in METHODS:
                    write(root / "actual" / tid / method / "slot.json", {"task_id": tid, "method": method,
                        "status": "NO_PLAN", "entered_actual": False, "full_task_success": False,
                        "original_independent_gates_passed": False, "clearance_30mm_met": None,
                        "quality": None, "unique_run": False, "plan_sha256": None,
                        "prediction_actual_consistent": None, "costs": None})
            protocol = {"tasks": tasks, "algorithm_producer_commit": "mock", "base_publication_commit": "mock"}
            with patch("v6_4.evaluate_route_optimizer.validate", return_value={"all_slots_terminal": True,
                    "tool_error_count": 0, "actual_metric_missing_count": 0}), patch("v6_4.evaluate_route_optimizer.verify_selections", return_value=protocol), patch("v6_4.evaluate_route_optimizer._figures"):
                summary = report(root)
            self.assertEqual(summary["full_task_success_by_method"], dict.fromkeys(METHODS, 0))
            self.assertEqual(summary["actual_method_slots"], 16)
            self.assertEqual(summary["actual_unique_runs"], 0)
            self.assertTrue(all(p["paired_complete_count"] == 0 for p in summary["improvement_over_zero"]))
            self.assertEqual(summary["teacher_rows"], 96)

    def test_candidate_calls_create_distinct_provider_and_config_objects(self):
        # Mock the runner before any integration. Separate calls still construct
        # independent provider/config instances; real runner owns fresh MjData/QP.
        calls = []
        def fake_runner(spec, cfg, qp, scene, traces, **kwargs):
            calls.append((cfg, qp, kwargs["reference_provider"]))
            raise RuntimeError("mock infrastructure stop before physics")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "source_identity.json").write_text("{}")
            config = read(HISTORY / "snapshot/frozen_execution_config.json")
            (root / "frozen_execution_config.json").write_text(__import__("json").dumps(config))
            (root / "frozen_run_config.json").write_text(__import__("json").dumps(read(HISTORY / "snapshot/attempts/EA_03/actual/run_metadata.json")["run_config"]))
            frozen = {"geometry_precheck_passed": True, "obstacle_name": "sphere"}
            with patch("v6_4.route_optimizer_protocol.verify_frozen"), patch("v6_4.residual_execution.start_run", return_value={}), patch("v6_4.residual_execution.finish_run"), patch("v6_lite.run_v6_lite.run_synchronous_scenario", side_effect=fake_runner), patch.object(mujoco, "mj_step", side_effect=AssertionError("no physical work")):
                evaluator = NominalCandidateEvaluator(root, self.task, frozen)
                plan = parameter_plan(self.task, "v1", [0.] * 4)
                a = evaluator(plan, "A"); b = evaluator(plan, "B")
                self.assertEqual(a["prediction_steps"], 0); self.assertEqual(b["prediction_steps"], 0)
                self.assertEqual(len(calls), 2)
                for i in range(3): self.assertIsNot(calls[0][i], calls[1][i])


if __name__ == "__main__": unittest.main()
