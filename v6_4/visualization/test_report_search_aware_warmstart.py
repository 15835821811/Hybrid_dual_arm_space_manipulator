"""Reporting audits on synthetic evidence; no simulation, training or media."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v6_4.visualization.report_search_aware_warmstart import (
    ENDPOINTS, GATES, Evidence, _comparison, _endpoint_cost, _hit, _near, _prefix, _sha, _write, build_report)
from v6_4.closed_loop_warmstart_validation import _seal, freeze_phase_selections


def full_slot(task, endpoint, preference, **extra):
    return {"task_id": task, "endpoint": endpoint, "preference": preference,
        "full_task_success": True, "full_27s_success": True, "actual_steps": 13500,
        "original_independent_gates_passed": True, "five_gates": {g: True for g in GATES},
        "quality": {"I_support": .1, "L_full": 1., "d_support": .04,
            "base_translation_peak_m": .002, "base_rotation_peak_rad": .003},
        "status": "ACTUAL_COMPLETE", "diagnostic_category": "FULL_TASK_AND_FIVE_GATES_PASSED",
        "unique_run": True, "clearance_30mm_met": True, "source_candidate_id": "C00",
        "selected_origin_source": "diffusion" if endpoint == "D8" else "zero",
        "source_attribution": "direct_learning_seed" if endpoint == "D8" else "rule_seed_or_descendant",
        "selected_lineage": [], "costs": {"private_preview_physics_steps": 3, "native_geometry_query_calls": 10,
            "independent_saved_torque_replay_steps": 13500, "qp_solve_calls": 1}, "elapsed_wall_s": 5., **extra}


def stream(root, task, method, stage="test", raw_illegal=0):
    directory = root / {"test": "test_search", "val": "closed_loop_val/search", "teacher": "teacher_search"}[stage] / task / method
    planning = directory / "planning" / task
    budget = 12 if method == "R" else 8
    rows = []
    for i in range(budget):
        row = {"candidate_id": f"C{i:02d}", "prediction_admissible": True, "prediction_task_passed": True,
            "online_guards_passed": True, "prediction_steps": 13500, "prediction_rollout_started": True,
            "status": "PREDICTION_ADMISSIBLE", "prediction_metrics": {"I_support": .1, "L_full": 1., "d_support": .04,
                "clearance_status": "MEASURED", "saved_horizon_s": 27., "native_state_count": 13501},
            "costs": {"prediction_physics_steps": 13500, "private_preview_physics_steps": 50,
                "native_geometry_query_calls": 10, "independent_saved_torque_replay_steps": 0, "qp_solve_calls": 1}}
        if i < raw_illegal:
            row.update(status="INITIALIZER_RAW_REJECTED", prediction_admissible=False, prediction_task_passed=False,
                prediction_steps=0, prediction_rollout_started=False, prediction_metrics=None, costs={"prediction_physics_steps": 0})
        rows.append(row)
    chosen = {p: {"prediction_metrics": {"I_support": .1, "L_full": 1., "d_support": .04},
        "selected_plan": {"mock_plan": task + p}} for p in ("A", "B")}
    if stage == "test" and task == "test0" and method == "D":
        chosen["B"]["selected_plan"] = None
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _write(planning / "candidate_registry.json", rows); _write(planning / "proposals.json", rows)
    common = {"task_id": task, "task_sha256": "sha-" + task, "selection_reads_final_actual": False,
        "preferences": chosen}
    _write(planning / "selection.json", {**common, "elapsed_wall_s": 90., "registry_content_sha256": digest,
        "budget": {"stop_reason": "CANDIDATE_BUDGET_EXHAUSTED", "slots_consumed": budget, "candidate_budget": budget}})
    _write(directory / "planning_cost.json", {"end_to_end_cold_planning_s": {"R": 100., "N": 80., "S": 80., "D": 70.}.get(method, 100.)})
    _write(directory / "outer_process.json", {"elapsed_wall_s": 105., "exit_code": 0})
    for b in ([4, 8, 12] if method == "R" else [4, 8]):
        pool = rows[:b]
        digest = hashlib.sha256(json.dumps(pool, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        _write(planning / f"prefix_{b:02d}.json", {**common, "budget": {"slots_consumed": b}, "registry_content_sha256": digest,
            "preferences": chosen, "protocol_completed": True, "elapsed_wall_s": b * 5.})


def fixture(root):
    tasks = [{"task_id": f"tr{i}", "split": "train", "mother_id": f"m{i//2}"} for i in range(6)]
    tasks += [{"task_id": f"val{i}", "split": "val", "mother_id": "mv"} for i in range(2)]
    tasks += [{"task_id": f"test{i}", "split": "test", "mother_id": f"mt{i//2}"} for i in range(4)]
    limits = {"teacher_candidate_slots": 96, "val_candidate_slots": 88, "test_candidate_slots": 144,
        "val_actual_slots": 20, "test_actual_slots": 40, "val_ddim_samples": 8, "test_ddim_samples": 8,
        "smoke_ddim_samples": 8, "diagnostic_ddim_samples": 8, "D_training_runs": 1, "S_training_runs": 1}
    _write(root / "plan.json", {"tasks": tasks, "budget_limits": limits})
    _write(root / "source_identity.json", {"algorithm_producer_commit": "producer"})
    _write(root / "dataset/manifest.json", {"search_effect_teacher_evidence_available": True,
        "route_quality_label_count": 10, "initializer_effect_label_count": 2, "label_count_views": 11, "zero_label_views": 0,
        "search_effect_status": "FINITE_SEARCH_EFFECT_EVIDENCE_AVAILABLE"})
    _write(root / "teacher_search/results.json", {"protocol_completed": True, "summaries": [{}] * 12, "slots_consumed": 96})
    for m in ("D", "S"):
        _write(root / f"models/{m}/training_report.json", {"status": "COMPLETED", "optimizer_updates_total": 4000,
            "sample_exposures_total": 128000, "elapsed_s": 20., "effective_parameter_count": 134412,
            "initialization_seed": 64321 if m == "D" else 64331, "paired_reference_indices_sha256": "shared", "ddim_sample_units": 0})
        for update in (250, 4000):
            (root / f"models/{m}/checkpoint_{update:04d}.pt").write_bytes(f"mock-{m}-{update}".encode())
    for row in tasks:
        tid = row["task_id"]
        if row["split"] == "train":
            for m in ("T_local", "T_transfer"):
                stream(root, tid, m, "teacher")
        elif row["split"] == "val":
            for m in ("R", "D250", "D4000", "S250", "S4000"):
                stream(root, tid, m, "val")
        else:
            for m in ("R", "N", "S", "D"):
                stream(root, tid, m, raw_illegal=int(m == "D"))
    checkpoint_files = {m + str(update): str(root / f"models/{m}/checkpoint_{update:04d}.pt")
        for m in ("D", "S") for update in (250, 4000)}
    def seal_phase(phase, phase_tasks, endpoints, bindings):
        entries = []
        for tid in phase_tasks:
            for endpoint in endpoints:
                method = endpoint if phase == "closed_loop_val" and endpoint != "R12" else endpoint[0]
                directory = root / ("closed_loop_val/search" if phase == "closed_loop_val" else "test_search") / tid / method
                planning = directory / "planning" / tid
                entries.append({"task_id": tid, "endpoint": endpoint,
                    "selection_path": str(planning / ("prefix_08.json" if endpoint == "R8" else "selection.json")),
                    "candidate_registry_path": str(planning / "candidate_registry.json"),
                    "proposals_path": str(planning / "proposals.json"), "planning_cost_path": str(directory / "planning_cost.json")})
        manifest = freeze_phase_selections(root / phase, entries, phase="VAL" if phase == "closed_loop_val" else "TEST",
            bindings=bindings, checkpoint_files=checkpoint_files if phase == "closed_loop_val" else {},
            expected_endpoints=endpoints)
        for entry in manifest["entries"]:
            tid, endpoint = entry["task_id"], entry["endpoint"]
            selected = json.loads(Path(entry["selection_path"]).read_text())["preferences"]
            for pref in ("A", "B"):
                directory = root / phase / "actual" / tid / f"{endpoint}_{pref}"; path = directory / "slot.json"
                plan_sha = hashlib.sha256(json.dumps(selected[pref]["selected_plan"], sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest() if selected[pref]["selected_plan"] else None
                binding = {"task_sha256": entry["task_sha256"], "selection_sha256": entry["selection_sha256"],
                    "plan_sha256": plan_sha, "alias_identity": tid + pref if plan_sha else None}
                s = full_slot(tid, endpoint, pref, phase=manifest["phase"], **binding, actual_binding=binding)
                if tid == "test0" and endpoint == "R12" and pref == "A":
                    original = root / phase / "actual" / tid / "R8_A/slot.json"
                    s.update(unique_run=False, alias_of_slot=str(original), alias_of_slot_sha256=_sha(original), elapsed_wall_s=0.)
                if tid == "test0" and endpoint == "D8" and pref == "B":
                    s.update(full_task_success=False, full_27s_success=False, actual_steps=0, original_independent_gates_passed=False,
                        unique_run=False, quality=None, clearance_30mm_met=None, status="NO_PLAN", diagnostic_category="NO_PLAN")
                _write(path, s); _seal(directory)
        return manifest
    source_binding = {"source_identity_sha256": _sha(root / "source_identity.json")}
    seal_phase("closed_loop_val", ["val0", "val1"], ("R12", "D250", "D4000", "S250", "S4000"), source_binding)
    chosen = {"D": checkpoint_files["D250"], "S": checkpoint_files["S4000"]}
    selection = {"selected_checkpoint_D": "D250", "selected_checkpoint_S": "S4000", "closed_loop_val_completed": True,
        "test_read": False, "selected_checkpoint_files": chosen, "selected_checkpoint_sha256": {m: _sha(p) for m, p in chosen.items()},
        "phase_manifest_sha256": _sha(root / "closed_loop_val/sealed_selections/all_selections.json"),
        "actual_slot_files": {str(p): _sha(p) for p in sorted((root / "closed_loop_val/actual").glob("*/*/slot.json"))}}
    _write(root / "model_selection.json", selection); _write(root / "closed_loop_val/model_selection.json", selection)
    files = [*(root / "models").rglob("*"), root / "dataset/manifest.json", root / "plan.json", root / "model_selection.json"]
    _write(root / "model_freeze.json", {"schema": "v64_c3_frozen_models_and_retrieval_v1", "selected_checkpoints": chosen,
        "artifacts": {str(p): _sha(p) for p in files if p.is_file()}})
    seal_phase("frozen_test", [f"test{i}" for i in range(4)], ENDPOINTS,
        {**source_binding, "model_freeze_sha256": _sha(root / "model_freeze.json")})
    test_paths = list((root / "frozen_test/actual").glob("*/*/slot.json"))
    test_slots = [json.loads(p.read_text()) for p in test_paths]
    all_slots = test_slots + [json.loads(p.read_text()) for p in (root / "closed_loop_val/actual").glob("*/*/slot.json")]
    _write(root / "actual_complete.json", {"logical_slots": 40, "unique_actual": sum(s["unique_run"] for s in test_slots),
        "slot_hashes": {p.relative_to(root).as_posix(): _sha(p) for p in test_paths}})
    counts = {k: limits[k] for k in limits}; counts["smoke_ddim_samples"] = counts["diagnostic_ddim_samples"] = 0
    for i, (category, count) in enumerate(counts.items()):
        if count:
            _write(root / f"budget_ledger/{i}.json", {"category": category, "count": count})
    counts.update(candidate_slots_total=328, actual_slots_total=60, ddim_samples_total=16)
    rows = [r for p in root.glob("**/candidate_registry.json") for r in json.loads(p.read_text())]
    _write(root / "validation/protocol_validation.json", {"all_terminal": True,
        "candidate_slots": {"teacher": 96, "val": 88, "test": 144}, "actual_logical_slots": {"val": 20, "test": 40},
        "budget_reservations": counts, "main_prediction_physics_steps": sum(r["prediction_steps"] for r in rows),
        "main_actual_physics_steps": sum(s["actual_steps"] for s in all_slots if s["unique_run"]), "tool_error_count": 0})


class ReportAuditTests(unittest.TestCase):
    def test_missing_reference_is_na_and_failure_cannot_be_near(self):
        a = full_slot("t", "D8", "A"); r = full_slot("t", "R12", "A")
        self.assertIsNone(_near(a, None, "A"))
        self.assertFalse(_near(None, r, "A"))
        a["quality"]["L_full"] = .1; a["actual_steps"] = 100
        self.assertFalse(_near(a, r, "A"))
        r["five_gates"]["native_geometry"] = None
        self.assertIsNone(_near(a, r, "A"))

    def test_first_miss_right_censored_never_zero_or_slot9(self):
        hit = _hit([{}] * 8, lambda r: False, 8, True)
        self.assertEqual(hit["status"], "RIGHT_CENSORED"); self.assertIsNone(hit["slot"])
        self.assertEqual(hit["sort_encoding"], 9); self.assertFalse(hit["encoding_is_observed_hit"])
        self.assertEqual(_hit([], lambda r: False, 8, False)["status"], "TECHNICAL_INCOMPLETE")
        self.assertEqual(_hit(None, lambda r: False, 8, False)["status"], "NOT_RUN")

    def test_report_keeps_denominators_aliases_cost_scope_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            with patch("v6_4.visualization.report_search_aware_warmstart._plots", return_value=[]):
                result = build_report(root)
            self.assertTrue(result["status"]["independent_test_completed"])
            self.assertTrue(result["status"]["research_execution_completed"])
            self.assertTrue(result["evidence_verification"]["complete"])
            self.assertEqual(result["status"]["full_actual_task_by_method_and_preference"]["D8"]["B"], {"passed": 3, "denominator": 4})
            row = next(r for r in result["endpoint_summary"] if r["endpoint"] == "D8" and r["preference"] == "B")
            self.assertEqual(row["NO_PLAN"], 1); self.assertEqual(row["near_R12_count"], 3)
            self.assertEqual(result["TEST_actual_aliases"], 1)
            self.assertEqual(result["TEST_unique_actual_attempts"], 38)
            r8 = next(r for r in result["paired_comparisons"] if r["method"] == "D8" and r["comparator"] == "R8")
            self.assertIsNone(r8["cost_reductions_all_four_tasks_including_failed_requests"]["end_to_end_cold_planning_s"]["comparator_total"])
            self.assertFalse(result["status"]["learning_benefit_over_simple_regression"])
            self.assertFalse(result["status"]["D_next_round_candidate_engineering_criteria_met"])
            self.assertEqual(result["cost_accounting"]["stages"]["val"]["streams_complete"], 10)
            self.assertEqual(result["cost_accounting"]["stages"]["test"]["consumed_candidate_slots"], 144)
            self.assertEqual(result["cost_accounting"]["stages"]["test"]["summed_outer_process_wall_s"], 16 * 105.)
            self.assertTrue((root / "result_tables/report_artifact_identity.json").is_file())
            identity = json.loads((root / "result_tables/report_artifact_identity.json").read_text())
            self.assertIn(str(root / "frozen_test/actual/test0/R8_A/manifest.json"), identity["input_files"])
            self.assertIn(str(root / "models/D/checkpoint_0250.pt"), identity["input_files"])
            (root / "frozen_test/actual/test1/D8_A/slot.json").unlink()
            with patch("v6_4.visualization.report_search_aware_warmstart._plots", return_value=[]):
                partial = build_report(root)
            self.assertFalse(partial["status"]["independent_test_completed"])
            missing = next(r for r in partial["endpoint_summary"] if r["endpoint"] == "D8" and r["preference"] == "A")
            self.assertEqual(missing["missing_slots"], 1); self.assertEqual(missing["denominator_tasks"], 4)

    def test_more_illegal_raw_cannot_win_on_saved_physics(self):
        tasks = ["t"]
        slots = {(t, e, p): full_slot(t, e, p) for t in tasks for e in ("D8", "S8", "R12") for p in ("A", "B")}
        costs = {("t", "D8"): {"end_to_end_cold_planning_s": 10., "prediction_physics_steps": 100, "raw_rejected": 1},
            ("t", "S8"): {"end_to_end_cold_planning_s": 20., "prediction_physics_steps": 200, "raw_rejected": 0}}
        result = _comparison("D8", "S8", tasks, slots, costs)
        self.assertTrue(result["cost_saving_confounded_by_more_raw_rejections"])
        self.assertFalse(result["engineering_10percent_goal_met"])
        costs[("t", "D8")]["raw_rejected"] = 0
        self.assertTrue(_comparison("D8", "S8", tasks, slots, costs)["engineering_10percent_goal_met"])

    def test_full_comparator_missing_quality_cannot_hide_capability_loss_or_win(self):
        slots = {("t", e, p): full_slot("t", e, p) for e in ("D8", "S8", "R12") for p in ("A", "B")}
        costs = {("t", "D8"): {"end_to_end_cold_planning_s": 10., "prediction_physics_steps": 100, "raw_rejected": 0},
            ("t", "S8"): {"end_to_end_cold_planning_s": 20., "prediction_physics_steps": 200, "raw_rejected": 0}}
        slots[("t", "S8", "A")]["quality"] = None
        slots[("t", "D8", "A")]["full_task_success"] = False
        result = _comparison("D8", "S8", ["t"], slots, costs)
        self.assertFalse(result["capability_preserved_same_tasks"])
        self.assertEqual(result["capability_lost"], [{"task_id": "t", "preference": "A"}])
        self.assertEqual(result["comparator_full_actual_endpoints"], 2)
        self.assertIsNone(result["near_quality_preserved_against_comparator"])
        self.assertFalse(result["engineering_10percent_goal_met"])
        slots[("t", "D8", "A")]["full_task_success"] = True
        result = _comparison("D8", "S8", ["t"], slots, costs)
        self.assertTrue(result["capability_preserved_same_tasks"])
        self.assertIsNone(result["near_quality_preserved_against_comparator"])
        self.assertFalse(result["engineering_10percent_goal_met"])

    def test_full_method_or_r12_missing_quality_is_na_and_withholds_benefit(self):
        slots = {("t", e, p): full_slot("t", e, p) for e in ("D8", "S8", "R12") for p in ("A", "B")}
        costs = {("t", "D8"): {"end_to_end_cold_planning_s": 10., "prediction_physics_steps": 100, "raw_rejected": 0},
            ("t", "S8"): {"end_to_end_cold_planning_s": 20., "prediction_physics_steps": 200, "raw_rejected": 0}}
        slots[("t", "D8", "A")]["quality"] = None
        self.assertIsNone(_near(slots[("t", "D8", "A")], slots[("t", "S8", "A")], "A"))
        result = _comparison("D8", "S8", ["t"], slots, costs)
        self.assertIsNone(result["near_quality_preserved_against_comparator"])
        self.assertIsNone(result["R12_near_quality_preserved"])
        self.assertFalse(result["engineering_10percent_goal_met"])
        slots[("t", "D8", "A")] = full_slot("t", "D8", "A")
        slots[("t", "R12", "A")]["quality"] = None
        result = _comparison("D8", "S8", ["t"], slots, costs)
        self.assertIsNone(result["R12_near_quality_preserved"])
        self.assertFalse(result["engineering_10percent_goal_met"])

    def test_actual_and_alias_source_tampering_are_rejected(self):
        for relative in ("frozen_test/actual/test1/D8_A/slot.json", "frozen_test/actual/test0/R8_A/slot.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fixture(root); path = root / relative
                slot = json.loads(path.read_text()); slot["quality"]["I_support"] = .001; _write(path, slot)
                with self.assertRaisesRegex(ValueError, "actual evidence changed|aliased original actual slot changed"):
                    build_report(root)

    def test_model_phase_and_stale_validation_tampering_are_rejected(self):
        cases = (("models/D/checkpoint_0250.pt", "model/data/config changed after freeze"),
            ("test_search/test1/D/planning/test1/candidate_registry.json", "sealed phase file changed"),
            ("validation/protocol_validation.json", "terminal validation no longer matches"))
        for relative, error in cases:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fixture(root); path = root / relative
                if path.suffix == ".pt":
                    path.write_bytes(b"changed")
                else:
                    value = json.loads(path.read_text())
                    if isinstance(value, list):
                        value[0]["prediction_steps"] = 1
                    else:
                        value["main_prediction_physics_steps"] = 1
                    _write(path, value)
                with self.assertRaisesRegex(ValueError, error):
                    build_report(root)

    def test_missing_seal_or_formal_process_receipt_keeps_report_incomplete(self):
        for relative in ("frozen_test/actual/test1/D8_A/manifest.json", "test_search/test1/D/outer_process.json",
                "frozen_test/sealed_selections/all_selections.json", "actual_complete.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fixture(root); (root / relative).unlink()
                with patch("v6_4.visualization.report_search_aware_warmstart._plots", return_value=[]):
                    result = build_report(root)
                self.assertFalse(result["status"]["independent_test_completed"])
                self.assertFalse(result["status"]["research_execution_completed"])
                self.assertFalse(result["evidence_verification"]["complete"])
                self.assertTrue(result["evidence_verification"]["missing_or_pending"])
                self.assertEqual(result["status"]["learning_benefit_over_rules"], "NOT_ESTABLISHED")

    def test_empty_run_can_still_produce_an_honest_partial_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("v6_4.visualization.report_search_aware_warmstart._plots", return_value=[]):
                result = build_report(Path(tmp))
            self.assertFalse(result["status"]["independent_test_completed"])
            self.assertFalse(result["evidence_verification"]["complete"])

    def test_stream_nominal_failures_are_counted_and_unbound_registry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); stream(root, "t", "D", raw_illegal=1)
            evidence = Evidence(root); result = evidence.stream("test", "t", "D", 8)
            self.assertEqual(result["nominal_status_counts"]["INITIALIZER_RAW_REJECTED"], 1)
            path = root / "test_search/t/D/planning/t/candidate_registry.json"
            rows = json.loads(path.read_text()); rows[1].update(status="EXECUTION_REFUSED", prediction_admissible=False,
                prediction_steps=123)
            _write(path, rows)
            with self.assertRaisesRegex(ValueError, "consumed candidate registry differ"):
                evidence.stream("test", "t", "D", 8)
            selection_path = path.parent / "selection.json"; selection = json.loads(selection_path.read_text())
            selection["registry_content_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(); _write(selection_path, selection)
            result = evidence.stream("test", "t", "D", 8)
            self.assertEqual(result["nominal_status_counts"]["EXECUTION_REFUSED"], 1)
            self.assertEqual(result["nominal_failed_rollouts"], 1)
            self.assertEqual(result["nominal_positive_step_incomplete_rollouts"], 1)

    def test_prefix_digest_and_request_time_are_not_reconstructed_from_r12(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); stream(root, "t", "R")
            evidence = Evidence(root); s = evidence.stream("test", "t", "R", 12)
            prefix = _prefix(evidence, s, 8); cost = _endpoint_cost(s, "R8", prefix)
            self.assertEqual(cost["optimizer_elapsed_s"], 40.)
            self.assertIsNone(cost["end_to_end_cold_planning_s"])
            path = root / "test_search/t/R/planning/t/candidate_registry.json"
            rows = json.loads(path.read_text()); rows[0]["prediction_steps"] = 1; _write(path, rows)
            with self.assertRaisesRegex(ValueError, "consumed candidate registry differ"):
                evidence.stream("test", "t", "R", 12)
            s["rows"] = rows
            with self.assertRaisesRegex(ValueError, "prefix candidate pool changed"):
                _prefix(evidence, s, 8)


if __name__ == "__main__":
    unittest.main()
