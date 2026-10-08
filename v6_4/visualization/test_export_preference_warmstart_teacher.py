"""Final teacher exporter checks using archived JSON, zero physics/inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import mujoco

from v6_4.preference_warmstart_protocol import HISTORY
from v6_4.route_optimizer_protocol import read, write, sha
from v6_4.task_protocol import TaskSpec
from v6_4.task_anchored_reference import TaskAnchoredResidualPlan
from v6_4.visualization.export_preference_warmstart_teacher import (
    SCHEMA, ENDPOINTS, bind_actual, compact_costs, merge_prediction, preference_views, _reuse, export_teacher_update)
from v6_4.route_optimizer_protocol import digest


class FinalTeacherTests(unittest.TestCase):
    def setUp(self):
        self.forbid_step = patch.object(mujoco, "mj_step", side_effect=AssertionError("no physics"))
        self.forbid_step2 = patch.object(mujoco, "mj_step2", side_effect=AssertionError("no physics"))
        self.forbid_step.start(); self.forbid_step2.start()
        self.addCleanup(self.forbid_step.stop); self.addCleanup(self.forbid_step2.stop)
        snapshot = HISTORY / "snapshot"
        tid = "c1_mother_00_plus"
        self.task = TaskSpec.from_dict(read(snapshot / "frozen_tasks" / tid / "task.json"))
        self.row = read(snapshot / "planning" / tid / "candidate_registry.json")[0]
        self.plan = TaskAnchoredResidualPlan.from_dict(self.row["plan"])
        self.frozen = {"mother_id": "synthetic_test_mother", "mother_source_sha256": "source_identity"}
        directory = snapshot / "actual" / tid / "Z0"
        self.slot = read(directory / "slot.json")
        self.slot.update(method="R8_A", endpoint="R8", preference="A")
        self.slot["_source_evidence"] = {
            "report": read(directory / "attempt/actual/evaluation/report.json"),
            "attempt": read(directory / "attempt/attempt_result.json"),
            "plan": read(directory / "selected_plan.json"), "source_actual_method": "R8_A"}

    def test_cross_method_physical_dedup_preserves_prediction_cost_and_ids(self):
        facts = {}
        merge_prediction(facts, self.row, self.task, self.frozen, "R", [], {"path": "R/registry", "sha256": "r"})
        other = copy.deepcopy(self.row)
        other.update(candidate_id="C07", parent_candidate_id="C05", source="adaptive_poll",
            proposal_lineage=[{"parent_candidate_id": "C05", "source": "adaptive_poll"}])
        other["elapsed_wall_s"] += 10.
        merge_prediction(facts, other, self.task, self.frozen, "D", [], {"path": "D/registry", "sha256": "d"})
        self.assertEqual(len(facts), 1)
        fact = next(iter(facts.values()))
        self.assertEqual([r["candidate_id"] for r in fact["prediction_runs"]], ["C00", "C07"])
        self.assertNotEqual(*[r["elapsed_wall_s"] for r in fact["prediction_runs"]])
        self.assertEqual(fact["evidence_tier"], "PREDICTED_COMPLETE")
        self.assertTrue(fact["forbidden_current_C2_training"])
        views = preference_views([fact], "frozen_model")
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0]["z_m"], [[0., 0.]] * 6)

    def test_validated_actual_below30mm_is_not_B_and_each_prediction_binds(self):
        binding = bind_actual(self.slot, self.row, self.task, self.plan)
        self.assertTrue(binding["independently_bound_and_validated"], binding["checks"])
        self.assertFalse(binding["B_clearance_30mm"])
        facts = {}
        merge_prediction(facts, self.row, self.task, self.frozen, "R", [binding], {"path": "registry"})
        views = preference_views(list(facts.values()), "frozen_model")
        self.assertTrue(views[1]["validated_execution_example"])
        self.assertFalse(views[1]["preference_qualified"])
        corrupt = copy.deepcopy(self.row)
        corrupt["prediction_metrics"]["I_support"] += .001
        self.assertFalse(bind_actual(self.slot, corrupt, self.task, self.plan)["independently_bound_and_validated"])
        missing = copy.deepcopy(self.slot)
        missing["_source_evidence"]["report"]["independent_interval"]["passed"] = False
        self.assertFalse(bind_actual(missing, self.row, self.task, self.plan)["independently_bound_and_validated"])

    def test_compact_costs_bind_full_receipt_and_immutable_reuse(self):
        original = {"elapsed_wall_s": 5., "prediction_physics_steps": 13500,
            "preview_records": [{"call": 0}], "phase_counts": {"actual": 13500}}
        compact = compact_costs(original)
        self.assertNotIn("preview_records", compact)
        self.assertEqual(compact["phase_counts"], original["phase_counts"])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); output = run / "teacher_update"
            write(run / "model_freeze.json", {"checkpoint": "immutable"})
            write(run / "source.json", {"source": "one"})
            write(output / "source_inventory.json", {"files": [{"path": "source.json", "sha256": sha(run / "source.json")}]})
            write(output / "manifest.json", {"schema": SCHEMA, "model_freeze_sha256": sha(run / "model_freeze.json"),
                "output_sha256": {"teacher_update/source_inventory.json": sha(output / "source_inventory.json")}})
            self.assertEqual(_reuse(run, output)["schema"], SCHEMA)
            (run / "source.json").write_text('{"source":"changed"}', encoding="utf8")
            with self.assertRaisesRegex(ValueError, "source changed"):
                _reuse(run, output)

    def test_complete_synthetic_export_keeps_planless_rejections_and_reuses(self):
        """Gate is mocked; every output derives from temporary archived JSON."""
        snapshot = HISTORY / "snapshot"
        frozen_tasks = [dict(r, mother_id=r["group_id"], split="test") for r in read(snapshot / "plan.json")["tasks"]]
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in ("source_identity.json", "frozen_execution_config.json", "frozen_run_config.json"):
                shutil.copyfile(snapshot / name, run / name)
            write(run / "model_freeze.json", {"checkpoint": "synthetic_only"})
            freeze_sha = sha(run / "model_freeze.json")
            complete = {"logical_slots": 32, "unique_actual_runs": 4, "slot_hashes": {}}
            for frozen in frozen_tasks:
                tid = frozen["task_id"]
                task_path = run / "frozen_tasks" / tid / "task.json"
                task_path.parent.mkdir(parents=True)
                shutil.copyfile(snapshot / "frozen_tasks" / tid / "task.json", task_path)
                row = read(snapshot / "planning" / tid / "candidate_registry.json")[0]
                original_actual = snapshot / "actual" / tid / "Z0"
                original_slot = read(original_actual / "slot.json")
                for endpoint in ENDPOINTS:
                    selection = run / "sealed_selections" / tid / (endpoint + ".json")
                    write(selection, {"synthetic": True})
                    for preference in ("A", "B"):
                        method = endpoint + "_" + preference
                        directory = run / "actual" / tid / method
                        directory.mkdir(parents=True)
                        slot = {**original_slot, "method": method, "endpoint": endpoint, "preference": preference,
                            "selection_sha256": sha(selection), "unique_run": method == "R8_A",
                            "alias_identity": digest({"task_sha256": original_slot["task_sha256"], "plan_sha256": original_slot["plan_sha256"],
                                "source_identity_sha256": sha(run / "source_identity.json"),
                                "config_sha256": sha(run / "frozen_execution_config.json"),
                                "run_config_sha256": sha(run / "frozen_run_config.json")})}
                        if method != "R8_A":
                            slot["alias_of_method"] = "R8_A"
                        else:
                            slot.pop("alias_of_method", None)
                            for relative in ("selected_plan.json", "attempt/attempt_result.json", "attempt/actual/evaluation/report.json"):
                                target = directory / relative
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(original_actual / relative, target)
                        write(directory / "slot.json", slot)
                        write(directory / "manifest.json", {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob("*") if p.is_file()})
                        complete["slot_hashes"][(directory / "slot.json").relative_to(run).as_posix()] = sha(directory / "slot.json")
                for method in ("R", "N", "D"):
                    root = run / "benchmark_search" / tid / method
                    planning = root / "planning" / tid
                    rows = [row]
                    if method == "D":
                        rows += [{"candidate_id": "C01", "source": "diffusion", "status": "INITIALIZER_RAW_REJECTED",
                            "plan": None, "plan_sha256": None, "raw_z_m": [0.] * 12,
                            "prediction_steps": 0, "prediction_rollout_started": False,
                            "raw_seed_diagnostics": {"raw_legal": False, "rejection_reason": "synthetic"}}]
                    write(planning / "candidate_registry.json", rows)
                    write(planning / "proposals.json", rows)
                    write(planning / "selection.json", {"synthetic": True})
                    write(root / "planning_cost.json", {"synthetic": True})
            target = "v6_4.visualization.export_preference_warmstart_teacher._gate"
            with patch(target, return_value=(frozen_tasks, complete, freeze_sha)):
                result = export_teacher_update(run)
            self.assertEqual(result["evaluation_slots"], 16)
            self.assertEqual(result["physical_candidates_unique"], 4)
            self.assertEqual(result["distinct_prediction_runs"], 12)
            self.assertEqual(result["rejected_planless_slots"], 4)
            self.assertEqual(result["preference_view_rows"], 8)
            self.assertEqual(result["physics_steps_added"], 0)
            self.assertEqual(export_teacher_update(run), result)
            rejected = read(run / "teacher_update/rejected_proposals.json")
            self.assertTrue(all(r["plan"] is None and r["evaluation_slot_consumed"] for r in rejected))
            self.assertEqual(sha(run / "model_freeze.json"), freeze_sha)


if __name__ == "__main__":
    unittest.main()
