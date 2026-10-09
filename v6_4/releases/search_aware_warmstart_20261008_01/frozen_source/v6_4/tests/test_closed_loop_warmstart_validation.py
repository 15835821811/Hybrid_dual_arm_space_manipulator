"""C.3 selection/execution contracts, using mock evidence and no physics."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4 import closed_loop_warmstart_validation as validation


class MockTask:
    model_contract_sha256 = "model"
    initial_qpos = [0.] * 81
    initial_qvel = [0.] * 79
    initial_planner_q = [0.] * 17
    initial_planner_dq = [0.] * 17

    def __init__(self, task_id):
        self.task_id = task_id

    def sha256(self):
        return "sha-" + self.task_id


class MockPlan:
    def __init__(self, value):
        self.value = value

    def sha256(self):
        return validation._digest(self.value)

    def to_dict(self):
        return self.value


class ClosedLoopValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.run = self.root / "run"
        self.run.mkdir(); self.phase = self.root / "val"
        for name in ("source_identity.json", "frozen_execution_config.json", "frozen_run_config.json"):
            validation._write(self.run / name, {"identity": name})
        self.checkpoints = {}
        for endpoint in validation.CHECKPOINTS:
            path = self.root / (endpoint + ".pt"); path.write_bytes(endpoint.encode())
            self.checkpoints[endpoint] = path
        self.physics = patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")).start()
        self.addCleanup(patch.stopall)
        self.entries = []

    def tearDown(self):
        self.physics.assert_not_called()

    def entry(self, task_id, endpoint, *, plan=True, raw=0, first=1, work=10, duration=2., distinct=False):
        directory = self.root / "search" / task_id / endpoint; directory.mkdir(parents=True)
        metrics = {"I_support": .01, "L_full": .3, "d_support": .031}
        rows = [{"candidate_id": "C00", "prediction_admissible": first == 1,
                 "prediction_metrics": metrics, "status": "INITIALIZER_RAW_REJECTED" if raw else "PREDICTED_COMPLETE",
                 "costs": {"prediction_physics_steps": work, "private_preview_physics_steps": work,
                           "native_geometry_query_calls": work}}]
        if first == 2:
            rows.append({**rows[0], "candidate_id": "C01", "status": "PREDICTED_COMPLETE", "prediction_admissible": True})
        selected = {"selected_plan": ({"endpoint": endpoint} if distinct else {"z": [0., 0.]}) if plan else None,
                    "source_candidate_id": "C00" if plan else None, "prediction_metrics": metrics if plan else None}
        selection = {"task_id": task_id, "task_sha256": MockTask(task_id).sha256(),
            "preferences": {"A": selected, "B": selected}, "selection_reads_final_actual": False,
            "budget": {"candidate_budget": 12 if endpoint == "R12" else 8, "slots_consumed": len(rows), "stop_reason": "MINIMUM_STEP_POLL_COMPLETE"},
            "registry_content_sha256": validation._digest(rows)}
        paths = {"selection_path": directory / "selection.json", "candidate_registry_path": directory / "candidate_registry.json",
                 "proposals_path": directory / "proposals.json", "planning_cost_path": directory / "planning_cost.json"}
        for name, value in (("selection_path", selection), ("candidate_registry_path", rows),
                            ("proposals_path", []), ("planning_cost_path", {"end_to_end_cold_planning_s": duration})):
            validation._write(paths[name], value)
        entry = {"task_id": task_id, "endpoint": endpoint, **paths}; self.entries.append(entry)
        return entry

    def freeze(self, **kwargs):
        if not self.entries:
            for task in ("val_a", "val_b"):
                for endpoint in validation.VAL_ENDPOINTS:
                    self.entry(task, endpoint, **kwargs)
        return validation.freeze_phase_selections(self.phase, self.entries, checkpoint_files=self.checkpoints)

    @staticmethod
    def runner(task, plan, directory, **kwargs):
        return {"status": "TASK_COMPLETED", "actual_runner_started": True, "entered_actual": True,
                "actual_steps": 13500, "full_task_success": True,
                "evaluation": {"evidence_valid": True, **{k: {"passed": True} for k in validation.FIVE_GATES}},
                "quality": {"I_support": .01, "L_full": .3, "d_support": .031}, "costs": {}, "elapsed_wall_s": 1.}

    def execute(self, task_id="val_a", **kwargs):
        return validation.execute_frozen_task(self.phase, MockTask(task_id), {}, execution_run=self.run,
            executor=kwargs.pop("executor", self.runner), plan_loader=MockPlan, **kwargs)

    def test_digest_matches_original_including_unicode(self):
        from v6_4.route_optimizer_protocol import digest
        value = [{"label": "净空", "x": .001, "finite": True}]
        self.assertEqual(validation._digest(value), digest(value))

    def test_whitelist_and_complete_phase_before_any_actual(self):
        self.entry("val_a", "D250")
        with self.assertRaises(ValueError): self.freeze()
        self.assertFalse((self.phase / "actual").exists())
        bad = dict(self.checkpoints); bad["D500"] = bad.pop("D250")
        with self.assertRaises(ValueError):
            validation.freeze_phase_selections(self.phase, [], checkpoint_files=bad)

    def test_checkpoint_and_registry_bytes_are_bound(self):
        self.freeze()
        self.checkpoints["D250"].write_bytes(b"changed")
        with self.assertRaises(ValueError): validation.verify_phase_selections(self.phase)

    def test_alias_same_task_config_plan_history_only_and_resume(self):
        self.freeze(); calls = []
        def runner(*args, **kwargs):
            calls.append(kwargs["slot_id"]); return self.runner(*args, **kwargs)
        slots = self.execute(executor=runner)
        self.assertEqual(len(slots), 10); self.assertEqual(len(calls), 1)
        self.assertEqual(sum(s["unique_run"] for s in slots), 1)
        self.assertEqual(sum("alias_of_slot" in s for s in slots), 9)
        resumed = self.execute(executor=runner)
        self.assertEqual(len(resumed), 10); self.assertEqual(len(calls), 1)
        self.execute("val_b", executor=runner)
        self.assertEqual(len(calls), 2)
        task = MockTask("val_a"); plan = MockPlan({"z": [0., 0.]})
        original = validation.actual_alias_identity(task, plan, self.run)
        changed = validation.cold_initial_history(task); changed["QP_history"] = [1.]
        self.assertNotEqual(original, validation.actual_alias_identity(task, plan, self.run, initial_history_identity=changed))
        (self.run / "frozen_execution_config.json").write_text('{"changed":true}')
        self.assertNotEqual(original, validation.actual_alias_identity(task, plan, self.run))
        with self.assertRaises(ValueError): self.execute(executor=runner)

    def test_no_plan_keeps_zero_steps_and_no_runner_call(self):
        self.freeze(plan=False)
        def forbidden(*args, **kwargs): raise AssertionError("NO_PLAN must not execute")
        slots = self.execute(executor=forbidden)
        self.assertTrue(all(s["diagnostic_category"] == "NO_PLAN" and s["actual_steps"] == 0 for s in slots))
        self.assertEqual(sum(s["unique_run"] for s in slots), 0)

    def test_tool_error_is_saved_separately_and_never_retried(self):
        self.freeze()
        def broken(*args, **kwargs): raise FileNotFoundError("tool unavailable")
        with self.assertRaises(RuntimeError): self.execute(executor=broken)
        saved = validation._read(self.phase / "actual/val_a/R12_A/slot.json")
        self.assertEqual(saved["diagnostic_category"], "TOOL_ERROR")
        self.assertEqual(saved["actual_steps"], 0)
        with self.assertRaises(RuntimeError): self.execute()

    def test_late_quality_tool_failure_retains_consumed_actual_count(self):
        result = self.runner(None, None, None)
        def broken_quality(*args): raise FileNotFoundError("quality artifact unavailable")
        with patch("v6_4.residual_execution.execute_residual_attempt", return_value=result):
            saved = validation._run_original(MockTask("val_a"), MockPlan({}), self.root / "attempt_slot",
                execution_run=self.run, frozen={"obstacle_name": "mock"}, slot_id="mock", quality_reader=broken_quality)
        self.assertEqual(saved["actual_steps"], 13500)
        self.assertTrue(saved["entered_actual"]); self.assertTrue(saved["actual_runner_started"])
        self.assertEqual(validation._diagnostic_category(saved), "TOOL_ERROR")

    def test_refusal_precheck_and_actual_failure_are_distinct(self):
        self.assertEqual(validation._diagnostic_category({"status": "REFERENCE_PRECHECK_REJECTED", "actual_steps": 0}), "ACTUAL_PRECHECK_REJECTED")
        self.assertEqual(validation._diagnostic_category({"status": "EXECUTION_REFUSED", "actual_steps": 10,
            "execution_failure": {"type": "UncertifiedExecutionError"}}), "ACTUAL_FAILURE")
        self.assertEqual(validation._diagnostic_category({"status": "EXECUTION_REFUSED", "actual_steps": 10,
            "execution_failure": {"type": "TypeError"}}), "TOOL_ERROR")
        incomplete = self.runner(None, None, None); incomplete["evaluation"]["native_geometry"] = {"passed": False}
        self.assertFalse(validation.five_gates_passed(incomplete["evaluation"]))
        self.assertEqual(validation._diagnostic_category(incomplete), "ACTUAL_FAILURE")

    def test_missing_R12_is_na_method_failure_is_false_and_hit_is_censored(self):
        baseline = {"full_task_success": True, "original_independent_gates_passed": True,
                    "quality": {"I_support": .01, "L_full": .3, "d_support": .031}}
        self.assertIsNone(validation.actual_near_quality({}, {}, "A"))
        self.assertFalse(validation.actual_near_quality({}, baseline, "A"))
        self.assertTrue(validation.actual_near_quality(baseline, baseline, "B"))
        miss = validation.first_near_quality([], baseline, "A")
        self.assertEqual(miss["status"], "RIGHT_CENSORED"); self.assertEqual(miss["sort_encoding"], 9)
        self.assertIsNone(miss["slot"]); self.assertFalse(miss["encoding_is_observed_hit"])

    def test_score_ties_select_earlier_update_no_test_reads(self):
        self.freeze(); self.execute(); self.execute("val_b")
        # A forbidden TEST location is never traversed by the scorer.
        (self.root / "test.json").write_text("not valid JSON")
        result = validation.score_checkpoints(self.phase)
        self.assertEqual(result["selected_checkpoint_D"], "D250")
        self.assertEqual(result["selected_checkpoint_S"], "S250")
        self.assertFalse(result["test_read"])
        self.assertEqual(result["scores"]["D250"]["full_27s_five_gate_endpoints"], 4)
        self.assertEqual(result["scores"]["D250"]["near_quality_endpoints"], 4)
        self.assertEqual(validation.score_checkpoints(self.phase), result)

    def test_quality_capability_precedes_cheaper_failure(self):
        for task in ("val_a", "val_b"):
            for endpoint in validation.VAL_ENDPOINTS:
                self.entry(task, endpoint, plan=endpoint != "D250", work=0 if endpoint == "D250" else 100,
                           duration=0. if endpoint == "D250" else 20.)
        self.freeze(); self.execute(); self.execute("val_b")
        result = validation.score_checkpoints(self.phase)
        self.assertEqual(result["selected_checkpoint_D"], "D4000")
        self.assertEqual(result["scores"]["D250"]["near_quality_endpoints"], 0)

    def test_first_hit_and_workload_precede_update_tie(self):
        for task in ("val_a", "val_b"):
            for endpoint in validation.VAL_ENDPOINTS:
                self.entry(task, endpoint, first=2 if endpoint == "D250" else 1, work=100 if endpoint == "S250" else 10)
        self.freeze(); self.execute(); self.execute("val_b")
        result = validation.score_checkpoints(self.phase)
        self.assertEqual(result["selected_checkpoint_D"], "D4000")
        self.assertEqual(result["selected_checkpoint_S"], "S4000")

    def test_raw_illegal_precedes_first_hit_and_cost(self):
        for task in ("val_a", "val_b"):
            for endpoint in validation.VAL_ENDPOINTS:
                self.entry(task, endpoint, raw=1 if endpoint == "D250" else 0,
                           first=2 if endpoint == "D4000" else 1)
        self.freeze(); self.execute(); self.execute("val_b")
        self.assertEqual(validation.score_checkpoints(self.phase)["selected_checkpoint_D"], "D4000")

    def test_B30_and_near_quality_precede_raw_and_cost(self):
        for task in ("val_a", "val_b"):
            for endpoint in validation.VAL_ENDPOINTS:
                self.entry(task, endpoint, distinct=True, raw=endpoint in ("D4000", "S4000"),
                           work=100 if endpoint in ("D4000", "S4000") else 0)
        self.freeze()
        def runner(*args, **kwargs):
            result = self.runner(*args, **kwargs); endpoint = args[1].value["endpoint"]
            if endpoint == "D250": result["quality"]["d_support"] = .029
            if endpoint == "S250": result["quality"]["I_support"] = .020
            return result
        self.execute(executor=runner); self.execute("val_b", executor=runner)
        result = validation.score_checkpoints(self.phase)
        self.assertEqual(result["selected_checkpoint_D"], "D4000")
        self.assertEqual(result["selected_checkpoint_S"], "S4000")

    def test_missing_reference_quality_is_na_and_shared_reference_set(self):
        self.assertIsNone(validation.actual_near_quality({}, {
            "full_task_success": True, "original_independent_gates_passed": True, "quality": {"L_full": .3}}, "A"))
        for task in ("val_a", "val_b"):
            for endpoint in validation.VAL_ENDPOINTS: self.entry(task, endpoint, distinct=True)
        self.freeze()
        def runner(*args, **kwargs):
            result = self.runner(*args, **kwargs)
            if args[1].value["endpoint"] == "R12": result["quality"]["d_support"] = None
            return result
        self.execute(executor=runner); self.execute("val_b", executor=runner)
        result = validation.score_checkpoints(self.phase)
        self.assertEqual(len(result["R12_qualified_reference_set"]), 2)
        self.assertTrue(all(row["preference"] == "A" for row in result["R12_qualified_reference_set"]))
        for score in result["scores"].values():
            b_rows = [r for r in score["near_quality_by_endpoint"] if r["preference"] == "B"]
            self.assertTrue(all(r["near_quality"] is None for r in b_rows))

    def test_missing_formal_workload_cannot_win_by_omission(self):
        self.freeze()
        # Byte-bound files must change the phase seal if omitted *after* freeze.
        entry = next(e for e in self.entries if e["endpoint"] == "D250")
        rows = validation._read(entry["candidate_registry_path"]); del rows[0]["costs"]["native_geometry_query_calls"]
        Path(entry["candidate_registry_path"]).write_text(json.dumps(rows), encoding="utf8")
        with self.assertRaises(ValueError): validation.score_checkpoints(self.phase)

    def test_test_phase_cannot_select_checkpoint(self):
        self.entry("test_a", "R8")
        validation.freeze_phase_selections(self.phase, self.entries, phase="TEST", expected_endpoints=["R8"])
        with self.assertRaises(ValueError): validation.score_checkpoints(self.phase)

    def test_original_consumed_command_check_rejects_nan_but_ignores_refused_row(self):
        from v6_4.evaluate_planning import _execution_trace_checks
        n = 10
        trace = {"torque": np.zeros((n, 67)), "command_velocity": np.zeros((n, 17)),
            "task_selected_command": np.vstack((np.zeros((1, 17)), np.full((1, 17), np.nan))),
            "task_failure_reason": np.array(["none", "refused"]), "task_execution_mode": np.array(["TRACK", "STOP"]),
            "reference_velocity_error_norm_rad_s": np.zeros(n)}
        for key in ("reference_q", "reference_velocity", "feedforward_acceleration",
                    "measured_planner_q_before_servo", "measured_velocity_before_servo"):
            trace[key] = np.zeros((n, 17))
        for key in ("reference_joint_limit_clip_count", "reference_measured_window_clip_count", "acceleration_clip_count", "torque_saturation_count"):
            trace[key] = np.zeros(n)
        spec = SimpleNamespace(planner_lower=np.full(17, -1.), planner_upper=np.ones(17), torque_limits=np.ones(67))
        self.assertTrue(_execution_trace_checks(MockTask("mock"), trace, spec, None)["passed"])
        trace["task_selected_command"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "consumed task_selected_command"):
            _execution_trace_checks(MockTask("mock"), trace, spec, None)


if __name__ == "__main__":
    unittest.main()
