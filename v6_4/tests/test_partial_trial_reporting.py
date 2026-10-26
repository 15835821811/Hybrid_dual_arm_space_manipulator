"""Attempt failure reporting with mocked execution and mocked evaluation only."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_4 import run_planning
from v6_4.tests.test_task_protocol import fixture_task


def partial_trace(task, steps=10):
    trace = {
        "torque": np.zeros((steps, 67)),
        "time": np.arange(1, steps + 1) * .002,
        "initial_qpos": np.asarray(task.initial_qpos),
        "initial_qvel": np.asarray(task.initial_qvel),
        "task_qpos": np.tile(task.initial_qpos, (steps // 10 + 1, 1)),
        "planner_q": np.zeros((steps, 17)),
        "command_velocity": np.zeros((steps, 17)),
        "task_selected_command": np.zeros((steps // 10, 17)),
        "task_failure_reason": np.full(steps // 10, "none"),
        "task_execution_mode": np.full(steps // 10, "TRACK"),
        "task_full_latency": np.full(steps // 10, .003),
        "torque_latency": np.full(steps, .001),
        "failure_time_s": np.asarray(steps * .002),
    }
    for key in ("reference_q", "reference_velocity", "feedforward_acceleration",
                "measured_planner_q_before_servo", "measured_velocity_before_servo"):
        trace[key] = np.zeros((steps, 17))
    for key in ("reference_joint_limit_clip_count", "reference_measured_window_clip_count",
                "acceleration_clip_count", "torque_saturation_count",
                "reference_velocity_error_norm_rad_s"):
        trace[key] = np.zeros(steps)
    return trace


class PartialTrialReportingTests(unittest.TestCase):
    def run_failed_attempt(self, output, trace=None, *, foreign_only=False,
                           existing_evaluation=None, evaluator_report=None):
        task = fixture_task()
        scenario = SimpleNamespace(scenario_id="current_scenario")
        failure_message = "accepted executor stopped before completing the task"
        partial_path = output / "failures" / "current_scenario_partial_trace.npz"
        report = evaluator_report or {"complete": False, "task_success": False,
                                      "evidence_valid": True, "metrics": {"physics_steps": 10}}

        def start(path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=False)
            return {"mock_run": True}

        def execute(*args, **kwargs):
            if trace is not None:
                partial_path.parent.mkdir()
                # A foreign scenario's valid trace must never be selected.
                foreign = partial_path.parent / "other_scenario_partial_trace.npz"
                np.savez_compressed(foreign, **partial_trace(task))
                if not foreign_only:
                    np.savez_compressed(partial_path, **trace)
            if existing_evaluation is not None:
                folder = output / "evaluation"
                folder.mkdir()
                (folder / "report.json").write_bytes(existing_evaluation)
            raise RuntimeError(failure_message)

        def evaluate(*args, **kwargs):
            folder = Path(kwargs["output_dir"])
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
            return report.copy()

        with ExitStack() as stack:
            stack.enter_context(patch.object(run_planning, "default_v6_lite_robot_spec",
                                             return_value=SimpleNamespace()))
            stack.enter_context(patch.object(run_planning, "scenario_from_task", return_value=scenario))
            stack.enter_context(patch.object(run_planning, "start_run", side_effect=start))
            finish = stack.enter_context(patch.object(run_planning, "finish_run"))
            executor = stack.enter_context(patch.object(run_planning, "run_synchronous_scenario",
                                                        side_effect=execute))
            evaluator = stack.enter_context(patch("v6_4.evaluate_planning.evaluate_trial",
                                                  side_effect=evaluate))
            result = run_planning.run_attempt(task, output, method="fixed_reference")
        executor.assert_called_once()
        self.assertTrue(result["proposal_accepted"])
        self.assertEqual(result["status"], "EXECUTION_OR_PIPELINE_FAILURE")
        self.assertFalse(result["task_success"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure"]["type"], "RuntimeError")
        self.assertEqual(result["failure"]["message"], failure_message)
        self.assertEqual(json.loads((output / "pipeline_failure.json").read_text(encoding="utf-8")),
                         result["failure"])
        self.assertEqual(json.loads((output / "result.json").read_text(encoding="utf-8"))["status"],
                         "EXECUTION_OR_PIPELINE_FAILURE")
        finish.assert_called_once()
        self.assertFalse(finish.call_args.kwargs["passed"])
        return result, evaluator, partial_path

    def test_complete_prefix_is_evaluated_once_without_upgrading_execution_failure(self):
        optimistic = {"complete": True, "task_success": True, "passed": True,
                      "evidence_valid": True, "metrics": {"physics_steps": 10}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "attempt"
            result, evaluator, partial_path = self.run_failed_attempt(
                output, partial_trace(fixture_task()), evaluator_report=optimistic)
            evaluator.assert_called_once()
            self.assertEqual(Path(evaluator.call_args.args[1]), partial_path)
            self.assertEqual(result["trace_path"], partial_path.resolve().as_posix())
            self.assertIsNone(evaluator.call_args.args[2])
            self.assertEqual(Path(evaluator.call_args.kwargs["output_dir"]), output / "evaluation")
            self.assertEqual(result["evaluation"]["metrics"]["physics_steps"], 10)
            self.assertTrue((output / "evaluation" / "report.json").is_file())

    def test_missing_foreign_incomplete_or_invalid_prefix_gets_diagnostic_only(self):
        task = fixture_task()
        bad_torque = partial_trace(task)
        bad_torque["torque"] = np.zeros((10, 66))
        missing_torque = partial_trace(task)
        del missing_torque["torque"]
        cases = (("missing", None, False), ("foreign_only", partial_trace(task), True),
                 ("incomplete_ramp", partial_trace(task, 9), False),
                 ("invalid_torque_shape", bad_torque, False),
                 ("missing_schema_field", missing_torque, False))
        for name, trace, foreign_only in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "attempt"
                result, evaluator, _ = self.run_failed_attempt(output, trace, foreign_only=foreign_only)
                evaluator.assert_not_called()
                self.assertIsNone(result.get("evaluation"))
                self.assertTrue(result.get("partial_evaluation_diagnostic"))
                self.assertNotEqual(result["partial_evaluation_diagnostic"], result["failure"])
                self.assertFalse((output / "evaluation").exists())

    def test_existing_evaluation_is_preserved_and_diagnosed_separately(self):
        original = b'{"existing_report":"preserve exact bytes"}\n'
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "attempt"
            result, evaluator, _ = self.run_failed_attempt(
                output, partial_trace(fixture_task()), existing_evaluation=original)
            evaluator.assert_not_called()
            self.assertEqual((output / "evaluation" / "report.json").read_bytes(), original)
            self.assertEqual({p.name for p in (output / "evaluation").iterdir()}, {"report.json"})
            self.assertIsNone(result.get("evaluation"))
            self.assertTrue(result.get("partial_evaluation_diagnostic"))
            self.assertNotEqual(result["partial_evaluation_diagnostic"], result["failure"])


if __name__ == "__main__":
    unittest.main()
