"""Saved-media provenance checks with synthetic arrays and no integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from .build_preference_warmstart_media import (C2RenderOnly, _angle, core_charts, dashboard,
    execution_model_binding, load_actual, resolve_unique, verify_render_model)
from . import render_residual_replays as legacy
from ..route_optimizer_protocol import HISTORY, TaskSpec, parameter_plan, read, sha
from ..task_anchored_reference import TaskAnchoredResidualReferenceProvider


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


class PreferenceMediaTests(unittest.TestCase):
    def test_alias_requires_full_identity_and_cannot_cycle(self):
        original = {"task_id": "task", "method": "R8_A", "task_sha256": "tasksha", "plan_sha256": "plan", "alias_identity": "full"}
        alias = {**original, "method": "D8_A", "alias_of_method": "R8_A"}
        self.assertEqual(resolve_unique([original, alias], "task", "D8_A"), original)
        with self.assertRaisesRegex(ValueError, "identity"):
            resolve_unique([original, {**alias, "alias_identity": "wrong"}], "task", "D8_A")
        with self.assertRaisesRegex(ValueError, "cycle"):
            resolve_unique([{**original, "alias_of_method": "D8_A"}, alias], "task", "D8_A")

    def test_saved_failed_prefix_loading_never_integrates_and_stops_exactly(self):
        task = TaskSpec.from_dict(read(HISTORY / "snapshot/tasks/b3_mother_00_c_plus/task.json"))
        plan = parameter_plan(task, "v1", [0., 0., 0., 0.])
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); directory = run / "actual" / task.task_id / "D8_A"; attempt = directory / "attempt"
            evaluation_dir = attempt / "actual" / "evaluation"; evaluation_dir.mkdir(parents=True)
            write(run / "frozen_tasks" / task.task_id / "task.json", task.to_dict())
            write(directory / "selected_plan.json", plan.to_dict()); write(attempt / "plan.json", plan.to_dict())
            qpos = np.tile(task.initial_qpos, (11, 1)); qvel = np.tile(task.initial_qvel, (11, 1)); times = np.arange(11) * .002
            reference = TaskAnchoredResidualReferenceProvider(task, plan).continuum_kinematics(times)[0]
            trace = attempt / "actual" / "failed_partial_trace.npz"
            trace_data = dict(initial_qpos=qpos[0].copy(), initial_qvel=qvel[0].copy(), actual_full_qpos=qpos[1:].copy(), actual_full_qvel=qvel[1:].copy(),
                time=times[1:], generated_continuum_position=reference[1:].copy())
            np.savez_compressed(trace, **trace_data)
            replay = {"qpos": qpos, "qvel": qvel, "time": times, "base_pose": np.zeros((11, 7)),
                **{k: np.zeros((11, 3)) for k in ("target_position", "rigid_position", "continuum_position")},
                **{k: np.tile(np.eye(3), (11, 1, 1)) for k in ("target_rotation", "rigid_rotation", "continuum_rotation")}}
            fresh = evaluation_dir / "fresh_replay.npz"; np.savez_compressed(fresh, **replay)
            write(evaluation_dir / "report.json", {"actual_physics_steps": 10, "fresh_replay_sha256": sha(fresh)})
            result = {"task_sha256": task.sha256(), "trace_path": str(trace), "plan_content_sha256": plan.sha256(),
                "trace_sha256": sha(trace), "plan_file_sha256": sha(attempt / "plan.json"),
                "evaluation_sha256": sha(evaluation_dir / "report.json")}
            write(attempt / "attempt_result.json", result)
            write(attempt / "actual" / "timing" / "commands.jsonl", {"certificate": {"source_model_hash": "a" * 64}})
            slot = {"task_id": task.task_id, "method": "D8_A", "task_sha256": task.sha256(), "plan_sha256": plan.sha256(), "actual_steps": 10}
            write(directory / "slot.json", slot)
            with patch.object(mujoco, "mj_step", side_effect=AssertionError("render input check cannot integrate")), patch.object(mujoco, "mj_geomDistance", side_effect=AssertionError("no new geometry")):
                loaded = load_actual(run, slot)
            self.assertEqual(loaded[4]["time"][-1], .020)
            self.assertEqual(len(loaded[4]["time"]), 11)
            self.assertTrue(loaded[-2]["qpos_bit_exact"])
            self.assertEqual(loaded[-2]["saved_execution_compiled_model_sha256"], "a" * 64)
            self.assertEqual(loaded[-2]["generated_continuum_reference_max_abs_difference_m"], 0.)
            # Altering a separately saved replay must fail before rendering.
            replay["qpos"][-1, 0] += .1; np.savez_compressed(fresh, **replay)
            write(evaluation_dir / "report.json", {"actual_physics_steps": 10, "fresh_replay_sha256": sha(fresh)})
            result["evaluation_sha256"] = sha(evaluation_dir / "report.json"); write(attempt / "attempt_result.json", result)
            with self.assertRaisesRegex(ValueError, "parity"):
                load_actual(run, slot)
            replay["qpos"] = np.tile(task.initial_qpos, (11, 1)); np.savez_compressed(fresh, **replay)
            write(evaluation_dir / "report.json", {"actual_physics_steps": 10, "fresh_replay_sha256": sha(fresh)})
            trace_data["generated_continuum_position"][-1, 0] += .01; np.savez_compressed(trace, **trace_data)
            result.update(trace_sha256=sha(trace), evaluation_sha256=sha(evaluation_dir / "report.json"))
            write(attempt / "attempt_result.json", result)
            with self.assertRaisesRegex(ValueError, "generated reference"):
                load_actual(run, slot)

    def test_guard_blocks_real_controller_qp_and_native_integrators(self):
        from v6_lite.hierarchical_qp import HierarchicalVelocityQP
        original = HierarchicalVelocityQP.solve
        with C2RenderOnly() as guard:
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_geomDistance"):
                with self.assertRaisesRegex(RuntimeError, "forbids"):
                    getattr(mujoco, name)(None, None)
            for name in ("solve", "_solve_qp_admm"):
                with self.assertRaisesRegex(RuntimeError, "QP"):
                    getattr(HierarchicalVelocityQP, name)(None)
        self.assertIs(HierarchicalVelocityQP.solve, original)
        self.assertEqual(guard.forward_calls, 0)
        self.assertEqual(len(guard.forbidden_attempts), 6)

    def test_compiled_model_binding_handles_failed_prefix_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); attempt = run / "attempt"; sources = {}
            certificate = attempt / "actual" / "timing" / "commands.jsonl"
            write(certificate, {"certificate": {"source_model_hash": "a" * 64}})
            self.assertEqual(execution_model_binding(run, attempt, sources), "a" * 64)
            self.assertIn("execution_model_certificate", sources)
            verify_render_model("a" * 64, {"saved_execution_compiled_model_sha256": "a" * 64})
            with self.assertRaisesRegex(ValueError, "render model"):
                verify_render_model("b" * 64, {"saved_execution_compiled_model_sha256": "a" * 64})
            write(attempt / "actual" / "historical_metric_observations.json",
                {"execution_contract": {"source_compiled_model_sha256": "b" * 64}})
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                execution_model_binding(run, attempt, sources)

    def test_orientation_uses_saved_current_target_rotation(self):
        actual = np.tile(np.eye(3), (2, 1, 1)); target = actual.copy()
        target[1] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        np.testing.assert_allclose(_angle(actual, target), [0., np.pi / 2])

    def test_seven_view_contract_and_non_grid_failed_endpoint(self):
        self.assertEqual(set(legacy.VIEWS), {"overview", "front", "side", "top", "iso"})
        times = np.arange(12) * .002
        requested, indices = legacy.schedule(times, fps=15.)
        self.assertEqual(indices[-1], 11)
        self.assertEqual(times[indices[-1]], .022)
        self.assertEqual(requested[-1], .022)
        self.assertLessEqual(float(np.max(times[indices])), times[-1])

    def test_core_charts_keep_all_four_failures_and_r8_prefix_cost(self):
        slots = []
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            run = Path(tmp); output = Path(out)
            for task in range(4):
                tid = f"task_{task}"
                for method in ("R", "N", "D"):
                    stream = run / "benchmark_search" / tid / method; planning = stream / "planning" / tid
                    write(stream / "planning_cost.json", {"end_to_end_cold_planning_s": 100., "warm_resident_model_path_estimate_s": 95.})
                    write(planning / "selection.json", {"elapsed_wall_s": 90.})
                    for budget in ((4, 8, 12) if method == "R" else (4, 8)):
                        write(planning / f"prefix_{budget:02d}.json", {"elapsed_wall_s": budget * 5.,
                            "preferences": {p: {"selected_plan": None, "prediction_metrics": None} for p in ("A", "B")}})
                for endpoint in ("R8", "R12", "N8", "D8"):
                    for pref in ("A", "B"):
                        slots.append({"task_id": tid, "endpoint": endpoint, "preference": pref, "full_task_success": False,
                            "original_independent_gates_passed": False, "clearance_30mm_met": None})
            jobs = [{"task_id": f"task_{i}", "method": method, "elapsed_wall_s": 120., "exit_code": 0}
                for i in range(4) for method in ("R", "N", "D")]
            write(run / "search_phase.json", {"passed": True, "workers": 4, "jobs": jobs})
            charts, costs = core_charts(run, output, slots)
            self.assertEqual(costs[("task_0", "R8")]["cold_lower_bound_s"], 50.)
            self.assertEqual(costs[("task_0", "R8")]["cold_upper_bound_s"], 70.)
            self.assertEqual(costs[("task_0", "R12")]["cold_lower_bound_s"], 100.)
            self.assertEqual(costs[("task_0", "R12")]["cold_upper_bound_s"], 120.)
            self.assertEqual(costs[("task_0", "R8")]["bound_width_s"], 20.)
            self.assertIsNone(costs[("task_0", "R8")]["warm_resident_model_path_estimate_s"])
            self.assertEqual(costs[("task_0", "R12")]["warm_resident_model_path_estimate_s"], 95.)
            self.assertTrue(all((output / path).is_file() for path in charts.values()))
            display = [{**r, "method": r["endpoint"] + "_" + r["preference"], "status": "NO_PLAN", "actual_steps": 0,
                "unique_run": False, "quality": None, "media": None, "figures": None,
                "planning_cold_bracket": costs[(r["task_id"], r["endpoint"])]} for r in slots]
            dashboard(output, {"tasks": [f"task_{i}" for i in range(4)], "charts": charts, "slots": display})
            page = (output / "index.html").read_text(encoding="utf8")
            self.assertIn("All 32 logical actual slots", page)
            self.assertIn("NO_PLAN", page)
            self.assertIn("Cold bracket [s]", page)
            self.assertIn("not isolated repeated-task", page)
            self.assertIn("Warm paths are decomposition estimates", page)
            jobs[0]["elapsed_wall_s"] = 80.
            write(run / "search_phase.json", {"passed": True, "workers": 4, "jobs": jobs})
            with self.assertRaisesRegex(ValueError, "final <= inner <= worker"):
                core_charts(run, output, slots)


if __name__ == "__main__": unittest.main()
