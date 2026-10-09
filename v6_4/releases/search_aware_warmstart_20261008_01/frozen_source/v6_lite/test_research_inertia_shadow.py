"""Stage and evidence gates for the inertia shadow; physics is mocked out."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_lite import run_research_inertia_shadow as runner


def clean_source():
    return {"git_commit": "committed-fixture", "tracked_worktree_dirty": False,
            "worktree_dirty": True, "files": {}, "capture_errors": []}


def source_result():
    return {"source_unchanged": True}


def plan_fixture():
    return {"schema": "inertia_shadow_plan_v1", "smoke_steps": None,
            "baseline_complete_manifest_sha256": runner.BASELINE_MANIFEST_SHA,
            "tool_sha256": runner.sha(runner.__file__), "current_source": clean_source(),
            "inputs": [{"fixture_scene": i} for i in range(5)],
            "runs": [{"scenario_index": i, "alpha": alpha,
                      "key": runner.run_key(i, alpha), "status": "NOT_RUN"}
                     for alpha in (1.0, .95, 1.05) for i in range(5)]}


def completed_rows(plan):
    return [{**item, "status": "REPLAY_COMPLETED", "full_27s_complete": True,
             "steps_completed": 13500,
             "nominal_parity": {"passed": True} if item["alpha"] == 1.0 else None}
            for item in plan["runs"]]


def seed_replay_evidence(output, plan, rows=None, complete=True):
    """A tiny stage fixture, never a fabricated numerical trajectory."""
    output.mkdir(parents=True)
    runner.write(output / "plan.json", plan)
    (output / "producer.py").write_bytes(Path(runner.__file__).read_bytes())
    runner.write(output / "input_identities.json", plan["inputs"])
    for item in plan["runs"]:
        directory = output / "replays" / item["key"]
        directory.mkdir(parents=True)
        for name in ("model_parameters.npz", "model.json", "trajectory.npz", "replay.json"):
            (directory / name).write_bytes(("stage fixture " + item["key"] + name).encode())
    runner.write(output / "replay_report.json", {"complete": complete,
        "physics_steps_completed": 202500, "runs": rows if rows is not None else completed_rows(plan)})
    manifest = {p.relative_to(output).as_posix(): runner.sha(p)
                for p in output.rglob("*") if p.is_file()}
    runner.write(output / "replay_manifest.json", manifest)
    return runner.sha(output / "replay_manifest.json")


def file_contents(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


class InertiaShadowStageTests(unittest.TestCase):
    def setUp(self):
        # Any unmocked route reaching physics is a test failure.
        self.no_physics = patch.object(runner.mujoco, "mj_step",
                                      side_effect=AssertionError("unit tests must not integrate physics"))
        self.no_physics.start()
        self.addCleanup(self.no_physics.stop)

    def test_tracked_dirty_is_rejected_before_planning_and_at_later_identity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plan_fixture()
            dirty = clean_source(); dirty["tracked_worktree_dirty"] = True
            fake_spec = SimpleNamespace(runtime_contract_sha256=lambda: runner.MODEL_CONTRACT_SHA,
                                        source_bundle_sha256=lambda: runner.MODEL_BUNDLE_SHA)
            with patch.object(runner, "frozen_inputs", return_value=({}, plan["inputs"])), \
                    patch.object(runner, "current_source_snapshot", return_value=dirty):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    runner.create_plan(root / "dirty-plan")
            self.assertFalse((root / "dirty-plan/plan.json").exists())
            output = root / "identity"; output.mkdir()
            (output / "producer.py").write_bytes(Path(runner.__file__).read_bytes())
            with patch.object(runner, "frozen_inputs", return_value=({}, plan["inputs"])), \
                    patch.object(runner, "default_v6_lite_robot_spec", return_value=fake_spec), \
                    patch.object(runner, "source_provenance", return_value=source_result()), \
                    patch.object(runner, "current_source_snapshot", return_value=clean_source()) as current:
                runner.identity_check(output, plan)  # unrelated untracked artifacts are allowed
                current.return_value = dirty
                with self.assertRaisesRegex(ValueError, "dirty"):
                    runner.identity_check(output, plan)
                current.return_value = clean_source()
                bad_plan = copy.deepcopy(plan); bad_plan["current_source"] = dirty
                with self.assertRaisesRegex(ValueError, "dirty"):
                    runner.identity_check(output, bad_plan)
                fake_spec.source_bundle_sha256 = lambda: "changed-model-asset"
                with self.assertRaisesRegex(ValueError, "asset identity changed"):
                    runner.identity_check(output, plan)

    def test_failed_last_nominal_scene_prevents_every_perturbed_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); plan = plan_fixture(); calls = []
            def replay(_output, _plan, index, alpha, _steps):
                calls.append((index, alpha))
                return {"key": runner.run_key(index, alpha), "scenario_index": index,
                        "alpha": alpha, "status": "REPLAY_FAILED" if index == 4 else "REPLAY_COMPLETED",
                        "full_27s_complete": index != 4, "steps_completed": 13500,
                        "nominal_parity": {"passed": index != 4},
                        "error": {"type": "NOMINAL_REPLAY_MISMATCH"} if index == 4 else None}
            with patch.object(runner, "identity_check", return_value=({}, plan["inputs"], source_result())), \
                    patch.object(runner, "replay_one", side_effect=replay):
                with self.assertRaisesRegex(ValueError, "replay failed"):
                    runner.replay_stage(output, plan)
            self.assertEqual(calls, [(i, 1.0) for i in range(5)])
            retained = runner.read(output / "replay_report.json")
            self.assertFalse(retained["complete"])
            self.assertFalse(retained["evidence_valid"])
            self.assertEqual(len(retained["runs"]), 15)
            self.assertEqual(sum(r["status"].startswith("NOT_STARTED") for r in retained["runs"]), 10)

    def test_complete_mock_observation_is_diagnostic_evidence_with_original_claim_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "full"; plan = plan_fixture()
            pin = seed_replay_evidence(output, plan)
            with patch.object(runner, "identity_check", return_value=({}, plan["inputs"], source_result())), \
                    patch.object(runner, "observe_one", side_effect=lambda _o, _p, r: {
                        "key": r["key"], "status": "OBSERVATION_COMPLETED"}) as observe, \
                    patch.object(runner, "paired_deltas", return_value=[]):
                result = runner.observe_stage(output, plan, pin)
            self.assertTrue(result["complete"])
            self.assertTrue(result["evidence_valid"])
            self.assertEqual(observe.call_count, 15)
            self.assertEqual(result["caller_pinned_replay_manifest_sha256"], pin)
            self.assertFalse(result["limitations"]["closed_loop_robustness_established"])
            self.assertFalse(result["limitations"]["runtime_certificate_reused"])
            self.assertFalse(result["limitations"]["continuous_time_collision_certified"])
            manifest = runner.read(output / "manifest.json")
            self.assertEqual(manifest["report.json"], runner.sha(output / "report.json"))

    def test_partial_smoke_duplicate_missing_or_unqualified_runs_never_start_observation(self):
        for kind in ("partial", "smoke", "duplicate", "missing", "order", "nominal_parity"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                plan = plan_fixture(); rows = completed_rows(plan)
                if kind == "smoke": plan["smoke_steps"] = 10
                elif kind == "duplicate": rows[-1] = copy.deepcopy(rows[-2])
                elif kind == "missing": rows.pop()
                elif kind == "order": rows[0], rows[1] = rows[1], rows[0]
                elif kind == "nominal_parity": rows[4]["nominal_parity"]["passed"] = False
                output = Path(directory) / "bad"
                pin = seed_replay_evidence(output, plan, rows, complete=kind != "partial")
                with patch.object(runner, "identity_check", return_value=({}, plan["inputs"], source_result())), \
                        patch.object(runner, "observe_one") as observe:
                    with self.assertRaises(ValueError):
                        runner.observe_stage(output, plan, pin)
                observe.assert_not_called()
                self.assertFalse((output / "observations").exists())
                self.assertFalse((output / "report.json").exists())

    def test_external_manifest_pin_and_file_coverage_reject_changed_evidence(self):
        for kind in ("wrong_pin", "changed_file", "omitted_file"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "tampered"; plan = plan_fixture()
                pin = seed_replay_evidence(output, plan)
                filename = "replays/scene_00_alpha_1.00/model.json"
                if kind == "wrong_pin": pin = "0" * 64
                elif kind == "changed_file": (output / filename).write_bytes(b"changed after external pin")
                else:
                    manifest = runner.read(output / "replay_manifest.json"); manifest.pop(filename)
                    (output / "replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                    pin = runner.sha(output / "replay_manifest.json")
                with patch.object(runner, "identity_check", return_value=({}, plan["inputs"], source_result())), \
                        patch.object(runner, "observe_one") as observe:
                    with self.assertRaises(ValueError):
                        runner.observe_stage(output, plan, pin)
                observe.assert_not_called()
                self.assertFalse((output / "observations").exists())

    def test_cli_repeated_or_terminal_stages_leave_existing_output_byte_for_byte_unchanged(self):
        for stage, marker in (("observe", "report.json"), ("observe", "manifest.json"),
                              ("observe", "failure.json"), ("observe", "observations"),
                              ("replay", "replays"), ("replay", "replay_report.json"),
                              ("replay", "replay_manifest.json")):
            with self.subTest(stage=stage, marker=marker), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); output = root / "v6_lite/output/runs/closed"
                output.mkdir(parents=True); runner.write(output / "plan.json", plan_fixture())
                target = output / marker
                if "." in marker: target.write_bytes(b"immutable terminal evidence\n")
                else:
                    target.mkdir(); (target / "retained.bin").write_bytes(b"retained partial evidence")
                before = file_contents(output)
                with patch.object(runner, "ROOT", root), \
                        patch.object(runner, "replay_stage") as replay, \
                        patch.object(runner, "observe_stage") as observe, \
                        patch("sys.argv", [runner.__file__, "--stage", stage, "--output-dir", str(output)]):
                    self.assertEqual(runner.main(), 1)
                replay.assert_not_called(); observe.assert_not_called()
                self.assertEqual(file_contents(output), before)

    def test_empty_incomplete_or_nonfinite_geometry_cannot_be_positive_evidence(self):
        pair = SimpleNamespace(geom_a=0, geom_b=1, geom_a_name="a", geom_b_name="b",
                               pair_class="continuum_target")
        for kind in ("valid", "empty_pairs", "zero_states", "short_prefix", "nan_distance",
                     "nan_state", "nan_phase"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                model = SimpleNamespace(nq=1, nv=1)
                observer = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1), ctrl=np.zeros(1), time=0.)
                qpos = np.array([float("nan") if kind == "nan_state" else 0.])
                phase = float("nan") if kind == "nan_phase" else 0.
                states = [] if kind == "zero_states" else [(qpos, np.zeros(1), phase)]
                count = 0 if kind == "zero_states" else 2 if kind == "short_prefix" else 1
                path = Path(directory) / "distances.npz"
                def query(*args):
                    args[-1][:] = 0.0
                    return float("nan") if kind == "nan_distance" else .02
                with patch.object(runner.mujoco, "MjData", return_value=observer), \
                        patch.object(runner.mujoco, "mj_forward"), \
                        patch.object(runner.mujoco, "mj_geomDistance", side_effect=query):
                    if kind == "valid":
                        report = runner.audit_pairs(model, (pair,), states, count, .25, path, "mock_states")
                        self.assertEqual(report["checked_state_count"], 1)
                        self.assertTrue(np.isfinite(report["minimum_clearance_m"]))
                        self.assertTrue(report["measured_discrete_clearance_at_least_gate"])
                    else:
                        with self.assertRaises(ValueError):
                            runner.audit_pairs(model, () if kind == "empty_pairs" else (pair,),
                                               states, count, .25, path, "mock_states")


if __name__ == "__main__":
    unittest.main()
