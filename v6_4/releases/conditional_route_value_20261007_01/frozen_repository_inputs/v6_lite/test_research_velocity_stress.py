"""Velocity stress qualification and failure evidence; no physics runs."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v6_lite import run_research_velocity_stress as runner
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_v6_lite import V6LiteRunConfig


def definitions():
    return [{"scenario_id": name, "seed": 20260801+104729*i,
        runner.VELOCITY_KEY: [.001+i*.0001, -.002, .003],
        "target_satellite_angular_velocity_rad_s": [.0001, -.0002, .0003],
        "workspace_obstacles": [], "unchanged_field": {"waypoints": [1, 2, 3]}}
        for i, name in enumerate(runner.SCENE_IDS)]


def doubled(original):
    value = copy.deepcopy(original)
    for scene in value:
        scene[runner.VELOCITY_KEY] = [2.0*v for v in scene[runner.VELOCITY_KEY]]
    return value


def config_and_qp():
    return (asdict(V6LiteRunConfig(pcc_mode="bounded_interval_pcc",
            dispatch_clock_policy="research_simulation", target_linear_velocity_scale=2.)),
            asdict(HierarchicalQPConfig(enable_capsule_cbf=True)))


def suite():
    cfg, qp = config_and_qp()
    return {"passed": True, "run_config": cfg, "qp_config": qp,
        "scenarios": [{"scenario": scene, "passed": True,
            "checks": {"functional": True},
            "execution_contract": {"dispatch_clock_policy": "research_simulation", "wall_deadline_enforced": False},
            "metrics": {"rates_and_latency": {"physics_steps": 13500, "task_ticks": 1350,
                "torque_update_count": 13500, "physics_hz": 500., "task_hz": 50.}}}
            for scene in doubled(definitions())]}


def snapshot():
    return {"git_commit": "fixture", "tracked_worktree_dirty": False, "files": {}, "capture_errors": []}


def identity():
    return {"passed": True, "required_committed_files": {"fixture": {"sha256_raw": "fixture"}},
            "errors": [], "untracked_inventory_sources": [], "untracked_inventory_sources_imported": []}


def reference():
    cfg, qp = config_and_qp()
    cfg.pop("target_linear_velocity_scale")
    return {"scenarios": definitions(), "run_config": cfg, "qp_config": qp,
            "source_commit": runner.REFERENCE_COMMIT, "manifest_sha256": runner.REFERENCE_MANIFEST_SHA}


class VelocityStressTests(unittest.TestCase):
    def test_only_declared_factor_and_exact_default_definitions_qualify(self):
        original = definitions()
        self.assertTrue(runner.compare_declared_scenes(original, copy.deepcopy(original), doubled(original))["passed"])
        for kind in ("angular", "waypoint", "default", "nan", "order", "subset"):
            one, two = copy.deepcopy(original), doubled(original)
            if kind == "angular": two[0]["target_satellite_angular_velocity_rad_s"][0] *= 2
            elif kind == "waypoint": two[0]["unchanged_field"]["waypoints"][0] = 5
            elif kind == "default": one[0][runner.VELOCITY_KEY][0] *= 1.01
            elif kind == "nan": two[0][runner.VELOCITY_KEY][0] = float("nan")
            elif kind == "order": two.reverse()
            else: two.pop()
            with self.subTest(kind=kind):
                self.assertFalse(runner.compare_declared_scenes(original, one, two)["passed"])

    def test_full_qualification_binds_physical_factor_and_all_research_gates(self):
        cfg, qp = config_and_qp()
        full = suite()
        self.assertTrue(runner.qualify_velocity_stress_suite(full, doubled(definitions()), cfg, qp, {"passed": True})["passed"])
        variations = []
        value = suite(); value["run_config"]["target_linear_velocity_scale"] = 1.; variations.append(value)
        value = suite(); value["scenarios"][0]["scenario"]["unchanged_field"]["waypoints"][0] = 4; variations.append(value)
        value = suite(); value["qp_config"]["enable_capsule_cbf"] = False; variations.append(value)
        value = suite(); value["scenarios"][0]["metrics"]["rates_and_latency"]["physics_steps"] -= 1; variations.append(value)
        value = suite(); value["scenarios"][0]["passed"] = False; variations.append(value)
        value = suite(); value["scenarios"].pop(); variations.append(value)
        value = suite(); value["scenarios"][0]["execution_contract"]["wall_deadline_enforced"] = True; variations.append(value)
        for value in variations:
            with self.subTest(value=value):
                self.assertFalse(runner.qualify_velocity_stress_suite(value, doubled(definitions()), cfg, qp, {"passed": True})["passed"])
        for factor in (None, {"passed": False}):
            self.assertFalse(runner.qualify_velocity_stress_suite(full, doubled(definitions()), cfg, qp, factor)["passed"])

    def test_actual_factor_audit_uses_named_nonterminal_slice_and_rejects_metadata_only_change(self):
        cfg, _ = config_and_qp()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            olddir, newdir = root/"baseline", root/"new"
            (olddir/"simulation").mkdir(parents=True); newdir.mkdir()
            originals, observed = [], suite()
            for i, (oldscene, scene) in enumerate(zip(definitions(), observed["scenarios"])):
                oldvel = np.zeros(18)
                oldvel[4:7] = oldscene[runner.VELOCITY_KEY]
                oldvel[7:10] = oldscene["target_satellite_angular_velocity_rad_s"]
                newvel = oldvel.copy(); newvel[4:7] *= 2
                initial_qpos = np.arange(24, dtype=float)*.001
                oldpath, newpath = olddir/f"old{i}.npz", newdir/f"new{i}.npz"
                np.savez(oldpath, initial_qvel=oldvel, initial_qpos=initial_qpos)
                np.savez(newpath, initial_qvel=newvel, initial_qpos=initial_qpos)
                originals.append({"scenario": oldscene, "trace": {"path": str(oldpath), "sha256": runner.sha(oldpath)}})
                scene["trace"] = {"path": str(newpath), "sha256": runner.sha(newpath)}
            (olddir/"simulation/v6_lite_metrics.json").write_text(json.dumps({"scenarios": originals}), encoding="utf-8")
            target_name = "named_target_joint_not_last"
            def located(model, name):
                self.assertEqual(name, target_name)
                return slice(2, 9), slice(4, 10)
            with patch("v6_lite.run_v6_lite.default_v6_lite_robot_spec", return_value=SimpleNamespace(target_free_joint_name=target_name)), \
                 patch("model_test.whole_body_verifier_v5.WholeBodyCollisionVerifier", return_value=SimpleNamespace(model=SimpleNamespace(nv=18, nq=24))), \
                 patch("v6_lite.hierarchical_qp.free_joint_slices", side_effect=located):
                self.assertTrue(runner.applied_factor_audit(observed, olddir, doubled(definitions()), newdir, cfg)["passed"])
                path = Path(observed["scenarios"][0]["trace"]["path"])
                with np.load(path) as z:
                    valid = z["initial_qvel"].copy()
                    valid_qpos = z["initial_qpos"].copy()
                for kind in ("old_linear", "angular", "robot", "nonfinite", "hash", "initial_position"):
                    bad = valid.copy()
                    bad_qpos = valid_qpos.copy()
                    if kind == "old_linear": bad[4:7] /= 2
                    elif kind == "angular": bad[7] *= 2
                    elif kind == "robot": bad[11] = .1
                    elif kind == "nonfinite": bad[5] = float("nan")
                    elif kind == "initial_position": bad_qpos[0] += .01
                    np.savez(path, initial_qvel=bad, initial_qpos=bad_qpos)
                    observed["scenarios"][0]["trace"]["sha256"] = "tampered" if kind == "hash" else runner.sha(path)
                    with self.subTest(kind=kind):
                        self.assertFalse(runner.applied_factor_audit(observed, olddir, doubled(definitions()), newdir, cfg)["passed"])

    def test_required_fixture_head_identity_is_checked_even_outside_python_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = {name: b"frozen\n" for name in runner.REQUIRED_COMMITTED_PATHS}
            for name, raw in blobs.items():
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
            def git(args, **_):
                return "\n".join(blobs).encode() if args[1] == "ls-files" else blobs[args[2].removeprefix("HEAD:")]
            snap = {"files": {"v6_lite/report_b2_latest_status.py": {}}}
            with patch.object(runner.subprocess, "check_output", side_effect=git):
                result = runner.committed_source_identity(root, snap)
                self.assertTrue(result["passed"])
                self.assertEqual(result["untracked_inventory_sources"], ["v6_lite/report_b2_latest_status.py"])
                self.assertFalse(result["all_inventory_sources_claimed_in_head"])
                (root/"v6_lite/test_fixtures/target_velocity_baseline.json").write_bytes(b"changed\n")
                self.assertFalse(runner.committed_source_identity(root, snap)["passed"])

    def _pipeline(self, root, action=None, tests=True):
        """Patch computational dependencies; exercise the real orchestration."""
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(runner, "current_source_snapshot", side_effect=lambda _: snapshot()))
        stack.enter_context(patch.object(runner, "committed_source_identity", side_effect=lambda *_: identity()))
        stack.enter_context(patch.object(runner, "load_frozen_reference", return_value=reference()))
        stack.enter_context(patch("v6_lite.run_v6_lite.default_v6_lite_robot_spec", return_value=None))
        def make(_, cfg):
            return [SimpleNamespace(to_dict=lambda v=copy.deepcopy(v): v) for v in
                    (definitions() if cfg.target_linear_velocity_scale == 1. else doubled(definitions()))]
        stack.enter_context(patch("v6_lite.run_v6_lite.build_scenarios", side_effect=make))
        profiles = stack.enter_context(patch("v6_lite.run_test_profiles.run", return_value=tests))
        simulation = stack.enter_context(patch("v6_lite.run_v6_lite.run_suite", side_effect=action or (lambda *_a, **_k: suite())))
        stack.enter_context(patch.object(runner, "applied_factor_audit", return_value={"passed": True}))
        native = stack.enter_context(patch("v6_lite.validate_v6_lite.validate_delivery", return_value={"passed": True}))
        stack.enter_context(patch("v6_lite.validate_v6_lite.validate_execution_contract", return_value={"passed": True}))
        stack.enter_context(patch("v6_lite.audit_b2_online_interval_recompute.run", return_value={"passed": True}))
        stack.enter_context(patch.object(runner, "audit_research_timing", return_value={"evidence_valid": True,
            "performance_target_met": False, "performance_is_functional_gate": False, "scenes": []}))
        return profiles, simulation, native

    def test_full_pipeline_runs_all_profiles_and_independent_gates_without_old_c1_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exclusive"
            profiles, simulation, native = self._pipeline(output)
            result = runner.run(output)
            self.assertTrue(result["passed"]); self.assertTrue(result["complete"])
            self.assertFalse(result["computational_performance"]["performance_target_met"])
            profiles.assert_called_once_with("all", output/"test_profiles")
            self.assertTrue(simulation.call_args.kwargs["continue_failed_scenarios"])
            self.assertEqual(native.call_args.kwargs["acceptance_profile"], "research_simulation")
            self.assertNotIn("exact_c1_nontiming_parity", [x["name"] for x in result["steps"]])
            self.assertTrue((output/"manifest.json").is_file())
            with self.assertRaises(FileExistsError): runner.run(output)

    def test_failed_regression_does_not_start_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exclusive"
            _, simulation, native = self._pipeline(output, tests=False)
            result = runner.run(output)
            self.assertFalse(result["passed"]); self.assertFalse(result["complete"])
            simulation.assert_not_called(); native.assert_not_called()
            self.assertTrue((output/"manifest.json").is_file())

    def test_each_independent_failure_or_invalid_clock_evidence_rejects_completed_workflow(self):
        targets = ["v6_lite.validate_v6_lite.validate_delivery", "v6_lite.validate_v6_lite.validate_execution_contract",
                   "v6_lite.audit_b2_online_interval_recompute.run",
                   "v6_lite.run_research_velocity_stress.applied_factor_audit",
                   "v6_lite.run_research_velocity_stress.audit_research_timing"]
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/"exclusive"
                self._pipeline(output)
                value = {"evidence_valid": False, "performance_target_met": True} if target.endswith("audit_research_timing") else {"passed": False}
                with patch(target, return_value=value):
                    result = runner.run(output)
                self.assertFalse(result["passed"])
                self.assertTrue(result["complete"])
                self.assertIn("acceptance additionally requires", result["complete_definition"])

    def test_source_change_during_regression_blocks_physics_and_marks_provenance_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exclusive"
            _, simulation, native = self._pipeline(output)
            calls = 0
            def changing(_):
                nonlocal calls
                calls += 1
                value = snapshot()
                if calls >= 3:
                    value["files"] = {"changed.py": {"sha256_raw": "other"}}
                return value
            with patch.object(runner, "current_source_snapshot", side_effect=changing):
                result = runner.run(output)
            self.assertFalse(result["passed"])
            self.assertFalse(result["source_provenance"]["source_unchanged"])
            simulation.assert_not_called(); native.assert_not_called()

    def test_rejected_cohort_preserves_all_attempts_and_never_qualifies_partial_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exclusive"
            def reject(_cfg, _qp, destination, **kwargs):
                self.assertTrue(kwargs["continue_failed_scenarios"])
                destination.mkdir()
                (destination/"research_trial_report.json").write_text(json.dumps({
                    "all_predeclared_scenes_attempted": True, "completed_scenes": suite()["scenarios"][:1],
                    "failed_scenarios": [{"scenario_id": name, "error": "uncertified"} for name in runner.SCENE_IDS[1:]]}), encoding="utf-8")
                raise RuntimeError("retained rejected cohort")
            _, _, native = self._pipeline(output, action=reject)
            result = runner.run(output)
            self.assertFalse(result["passed"]); self.assertFalse(result["complete"])
            self.assertEqual(result["trial_summary"]["observed_attempt_count"], 5)
            self.assertEqual(result["trial_summary"]["reported_rejected_scene_count"], 4)
            self.assertFalse(result["computational_performance"]["evidence_valid"])
            self.assertIsNone(result["computational_performance"]["performance_target_met"])
            self.assertFalse(result["trial_summary"]["partial_is_full_acceptance"])
            native.assert_not_called()

    def test_interruption_records_unattempted_scenes_and_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exclusive"
            def interrupt(_cfg, _qp, destination, **_):
                destination.mkdir()
                (destination/"run_failure.json").write_text(json.dumps({"scenario_id": runner.SCENE_IDS[0],
                    "status": "INTERRUPTED", "message": "fixture interrupt"}), encoding="utf-8")
                raise KeyboardInterrupt()
            _, _, native = self._pipeline(output, action=interrupt)
            result = runner.run(output)
            self.assertFalse(result["passed"]); self.assertFalse(result["complete"])
            self.assertEqual(result["trial_summary"]["observed_attempt_count"], 1)
            self.assertEqual(result["trial_summary"]["scenes"][0]["status"], "ABORTED_PARTIAL")
            self.assertEqual(result["trial_summary"]["scenes"][1]["status"], "NOT_OBSERVED_ATTEMPTED")
            self.assertEqual(result["steps"][1]["status"], "ABORTED")
            native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
