import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from v6_lite.c11_wall_qualification import qualify_scenes
from v6_lite import run_c11_qualified_acceptance, audit_c11_wall_repeats


def scene(index=0):
    latency = {"count": 1350, "p95_ms": 19.9, "first_cycle_ms": 100.0}
    return {"scenario": {"scenario_id": f"v6_lite_scenario_{index:02d}"}, "passed": True,
            "metrics": {"rates_and_latency": {
                "physics_steps": 13500, "task_ticks": 1350, "torque_update_count": 13500,
                "physics_hz": 500.0, "task_hz": 50.0,
                "dispatch": copy.deepcopy(latency),
                "acquisition_to_first_application": copy.deepcopy(latency)}}}


class WallQualificationTests(unittest.TestCase):
    def fixture_root(self, directory):
        root = Path(directory)/"repo"
        for package in ("v6_lite", "model_test"):
            (root/package).mkdir(parents=True)
            (root/package/"fixture.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
        self.fixture_commit(root, "initial fixture")
        return root

    def fixture_commit(self, root, message):
        subprocess.run(["git", "-c", "user.name=Wall Qualification Test", "-c",
                        "user.email=wall-qualification@example.invalid", "-c",
                        "commit.gpgsign=false", "commit", "--quiet", "-m", message],
                       cwd=root, check=True, capture_output=True)

    def mutate_source(self, root, kind):
        if kind == "source_edit":
            (root/"v6_lite/fixture.py").write_text("value = 2\n", encoding="utf-8")
        else:
            # A clean new commit leaves Python bytes unchanged, so HEAD binding
            # must reject it independently of both dirty and content hashes.
            (root/"notes.txt").write_text("new commit\n", encoding="utf-8")
            subprocess.run(["git", "add", "notes.txt"], cwd=root, check=True, capture_output=True)
            self.fixture_commit(root, "changed commit during declared cohort")

    def fake_subprocess(self, invoke):
        # Replace the script's subprocess namespace, leaving real Git commands
        # used by the source inventory and temporary repository intact.
        return SimpleNamespace(run=invoke, STDOUT=subprocess.STDOUT)

    def test_full_horizons_keep_startup_tail(self):
        report = qualify_scenes([scene(index) for index in range(5)], 5)
        self.assertTrue(report["passed"])
        self.assertEqual(report["scenes"][0]["latency"]["dispatch"]["first_cycle_ms"], 100.0)

    def test_legacy_pass_does_not_override_late_actual_application(self):
        item = scene()
        item["metrics"]["rates_and_latency"]["acquisition_to_first_application"]["p95_ms"] = 20.1
        self.assertFalse(qualify_scenes([item], 1)["passed"])

    def test_incomplete_or_duplicate_scene_suite_rejected(self):
        for items in ([], [scene(index) for index in range(4)], [scene() for _ in range(5)]):
            with self.subTest(count=len(items)):
                self.assertFalse(qualify_scenes(items, 5)["passed"])

    def test_truncated_horizon_or_dropped_startup_rejected(self):
        for name in ("physics_steps", "torque_update_count", "task_ticks"):
            item = scene()
            item["metrics"]["rates_and_latency"][name] -= 1
            self.assertFalse(qualify_scenes([item], 1)["passed"])
        item = scene()
        item["metrics"]["rates_and_latency"]["dispatch"]["count"] = 1349
        self.assertFalse(qualify_scenes([item], 1)["passed"])

    def test_nonfinite_latency_rejected(self):
        for name in ("dispatch", "acquisition_to_first_application"):
            for field in ("p95_ms", "first_cycle_ms"):
                for value in (float("nan"), float("inf"), -1, None):
                    with self.subTest(name=name, field=field, value=value):
                        item = scene()
                        item["metrics"]["rates_and_latency"][name][field] = value
                        self.assertFalse(qualify_scenes([item], 1)["passed"])

    def test_pipeline_stops_before_five_and_formal_on_failed_single_gate(self):
        calls = []
        def invoke(command, **kwargs):
            calls.append(command[2])
            if command[2] == "v6_lite.run_c11_wall_single":
                output = Path(command[command.index("--output-dir")+1])
                output.mkdir()
                item = scene()
                item["metrics"]["rates_and_latency"]["acquisition_to_first_application"]["p95_ms"] = 20.2
                (output/"single_scene_metrics.json").write_text(json.dumps(item), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        scheduler = SimpleNamespace(record={"applied": True, "process_priority_set": True,
            "actual_process_priority_class": 0x80}, restore=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture_root(directory)
            with patch.object(run_c11_qualified_acceptance, "__file__", str(root/"v6_lite/run_c11_qualified_acceptance.py")), \
                    patch.object(run_c11_qualified_acceptance, "subprocess", self.fake_subprocess(invoke)), \
                    patch("v6_lite.runtime_scheduler_environment.ThreadScheduling", return_value=scheduler):
                output = Path(directory)/"acceptance"
                self.assertFalse(run_c11_qualified_acceptance.run(output, "high"))
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(calls, ["v6_lite.run_test_profiles", "v6_lite.run_c11_wall_single"])
            self.assertEqual(report["steps"][1]["status"], "FAILED")
            self.assertTrue(all(row["status"] == "NOT_STARTED_FAILED_PREREQUISITE" for row in report["steps"][2:]))

    def test_pipeline_rejects_clean_new_head_or_source_edit_before_formal(self):
        for kind in ("clean_new_head", "source_edit"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = self.fixture_root(directory)
                calls = []
                def invoke(command, **kwargs):
                    name = command[2]
                    calls.append(name)
                    if name in ("v6_lite.run_c11_wall_single", "v6_lite.run_v6_lite"):
                        output = Path(command[command.index("--output-dir")+1])
                        output.mkdir()
                        if name == "v6_lite.run_c11_wall_single":
                            payload, file = scene(), "single_scene_metrics.json"
                        else:
                            payload = {"passed": True, "scenarios": [scene(index) for index in range(5)]}
                            file = "v6_lite_metrics.json"
                            self.mutate_source(root, kind)
                        (output/file).write_text(json.dumps(payload), encoding="utf-8")
                    return SimpleNamespace(returncode=0)
                scheduler = SimpleNamespace(record={"applied": True, "process_priority_set": True,
                    "actual_process_priority_class": 0x80}, restore=lambda: None)
                with patch.object(run_c11_qualified_acceptance, "__file__", str(root/"v6_lite/run_c11_qualified_acceptance.py")), \
                        patch.object(run_c11_qualified_acceptance, "subprocess", self.fake_subprocess(invoke)), \
                        patch("v6_lite.runtime_scheduler_environment.ThreadScheduling", return_value=scheduler):
                    output = Path(directory)/"acceptance"
                    self.assertFalse(run_c11_qualified_acceptance.run(output, "high"))
                report = json.loads((output/"report.json").read_text())
                self.assertEqual(calls, ["v6_lite.run_test_profiles", "v6_lite.run_c11_wall_single",
                                         "v6_lite.run_v6_lite"])
                self.assertEqual(report["steps"][2]["reason"], "SOURCE_IDENTITY_CHANGED_AFTER_DECLARATION")
                self.assertEqual(report["steps"][3]["status"], "NOT_STARTED_FAILED_PREREQUISITE")
                source = json.loads((output/"five_source_provenance.json").read_text())
                self.assertTrue(source["before_step"]["passed"])
                self.assertFalse(source["after_step"]["passed"])
                if kind == "clean_new_head":
                    self.assertTrue(source["after_step"]["tracked_worktree_clean"])
                    self.assertFalse(source["after_step"]["git_commit_unchanged"])

    def test_pipeline_checks_source_again_before_starting_next_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture_root(directory)
            captures, calls = [], []
            capture = run_c11_qualified_acceptance.current_source_snapshot
            def capture_then_mutate(path):
                result = capture(path)
                captures.append(result)
                if len(captures) == 3:
                    self.mutate_source(root, "source_edit")
                return result
            def invoke(command, **kwargs):
                calls.append(command[2])
                return SimpleNamespace(returncode=0)
            scheduler = SimpleNamespace(record={"applied": True, "process_priority_set": True,
                "actual_process_priority_class": 0x80}, restore=lambda: None)
            with patch.object(run_c11_qualified_acceptance, "__file__", str(root/"v6_lite/run_c11_qualified_acceptance.py")), \
                    patch.object(run_c11_qualified_acceptance, "subprocess", self.fake_subprocess(invoke)), \
                    patch.object(run_c11_qualified_acceptance, "current_source_snapshot", side_effect=capture_then_mutate), \
                    patch("v6_lite.runtime_scheduler_environment.ThreadScheduling", return_value=scheduler):
                output = Path(directory)/"acceptance"
                self.assertFalse(run_c11_qualified_acceptance.run(output, "high"))
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(calls, ["v6_lite.run_test_profiles"])
            self.assertEqual(report["steps"][1]["status"], "NOT_STARTED_SOURCE_CHANGED")

    def test_formal_retains_all_rounds_but_rejects_incomplete_suites(self):
        calls = []
        def invoke(command, **kwargs):
            output = Path(command[command.index("--output-dir")+1])
            output.mkdir()
            calls.append(output.name)
            items = [scene(index) for index in range(4 if len(calls) == 1 else 5)]
            (output/"v6_lite_metrics.json").write_text(json.dumps({"passed": True, "scenarios": items}))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture_root(directory)
            with patch.object(audit_c11_wall_repeats, "__file__", str(root/"v6_lite/audit_c11_wall_repeats.py")), \
                    patch.object(audit_c11_wall_repeats, "subprocess", self.fake_subprocess(invoke)):
                output = Path(directory)/"repeats"
                self.assertFalse(audit_c11_wall_repeats.run(output))
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(len(calls), 3)
            self.assertFalse(report["rounds"][0]["passed"])
            self.assertFalse(report["rounds"][0]["complete_five_scenes"])
            self.assertTrue(report["complete"])

    def test_formal_source_changes_retain_three_records_without_launching_remaining_rounds(self):
        for kind in ("clean_new_head", "source_edit"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = self.fixture_root(directory)
                calls = []
                def invoke(command, **kwargs):
                    output = Path(command[command.index("--output-dir")+1])
                    output.mkdir()
                    calls.append(output.name)
                    (output/"v6_lite_metrics.json").write_text(json.dumps({"passed": True,
                        "scenarios": [scene(index) for index in range(5)]}), encoding="utf-8")
                    self.mutate_source(root, kind)
                    return SimpleNamespace(returncode=0)
                with patch.object(audit_c11_wall_repeats, "__file__", str(root/"v6_lite/audit_c11_wall_repeats.py")), \
                        patch.object(audit_c11_wall_repeats, "subprocess", self.fake_subprocess(invoke)):
                    output = Path(directory)/"repeats"
                    self.assertFalse(audit_c11_wall_repeats.run(output))
                report = json.loads((output/"report.json").read_text())
                self.assertEqual(calls, ["round_01"])
                self.assertEqual(len(report["rounds"]), 3)
                self.assertFalse(report["complete"])
                self.assertTrue(report["all_declared_rounds_accounted_for"])
                self.assertFalse(report["passed"])
                self.assertEqual(report["rounds"][0]["status"], "FAILED_SOURCE_CHANGED")
                self.assertTrue(all(row["status"] == "NOT_STARTED_SOURCE_CHANGED"
                                    for row in report["rounds"][1:]))


if __name__ == "__main__":
    unittest.main()
