"""Light saved-media contract tests; no rendering or model execution."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from . import build_search_aware_warmstart_media as media
from ..closed_loop_warmstart_validation import _seal


class SavedMediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.run = self.root / "run"
        self.run.mkdir()
        self.physics = patch("mujoco.mj_step", side_effect=AssertionError("physics forbidden")).start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        self.physics.assert_not_called()

    def no_plan_run(self):
        tasks = [{"task_id": "test_" + str(i), "task_sha256": "tasksha" + str(i), "split": "test"} for i in range(4)]
        media.write(self.run / "plan.json", {"tasks": tasks})
        for name in ("source_identity", "model_freeze"):
            media.write(self.run / (name + ".json"), {"mock": name})
        entries, files, slot_files = [], {}, {}
        for task in tasks:
            for endpoint in media.ENDPOINTS:
                selection_path = self.run / "frozen_test/sealed_selections" / task["task_id"] / (endpoint + ".json")
                media.write(selection_path, {"preferences": {p: {"selected_plan": None} for p in ("A", "B")}})
                files[str(selection_path)] = media.sha(selection_path)
                entries.append({"task_id": task["task_id"], "endpoint": endpoint, "selection_path": str(selection_path)})
                for preference in ("A", "B"):
                    path = self.run / "frozen_test/actual" / task["task_id"] / (endpoint + "_" + preference) / "slot.json"
                    media.write(path, {**task, "endpoint": endpoint, "preference": preference,
                        "method": endpoint + "_" + preference, "selection_sha256": media.sha(selection_path),
                        "plan_sha256": None, "status": "NO_PLAN", "diagnostic_category": "NO_PLAN",
                        "actual_steps": 0, "full_task_success": False, "unique_run": False,
                        "original_independent_gates_passed": False, "quality": None,
                        "clearance_30mm_met": None})
                    _seal(path.parent); slot_files[path.relative_to(self.run).as_posix()] = media.sha(path)
        media.write(self.run / "frozen_test/sealed_selections/all_selections.json", {
            "phase": "TEST", "endpoints": list(media.ENDPOINTS), "entries": entries, "files": files,
            "bindings": {name + "_sha256": media.sha(self.run / (name + ".json")) for name in ("source_identity", "model_freeze")}})
        media.write(self.run / "actual_complete.json", {"logical_slots": 40, "slot_hashes": slot_files})

    def test_complete_no_plan_dashboard_has_all_five_methods_and_no_video(self):
        self.no_plan_run()
        slots = media.terminal_slots(self.run)
        self.assertEqual(len(slots), 40)
        self.assertTrue(all(s["actual_steps"] == 0 for s in slots))
        result = media.build(self.run, self.root / "media")
        self.assertEqual(result["logical_slots"], 40); self.assertEqual(result["video_count"], 0)
        payload = media.read(self.root / "media/dashboard_data.json")
        self.assertEqual(payload["endpoints"], ["R8", "R12", "N8", "S8", "D8"])
        self.assertEqual(len(payload["views"]), 7)
        self.assertTrue(all(s["media"] is None for s in payload["slots"]))
        manifest = media.read(self.root / "media/visualization_manifest.json")
        self.assertEqual(manifest["no_plan_slots"], 40)
        for name, record in manifest["files"].items():
            self.assertEqual(media.sha(self.root / "media" / name), record["sha256"])

    def test_portable_complete_run_rebinds_old_absolute_paths(self):
        self.no_plan_run()
        copied = self.root / "portable_run"; shutil.copytree(self.run, copied)
        # Corrupt the *old* copy after the portable copy was sealed. Reading the
        # old absolute source would fail; the local relative source remains valid.
        next((self.run / "frozen_test/sealed_selections").glob("*/*.json")).write_text("changed old copy")
        self.assertEqual(len(media.terminal_slots(copied)), 40)
        with self.assertRaises(ValueError): media.terminal_slots(self.run)

    def test_missing_terminal_or_changed_selection_blocks_media(self):
        with self.assertRaises(ValueError): media.terminal_slots(self.run)
        self.no_plan_run()
        selection = self.run / "frozen_test/sealed_selections/test_0/S8.json"
        selection.write_text("{}")
        with self.assertRaises(ValueError): media.terminal_slots(self.run)

    def test_render_implementation_reused_without_mutating_historical_globals(self):
        self.assertIs(media._render_impl.__code__, media.c2.render_actual.__code__)
        self.assertIs(media._figure_impl.__code__, media.c2.trajectory_figures.__code__)
        self.assertIs(media._render_impl.__globals__["load_actual"], media.load_actual)
        self.assertIsNot(media.c2.render_actual.__globals__["load_actual"], media.load_actual)
        self.assertEqual(media._render_impl.__globals__["SCHEMA"], media.SCHEMA)
        self.assertEqual(media.c2.SCHEMA, "v64_c2_saved_actual_media_v1")

    def test_alias_requires_same_task_plan_and_complete_identity(self):
        original = {"task_id": "t", "method": "R12_A", "task_sha256": "tsha", "plan_sha256": "psha", "alias_identity": "full"}
        alias = {**original, "method": "D8_A", "alias_of_slot": str(self.run / "actual/t/R12_A/slot.json")}
        self.assertIs(media.resolve_unique([original, alias], "t", "D8_A"), original)
        bad = {**alias, "alias_identity": "different_history"}
        with self.assertRaises(ValueError): media.resolve_unique([original, bad], "t", "D8_A")

    def test_saved_schedule_contains_exact_prefix_endpoint(self):
        times = np.arange(12) * .002
        requested, indices = media.c2.legacy.schedule(times, 15.)
        self.assertEqual(indices[-1], len(times) - 1)
        self.assertEqual(requested[-1], times[-1])
        self.assertTrue(np.all(times[indices] <= times[-1]))

    def test_render_only_guard_blocks_integrator_geometry_and_qp(self):
        from v6_lite.hierarchical_qp import HierarchicalVelocityQP
        with media.c2.C2RenderOnly() as guard:
            with self.assertRaises(RuntimeError): mujoco.mj_step(None, None)
            with self.assertRaises(RuntimeError): mujoco.mj_geomDistance(None, None, 0, 0, 1., None)
            with self.assertRaises(RuntimeError): HierarchicalVelocityQP.solve(None)
        self.assertEqual(guard.forward_calls, 0); self.assertEqual(len(guard.forbidden_attempts), 3)


if __name__ == "__main__":
    unittest.main()
