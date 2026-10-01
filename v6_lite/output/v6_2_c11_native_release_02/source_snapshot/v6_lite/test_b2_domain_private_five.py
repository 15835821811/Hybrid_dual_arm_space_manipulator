"""Check new five-scene evidence against raw traces and the old failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path("v6_lite/output/v6_2_b2")
GATE = ROOT / "domain_endpoint_private_five_gate"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DomainPrivateFiveTests(unittest.TestCase):
    def test_new_trace_passes_full_private_scope_with_online_gate_closed(self) -> None:
        report = _read(GATE / "domain_private_five_summary.json")
        manifest = _read(GATE / "domain_private_five_manifest.json")
        self.assertEqual(manifest["summary_sha256"],
                         _sha(GATE / "domain_private_five_summary.json"))
        self.assertEqual(report["private_diagnostic_status"],
                         "PASS_PRIVATE_FIVE_WITH_LIMITS")
        self.assertEqual(report["stage2_online_gate_status"], "GATE_NOT_MET")
        self.assertFalse(report["stage3_admission"])
        self.assertFalse(report["original_26_11_online_acceptance_completed"])
        self.assertEqual(report["executed_task_ticks"], 6750)
        self.assertEqual(report["independent_recomputed_task_states"], 6755)
        self.assertEqual(report["independent_checked_2ms_states"], 67505)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(report["preflight_plus_qp_over_20ms_tick_count"], 6)
        interval_rows = 0
        for index, item in enumerate(report["scenes"]):
            scene_id = f"v6_lite_scenario_{index:02d}"
            prefix = f"domain_endpoint_cholesky_scene{index:02d}_1350"
            private_dir = ROOT / prefix
            replay = _read(ROOT / f"{prefix}_recompute" /
                           "private_recompute_summary.json")
            domain = _read(ROOT / f"{prefix}_domain_recompute" /
                           "domain_endpoint_recompute.json")
            task = _read(ROOT / f"{prefix}_task_metrics" /
                         "private_task_metrics.json")
            budget = _read(ROOT / f"{prefix}_domain_budget" /
                           "unified_budget_summary.json")
            self.assertEqual(item["scenario_id"], scene_id)
            self.assertTrue(all(item["checks"].values()))
            self.assertTrue(replay["pass_recompute"])
            self.assertTrue(domain["pass_domain_recompute"])
            self.assertTrue(task["original_task_and_clearance_thresholds_passed"])
            self.assertEqual(budget["budget_overrun_tick_count"], 0)
            self.assertEqual(budget["parity_failure_count"], 0)
            self.assertGreater(domain["minimum_executed_domain_margin_rad"], 0)
            with np.load(private_dir / "private_rollout_trace.npz",
                         allow_pickle=False) as trace:
                self.assertEqual(trace["torque"].shape, (13500, 67))
                self.assertEqual(len(trace["qpos_states"]), 13501)
            interval_rows += replay["checked_interval_rows"]
        self.assertEqual(interval_rows,
                         report["independent_recomputed_interval_rows"])

    def test_scene01_old_rejection_is_replaced_by_new_executable_candidate(self) -> None:
        old = _read(ROOT / "strict_cholesky_scene01_900" /
                    "private_rollout_summary.json")
        new_dir = ROOT / "domain_endpoint_cholesky_scene01_1350"
        new = _read(new_dir / "private_rollout_summary.json")
        records = [json.loads(line) for line in (new_dir /
            "private_rollout_records.jsonl").read_text(
                encoding="utf-8").splitlines()]
        self.assertEqual(old["first_preflight_domain_rejection_tick"], 887)
        self.assertEqual(old["executed_ticks"], 887)
        self.assertEqual(new["executed_ticks"], 1350)
        self.assertEqual(new["stop_reason"], "HORIZON_COMPLETE")
        self.assertIsNone(new["first_preflight_domain_rejection_tick"])
        self.assertEqual(records[887]["failure_reason"], "none")
        self.assertTrue(records[887]["ramp_valid"])
        self.assertEqual(len(records[887]["selected_command"]), 17)
        self.assertEqual(records[887]["ramp_domain_failures"], [])
        self.assertEqual(new["source_sha256"]["b2_work_domain_velocity_box.py"],
                         hashlib.sha256(Path(
                             "v6_lite/b2_work_domain_velocity_box.py")
                             .read_bytes().replace(b"\r\n", b"\n")).hexdigest())


if __name__ == "__main__":
    unittest.main()
