"""Independent checks on the complete private five-scene gate evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path("v6_lite/output/v6_2_b2")
GATE = ROOT / "full_private_five_gate"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class FullPrivateFiveGateTests(unittest.TestCase):
    def test_strict_domain_gate_rejects_before_next_torque_step(self) -> None:
        strict_dir = ROOT / "strict_cholesky_scene01_900"
        replay_dir = ROOT / "strict_domain_replay_scene01_900"
        summary = _read(strict_dir / "private_rollout_summary.json")
        manifest = _read(strict_dir / "private_rollout_manifest.json")
        replay = _read(replay_dir / "strict_domain_replay.json")
        trace_path = strict_dir / "private_rollout_trace.npz"
        records_path = strict_dir / "private_rollout_records.jsonl"
        self.assertEqual(manifest["summary_sha256"],
                         _sha(strict_dir / "private_rollout_summary.json"))
        self.assertEqual(manifest["records_sha256"], _sha(records_path))
        self.assertEqual(manifest["trace_sha256"], _sha(trace_path))
        self.assertEqual(summary["attempted_ticks"], 888)
        self.assertEqual(summary["executed_ticks"], 887)
        self.assertEqual(summary["first_preflight_domain_rejection_tick"], 887)
        self.assertTrue(summary["strict_gate_rejected_before_servo"])
        rejected = summary["failure_record"]
        self.assertEqual(rejected["action_mode"], "UNCERTIFIED")
        self.assertEqual(rejected["failure_reason"],
                         "ramp_work_domain_unsupported")
        self.assertIsNone(rejected["selected_command"])
        self.assertFalse(rejected["next_servo_step_executed"])
        self.assertEqual(rejected["ramp_domain_failures"][0]["servo_substep"], 4)
        self.assertEqual(len(rejected["rejected_candidate_command"]), 17)
        with np.load(trace_path, allow_pickle=False) as trace:
            self.assertEqual(trace["torque"].shape, (8870, 67))
            self.assertEqual(len(trace["qpos_states"]), 8871)
            with np.load(ROOT / "cholesky_private_full_scene01_1350" /
                         "private_rollout_trace.npz", allow_pickle=False) as old:
                np.testing.assert_array_equal(trace["torque"],
                                              old["torque"][:8870])
                np.testing.assert_array_equal(trace["qpos_states"],
                                              old["qpos_states"][:8871])
        self.assertEqual(replay["native_replay_max_state_error"], 0.0)
        self.assertGreater(replay["minimum_executed_work_domain_margin_rad"], 0)
        self.assertTrue(replay["rejected_candidate_excluded_from_torque_trace"])
        self.assertFalse(replay["production_online_admitted"])

    def test_domain_counterexample_is_bound_to_all_five_raw_traces(self) -> None:
        summary = _read(GATE / "private_five_gate_summary.json")
        manifest = _read(GATE / "private_five_gate_manifest.json")
        failure_path = GATE / "private_five_gate_failures.jsonl"
        self.assertEqual(manifest["summary_sha256"],
                         _sha(GATE / "private_five_gate_summary.json"))
        self.assertEqual(manifest["failures_sha256"], _sha(failure_path))
        self.assertEqual(summary["failure_records_sha256"], _sha(failure_path))
        self.assertEqual(summary["scene_count"], 5)
        self.assertEqual(summary["executed_task_ticks"], 6750)
        self.assertEqual(summary["recomputed_task_states"], 6755)
        self.assertFalse(summary["stage3_admission"])
        self.assertEqual(summary["stage2_status"], "GATE_NOT_MET")

        observed = []
        row_total = 0
        for index, scene in enumerate(summary["scenes"]):
            scene_id = f"v6_lite_scenario_{index:02d}"
            self.assertEqual(scene["scenario_id"], scene_id)
            prefix = f"cholesky_private_full_scene{index:02d}"
            private_dir = ROOT / f"{prefix}_1350"
            replay_dir = ROOT / f"{prefix}_recompute"
            budget_dir = ROOT / f"{prefix}_budget"
            records_path = private_dir / "private_rollout_records.jsonl"
            records = [json.loads(line) for line in records_path.read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1350)
            self.assertEqual(scene["inputs"]["private_records_sha256"],
                             _sha(records_path))
            self.assertEqual(scene["inputs"]["private_trace_sha256"],
                             _sha(private_dir / "private_rollout_trace.npz"))
            replay = _read(replay_dir / "private_recompute_summary.json")
            budget = _read(budget_dir / "unified_budget_summary.json")
            self.assertEqual(scene["inputs"]["recompute_summary_sha256"],
                             _sha(replay_dir / "private_recompute_summary.json"))
            self.assertEqual(scene["inputs"]["budget_summary_sha256"],
                             _sha(budget_dir / "unified_budget_summary.json"))
            self.assertTrue(replay["pass_recompute"])
            self.assertEqual(replay["failure_count"], 0)
            self.assertEqual(budget["budget_overrun_tick_count"], 0)
            self.assertEqual(budget["parity_failure_count"], 0)
            row_total += replay["checked_interval_rows"]
            times = [row["preflight_plus_qp_ms"] for row in records]
            self.assertAlmostEqual(float(np.percentile(times, 95)),
                                   scene["preflight_plus_qp_timing"]["p95_ms"],
                                   places=9)
            for row in records:
                if not row["strict_online_domain_met"]:
                    observed.append((scene_id, row["tick"], row["action_mode"],
                                     row["geometry_domain_status"]))
        self.assertEqual(row_total, summary["recomputed_interval_rows"])
        self.assertEqual(len(observed), summary["strict_domain_violation_tick_count"])
        self.assertEqual(observed, [
            ("v6_lite_scenario_01", tick, "TRACK",
             "OUTSIDE_DECLARED_WORK_DOMAIN;ON_DECLARED_SHAPE_SUBSPACE")
            for tick in range(888, 951)])
        failure_rows = [json.loads(line) for line in failure_path.read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual([(row["scenario_id"], row["tick"], row["action_mode"],
                           row["geometry_domain_status"]) for row in failure_rows],
                         observed)


if __name__ == "__main__":
    unittest.main()
