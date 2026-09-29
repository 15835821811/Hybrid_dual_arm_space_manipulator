"""Verify private multi-cycle evidence, replay provenance and gate limits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path("v6_lite/output/v6_2_b2")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrivateRolloutEvidenceTests(unittest.TestCase):
    def test_five_divergent_private_traces_do_not_claim_admission(self) -> None:
        parent = ROOT / "private_rollout_400"
        summary_dir = ROOT / "private_rollout_400_summary"
        summary = json.loads((summary_dir / "private_five_scene_summary.json")
                             .read_text(encoding="utf-8"))
        manifest = json.loads((summary_dir / "private_five_scene_manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(summary["complete_horizon_count"], 5)
        self.assertEqual(summary["old_violation_tick_reached_count"], 5)
        self.assertEqual(summary["old_violation_tick_start_satisfied_count"], 5)
        self.assertFalse(summary["strict_online_admission"])
        self.assertFalse(summary["new_mode_five_scene_acceptance"])
        self.assertFalse(summary["full_cycle_20ms_acceptance"])
        self.assertEqual(manifest["summary_sha256"], _sha(
            summary_dir / "private_five_scene_summary.json"))
        self.assertEqual(manifest["document_sha256"], _sha(
            summary_dir / "PRIVATE_FIVE_SCENE.md"))
        for index, scene in enumerate(summary["scenes"]):
            folder = parent / f"scene_{index:02d}"
            report = json.loads((folder / "private_rollout_summary.json")
                                .read_text(encoding="utf-8"))
            own_manifest = json.loads((folder / "private_rollout_manifest.json")
                                      .read_text(encoding="utf-8"))
            self.assertEqual(report["scenario_id"], f"v6_lite_scenario_{index:02d}")
            self.assertEqual(report["executed_ticks"], 400)
            self.assertEqual(report["stop_reason"], "HORIZON_COMPLETE")
            self.assertEqual(report["native_replay_max_qpos_error"], 0)
            self.assertEqual(report["first_tick_outside_strict_online_domain"], 1)
            self.assertGreater(report["maximum_qpos_linf_difference_from_old_a1"], .01)
            self.assertGreater(report["private_preflight_plus_qp_timing"]["p95_ms"], 20)
            self.assertFalse(report["strict_online_domain_all_executed_ticks"])
            self.assertFalse(report["stage3_admission"])
            self.assertFalse(report["production_online_controller_changed"])
            for label, filename in (("summary", "private_rollout_summary.json"),
                                    ("records", "private_rollout_records.jsonl"),
                                    ("trace", "private_rollout_trace.npz"),
                                    ("document", "PRIVATE_ROLLOUT.md")):
                self.assertEqual(own_manifest[f"{label}_sha256"], _sha(
                    folder / filename))
            rows = [json.loads(line) for line in (folder /
                    "private_rollout_records.jsonl").read_text(
                        encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 400)
            self.assertTrue(all(row["realized_next_start"]["frozen_rows_status"]
                                == "START_ROWS_SATISFIED" for row in rows))
            self.assertTrue(all(row["ramp_covered_microstates"] == 11
                                for row in rows))
            prior = rows[scene["old_first_sampled_start_violation_tick"] - 1]
            self.assertGreater(prior["realized_next_start"][
                "minimum_start_slack_m_s"], 0)
            with np.load(folder / "private_rollout_trace.npz",
                         allow_pickle=False) as trace:
                self.assertEqual(trace["torque"].shape, (4000, 67))
                self.assertEqual(trace["qpos_states"].shape[0], 4001)
                self.assertEqual(trace["task_qvel_states"].shape[0], 401)

    def test_separate_torque_replay_and_interval_recompute_is_complete(self) -> None:
        folder = ROOT / "private_recompute_400"
        report = json.loads((folder / "private_recompute_summary.json")
                            .read_text(encoding="utf-8"))
        manifest = json.loads((folder / "private_recompute_manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertTrue(report["pass_recompute"])
        self.assertEqual(report["checked_task_states"], 2005)
        self.assertEqual(report["checked_interval_rows"], 5257)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(report["maximum_native_state_error"], 0)
        self.assertEqual(report["maximum_query_error_m"], 0)
        self.assertEqual(report["maximum_start_slack_error_m_s"], 0)
        self.assertTrue(report["shared_model_geometry_with_private_controller"])
        self.assertFalse(report["new_mode_online_admitted"])
        for label, filename in (("summary", "private_recompute_summary.json"),
                                ("checks", "private_recompute_checks.jsonl"),
                                ("interval_rows", "private_recompute_interval_rows.jsonl"),
                                ("failures", "private_recompute_failures.jsonl"),
                                ("document", "PRIVATE_RECOMPUTE.md")):
            self.assertEqual(manifest[f"{label}_sha256"], _sha(folder / filename))
        self.assertEqual((folder / "private_recompute_failures.jsonl").stat().st_size, 0)
        checks = [json.loads(line) for line in (folder /
                  "private_recompute_checks.jsonl").read_text(
                      encoding="utf-8").splitlines()]
        self.assertEqual(len(checks), 2005)
        self.assertTrue(all(not item["errors"] for item in checks))
        self.assertEqual([sum(item["scenario_id"] == f"v6_lite_scenario_{i:02d}"
                              for item in checks) for i in range(5)], [401] * 5)


if __name__ == "__main__":
    unittest.main()
