"""Check the frozen read-only QP sphere-screen evidence and its limits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path("v6_lite/output/v6_2_b2/qp_sphere_trial")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WeightedQPSphereTrialEvidenceTest(unittest.TestCase):
    def test_four_groups_preserve_candidates_and_independent_rows(self) -> None:
        summary = json.loads((ROOT / "qp_sphere_summary.json").read_text(
            encoding="utf-8"))
        self.assertEqual(summary["complete_group_count"], 4)
        self.assertEqual(summary["candidate_mismatch_count"], 0)
        self.assertEqual(summary["probe_ticks"], [50, 100, 150])
        self.assertFalse(summary["online_controller_changed"])
        self.assertFalse(summary["new_interval_mode_executed"])
        self.assertFalse(summary["full_control_cycle_measured"])
        self.assertEqual(summary["input_original_probe_sha256"], _sha(
            Path("v6_lite/output/v6_2_b2/qp_probe_early/weighted_qp_probe.json")))
        manifest = json.loads((ROOT / "qp_sphere_manifest.json").read_text(
            encoding="utf-8"))
        for label, filename in (("summary", "qp_sphere_summary.json"),
                                ("records", "qp_sphere_records.jsonl"),
                                ("document", "QP_SPHERE_TRIAL.md")):
            self.assertEqual(manifest[f"{label}_sha256"], _sha(ROOT / filename))
        self.assertEqual(summary["records_sha256"], _sha(
            ROOT / "qp_sphere_records.jsonl"))
        reference = json.loads(Path(
            "v6_lite/output/v6_2_b2/qp_probe_early/weighted_qp_probe.json"
        ).read_text(encoding="utf-8"))
        by_state = {(row["mode"], row["scenario_id"], row["tick"]): row
                    for row in reference["records"]}
        self.assertEqual(len(by_state), 30)
        for group in summary["groups"]:
            self.assertEqual(group["status"], "COMPLETE")
            self.assertEqual(group["record_count"], 30)
            folder = ROOT / group["name"]
            for field, filename in (("report", "weighted_qp_probe.json"),
                                    ("document", "WEIGHTED_QP_PROBE.md"),
                                    ("manifest", "weighted_qp_probe_manifest.json")):
                self.assertEqual(group[f"{field}_sha256"], _sha(folder / filename))
            self.assertEqual(group["stdout_sha256"], _sha(
                ROOT / f"{group['name']}_stdout.txt"))
            report = json.loads((folder / "weighted_qp_probe.json").read_text(
                encoding="utf-8"))
            self.assertTrue(report["passed_as_read_only_integrity"])
            self.assertEqual(len(report["records"]), 30)
            for row in report["records"]:
                expected = by_state[(row["mode"], row["scenario_id"], row["tick"])]
                self.assertEqual(row["action_mode"], expected["action_mode"])
                self.assertEqual(row["failure_reason"], expected["failure_reason"])
                np.testing.assert_allclose(row["solver_candidate"],
                                           expected["solver_candidate"],
                                           rtol=0, atol=1e-8)
                if row["selected_command"] is None:
                    self.assertIsNone(expected["selected_command"])
                else:
                    np.testing.assert_allclose(row["selected_command"],
                                               expected["selected_command"],
                                               rtol=0, atol=1e-8)
                for name in ("matrix_max_abs_error", "lower_max_abs_error",
                             "drift_max_abs_error", "gain_max_abs_error"):
                    self.assertLessEqual(row["row_parity"][name], 1e-7)
        for mode in ("baseline", "enabled"):
            full = summary["modes"][mode]["reference"]
            screened = summary["modes"][mode]["sphere_screen"]
            self.assertEqual(full["record_count"], 30)
            self.assertEqual(screened["record_count"], 30)
            self.assertEqual(full["exact_call_count"]["p95"], 2927)
            self.assertLess(screened["exact_call_count"]["p95"], 2927)
            self.assertGreater(screened["query_plus_qp_probe_ms"]["p95"], 20)


if __name__ == "__main__":
    unittest.main()
