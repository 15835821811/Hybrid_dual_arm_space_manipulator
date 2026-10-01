"""Check the complete private-state QP sphere-screen census."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "v6_lite/output/v6_2_b2"
OUTPUT = BASE / "private_qp_sphere_full"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrivateFullQPSphereTests(unittest.TestCase):
    def test_full_trace_pair_parity_and_hashes(self) -> None:
        report = json.loads((OUTPUT / "private_full_qp_sphere_summary.json")
                            .read_text(encoding="utf-8"))
        manifest = json.loads((OUTPUT / "private_full_qp_sphere_manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "v6_2_b2_private_full_qp_sphere_v1")
        self.assertEqual(report["scope"], "read_only_all_saved_private_planning_states")
        self.assertEqual(report["max_ticks_per_scene"], 400)
        self.assertEqual(report["scene_count"], 5)
        self.assertEqual(report["record_count"], 4000)
        self.assertEqual(report["failure_count"], 0)
        for flag in ("production_online_controller_changed",
                     "private_servo_commanded_by_trial", "full_cycle_timing_measured",
                     "strict_online_domain_accepted"):
            self.assertFalse(report[flag])
        for label, filename in (("summary", "private_full_qp_sphere_summary.json"),
                                ("records", "private_full_qp_sphere_records.jsonl"),
                                ("failures", "private_full_qp_sphere_failures.jsonl")):
            self.assertEqual(manifest[f"{label}_sha256"], _sha(OUTPUT / filename))
        self.assertEqual(report["records_sha256"], manifest["records_sha256"])
        self.assertEqual(report["failures_sha256"], manifest["failures_sha256"])
        self.assertEqual(report["source_hash_newline_policy"], "LF_NORMALIZED")
        for name, digest in report["source_sha256"].items():
            source = (ROOT / "v6_lite" / name).read_bytes()
            self.assertEqual(hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
                             digest, name)
        rows = [json.loads(line) for line in
                (OUTPUT / "private_full_qp_sphere_records.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4000)
        by = {(row["scenario_id"], row["tick"], row["method"]): row
              for row in rows}
        self.assertEqual(len(by), 4000)
        for index in range(5):
            scene_id = f"v6_lite_scenario_{index:02d}"
            self.assertEqual(report["method_order_by_scene"][scene_id],
                             ["reference", "sphere_screen"] if index % 2 == 0
                             else ["sphere_screen", "reference"])
            folder = BASE / "private_rollout_400" / f"scene_{index:02d}"
            for label, filename in (("trace", "private_rollout_trace.npz"),
                                    ("records", "private_rollout_records.jsonl"),
                                    ("summary", "private_rollout_summary.json")):
                self.assertEqual(report["input_private_hashes"][scene_id]
                                 [f"{label}_sha256"], _sha(folder / filename))
            for tick in range(400):
                reference = by[scene_id, tick, "reference"]
                screen = by[scene_id, tick, "sphere_screen"]
                for field in ("constraint_row_sha256", "solver_status",
                              "action_mode", "failure_reason",
                              "selected_interval_count"):
                    self.assertEqual(reference[field], screen[field],
                                     (scene_id, tick, field))
                np.testing.assert_allclose(reference["candidate"],
                                           screen["candidate"], atol=1e-8, rtol=0)
                np.testing.assert_allclose(reference["selected_command"],
                                           screen["selected_command"], atol=1e-8, rtol=0)
                self.assertLessEqual(reference["saved_command_max_error"], 1e-8)
                self.assertLessEqual(screen["saved_command_max_error"], 1e-8)
        for method in ("reference", "sphere_screen"):
            values = [row["preflight_plus_qp_ms"] for row in rows
                      if row["method"] == method]
            timing = report["summary"][method]["preflight_plus_qp_ms"]
            self.assertEqual(timing["count"], 2000)
            self.assertAlmostEqual(timing["p95"], float(np.percentile(values, 95)))
            self.assertEqual(timing["over_20ms_count"],
                             sum(value > 20 for value in values))
        self.assertLess(report["summary"]["sphere_screen"]
                        ["preflight_plus_qp_ms"]["p95"], 20.0)
        self.assertGreater(report["summary"]["sphere_screen"]
                           ["preflight_plus_qp_ms"]["max"], 20.0)


if __name__ == "__main__":
    unittest.main()
