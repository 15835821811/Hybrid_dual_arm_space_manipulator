"""Verify the frozen private-state QP comparison and its provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "v6_lite/output/v6_2_b2/private_qp_sphere_trial"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrivateQPSphereEvidenceTests(unittest.TestCase):
    def test_hash_bound_abba_parity_and_scope(self) -> None:
        report = json.loads((OUTPUT / "private_qp_sphere_summary.json").read_text(
            encoding="utf-8"))
        manifest = json.loads((OUTPUT / "private_qp_sphere_manifest.json").read_text(
            encoding="utf-8"))
        for label, filename in (("summary", "private_qp_sphere_summary.json"),
                                ("records", "private_qp_sphere_records.jsonl"),
                                ("failures", "private_qp_sphere_failures.jsonl"),
                                ("document", "PRIVATE_QP_SPHERE.md")):
            self.assertEqual(manifest[f"{label}_sha256"], _sha(OUTPUT / filename))
        self.assertEqual(report["records_sha256"], manifest["records_sha256"])
        self.assertEqual(report["failures_sha256"], manifest["failures_sha256"])
        self.assertEqual(report["source_hash_newline_policy"], "LF_NORMALIZED")
        for name, digest in report["source_sha256"].items():
            raw = (ROOT / "v6_lite" / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
                             digest, name)
        self.assertEqual(report["groups_predeclared"], [
            {"name": "a1", "method": "reference"},
            {"name": "b1", "method": "sphere_screen"},
            {"name": "b2", "method": "sphere_screen"},
            {"name": "a2", "method": "reference"},
        ])
        self.assertEqual(report["ticks_per_scene"], [1, 50, 100, 200, 300, 399])
        self.assertEqual(report["record_count"], 120)
        self.assertEqual(report["failure_count"], 0)
        self.assertFalse(report["online_controller_changed"])
        self.assertFalse(report["private_servo_commanded_by_trial"])
        self.assertFalse(report["strict_online_domain_accepted"])
        self.assertFalse(report["full_cycle_timing_measured"])
        self.assertTrue(all(group["record_count"] == 30
                            and group["mismatch_count"] == 0
                            for group in report["groups"]))
        rows = [json.loads(line) for line in (OUTPUT / "private_qp_sphere_records.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 120)
        by_key = {(r["group"], r["scenario_id"], r["tick"]): r for r in rows}
        self.assertEqual(len(by_key), 120)
        for index in range(5):
            scene = f"v6_lite_scenario_{index:02d}"
            folder = ROOT / "v6_lite/output/v6_2_b2/private_rollout_400" / f"scene_{index:02d}"
            for label, filename in (("trace", "private_rollout_trace.npz"),
                                    ("summary", "private_rollout_summary.json"),
                                    ("records", "private_rollout_records.jsonl")):
                self.assertEqual(report["inputs"][scene][f"{label}_sha256"],
                                 _sha(folder / filename))
            for tick in report["ticks_per_scene"]:
                a = by_key[("a1", scene, tick)]
                for group in ("b1", "b2", "a2"):
                    b = by_key[(group, scene, tick)]
                    for field in ("sources", "solver_status", "action_mode",
                                  "failure_reason", "strict_subspace_domain"):
                        self.assertEqual(a[field], b[field], (scene, tick, field))
                    for field in ("candidate", "selected_command", "matrix", "lower",
                                  "drifts", "gains"):
                        if a[field] is None:
                            self.assertIsNone(b[field])
                        else:
                            np.testing.assert_allclose(a[field], b[field], atol=1e-8,
                                                       rtol=0)
        self.assertLess(report["summary"]["sphere_screen"]["exact_pair_calls"]["p95"],
                        report["summary"]["reference"]["exact_pair_calls"]["p95"])
        self.assertGreater(report["summary"]["sphere_screen"]
                           ["preflight_plus_qp_ms"]["max"], 20.0)


if __name__ == "__main__":
    unittest.main()
