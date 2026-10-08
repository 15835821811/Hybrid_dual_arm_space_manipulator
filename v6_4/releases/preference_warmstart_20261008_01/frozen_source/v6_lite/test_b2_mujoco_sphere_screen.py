"""Regression checks for the read-only MuJoCo pair screening audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from v6_lite.audit_b2_mujoco_sphere_screen import _screened


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class B2MuJoCoSphereScreenTests(unittest.TestCase):
    def test_all_far_pairs_fall_back_to_exact_minimum(self) -> None:
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody>
              <geom type="sphere" pos="0 0 0" size="0.1"/>
              <geom type="sphere" pos="2 0 0" size="0.1"/>
            </worldbody></mujoco>
        """)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        a = np.array([0], dtype=np.int32)
        b = np.array([1], dtype=np.int32)
        expected = mujoco.mj_geomDistance(model, data, 0, 1, .09, np.zeros(6))
        result = _screened(model, data, a, b, .09)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["exact_call_count"], 1)
        self.assertEqual(result["distances"][0], expected)

    def test_native_replay_artifact_preserves_active_evidence(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2" / "mujoco_sphere_screen"
        summary = json.loads((root / "sphere_screen_summary.json").read_text(
            encoding="utf-8"))
        manifest = json.loads((root / "sphere_screen_manifest.json").read_text(
            encoding="utf-8"))
        for label, filename in (("summary", "sphere_screen_summary.json"),
                                ("records", "sphere_screen_records.jsonl"),
                                ("failures", "sphere_screen_failures.jsonl"),
                                ("document", "MUJOCO_SPHERE_SCREEN.md")):
            self.assertEqual(manifest[f"{label}_sha256"], _sha(root / filename))
        self.assertEqual(summary["records_sha256"],
                         _sha(root / "sphere_screen_records.jsonl"))
        self.assertEqual(summary["failures_sha256"],
                         _sha(root / "sphere_screen_failures.jsonl"))
        self.assertEqual(summary["record_count"], 70)
        self.assertEqual(summary["failure_count"], 0)
        self.assertFalse(summary["online_controller_changed"])
        self.assertFalse(summary["new_interval_mode_executed"])
        self.assertFalse(summary["full_control_cycle_measured"])
        self.assertEqual(summary["query_max_m"], .09)
        self.assertEqual(summary["activation_m"], .08)
        for name, digest in summary["source_sha256"].items():
            path = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(path.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        records = [json.loads(line) for line in
                   (root / "sphere_screen_records.jsonl").read_text(
                       encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 70)
        self.assertEqual(len({(row["mode"], row["scenario_id"], row["tick"])
                              for row in records}), 70)
        self.assertTrue(all(row["pair_count"] == 2927 for row in records))
        self.assertTrue(all(row["exact_call_count"] < row["pair_count"]
                            for row in records))
        self.assertTrue(all(row["native_replay_error"] <= 1e-8 for row in records))
        self.assertTrue(all(row["maximum_active_distance_error_m"] == 0.0
                            and row["maximum_active_witness_error_m"] == 0.0
                            and row["maximum_minimum_error_m"] == 0.0
                            and row["missed_active_count"] == 0
                            for row in records))


if __name__ == "__main__":
    unittest.main()
