from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPOSITORY_ROOT / "v6_lite" / "output" / "v6_1a"
FROZEN_V6_CONTROL_SOURCE_SHA256 = {
    "v6_lite/hierarchical_qp.py": (
        "3ac5a550ccb8f2362dafaa709410de9bdc773169f09f943a9a2e9bf52043435e"
    ),
    "v6_lite/run_v6_lite.py": (
        "112b461fa032ea1b3155aa29bc5d64b4b1ead89e072b7ba4558713332b07fc93"
    ),
    "v6_lite/irregular_waypoints.py": (
        "4ed7f2a4ed771604827f1908ad67020eae9477028a73cac770c1e7e0e8f36875"
    ),
}


def _load_json(name: str) -> dict:
    with (AUDIT_ROOT / name).open("r", encoding="utf-8") as stream:
        return json.load(stream)


class V61AArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_json("v6_1a_manifest.json")
        cls.shape = _load_json("shape_model_audit.json")
        cls.geometry = _load_json("geometry_envelope_audit.json")
        cls.comparison = _load_json("shape_vs_geom_comparison.json")

    def test_manifest_and_all_reports_pass(self) -> None:
        self.assertTrue(self.manifest["passed"])
        self.assertTrue(self.shape["passed"])
        self.assertTrue(self.geometry["passed"])
        self.assertTrue(self.comparison["passed"])
        contract_hash = self.manifest["model_contract_sha256"]
        self.assertEqual(self.shape["model_contract_sha256"], contract_hash)
        self.assertEqual(self.geometry["model_contract_sha256"], contract_hash)
        self.assertEqual(self.comparison["model_contract_sha256"], contract_hash)

    def test_shape_audit_has_required_scale_and_accuracy(self) -> None:
        self.assertGreaterEqual(self.shape["configuration_count"], 10_000)
        checks = self.shape["checks"]
        self.assertTrue(all(checks.values()))
        fk = self.shape["discrete_fk_vs_mujoco"]
        self.assertLessEqual(
            fk["position_error_m_max"], fk["position_threshold_m"]
        )
        self.assertLessEqual(
            fk["orientation_error_rad_max"], fk["orientation_threshold_rad"]
        )
        distribution = self.shape["pcc_vs_discrete_full_arm"]
        self.assertEqual(distribution["position_error_m"]["count"], 310_000)
        self.assertEqual(distribution["orientation_error_rad"]["count"], 310_000)

    def test_geometry_ownership_and_holdout_coverage(self) -> None:
        self.assertEqual(self.geometry["calibration_configuration_count"], 8_000)
        self.assertEqual(self.geometry["held_out_configuration_count"], 2_000)
        self.assertEqual(self.geometry["continuum_collision_geom_count"], 62)
        self.assertEqual(self.geometry["capsule_count"], 61)
        self.assertEqual(self.geometry["fallback_geom_names"], ["collision_0003"])
        self.assertTrue(all(self.geometry["checks"].values()))
        self.assertEqual(
            self.geometry["held_out"]["uncovered_configuration_count"], 0
        )

    def test_shadow_comparison_has_no_finite_false_safe_cases(self) -> None:
        self.assertGreaterEqual(self.comparison["target_case_count"], 10_000)
        expected_categories = {
            "face",
            "edge",
            "corner",
            "rotated",
            "mid_arm",
            "penetration",
        }
        self.assertEqual(
            set(self.comparison["target_case_categories"]), expected_categories
        )
        self.assertTrue(all(self.comparison["checks"].values()))
        self.assertEqual(self.comparison["false_safe"]["capsule_count"], 0)
        self.assertEqual(self.comparison["false_safe"]["pcc_tube_count"], 0)
        trace = self.comparison["formal_moving_target_trace_shadow"]
        self.assertGreater(trace["sample_count"], 0)
        self.assertEqual(trace["pcc_false_safe_count"], 0)
        self.assertIn("distmax", self.comparison["query_truncation"]["rule"])

    def test_manifest_hashes_match_the_published_reports(self) -> None:
        for artifact in self.manifest["artifacts"]:
            path = REPOSITORY_ROOT / artifact["path"]
            payload = path.read_bytes()
            self.assertEqual(len(payload), artifact["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), artifact["sha256"])

    def test_v61a_does_not_modify_the_frozen_v6_control_sources(self) -> None:
        self.assertEqual(
            self.shape["controller_source_sha256"],
            FROZEN_V6_CONTROL_SOURCE_SHA256,
        )
        for relative_path, expected_hash in FROZEN_V6_CONTROL_SOURCE_SHA256.items():
            payload = (REPOSITORY_ROOT / relative_path).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)


if __name__ == "__main__":
    unittest.main()
