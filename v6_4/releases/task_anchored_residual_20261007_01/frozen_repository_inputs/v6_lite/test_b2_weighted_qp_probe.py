"""Integrity checks for the frozen early-state weighted-QP probe."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


class WeightedQPProbeTests(unittest.TestCase):
    def test_native_replay_rows_and_action_validation_are_hash_bound(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2"
        path = root / "qp_probe_early" / "weighted_qp_probe.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((root / "qp_probe_early"
                               / "weighted_qp_probe_manifest.json").read_text(
                                   encoding="utf-8"))
        self.assertEqual(manifest["report_sha256"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest["report_bytes"], path.stat().st_size)
        self.assertFalse(report["online_control_changed"])
        self.assertFalse(report["new_mode_closed_loop_acceptance"])
        self.assertFalse(report["solver_dual_reused_between_probes"])
        self.assertTrue(report["passed_as_read_only_integrity"])
        self.assertEqual(report["probe_ticks"], [50, 100, 150])
        self.assertEqual(report["point_budget"], 63)
        self.assertEqual(report["residual_bound_scope"],
                         "analytic_URDF_discrete_backbone_actual_vs_projected_only")
        self.assertEqual(report["residual_bound_sampled_material_points_per_state"],
                         31)
        self.assertEqual(report["state_local_envelope_scope"],
                         "actual_MuJoCo_continuum_capsules_inside_existing_PCC_tubes_at_frozen_state_only")
        self.assertEqual(report["state_local_envelope_pcc_points_per_segment"], 17)
        self.assertEqual(report["state_local_envelope_axis_points_per_capsule"], 5)
        self.assertEqual(report["state_local_envelope_fallback_geom_names"],
                         ["collision_0003"])
        self.assertEqual(len(report["records"]), 30)
        self.assertEqual(len(report["native_replay_checks"]), 10)
        self.assertTrue(all(item["max_state_error"] <= 1e-8
                            for item in report["native_replay_checks"]))
        self.assertEqual(report["input_refined_start_sha256"], hashlib.sha256(
            (root / "refined_start" / "refined_start_report.json").read_bytes()
        ).hexdigest())
        self.assertEqual(report["source_hash_newline_policy"], "LF_NORMALIZED")
        for name, digest in report["source_sha256"].items():
            self.assertEqual(hashlib.sha256((Path(__file__).parent / name)
                                            .read_bytes().replace(b"\r\n", b"\n")
                                            ).hexdigest(), digest)
        for item in report["inputs"].values():
            self.assertEqual(hashlib.sha256(Path(item["metrics_path"]).read_bytes())
                             .hexdigest(), item["metrics_sha256"])
            for trace in item["traces"]:
                self.assertEqual(hashlib.sha256(Path(trace["path"]).read_bytes())
                                 .hexdigest(), trace["sha256"])
        for mode in ("baseline", "enabled"):
            own = [item for item in report["records"] if item["mode"] == mode]
            self.assertEqual(len(own), 15)
            self.assertEqual({item["tick"] for item in own}, {50, 100, 150})
            self.assertEqual(len({item["scenario_id"] for item in own}), 5)
            self.assertEqual(report["summary"][mode]["query_budget_ok_count"], 15)
            self.assertEqual(report["summary"][mode]["proxy_safe_count"], 15)
            self.assertEqual(report["summary"][mode]["envelope_supported_count"], 0)
            self.assertEqual(report["summary"][mode]["admission_preconditions_met_count"], 0)
            self.assertEqual(report["summary"][mode]["validated_command_count"], 15)
            self.assertEqual(report["summary"][mode]["state_local_capsule_covered_count"], 15)
            for item in own:
                self.assertTrue(item["query_budget_ok"])
                self.assertTrue(item["proxy_safe"])
                self.assertFalse(item["envelope_supported"])
                self.assertFalse(item["admission_preconditions_met"])
                self.assertEqual(item["query_status"],
                                 "PROXY_CLEARANCE_AT_LEAST_GATE")
                self.assertEqual(item["geometry_domain_status"],
                                 "INSIDE_DECLARED_WORK_DOMAIN;"
                                 "OUTSIDE_DECLARED_SHAPE_SUBSPACE")
                self.assertGreater(item["subspace_residual_linf_rad"],
                                   report["shape_subspace_membership_tolerance_rad"])
                self.assertGreater(item["discrete_backbone_residual_upper_m"], 0.0)
                self.assertLessEqual(
                    item["sampled_discrete_backbone_residual_max_m"],
                    item["discrete_backbone_residual_upper_m"] + 2e-12,
                )
                self.assertEqual(item["state_local_capsule_envelope_status"],
                                 "COVERED_AT_THIS_STATE")
                self.assertGreater(item["state_local_capsule_envelope_min_margin_m"],
                                   0.0)
                self.assertEqual(len(item["state_local_capsule_envelope_rows"]), 61)
                self.assertEqual(item["state_local_fallback_geom_names"],
                                 ["collision_0003"])
                for row in item["state_local_capsule_envelope_rows"]:
                    self.assertGreaterEqual(row["margin_m"], 0.0)
                    self.assertLessEqual(row["required_tube_radius_upper_m"],
                                         row["declared_tube_radius_m"])
                self.assertEqual(item["floating_point_certification"],
                                 "NOT_FORMALLY_CERTIFIED")
                self.assertGreater(item["selected_interval_count"], 0)
                self.assertEqual(item["failure_reason"], "none")
                self.assertTrue(item["candidate_valid"])
                self.assertTrue(item["ramp_valid"])
                self.assertEqual(np.asarray(item["selected_command"]).shape, (17,))
                self.assertTrue(np.all(np.isfinite(item["selected_command"])))
                for key in ("matrix_max_abs_error", "lower_max_abs_error",
                            "drift_max_abs_error", "gain_max_abs_error"):
                    self.assertLessEqual(item["row_parity"][key], 1e-7)


if __name__ == "__main__":
    unittest.main()
