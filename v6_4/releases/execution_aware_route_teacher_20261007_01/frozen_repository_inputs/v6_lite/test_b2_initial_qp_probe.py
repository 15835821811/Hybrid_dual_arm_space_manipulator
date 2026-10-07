"""Check the five-scene initial QP preflight and its retained script failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InitialQPProbeTests(unittest.TestCase):
    def test_initial_empty_pcc_row_set_keeps_original_qp_and_validation(self) -> None:
        root = Path(__file__).parent / "output/v6_2_b2"
        folder = root / "qp_probe_initial255"
        path = folder / "weighted_qp_probe.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((folder / "weighted_qp_probe_manifest.json").read_text(
            encoding="utf-8"))
        self.assertEqual(manifest["report_sha256"], _sha(path))
        self.assertEqual(manifest["document_sha256"], _sha(
            folder / "WEIGHTED_QP_PROBE.md"))
        self.assertTrue(report["passed_as_read_only_integrity"])
        self.assertFalse(report["online_control_changed"])
        self.assertFalse(report["new_mode_closed_loop_acceptance"])
        self.assertEqual(report["probe_protocol"], "initial_preflight")
        self.assertEqual(report["probe_ticks"], [0])
        self.assertEqual(report["point_budget"], 255)
        self.assertEqual(report["input_refined_start_sha256"], _sha(
            root / "refined_start/refined_start_report.json"))
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        self.assertEqual(len(report["native_replay_checks"]), 10)
        self.assertTrue(all(item["max_state_error"] <= 1e-8
                            for item in report["native_replay_checks"]))
        self.assertEqual(len(report["records"]), 10)
        for mode in ("baseline", "enabled"):
            own = [row for row in report["records"] if row["mode"] == mode]
            self.assertEqual(len(own), 5)
            self.assertEqual(len({row["scenario_id"] for row in own}), 5)
            item = report["summary"][mode]
            self.assertEqual(item["proxy_safe_count"], 5)
            self.assertEqual(item["envelope_supported_count"], 5)
            self.assertEqual(item["validated_command_count"], 5)
            self.assertEqual(item["admission_preconditions_met_count"], 5)
            for row in own:
                self.assertEqual(row["tick"], 0)
                self.assertEqual(row["selected_interval_count"], 0)
                self.assertEqual(row["query_points"], 5)
                self.assertEqual(row["query_status"],
                                 "PROXY_CLEARANCE_AT_LEAST_GATE")
                self.assertTrue(row["candidate_valid"])
                self.assertTrue(row["ramp_valid"])
                self.assertTrue(row["admission_preconditions_met"])
                self.assertEqual(np.asarray(row["selected_command"]).shape, (17,))
                for key in ("matrix_max_abs_error", "lower_max_abs_error",
                            "drift_max_abs_error", "gain_max_abs_error"):
                    self.assertLessEqual(row["row_parity"][key], 1e-7)
        failure = json.loads((root / "qp_probe_initial255_failures/attempt1.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "SCRIPT_ERROR_BEFORE_QP_SOLVE")
        self.assertEqual(failure["failure_output_file_count"], 0)
        self.assertEqual(len(failure["source_sha256_lf_normalized_before_fix"]), 64)


if __name__ == "__main__":
    unittest.main()
