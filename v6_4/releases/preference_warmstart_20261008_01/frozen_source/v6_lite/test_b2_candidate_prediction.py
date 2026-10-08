"""Verify fixed-interval candidate prediction rows and failure provenance."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidatePredictionTests(unittest.TestCase):
    def test_same_interval_rows_are_complete_and_hash_bound(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2"
        folder = root / "candidate_prediction_early"
        files = {
            "summary": folder / "candidate_prediction_summary.json",
            "rows": folder / "candidate_prediction_rows.jsonl",
            "failures": folder / "candidate_prediction_failures.jsonl",
            "document": folder / "CANDIDATE_PREDICTION.md",
        }
        report = json.loads(files["summary"].read_text(encoding="utf-8"))
        manifest = json.loads((folder / "candidate_prediction_manifest.json")
                              .read_text(encoding="utf-8"))
        for name, path in files.items():
            self.assertEqual(manifest[f"{name}_sha256"], _sha(path))
        self.assertEqual(report["row_records_sha256"], _sha(files["rows"]))
        self.assertEqual(report["failure_records_sha256"], _sha(files["failures"]))
        self.assertEqual(report["scope"],
                         "thirty_predeclared_private_candidate_ramps_same_interval_ids_both_ends")
        self.assertEqual(report["probe_ticks"], [50, 100, 150])
        self.assertEqual(report["servo_steps_per_task"], 10)
        self.assertTrue(report["margin_is_empirical_not_certified_bound"])
        self.assertFalse(report["new_interval_mode_executed"])
        self.assertFalse(report["continuous_time_certified"])
        self.assertEqual(report["input_probe_sha256"], _sha(
            root / "qp_probe_early" / "weighted_qp_probe.json"))
        self.assertEqual(report["input_candidate_ramp_sha256"], _sha(
            root / "candidate_ramp_early" / "candidate_ramp_summary.json"))
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        for inputs in report["inputs"].values():
            self.assertEqual(_sha(Path(inputs["metrics_path"])), inputs["metrics_sha256"])
            self.assertEqual(len(inputs["traces"]), 5)
            for trace in inputs["traces"]:
                self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])

        probe = json.loads((root / "qp_probe_early" / "weighted_qp_probe.json")
                           .read_text(encoding="utf-8"))
        expected = {(r["mode"], r["scenario_id"], r["tick"]):
                    r["selected_interval_count"] for r in probe["records"]}
        cases = {(r["mode"], r["scenario_id"], r["tick"]): r
                 for r in report["cases"]}
        self.assertEqual(len(cases), 30)
        self.assertEqual(set(cases), set(expected))
        grouped = defaultdict(list)
        with files["rows"].open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                key = item["mode"], item["scenario_id"], item["tick"]
                self.assertIn(key, expected)
                grouped[key].append(item)
                self.assertEqual(item["derivative_status_start"], "SUPPORTED")
                self.assertEqual(item["derivative_status_end"], "SUPPORTED")
                self.assertAlmostEqual(
                    item["h_prediction_error_rate_m_s"],
                    (item["realized_next_h_m"]-item["predicted_next_h_m"])/0.02,
                )
                self.assertAlmostEqual(
                    item["cbf_start_slack_model_error_m_s"],
                    item["realized_start_slack_m_s"]
                    - item["predicted_start_slack_before_margin_m_s"],
                )
                self.assertAlmostEqual(
                    item["predicted_start_slack_after_margin_m_s"],
                    item["predicted_start_slack_before_margin_m_s"]
                    - item["frozen_margin_m_s"],
                )
                self.assertEqual(item["frozen_margin_m_s"], 0.005)
        self.assertEqual(set(grouped), set(expected))
        self.assertEqual(sum(map(len, grouped.values())), sum(expected.values()))
        for key, rows in grouped.items():
            self.assertEqual(len(rows), expected[key])
            self.assertEqual(len({x["interval_id"] for x in rows}), len(rows))
            self.assertEqual(cases[key]["selected_interval_count"], len(rows))
            self.assertLessEqual(cases[key]["native_replay_max_state_error"], 1e-8)
            self.assertLessEqual(cases[key]["candidate_final_qpos_parity_error"], 1e-8)
        for mode in ("baseline", "enabled"):
            own = [r for rows in grouped.values() for r in rows if r["mode"] == mode]
            item = report["modes"][mode]
            self.assertEqual(item["case_count"], 15)
            self.assertEqual(item["selected_row_count"], len(own))
            self.assertEqual(item["cbf_optimism_over_frozen_margin_count"], sum(
                -r["cbf_start_slack_model_error_m_s"] > r["frozen_margin_m_s"]
                for r in own))
        failed_attempt = json.loads((root / "candidate_prediction_early_failures"
                                     / "attempt1.json").read_text(encoding="utf-8"))
        partial = root / "candidate_prediction_early_failures" / "attempt1_partial"
        self.assertFalse(failed_attempt["full_result_generated"])
        self.assertEqual(failed_attempt["partial_rows_sha256"], _sha(
            partial / "candidate_prediction_rows.jsonl"))
        self.assertEqual(failed_attempt["partial_failure_rows_sha256"], _sha(
            partial / "candidate_prediction_failures.jsonl"))


if __name__ == "__main__":
    unittest.main()
