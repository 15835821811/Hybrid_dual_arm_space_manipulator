"""Audit saved B.2 query optimization trials and full-trace parity."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QueryOptimizationEvidenceTests(unittest.TestCase):
    def test_trials_are_hash_bound_and_do_not_claim_online_admission(self) -> None:
        root = Path(__file__).parent / "output/v6_2_b2"
        cases = (
            ("batched_query_trial", "batched_query_summary.json",
             "BATCHED_QUERY_TRIAL.md", "batched_query_manifest.json", 60),
            ("prepared_query_trial", "prepared_query_summary.json",
             "PREPARED_QUERY_TRIAL.md", "prepared_query_manifest.json", 136),
        )
        for folder_name, summary_name, document_name, manifest_name, count in cases:
            folder = root / folder_name
            summary_path = folder / summary_name
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads((folder / manifest_name).read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary_sha256"], _sha(summary_path))
            self.assertEqual(manifest["document_sha256"], _sha(folder / document_name))
            self.assertEqual(len(report["records"]), count)
            self.assertFalse(report["online_control_changed"])
            self.assertFalse(report["new_interval_mode_executed"])
            self.assertEqual(report["source_hash_newline_policy"], "LF_NORMALIZED")
            for name, digest in report["source_sha256"].items():
                self.assertEqual(hashlib.sha256((Path(__file__).parent / name)
                    .read_bytes().replace(b"\r\n", b"\n")).hexdigest(), digest)
            for inputs in report["inputs"].values():
                self.assertEqual(_sha(Path(inputs["metrics_path"])), inputs["metrics_sha256"])
                self.assertEqual(len(inputs["traces"]), 5)
                for trace in inputs["traces"]:
                    self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])
            for item in report["records"]:
                self.assertLessEqual(max(item["parity"].values()), 1e-12)
        batched = json.loads((root / cases[0][0] / cases[0][1]).read_text(
            encoding="utf-8"))
        prepared = json.loads((root / cases[1][0] / cases[1][1]).read_text(
            encoding="utf-8"))
        self.assertEqual(batched["input_refined_start_sha256"], _sha(
            root / "refined_start/refined_start_report.json"))
        self.assertEqual(prepared["input_warm_shadow_sha256"], _sha(
            root / "shadow_warm/shadow_report.json"))
        self.assertEqual(Counter(item["state_group"] for item in
                                 prepared["records"]),
                         {"early_probe": 20, "warm_unknown_sample": 116})

    def test_every_saved_state_matches_reference_partition_and_decision(self) -> None:
        root = Path(__file__).parent / "output/v6_2_b2"
        folder = root / "prepared_full_trace"
        paths = {
            "summary": folder / "prepared_full_trace_summary.json",
            "states": folder / "prepared_full_trace_states.jsonl",
            "failures": folder / "prepared_full_trace_failures.jsonl",
            "document": folder / "PREPARED_FULL_TRACE.md",
        }
        manifest = json.loads((folder / "prepared_full_trace_manifest.json")
                              .read_text(encoding="utf-8"))
        for name, path in paths.items():
            self.assertEqual(manifest[f"{name}_sha256"], _sha(path))
        self.assertEqual(paths["failures"].stat().st_size, 0)
        report = json.loads(paths["summary"].read_text(encoding="utf-8"))
        self.assertEqual(report["state_records_sha256"], _sha(paths["states"]))
        self.assertEqual(report["failure_records_sha256"], _sha(paths["failures"]))
        self.assertEqual(report["input_warm_shadow_sha256"], _sha(
            root / "shadow_warm/shadow_report.json"))
        self.assertFalse(report["online_control_changed"])
        self.assertFalse(report["new_interval_mode_executed"])
        for name, digest in report["source_sha256"].items():
            self.assertEqual(hashlib.sha256((Path(__file__).parent / name)
                .read_bytes().replace(b"\r\n", b"\n")).hexdigest(), digest)
        rows = [json.loads(line) for line in paths["states"].read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 13500)
        self.assertEqual(len({(row["mode"], row["scenario_id"], row["tick"])
                              for row in rows}), 13500)
        warm = json.loads((root / "shadow_warm/shadow_report.json")
                          .read_text(encoding="utf-8"))
        for mode in ("baseline", "enabled"):
            own = [row for row in rows if row["mode"] == mode]
            self.assertEqual(len(own), 6750)
            self.assertTrue(all(row["parity_failure"] is None for row in own))
            errors = [max(row["parity"].values()) for row in own]
            self.assertLessEqual(max(errors), 1e-12)
            count = sum(row["status"] == "UNKNOWN_CROSSES_GATE" for row in own)
            self.assertEqual(count, warm["modes"][mode]["counts"][
                "warm_all_task_unknown"])
            summary = report["modes"][mode]
            self.assertEqual(summary["counts"]["task_states"], 6750)
            self.assertEqual(summary["counts"]["UNKNOWN_CROSSES_GATE"], count)
            self.assertAlmostEqual(summary["maximum_parity_error_m"], max(errors))
            for key, field in (("reference_query_ms", "reference_query_ms"),
                               ("prepared_query_ms", "prepared_query_ms")):
                self.assertAlmostEqual(summary[key]["p95"],
                                       float(np.percentile([row[field] for row in own], 95)))
        for inputs in report["inputs"].values():
            self.assertEqual(_sha(Path(inputs["metrics_path"])), inputs["metrics_sha256"])
            self.assertEqual(len(inputs["traces"]), 5)
            for trace in inputs["traces"]:
                self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])


if __name__ == "__main__":
    unittest.main()
