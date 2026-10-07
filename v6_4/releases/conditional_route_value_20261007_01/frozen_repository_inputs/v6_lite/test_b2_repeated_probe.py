"""Verify the five predeclared read-only QP timing rounds and hashes."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RepeatedProbeTests(unittest.TestCase):
    def test_repeated_timings_have_all_rounds_and_original_candidates(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2"
        folder = root / "repeated_qp_probe_early"
        files = {
            "summary": folder / "repeated_probe_summary.json",
            "records": folder / "repeated_probe_records.jsonl",
            "document": folder / "REPEATED_QP_PROBE.md",
        }
        report = json.loads(files["summary"].read_text(encoding="utf-8"))
        manifest = json.loads((folder / "repeated_probe_manifest.json")
                              .read_text(encoding="utf-8"))
        for name, path in files.items():
            self.assertEqual(manifest[f"{name}_sha256"], _sha(path))
        self.assertEqual(report["records_sha256"], _sha(files["records"]))
        self.assertEqual(report["scope"],
                         "five_predeclared_repeats_of_thirty_frozen_old_A1_states")
        self.assertEqual(report["round_count"], 5)
        self.assertEqual(report["probe_ticks"], [50, 100, 150])
        self.assertEqual(report["point_budget"], 63)
        self.assertEqual(report["period_reference_ms"], 20.0)
        self.assertFalse(report["new_interval_mode_executed"])
        self.assertFalse(report["full_control_cycle_measured"])
        self.assertEqual(report["complete_round_count"], 5)
        self.assertEqual(report["candidate_mismatch_count"], 0)
        self.assertEqual(report["input_original_probe_sha256"], _sha(
            root / "qp_probe_early" / "weighted_qp_probe.json"))
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        for round_index, item in enumerate(report["rounds"], 1):
            self.assertEqual(item["round"], round_index)
            self.assertEqual(item["status"], "COMPLETE")
            self.assertEqual(item["record_count"], 30)
            self.assertEqual(item["candidate_mismatch_count"], 0)
            self.assertEqual(item["report_sha256"], _sha(Path(item["report_path"])))
            folder_for_round = Path(item["report_path"]).parent
            self.assertEqual(item["document_sha256"], _sha(
                folder_for_round / "WEIGHTED_QP_PROBE.md"))
            self.assertEqual(item["manifest_sha256"], _sha(
                folder_for_round / "weighted_qp_probe_manifest.json"))
            own = json.loads(Path(item["report_path"]).read_text(encoding="utf-8"))
            self.assertTrue(own["passed_as_read_only_integrity"])
            self.assertEqual(len(own["records"]), 30)
            self.assertEqual(own["probe_ticks"], [50, 100, 150])
            self.assertEqual(own["point_budget"], 63)
        records = [json.loads(line) for line in files["records"].read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 150)
        keys = {(r["round"], r["mode"], r["scenario_id"], r["tick"])
                for r in records}
        self.assertEqual(len(keys), 150)
        self.assertTrue(all(r["candidate_parity_ok"] for r in records))
        self.assertEqual(Counter(r["round"] for r in records),
                         {round_index: 30 for round_index in range(1, 6)})
        for mode in ("baseline", "enabled"):
            own = [r for r in records if r["mode"] == mode]
            self.assertEqual(len(own), 75)
            item = report["modes"][mode]
            self.assertEqual(item["record_count"], 75)
            values = np.asarray([r["query_plus_qp_probe_ms"] for r in own])
            self.assertAlmostEqual(item["query_plus_qp_probe_ms"]["p95"],
                                   np.percentile(values, 95))
            self.assertEqual(item["query_plus_qp_probe_ms"]["over_20ms_count"],
                             int(np.count_nonzero(values > 20.0)))
            self.assertTrue(all(r["probe_plus_envelope_ms"]
                                >= r["query_plus_qp_probe_ms"] for r in own))


if __name__ == "__main__":
    unittest.main()
