"""Checks for the immutable, read-only OpenBLAS ABBA timing trial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class B2BLASThreadTrialTests(unittest.TestCase):
    def test_abba_groups_and_raw_hashes(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2" / "blas_thread_trial"
        report = json.loads((root / "abba_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "abba_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "v6_2_b2_openblas_threads_abba_v1")
        self.assertEqual([group["requested_openblas_threads"]
                          for group in report["groups"]], [2, 1, 1, 2])
        self.assertEqual(report["complete_group_count"], 4)
        self.assertEqual(report["candidate_mismatch_count"], 0)
        self.assertFalse(report["new_interval_mode_executed"])
        self.assertFalse(report["full_control_cycle_measured"])
        self.assertEqual(manifest["summary_sha256"], _sha(root / "abba_summary.json"))
        self.assertEqual(manifest["records_sha256"], _sha(root / "abba_records.jsonl"))
        self.assertEqual(manifest["document_sha256"], _sha(root / "BLAS_THREAD_TRIAL.md"))
        self.assertEqual(report["records_sha256"], _sha(root / "abba_records.jsonl"))
        for name, digest in report["source_sha256"].items():
            path = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(path.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        records = [json.loads(line) for line in
                   (root / "abba_records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 120)
        self.assertTrue(all(row["candidate_parity_ok"] for row in records))
        for group in report["groups"]:
            self.assertEqual(group["status"], "COMPLETE")
            self.assertEqual(group["record_count"], 30)
            self.assertEqual(group["candidate_mismatch_count"], 0)
            folder = root / group["name"]
            self.assertEqual(group["report_sha256"], _sha(folder / "weighted_qp_probe.json"))
            self.assertEqual(group["document_sha256"], _sha(folder / "WEIGHTED_QP_PROBE.md"))
            self.assertEqual(group["manifest_sha256"], _sha(folder / "weighted_qp_probe_manifest.json"))
            self.assertEqual(group["stdout_sha256"], _sha(root / f"{group['name']}_stdout.txt"))
            self.assertEqual(group["stderr_sha256"], _sha(root / f"{group['name']}_stderr.txt"))
        for mode in ("baseline", "enabled"):
            for threads in (1, 2):
                self.assertEqual(report["modes"][mode][str(threads)]["record_count"], 30)


if __name__ == "__main__":
    unittest.main()
