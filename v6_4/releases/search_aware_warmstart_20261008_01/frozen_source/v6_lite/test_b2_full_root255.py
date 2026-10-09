"""Check the complete frozen A.1 root-255 diagnostic census."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FullRoot255CensusTests(unittest.TestCase):
    def test_all_saved_states_have_reference_prepared_parity(self) -> None:
        root = Path(__file__).parent / "output/v6_2_b2"
        folder = root / "full_root255_census"
        outputs = {
            "summary": folder / "full_root255_summary.json",
            "states": folder / "full_root255_states.jsonl",
            "failures": folder / "full_root255_failures.jsonl",
            "document": folder / "FULL_ROOT255_CENSUS.md",
        }
        manifest = json.loads((folder / "full_root255_manifest.json").read_text(
            encoding="utf-8"))
        for name, path in outputs.items():
            self.assertEqual(_sha(path), manifest[f"{name}_sha256"])
        self.assertEqual(outputs["failures"].stat().st_size, 0)
        report = json.loads(outputs["summary"].read_text(encoding="utf-8"))
        self.assertEqual(report["state_records_sha256"], _sha(outputs["states"]))
        self.assertEqual(report["failure_records_sha256"], _sha(outputs["failures"]))
        self.assertEqual(report["point_budget"], 255)
        self.assertEqual(report["max_leaves"], 256)
        self.assertFalse(report["online_control_changed"])
        self.assertFalse(report["new_interval_mode_executed"])
        self.assertEqual(report["input_warm_shadow_sha256"], _sha(
            root / "shadow_warm/shadow_report.json"))
        self.assertEqual(report["input_root_rescue_summary_sha256"], _sha(
            root / "root_rescue_frontier/root_rescue_summary.json"))
        self.assertEqual(report["input_root_rescue_states_sha256"], _sha(
            root / "root_rescue_frontier/root_rescue_states.jsonl"))
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        for item in report["inputs"].values():
            self.assertEqual(_sha(Path(item["metrics_path"])),
                             item["metrics_sha256"])
            self.assertEqual(len(item["traces"]), 5)
            for trace in item["traces"]:
                self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])

        rows = [json.loads(line) for line in outputs["states"].read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 13500)
        keys = {(row["mode"], row["scenario_id"], row["tick"]) for row in rows}
        self.assertEqual(len(keys), len(rows))
        for row in rows:
            self.assertIsNone(row["parity_failure"])
            self.assertLessEqual(max(row["parity"].values()), 1e-12)
            self.assertLessEqual(row["point_evaluations"], 255)
            self.assertGreater(row["reference_query_ms"], 0)
            self.assertGreater(row["prepared_query_ms"], 0)
        for mode in ("baseline", "enabled"):
            own = [row for row in rows if row["mode"] == mode]
            counts = Counter(row["status"] for row in own)
            summary = report["modes"][mode]
            self.assertEqual(len(own), 6750)
            self.assertEqual(summary["counts"]["task_states"], 6750)
            self.assertEqual(summary["counts"]["root_rescue_matches"],
                             124 if mode == "baseline" else 874)
            for status, count in counts.items():
                self.assertEqual(summary["counts"][status], count)
            self.assertEqual(sum(row["budget_exhausted"] for row in own),
                             summary["counts"]["budget_exhausted"])
            self.assertLessEqual(summary["maximum_parity_error_m"], 1e-12)
        self.assertEqual(report["modes"]["baseline"]["counts"][
            "PROXY_CLEARANCE_BELOW_GATE"], 2557)
        self.assertEqual(report["modes"]["baseline"]["counts"][
            "UNKNOWN_CROSSES_GATE"], 17)
        self.assertEqual(report["modes"]["enabled"]["counts"].get(
            "PROXY_CLEARANCE_BELOW_GATE", 0), 0)
        self.assertEqual(report["modes"]["enabled"]["counts"].get(
            "UNKNOWN_CROSSES_GATE", 0), 0)


if __name__ == "__main__":
    unittest.main()
