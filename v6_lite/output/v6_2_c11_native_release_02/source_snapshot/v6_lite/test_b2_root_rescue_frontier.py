"""Verify the frozen 998-state diagnostic root-query budget ladder."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RootRescueFrontierTests(unittest.TestCase):
    def test_all_unresolved_root64_states_have_hash_bound_127_255_results(self) -> None:
        root = Path(__file__).parent / "output/v6_2_b2"
        folder = root / "root_rescue_frontier"
        paths = {
            "summary": folder / "root_rescue_summary.json",
            "states": folder / "root_rescue_states.jsonl",
            "failures": folder / "root_rescue_failures.jsonl",
            "document": folder / "ROOT_RESCUE_FRONTIER.md",
        }
        manifest = json.loads((folder / "root_rescue_manifest.json").read_text(
            encoding="utf-8"))
        for name, path in paths.items():
            self.assertEqual(manifest[f"{name}_sha256"], _sha(path))
        self.assertEqual(paths["failures"].stat().st_size, 0)
        report = json.loads(paths["summary"].read_text(encoding="utf-8"))
        self.assertEqual(report["state_records_sha256"], _sha(paths["states"]))
        self.assertEqual(report["failure_records_sha256"], _sha(paths["failures"]))
        self.assertEqual(report["scope"],
                         "998_frozen_A1_states_unknown_after_64_point_root_requery")
        self.assertEqual(report["budgets"], [127, 255])
        self.assertEqual(report["max_leaves"], 256)
        self.assertFalse(report["online_control_changed"])
        self.assertFalse(report["new_interval_mode_executed"])
        counter_path = root / "repartition_counterfactual/repartition_counterfactual.json"
        self.assertEqual(report["input_repartition_counterfactual_sha256"],
                         _sha(counter_path))
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        for inputs in report["inputs"].values():
            self.assertEqual(_sha(Path(inputs["metrics_path"])), inputs["metrics_sha256"])
            self.assertEqual(len(inputs["traces"]), 5)
            for trace in inputs["traces"]:
                self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])

        prior = json.loads(counter_path.read_text(encoding="utf-8"))
        expected = {(item["mode"], item["scenario_id"], item["tick"])
                    for item in prior["entries"]
                    if item["cold_status"] == "UNKNOWN_CROSSES_GATE"}
        self.assertEqual(len(expected), 998)
        rows = [json.loads(line) for line in paths["states"].read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1996)
        grouped = defaultdict(dict)
        for row in rows:
            key = row["mode"], row["scenario_id"], row["tick"]
            self.assertIn(key, expected)
            self.assertNotIn(row["budget"], grouped[key])
            grouped[key][row["budget"]] = row
            self.assertEqual(row["prior_root64_status"], "UNKNOWN_CROSSES_GATE")
            self.assertIsNone(row["parity_failure"])
            self.assertLessEqual(max(row["parity"].values()), 1e-12)
            self.assertLessEqual(row["point_evaluations"], row["budget"])
            self.assertGreater(row["reference_query_ms"], 0)
            self.assertGreater(row["prepared_query_ms"], 0)
        self.assertEqual(set(grouped), expected)
        self.assertTrue(all(set(items) == {127, 255} for items in grouped.values()))
        for mode in ("baseline", "enabled"):
            own = [row for row in rows if row["mode"] == mode]
            summary = report["modes"][mode]
            self.assertEqual(summary["counts"]["frozen_states"],
                             124 if mode == "baseline" else 874)
            for budget in (127, 255):
                trial = [row for row in own if row["budget"] == budget]
                self.assertEqual(len(trial), summary["counts"]["frozen_states"])
                counts = Counter(row["status"] for row in trial)
                for status, count in counts.items():
                    self.assertEqual(summary["counts"][f"{budget}_{status}"], count)
                self.assertAlmostEqual(
                    summary["budgets"][str(budget)]["prepared_query_ms"]["p95"],
                    float(np.percentile([row["prepared_query_ms"]
                                         for row in trial], 95)),
                )
        self.assertEqual(report["modes"]["enabled"]["counts"].get(
            "255_UNKNOWN_CROSSES_GATE", 0), 0)


if __name__ == "__main__":
    unittest.main()
