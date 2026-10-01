"""Verify full saved-state envelope sweep against its traces and early replay probe."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


class FullTraceEnvelopeTests(unittest.TestCase):
    def test_all_task_states_and_early_probe_are_hash_bound(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2"
        folder = root / "full_state_envelope"
        summary_path = folder / "envelope_sweep_summary.json"
        states_path = folder / "envelope_sweep_states.jsonl"
        failures_path = folder / "envelope_sweep_failures.jsonl"
        doc_path = folder / "ENVELOPE_SWEEP.md"
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads((folder / "envelope_sweep_manifest.json").read_text(
            encoding="utf-8"))
        for key, path in (("summary", summary_path), ("states", states_path),
                          ("failures", failures_path), ("document", doc_path)):
            self.assertEqual(manifest[f"{key}_sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(manifest[f"{key}_bytes"], path.stat().st_size)
        self.assertEqual(report["state_records_sha256"], manifest["states_sha256"])
        self.assertEqual(report["failure_records_sha256"], manifest["failures_sha256"])
        self.assertEqual(report["source_hash_newline_policy"], "LF_NORMALIZED")
        self.assertEqual(report["capsule_count_per_state"], 61)
        self.assertEqual(report["fallback_geom_names"], ["collision_0003"])
        self.assertFalse(report["online_control_changed"])
        self.assertEqual(report["numerical_certification"], "NOT_FORMALLY_CERTIFIED")
        warm_path = root / "shadow_warm" / "shadow_report.json"
        self.assertEqual(report["warm_shadow_sha256"],
                         hashlib.sha256(warm_path.read_bytes()).hexdigest())
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)

        early = json.loads((root / "qp_probe_early" / "weighted_qp_probe.json")
                           .read_text(encoding="utf-8"))
        early_by_key = {(item["mode"], item["scenario_id"], item["tick"]): item
                        for item in early["records"]}
        self.assertEqual(len(early_by_key), 30)
        expected_qpos_hashes = {}
        for mode in ("baseline", "enabled"):
            for trace_item in report["inputs"][mode]["traces"]:
                with np.load(Path(trace_item["path"]), allow_pickle=False) as trace:
                    for tick in (50, 100, 150):
                        expected_qpos_hashes[mode, trace_item["scenario_id"], tick] = (
                            hashlib.sha256(trace["task_qpos"][tick].tobytes()).hexdigest()
                        )
        counts: Counter[tuple[str, str]] = Counter()
        ticks: dict[tuple[str, str], set[int]] = {}
        margins: dict[str, list[float]] = {"baseline": [], "enabled": []}
        checked_early = set()
        with states_path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                mode, scenario, tick = item["mode"], item["scenario_id"], item["tick"]
                key = mode, scenario
                counts[key] += 1
                ticks.setdefault(key, set()).add(tick)
                margins[mode].append(item["minimum_margin_m"])
                self.assertEqual(item["status"], "COVERED_AT_THIS_STATE")
                self.assertGreater(item["minimum_margin_m"], 0.0)
                self.assertAlmostEqual(item["minimum_margin_m"],
                                       item["worst_capsule"]["margin_m"], places=12)
                self.assertEqual(len(item["task_qpos_sha256"]), 64)
                probe = early_by_key.get((mode, scenario, tick))
                if probe is not None:
                    checked_early.add((mode, scenario, tick))
                    self.assertEqual(item["task_qpos_sha256"],
                                     expected_qpos_hashes[mode, scenario, tick])
                    self.assertAlmostEqual(
                        item["minimum_margin_m"],
                        probe["state_local_capsule_envelope_min_margin_m"], places=10,
                    )
        self.assertEqual(checked_early, set(early_by_key))
        self.assertEqual(len(counts), 10)
        self.assertTrue(all(value == 1350 for value in counts.values()))
        self.assertTrue(all(values == set(range(1350)) for values in ticks.values()))
        self.assertEqual(failures_path.stat().st_size, 0)
        for mode in ("baseline", "enabled"):
            item = report["modes"][mode]
            self.assertEqual(item["state_count"], 6750)
            self.assertEqual(item["status_counts"], {"COVERED_AT_THIS_STATE": 6750})
            self.assertEqual(item["first_uncovered_tick_by_scenario"], {})
            self.assertAlmostEqual(item["minimum_margin_m"]["min"],
                                   float(np.min(margins[mode])), places=12)
            self.assertAlmostEqual(item["minimum_margin_m"]["p95"],
                                   float(np.percentile(margins[mode], 95)), places=12)
            inputs = report["inputs"][mode]
            self.assertEqual(hashlib.sha256(Path(inputs["metrics_path"]).read_bytes()
                                            ).hexdigest(), inputs["metrics_sha256"])
            self.assertEqual(len(inputs["traces"]), 5)
            for trace in inputs["traces"]:
                self.assertEqual(hashlib.sha256(Path(trace["path"]).read_bytes()
                                                ).hexdigest(), trace["sha256"])


if __name__ == "__main__":
    unittest.main()
