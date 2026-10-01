"""Check the frozen 500 Hz ramp audit and its source/trace provenance."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unittest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MicrostepEnvelopeTests(unittest.TestCase):
    def test_selected_native_ramps_are_complete_and_hash_bound(self) -> None:
        folder = Path(__file__).parent / "output" / "v6_2_b2" / "microstep_envelope"
        files = {
            "summary": folder / "microstep_envelope_summary.json",
            "states": folder / "microstep_envelope_states.jsonl",
            "failures": folder / "microstep_envelope_failures.jsonl",
            "document": folder / "MICROSTEP_ENVELOPE.md",
        }
        report = json.loads(files["summary"].read_text(encoding="utf-8"))
        manifest = json.loads((folder / "microstep_envelope_manifest.json").read_text(
            encoding="utf-8"))
        for name, path in files.items():
            self.assertEqual(manifest[f"{name}_sha256"], _sha(path))
        self.assertEqual(report["state_records_sha256"], _sha(files["states"]))
        self.assertEqual(report["failure_records_sha256"], _sha(files["failures"]))
        self.assertEqual(files["failures"].stat().st_size, 0)
        self.assertEqual(report["scope"],
                         "two_predeclared_old_A1_ten_step_ramps_per_mode_and_scene")
        self.assertEqual(report["numerical_certification"], "NOT_FORMALLY_CERTIFIED")
        self.assertFalse(report["continuous_time_certified"])
        self.assertFalse(report["new_interval_mode_executed"])
        self.assertEqual(report["early_tick"], 50)
        self.assertEqual(report["servo_steps_per_task"], 10)
        self.assertEqual(report["source_hash_newline_policy"], "LF_NORMALIZED")
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)

        root = folder.parent
        self.assertEqual(report["input_refined_start_sha256"], _sha(
            root / "refined_start" / "refined_start_report.json"))
        self.assertEqual(report["input_full_sweep_sha256"], _sha(
            root / "full_state_envelope" / "envelope_sweep_summary.json"))
        self.assertEqual(report["input_full_sweep_states_sha256"], _sha(
            root / "full_state_envelope" / "envelope_sweep_states.jsonl"))
        refined = json.loads((root / "refined_start" / "refined_start_report.json")
                             .read_text(encoding="utf-8"))
        first = {}
        for item in refined["records"]:
            if item["feasibility"]["status"] == "START_CLEARANCE_VIOLATION":
                key = item["mode"], item["scenario_id"]
                first[key] = min(first.get(key, item["tick"]), item["tick"])
        self.assertEqual(len(first), 10)
        for inputs in report["inputs"].values():
            self.assertEqual(_sha(Path(inputs["metrics_path"])), inputs["metrics_sha256"])
            self.assertEqual(len(inputs["traces"]), 5)
            for trace in inputs["traces"]:
                self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])
        self.assertEqual(len(report["native_replay_checks"]), 10)
        self.assertTrue(all(item["max_state_error"] <= 1e-8
                            for item in report["native_replay_checks"]))

        records = [json.loads(line) for line in files["states"].read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 220)
        grouped = defaultdict(list)
        for item in records:
            key = item["mode"], item["scenario_id"], item["window"]
            grouped[key].append(item)
            self.assertEqual(item["status"], "COVERED_AT_THIS_STATE")
            self.assertGreater(item["minimum_margin_m"], 0)
            self.assertEqual(item["physics_step"], 10 * item["task_tick"]
                             + item["servo_substep"])
        self.assertEqual(len(grouped), 20)
        self.assertEqual(len(report["windows"]), 20)
        for window in report["windows"]:
            key = window["mode"], window["scenario_id"], window["window"]
            values = sorted(grouped[key], key=lambda item: item["servo_substep"])
            self.assertEqual([item["servo_substep"] for item in values], list(range(11)))
            expected_tick = (50 if window["window"] == "early_probe" else
                             first[key[:2]])
            self.assertEqual(window["task_tick"], expected_tick)
            self.assertTrue(all(item["task_tick"] == expected_tick for item in values))
            self.assertAlmostEqual(window["start_margin_m"],
                                   values[0]["minimum_margin_m"])
            self.assertAlmostEqual(window["end_margin_m"],
                                   values[-1]["minimum_margin_m"])
            self.assertAlmostEqual(window["minimum_margin_m"],
                                   min(item["minimum_margin_m"] for item in values))
            self.assertEqual(window["covered_microstates"], 11)
            self.assertEqual(window["interior_below_both_endpoints"], any(
                item["minimum_margin_m"] < min(window["start_margin_m"],
                                                 window["end_margin_m"])
                for item in values[1:-1]))
        for mode in ("baseline", "enabled"):
            item = report["modes"][mode]
            self.assertEqual(item["window_count"], 10)
            self.assertEqual(item["microstate_count"], 110)
            self.assertEqual(item["covered_microstate_count"], 110)
            self.assertGreater(item["minimum_margin_m"]["min"], 0)


if __name__ == "__main__":
    unittest.main()
