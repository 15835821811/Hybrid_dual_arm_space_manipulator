"""Check the full old-torque envelope replay and its cross-audit evidence."""

from __future__ import annotations

from collections import Counter
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


class FullTorqueEnvelopeTests(unittest.TestCase):
    def test_full_native_replay_is_complete_and_hash_bound(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2"
        folder = root / "full_torque_envelope"
        files = {
            "summary": folder / "full_torque_envelope_summary.json",
            "states": folder / "full_torque_envelope_states.jsonl",
            "failures": folder / "full_torque_envelope_failures.jsonl",
            "document": folder / "FULL_TORQUE_ENVELOPE.md",
        }
        report = json.loads(files["summary"].read_text(encoding="utf-8"))
        manifest = json.loads((folder / "full_torque_envelope_manifest.json")
                              .read_text(encoding="utf-8"))
        for name, path in files.items():
            self.assertEqual(manifest[f"{name}_sha256"], _sha(path))
        self.assertEqual(report["state_records_sha256"], _sha(files["states"]))
        self.assertEqual(report["failure_records_sha256"], _sha(files["failures"]))
        self.assertEqual(report["scope"],
                         "all_500hz_states_of_ten_historical_A1_native_torque_replays")
        self.assertFalse(report["continuous_time_certified"])
        self.assertFalse(report["new_interval_mode_executed"])
        self.assertEqual(report["numerical_certification"], "NOT_FORMALLY_CERTIFIED")
        self.assertEqual(report["servo_steps_per_task"], 10)
        self.assertEqual(report["cross_checks"], {
            "saved_task_states": 13500, "selected_microstates": 220,
        })
        self.assertEqual(report["input_full_sweep_summary_sha256"],
                         _sha(root / "full_state_envelope" / "envelope_sweep_summary.json"))
        self.assertEqual(report["input_full_sweep_states_sha256"],
                         _sha(root / "full_state_envelope" / "envelope_sweep_states.jsonl"))
        self.assertEqual(report["input_selected_summary_sha256"],
                         _sha(root / "microstep_envelope" / "microstep_envelope_summary.json"))
        self.assertEqual(report["input_selected_states_sha256"],
                         _sha(root / "microstep_envelope" / "microstep_envelope_states.jsonl"))
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        for inputs in report["inputs"].values():
            self.assertEqual(_sha(Path(inputs["metrics_path"])), inputs["metrics_sha256"])
            self.assertEqual(len(inputs["traces"]), 5)
            for trace in inputs["traces"]:
                self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])

        counts = Counter()
        previous = {}
        with files["states"].open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                key = item["mode"], item["scenario_id"]
                step = item["physics_step"]
                self.assertEqual(step, previous.get(key, -1) + 1)
                previous[key] = step
                counts[key] += 1
                self.assertEqual(item["status"], "COVERED_AT_THIS_STATE")
                self.assertGreater(item["minimum_margin_m"], 0)
        self.assertEqual(len(counts), 10)
        self.assertEqual(set(counts.values()), {13501})
        self.assertEqual(set(previous.values()), {13500})
        self.assertEqual(files["failures"].stat().st_size, 0)
        self.assertEqual(len(report["scenes"]), 10)
        for scene in report["scenes"]:
            self.assertEqual(scene["torque_steps"], 13500)
            self.assertEqual(scene["checked_states"], 13501)
            self.assertEqual(scene["status_counts"], {"COVERED_AT_THIS_STATE": 13501})
            self.assertLessEqual(scene["native_replay_max_state_error"], 1e-8)
        for mode in ("baseline", "enabled"):
            item = report["modes"][mode]
            self.assertEqual(item["scene_count"], 5)
            self.assertEqual(item["torque_steps"], 67500)
            self.assertEqual(item["checked_states"], 67505)
            self.assertEqual(item["covered_states"], 67505)
            self.assertGreater(item["minimum_margin_m"], 0)


if __name__ == "__main__":
    unittest.main()
