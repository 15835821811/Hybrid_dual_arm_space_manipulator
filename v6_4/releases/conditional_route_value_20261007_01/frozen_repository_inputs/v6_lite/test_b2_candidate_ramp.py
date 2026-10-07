"""Verify the hash-bound one-ramp candidate branches and old-servo parity."""

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


class CandidateRampTests(unittest.TestCase):
    def test_private_branches_reproduce_old_servo_and_record_new_ramp(self) -> None:
        root = Path(__file__).parent / "output" / "v6_2_b2"
        folder = root / "candidate_ramp_early"
        files = {
            "summary": folder / "candidate_ramp_summary.json",
            "microstates": folder / "candidate_ramp_states.jsonl",
            "failures": folder / "candidate_ramp_failures.jsonl",
            "document": folder / "CANDIDATE_RAMP.md",
        }
        report = json.loads(files["summary"].read_text(encoding="utf-8"))
        manifest = json.loads((folder / "candidate_ramp_manifest.json")
                              .read_text(encoding="utf-8"))
        for name, path in files.items():
            self.assertEqual(manifest[f"{name}_sha256"], _sha(path))
        self.assertEqual(report["microstate_records_sha256"], _sha(files["microstates"]))
        self.assertEqual(report["failure_records_sha256"], _sha(files["failures"]))
        self.assertEqual(report["scope"],
                         "thirty_predeclared_early_old_A1_states_private_one_ramp_counterfactual")
        self.assertEqual(report["probe_ticks"], [50, 100, 150])
        self.assertEqual(report["servo_steps_per_task"], 10)
        self.assertFalse(report["new_interval_mode_executed"])
        self.assertFalse(report["continuous_time_certified"])
        self.assertTrue(report["private_counterfactual_servo_executed"])
        self.assertEqual(report["input_probe_sha256"], _sha(
            root / "qp_probe_early" / "weighted_qp_probe.json"))
        for name, digest in report["source_sha256"].items():
            source = Path(__file__).parent / name
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(
                b"\r\n", b"\n")).hexdigest(), digest)
        for inputs in report["inputs"].values():
            self.assertEqual(_sha(Path(inputs["metrics_path"])), inputs["metrics_sha256"])
            self.assertEqual(len(inputs["traces"]), 5)
            for trace in inputs["traces"]:
                self.assertEqual(_sha(Path(trace["path"])), trace["sha256"])

        records = report["records"]
        self.assertEqual(len(records), 30)
        by_key = {(x["mode"], x["scenario_id"], x["tick"]): x for x in records}
        self.assertEqual(len(by_key), 30)
        steps = defaultdict(list)
        with files["microstates"].open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                key = item["mode"], item["scenario_id"], item["tick"]
                self.assertIn(key, by_key)
                steps[key].append(item)
        self.assertEqual(len(steps), 30)
        for key, record in by_key.items():
            self.assertIn(key[2], [50, 100, 150])
            values = steps[key]
            self.assertEqual([x["servo_substep"] for x in values], list(range(11)))
            self.assertEqual(record["candidate_covered_microstate_count"], sum(
                x["status"] == "COVERED_AT_THIS_STATE" for x in values))
            self.assertAlmostEqual(record["candidate_minimum_envelope_margin_m"],
                                   min(x["minimum_margin_m"] for x in values))
            self.assertLessEqual(record["old_torque_max_abs_error"], 1e-8)
            self.assertLessEqual(record["old_next_qpos_max_abs_error"], 1e-8)
            self.assertLessEqual(record["native_replay_max_state_error"], 1e-8)
            self.assertGreater(record["command_l2_difference_rad_s"], 0)
            self.assertFalse(record["realized_next_start"]["budget_exhausted"]
                             and record["realized_next_start"]["point_evaluations"] > 255)
        for mode in ("baseline", "enabled"):
            own = [x for x in records if x["mode"] == mode]
            item = report["modes"][mode]
            self.assertEqual(len(own), item["ramp_count"])
            self.assertEqual(item["ramp_count"], 15)
            self.assertEqual(item["covered_ramp_count"], sum(
                x["candidate_covered_microstate_count"] == 11 for x in own))
            self.assertEqual(item["realized_next_start_satisfied_count"], sum(
                x["realized_next_start"]["frozen_rows_status"]
                == "START_ROWS_SATISFIED" for x in own))
            self.assertEqual(item["next_proxy_safe_count"], sum(
                x["realized_next_start"]["proxy_status"]
                == "PROXY_CLEARANCE_AT_LEAST_GATE" for x in own))
            self.assertEqual(item["whole_body_violation_ramp_count"], sum(
                x["candidate_whole_body_clearance_violation_count"] > 0
                for x in own))
            self.assertEqual(item["next_start_status_counts"], dict(Counter(
                x["realized_next_start"]["frozen_rows_status"] for x in own)))


if __name__ == "__main__":
    unittest.main()
