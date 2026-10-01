"""Independent evidence checks reject clock and stopped-state corruption."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v6_lite.audit_c11_native_partial import native_failure_audit, run as partial_audit
from v6_lite.audit_c11_wall_evidence import native_clock_audit, timeline_audit
from v6_lite.runtime_timing import latency_summary


class NativeClockEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.count = 3
        epoch, acquired = 100., 99.9
        scheduled = epoch+np.arange(self.count*10)*.002
        started = scheduled+.00001
        consumed = scheduled+.00002
        finished = scheduled+.0001
        timings = np.column_stack((scheduled, started, consumed, finished))
        source_times = finished[::10].copy()
        sources = np.concatenate(([acquired], source_times[:-1]))
        publications = sources+.015
        self.rows = [{
            "certificate": {"source_acquisition_time": float(sources[i]),
                "execution_start": float(scheduled[i*10]),
                "publish_deadline": float(scheduled[i*10]),
                "execution_end": float(scheduled[i*10]+.020),
                "execution_start_simulation_s": i*.020},
            "first_servo_application_monotonic_s": float(consumed[i*10]),
            "actual_publish_monotonic_s": float(publications[i]),
            "dispatch_latency_s": float(publications[i]-sources[i]),
            "source_simulation_time_s": max(i-1, 0)*.020+(.002 if i else 0),
        } for i in range(self.count)]
        self.native = {"epoch": epoch, "acquired": acquired,
            "source_times": source_times, "timings": timings}
        # Runtime outputs are inputs to this auditor, independently recomputed
        # by its own implementation. The fixture retains the startup >20 ms.
        self.rates = {
            "dispatch": latency_summary(publications-sources),
            "acquisition_to_first_application": latency_summary(consumed[::10]-sources),
            "publication_to_first_application": latency_summary(consumed[::10]-publications),
            "application_start_offset": latency_summary(consumed[::10]-scheduled[::10], .002),
            "servo_jitter": latency_summary(started-scheduled, .002),
            "full_servo_cycle": latency_summary(finished-started, .002),
        }

    def audit(self):
        return native_clock_audit(self.rows, self.native, self.rates)

    def test_native_clocks_match_statistics_and_keep_startup(self):
        result = self.audit()
        self.assertTrue(result["passed"], result["errors"])
        application = result["statistics"]["acquisition_to_first_application"]
        self.assertEqual(application["count"], 3)
        self.assertGreater(application["first_cycle_ms"], 100.)
        self.assertEqual(application["over_deadline_count"], 1)
        self.assertEqual(application["longest_consecutive_over_deadline"], 1)
        self.assertEqual(application["steady_after_first_cycle"]["count"], 2)
        # Consistency does not turn an over-budget fixture into timing acceptance.
        self.assertFalse(application["passed"])

    def test_retimestamped_source_is_rejected(self):
        self.rows[1]["certificate"]["source_acquisition_time"] += .001
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertIn("1:SOURCE_ACQUISITION_TIME_DIFFERS_FROM_NATIVE", result["errors"])

    def test_retimestamped_raw_source_and_certificate_are_still_rejected(self):
        self.native["source_times"][0] += .001
        self.rows[1]["certificate"]["source_acquisition_time"] += .001
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertIn("SOURCE_CAPTURE_TIME_DIFFERS_FROM_COMPLETED_NATIVE_STEP", result["errors"])

    def test_reported_application_must_equal_real_consumption(self):
        self.rows[1]["first_servo_application_monotonic_s"] += .0001
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertIn("1:FIRST_APPLICATION_TIME_DIFFERS_FROM_NATIVE", result["errors"])

    def test_underreported_p95_or_removed_startup_is_rejected(self):
        for mutate in (
            lambda rates: rates["acquisition_to_first_application"].update(p95_ms=19.),
            lambda rates: rates["acquisition_to_first_application"].update(count=2),
            lambda rates: rates["acquisition_to_first_application"].update(over_deadline_count=0),
        ):
            with self.subTest(mutate=mutate):
                rates = copy.deepcopy(self.rates)
                mutate(rates)
                result = native_clock_audit(self.rows, self.native, rates)
                self.assertFalse(result["passed"])
                self.assertTrue(any("REPORTED_CLOCK_STATISTIC_MISMATCH" in error
                    for error in result["errors"]))

    def test_missing_or_truncated_clock_capture_is_rejected(self):
        del self.native["source_times"]
        self.assertIn("MISSING_NATIVE_CLOCK_FIELDS", self.audit()["errors"])
        self.native["source_times"] = np.zeros(self.count-1)
        self.assertIn("INCOMPLETE_OR_INVALID_NATIVE_CLOCK_CAPTURE", self.audit()["errors"])

    def test_late_real_consumption_is_rejected_even_with_unchanged_report(self):
        self.native["timings"][5, 2] = self.native["timings"][5, 0]+.0021
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertIn("NATIVE_ALL_STEP_CLOCK_CONTRACT", result["errors"])

    def test_nonfinite_certificate_or_row_clock_is_rejected(self):
        for container, key in (
            ("certificate", "source_acquisition_time"),
            ("certificate", "publish_deadline"),
            ("certificate", "execution_start"),
            ("certificate", "execution_end"),
            ("certificate", "execution_start_simulation_s"),
            ("row", "source_simulation_time_s"),
            ("row", "dispatch_latency_s"),
            ("row", "actual_publish_monotonic_s"),
            ("row", "first_servo_application_monotonic_s"),
        ):
            with self.subTest(container=container, key=key):
                rows = copy.deepcopy(self.rows)
                target = rows[1]["certificate"] if container == "certificate" else rows[1]
                target[key] = float("nan")
                result = native_clock_audit(rows, self.native, self.rates)
                self.assertFalse(result["passed"])
                self.assertIn("1:NONFINITE_OR_MISSING_NATIVE_TIMELINE_CLOCK", result["errors"])


class TimelineFiniteClockTests(unittest.TestCase):
    def test_nonfinite_phase_or_certificate_clock_is_rejected(self):
        row = {
            "certificate": {"command_id": 0, "predecessor_command_id": -1,
                "source_acquisition_time": .9, "planning_release": .9,
                "solve_started": .91, "solve_finished": .92, "validation_finished": .93,
                "publish_deadline": 1., "execution_start": 1., "execution_end": 1.020,
                "execution_start_simulation_s": 0., "controller_config_hash": "config"},
            "actual_publish_monotonic_s": .94, "first_servo_application_monotonic_s": 1.0001,
            "source_simulation_time_s": 0., "dispatch_latency_s": .04,
            "phases": [{"start_ns": 900000000, "end_ns": 940000000,
                "wall_s": .04, "thread_cpu_s": None}],
            "publication_check": {"accepted": True}, "handoff_check": {"accepted": True},
            "servo_dispatch_checks": [{"accepted": True} for _ in range(10)],
        }
        metrics = {"scenarios": [{"scenario": {"scenario_id": "scene"},
            "execution_contract": {"controller_config_hash": "config"},
            "metrics": {"rates_and_latency": {}}}]}
        # Incomplete toy duration remains rejected independently; this checks
        # that NaN also yields an explicit clock-integrity rejection.
        for container, key in (
            ("certificate", "execution_start_simulation_s"),
            ("certificate", "execution_end"),
            ("certificate", "solve_finished"),
            ("row", "source_simulation_time_s"),
            ("row", "dispatch_latency_s"),
            ("phase", "start_ns"),
            ("phase", "end_ns"),
            ("phase", "wall_s"),
            ("phase", "thread_cpu_s"),
        ):
            with self.subTest(container=container, key=key), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(row)
                target = changed["certificate"] if container == "certificate" else (
                    changed["phases"][0] if container == "phase" else changed)
                target[key] = float("nan")
                root = Path(directory)
                (root/"timing").mkdir()
                (root/"timing/scene.jsonl").write_text(json.dumps(changed)+"\n", encoding="utf-8")
                result = timeline_audit(root, metrics)
                self.assertFalse(result["passed"])
                self.assertIn("0:NONFINITE_TIMELINE_CLOCK", result["records"][0]["errors"])


class StoppedNativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="c11_native_audit_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trace = self.root/"native.npz"
        self.failure_path = self.root/"failure.json"
        self.states = np.array([[0., 1., 2.], [.002, 1.1, 2.1]])
        self.final = self.states[-1].copy()
        self.failure = {
            "scenario_id": "scene", "failure_reason": "MISSED_SERVO_WINDOW",
            "physics_steps_executed": 1, "next_servo_step_executed": False,
            "rejected_servo_attempt": {"physics_step": 1,
                "scheduled": 100.002, "actual_start": 100.0041,
                "torque_consumed": False},
        }
        self.write()

    def write(self, *, include_final=True):
        values = {"applied_torques": np.zeros((1, 67)),
            "timings": np.array([[100., 100.00001, 100.00002, 100.0001]]),
            "integration_states": self.states}
        if include_final:
            values["final_actual_integration_state"] = self.final
        np.savez(self.trace, **values)
        self.failure["partial_trace"] = {"path": str(self.trace),
            "sha256": hashlib.sha256(self.trace.read_bytes()).hexdigest()}
        self.failure_path.write_text(json.dumps(self.failure), encoding="utf-8")

    def test_stopped_real_state_matches_last_capture_without_tolerance(self):
        result = native_failure_audit(self.failure_path)
        self.assertTrue(result["no_next_physics_step"])
        self.assertTrue(result["final_actual_matches_last_capture"])
        self.assertTrue(result["native_trace_hash_matches_failure"])

    def test_counter_agreement_cannot_hide_a_changed_live_state(self):
        self.final[-1] = np.nextafter(self.final[-1], np.inf)
        self.write()
        result = native_failure_audit(self.failure_path)
        self.assertFalse(result["no_next_physics_step"])
        self.assertFalse(result["final_actual_matches_last_capture"])
        self.assertTrue(result["state_capture_complete"])
        self.assertTrue(result["native_trace_hash_matches_failure"])

    def test_extra_unreported_physics_is_rejected(self):
        self.final[0] += .002
        self.write()
        result = native_failure_audit(self.failure_path)
        self.assertFalse(result["no_next_physics_step"])
        self.assertAlmostEqual(result["final_actual_physics_time_s"], .004)

    def test_missing_actual_state_or_wrong_trace_identity_is_rejected(self):
        self.write(include_final=False)
        self.assertFalse(native_failure_audit(self.failure_path)["no_next_physics_step"])
        self.write()
        self.failure["partial_trace"]["sha256"] = "0"*64
        self.failure_path.write_text(json.dumps(self.failure), encoding="utf-8")
        result = native_failure_audit(self.failure_path)
        self.assertFalse(result["no_next_physics_step"])
        self.assertFalse(result["native_trace_hash_matches_failure"])

    def test_not_attempted_record_has_no_invented_wake_offset(self):
        self.failure["rejected_servo_attempt"].update(
            physics_step=None, actual_start=None, scheduled=None, not_attempted=True)
        self.write()
        result = native_failure_audit(self.failure_path)
        self.assertTrue(result["no_next_physics_step"])
        self.assertFalse(result["rejected_attempt_timing_available"])
        self.assertIsNone(result["wake_offset_ms"])

    def test_poststep_rejection_preserves_the_consumed_physics_step(self):
        self.failure["failure_reason"] = "NATIVE_REALIZATION_MISMATCH"
        self.failure["rejected_servo_attempt"].update(physics_step=0, torque_consumed=True)
        self.write()
        result = native_failure_audit(self.failure_path)
        self.assertTrue(result["no_next_physics_step"])
        self.assertTrue(result["rejected_operation_consumed"])
        self.assertTrue(result["attempted_step_count_consistent"])
        self.failure["rejected_servo_attempt"]["physics_step"] = 1
        self.write()
        result = native_failure_audit(self.failure_path)
        self.assertFalse(result["no_next_physics_step"])
        self.assertFalse(result["attempted_step_count_consistent"])

    def test_consistent_partial_evidence_keeps_the_full_trial_failed(self):
        (self.root/"failures").mkdir()
        (self.root/"failures/scene_interval_failure.json").write_text(
            self.failure_path.read_text(encoding="utf-8"), encoding="utf-8")
        (self.root/"wall_trial_report.json").write_text(
            json.dumps({"completed_scenes": [{"scenario": {"scenario_id": "complete_scene"}}]}),
            encoding="utf-8")
        (self.root/"run_metadata.json").write_text(
            json.dumps({"run_config": {}, "source": {"git_commit": "fixed"}}), encoding="utf-8")
        output = self.root/"partial_audit"
        with patch("v6_lite.audit_c11_native_partial.timeline_audit", return_value={"passed": True}), \
                patch("v6_lite.audit_c11_native_partial.packet_replay", return_value={"passed": True}):
            self.assertTrue(partial_audit(self.root, output))
        report = json.loads((output/"report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["evidence_consistent"])
        self.assertFalse(report["complete_trial_passed"])
        self.assertTrue(report["whole_26_and_11_acceptance"].startswith("NOT_MET"))
        self.assertEqual(report["source_trial_commit"], "fixed")
        self.assertIn("auditor_source_commit", report)
        self.assertEqual(len(report["auditor_source_raw_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
