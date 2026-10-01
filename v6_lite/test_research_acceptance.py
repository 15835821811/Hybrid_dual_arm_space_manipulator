"""A performance miss must not mask genuine research or evidence failures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from v6_lite.run_research_acceptance import qualify_research_suite, audit_research_timing
from v6_lite.runtime_timing import latency_summary
from v6_lite.validate_v6_lite import delivery_acceptance_checks


def full_suite():
    return {"passed": True, "run_config": {"dispatch_clock_policy": "research_simulation",
                "pcc_mode": "bounded_interval_pcc", "physics_period_s": .002,
                "task_period_s": .020, "duration_s": 27.0, "seed": 20260801},
            "scenarios": [{"passed": True, "scenario": {"scenario_id": f"v6_lite_scenario_{i:02d}"},
                "performance_checks": {"task_controller_runs_within_50hz_p95": False},
                "execution_contract": {"dispatch_clock_policy": "research_simulation", "wall_deadline_enforced": False},
                "metrics": {"rates_and_latency": {"physics_steps": 13500, "task_ticks": 1350,
                    "torque_update_count": 13500, "physics_hz": 500.0, "task_hz": 50.0}}}
                for i in range(5)]}


class ResearchAcceptanceTests(unittest.TestCase):
    def test_timing_only_failure_is_observed_without_research_rejection(self):
        gates, perf = delivery_acceptance_checks({"collision": True, "measured_rate_deadlines": False},
            profile="research_simulation", dispatch_clock_policy="research_simulation")
        self.assertTrue(all(gates.values()))
        self.assertFalse(perf["measured_rate_deadlines"])
        self.assertTrue(qualify_research_suite(full_suite())["passed"])

    def test_functional_failure_remains_failure_and_legacy_gate_is_unchanged(self):
        observed = {"collision": False, "measured_rate_deadlines": False}
        gates, _ = delivery_acceptance_checks(observed, profile="research_simulation",
                                              dispatch_clock_policy="research_simulation")
        self.assertFalse(all(gates.values()))
        gates, perf = delivery_acceptance_checks(observed, profile="legacy_combined",
                                                 dispatch_clock_policy="offline_replay")
        self.assertEqual(gates, observed)
        self.assertFalse(perf)

    def test_research_cannot_relabel_wall_or_historical_evidence(self):
        for policy in ("wall_deadline", "offline_replay", None):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                delivery_acceptance_checks({"measured_rate_deadlines": True},
                    profile="research_simulation", dispatch_clock_policy=policy)

    def test_incomplete_wrong_rate_duplicate_or_failed_scenes_are_rejected(self):
        variations = []
        suite = full_suite(); suite["scenarios"][0]["metrics"]["rates_and_latency"]["physics_steps"] -= 1; variations.append(suite)
        suite = full_suite(); suite["scenarios"][1]["scenario"]["scenario_id"] = "v6_lite_scenario_00"; variations.append(suite)
        suite = full_suite(); suite["run_config"]["task_period_s"] = .040; variations.append(suite)
        suite = full_suite(); suite["scenarios"][2]["passed"] = False; variations.append(suite)
        suite = full_suite(); suite["scenarios"][0]["execution_contract"]["wall_deadline_enforced"] = True; variations.append(suite)
        for suite in variations:
            with self.subTest(suite=suite):
                self.assertFalse(qualify_research_suite(copy.deepcopy(suite))["passed"])

    def test_existing_wall_single_entry_keeps_explicit_wall_policy(self):
        from v6_lite import run_c11_wall_single
        captured = []
        def inspect(_spec, config, _qp, _scene, _trace):
            captured.append(config.dispatch_clock_policy)
            return {"passed": False, "checks": {"fixture": False}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_c11_wall_single, "default_v6_lite_robot_spec", return_value=None), \
                patch.object(run_c11_wall_single, "build_scenarios", return_value=[SimpleNamespace(scenario_id="fixture")]), \
                patch.object(run_c11_wall_single, "start_run", return_value={}), \
                patch.object(run_c11_wall_single, "finish_run"), \
                patch.object(run_c11_wall_single, "run_scenario", side_effect=inspect):
            self.assertFalse(run_c11_wall_single.run(Path(directory)))
        self.assertEqual(captured, ["wall_deadline"])

    def test_nonfinite_or_falsified_raw_timeline_cannot_pass_evidence_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "timing").mkdir()
            payload = full_suite()
            timelines = []
            for i in range(1350):
                start, end = 1000000000+i*20000000, 1010000000+i*20000000
                timelines.append({"state_acquisition_monotonic_ns": start,
                    "actual_dispatch_monotonic_ns": end, "source_simulation_time_s": i*.020,
                    "dispatch_latency_s": .010, "thread_cpu_s": .008,
                    "phases": [{"start_ns": start, "end_ns": end, "wall_s": .010, "thread_cpu_s": .008}]})
            for scene in payload["scenarios"]:
                name = scene["scenario"]["scenario_id"]
                path = root / (name + ".npz")
                np.savez_compressed(path, task_full_latency=np.full(1350, .010),
                                    torque_latency=np.full(13500, .0001))
                scene["trace"] = {"path": str(path)}
                scene["metrics"]["rates_and_latency"].update({"initialization_latency_s": .1,
                    "algorithm": latency_summary(np.full(1350, .010)),
                    "dispatch": latency_summary(np.full(1350, .010))})
                (root/"timing"/(name+".jsonl")).write_text(
                    "\n".join(json.dumps(row) for row in timelines), encoding="utf-8")
            self.assertTrue(audit_research_timing(payload, root)["evidence_valid"])
            target = root / "timing/v6_lite_scenario_00.jsonl"
            for field in ("source", "clocks", "phase_wall", "phase_cpu", "wrong_phase_duration"):
                changed = copy.deepcopy(timelines)
                first = changed[0]
                if field == "source": first["source_simulation_time_s"] = float("nan")
                elif field == "clocks":
                    first["state_acquisition_monotonic_ns"] = first["actual_dispatch_monotonic_ns"] = float("inf")
                    first["phases"][0]["start_ns"] = first["phases"][0]["end_ns"] = float("inf")
                elif field == "phase_wall": first["phases"][0]["wall_s"] = float("nan")
                elif field == "phase_cpu": first["phases"][0]["thread_cpu_s"] = float("inf")
                else: first["phases"][0]["wall_s"] = .020
                target.write_text("\n".join(json.dumps(row) for row in changed), encoding="utf-8")
                with self.subTest(field=field):
                    self.assertFalse(audit_research_timing(payload, root)["evidence_valid"])
