"""Complete simulation research acceptance, independent of wall deployment.

No scheduling priority changes, real-time sleeps or wall executor are used.
The simulation still uses a 20 ms planning period and ten 2 ms torque steps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from v6_lite.run_evidence import _write_json
from v6_lite.run_test_profiles import current_source_snapshot, source_provenance
from v6_lite.runtime_timing import latency_summary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "v6_lite/output/runs/c1_formal_three_by_five/round_01"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualify_research_suite(payload):
    """Reject shortened, relabelled or functionally failed simulation suites."""
    cfg = payload.get("run_config", {})
    scenes = payload.get("scenarios", [])
    checks = {
        "explicit_research_simulation": cfg.get("dispatch_clock_policy") == "research_simulation",
        "bounded_interval_pcc": cfg.get("pcc_mode") == "bounded_interval_pcc",
        "fixed_simulation_periods": cfg.get("physics_period_s") == .002 and cfg.get("task_period_s") == .020,
        "full_27_second_horizon": cfg.get("duration_s") == 27.0,
        "original_seed": cfg.get("seed") == 20260801,
        "original_five_scene_set": sorted(s.get("scenario", {}).get("scenario_id", "") for s in scenes)
            == [f"v6_lite_scenario_{i:02d}" for i in range(5)],
        "reported_functional_acceptance": payload.get("passed") is True,
        "all_scene_functional_acceptance": len(scenes) == 5 and all(s.get("passed") is True for s in scenes),
        "complete_fixed_rate_execution": len(scenes) == 5 and all(
            s.get("metrics", {}).get("rates_and_latency", {}).get("physics_steps") == 13500
            and s["metrics"]["rates_and_latency"].get("task_ticks") == 1350
            and s["metrics"]["rates_and_latency"].get("torque_update_count") == 13500
            and s["metrics"]["rates_and_latency"].get("physics_hz") == 500.0
            and s["metrics"]["rates_and_latency"].get("task_hz") == 50.0 for s in scenes),
        "all_scene_simulation_scopes": len(scenes) == 5 and all(
            s.get("execution_contract", {}).get("dispatch_clock_policy") == "research_simulation"
            and s["execution_contract"].get("wall_deadline_enforced") is False for s in scenes),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "failures": [key for key, value in checks.items() if not value]}


def audit_research_timing(payload, run_dir):
    """Recompute startup-inclusive performance and verify raw evidence clocks."""
    records = []
    for scene in payload["scenarios"]:
        scene_id = scene["scenario"]["scenario_id"]
        trace_path = Path(scene["trace"]["path"])
        timeline_path = run_dir / "timing" / (scene_id + ".jsonl")
        timelines = [json.loads(line) for line in timeline_path.read_text(encoding="utf-8").splitlines()]
        with np.load(trace_path, allow_pickle=False) as trace:
            algorithm = trace["task_full_latency"].copy()
            torque = trace["torque_latency"].copy()
        dispatch = np.asarray([row["dispatch_latency_s"] for row in timelines], dtype=float)
        arrays = {"algorithm": algorithm, "dispatch": dispatch, "torque": torque}
        errors = []
        if len(algorithm) != 1350 or len(dispatch) != 1350 or len(torque) != 13500:
            errors.append("INCOMPLETE_TIMING_SAMPLES")
        if any(not np.all(np.isfinite(values)) or np.any(values < 0) for values in arrays.values()):
            errors.append("INVALID_TIMING_SAMPLES")
        for index, row in enumerate(timelines):
            start, end = row["state_acquisition_monotonic_ns"], row["actual_dispatch_monotonic_ns"]
            phases = row["phases"]
            scalars = [start, end, row["source_simulation_time_s"], row["dispatch_latency_s"],
                       row["thread_cpu_s"]]
            scalars.extend(value for phase in phases for value in (
                phase["start_ns"], phase["end_ns"], phase["wall_s"], phase["thread_cpu_s"]))
            if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                   for value in scalars):
                errors.append(f"NONFINITE_OR_NEGATIVE_TIMELINE_{index}")
                continue
            if (not phases or end < start or phases[0]["start_ns"] != start
                    or phases[-1]["end_ns"] != end
                    or abs(row["source_simulation_time_s"] - index * .020) > 1e-9
                    or abs((end-start)*1e-9 - row["dispatch_latency_s"]) > 1e-12
                    or any(phase["end_ns"] < phase["start_ns"] for phase in phases)
                    or any(abs((phase["end_ns"]-phase["start_ns"])*1e-9 - phase["wall_s"]) > 1e-12
                           for phase in phases)
                    or abs(sum(phase["thread_cpu_s"] for phase in phases) - row["thread_cpu_s"]) > 1e-12
                    or any(a["end_ns"] != b["start_ns"] for a, b in zip(phases, phases[1:]))):
                errors.append(f"INVALID_TIMELINE_{index}")
        summaries = {name: latency_summary(values, .002 if name == "torque" else .020)
                     for name, values in arrays.items()}
        reported = scene["metrics"]["rates_and_latency"]
        if (not np.isfinite(reported["initialization_latency_s"])
                or reported["initialization_latency_s"] < 0):
            errors.append("INVALID_INITIALIZATION_TIMING")
        for name in ("algorithm", "dispatch"):
            if summaries[name] != reported[name]:
                errors.append("REPORTED_TIMING_MISMATCH_" + name.upper())
        records.append({"scenario_id": scene_id, "passed": not errors, "errors": errors,
                        "trace_sha256": sha(trace_path), "timeline_sha256": sha(timeline_path),
                        "initialization_latency_s": reported["initialization_latency_s"],
                        **summaries})
    return {"schema": "research_simulation_timing_audit_v1",
            "evidence_valid": len(records) == 5 and all(row["passed"] for row in records),
            "performance_target_met": all(row[name]["passed"] for row in records
                                           for name in ("algorithm", "dispatch", "torque")),
            "performance_is_functional_gate": False,
            "startup_samples_excluded": False,
            "wall_continuation_claim": False, "scenes": records}


def run(output_dir, reference_dir=DEFAULT_REFERENCE):
    from v6_lite.run_v6_lite import V6LiteRunConfig, run_suite
    from v6_lite.hierarchical_qp import HierarchicalQPConfig
    output_dir, reference_dir = Path(output_dir).resolve(), Path(reference_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = current_source_snapshot(ROOT)
    config = V6LiteRunConfig(pcc_mode="bounded_interval_pcc", dispatch_clock_policy="research_simulation")
    qp = HierarchicalQPConfig(enable_capsule_cbf=True)
    plan = {"schema": "v6_2_research_acceptance_plan_v1", "supplements_previous_c11_goal": True,
            "run_config": asdict(config), "qp_config": asdict(qp),
            "source": frozen, "scenario_order": list(range(5)), "reference_dir": str(reference_dir),
            "steps": ["current_and_historical_tests", "complete_five_scene_simulation",
                      "simulation_qualification", "independent_torque_delivery",
                      "independent_execution_contract", "independent_interval_recompute",
                      "exact_c1_nontiming_parity", "raw_timing_audit"],
            "research_gates": "functional checks, full horizon, source identity and all independent evidence",
            "performance_targets": {"planning_p95_ms": 20, "torque_p95_ms": 2, "research_blocking": False},
            "wall_deployment_status": "NOT_MET", "administrator_trial": "DEFERRED_BLOCKED_PLATFORM_CONDITION",
            "hardware_deployment": "NOT_ESTABLISHED", "hard_realtime_certified": False,
            "simulation_during_compute": "frozen by normal synchronous numerical simulation",
            "failure_policy": "preserve failures in exclusive new output directory"}
    _write_json(output_dir / "plan.json", plan)
    result = {"schema": "v6_2_research_acceptance_result_v1", "passed": False, "complete": False,
              "steps": [], "claim_scope": "nominal seeded simulation; does not complete prior wall goal"}

    def step(name, action):
        print("[research] " + name, flush=True)
        before = current_source_snapshot(ROOT)
        if (not source_provenance(frozen, before)["source_unchanged"]
                or before["tracked_worktree_dirty"]):
            raise RuntimeError("declared source changed before " + name)
        try:
            payload = action()
            record = {"name": name, "status": "COMPLETED", "result": payload}
        except Exception as error:
            record = {"name": name, "status": "FAILED", "error_type": type(error).__name__,
                      "message": str(error), "traceback": traceback.format_exc()}
            payload = None
        after = current_source_snapshot(ROOT)
        provenance = source_provenance(frozen, after)
        record["source_unchanged"] = provenance["source_unchanged"]
        result["steps"].append(record)
        _write_json(output_dir / "report.json", result)
        if not provenance["source_unchanged"] or after["tracked_worktree_dirty"]:
            raise RuntimeError("declared source changed during " + name)
        return payload

    try:
        if frozen["capture_errors"] or frozen["tracked_worktree_dirty"]:
            raise RuntimeError("freeze tracked source before research acceptance")
        from v6_lite.run_test_profiles import run as test_profiles
        tests = step("current_and_historical_tests", lambda: test_profiles("all", output_dir / "test_profiles"))
        if tests is not True:
            raise RuntimeError("research regression prerequisite failed")
        suite_dir = output_dir / "simulation"
        payload = step("complete_five_scene_simulation", lambda: run_suite(
            config, qp, suite_dir, continue_failed_scenarios=True))
        if payload is None:
            raise RuntimeError("complete five-scene simulation unavailable")
        qualification = step("simulation_qualification", lambda: qualify_research_suite(payload))
        from v6_lite.validate_v6_lite import validate_delivery, validate_execution_contract
        native = step("independent_torque_delivery", lambda: validate_delivery(
            ROOT / "v6_lite", suite_dir, acceptance_profile="research_simulation"))
        contract = step("independent_execution_contract", lambda: validate_execution_contract(suite_dir))
        from v6_lite.audit_b2_online_interval_recompute import run as interval_recompute
        interval = step("independent_interval_recompute", lambda: interval_recompute(
            suite_dir, output_dir / "interval_recompute"))
        from v6_lite.audit_c1_trace_parity import run as parity
        same = step("exact_c1_nontiming_parity", lambda: parity(reference_dir, suite_dir, output_dir / "parity"))
        timing = step("raw_timing_audit", lambda: audit_research_timing(payload, suite_dir))
        result.update({"complete": True,
                       "algorithm_simulation": {"qualification": qualification, "delivery": native,
                           "execution": contract, "interval": interval, "c1_exact_parity": same},
                       "computational_performance": timing,
                       "wall_continuation": {"status": "NOT_MET", "retested": False,
                           "previous_goal_completed": False},
                       "hardware_deployment": {"status": "NOT_ESTABLISHED"},
                       "hard_realtime_certified": False})
        result["passed"] = (all(item and item.get("passed") for item in (qualification, native, contract, interval))
                            and bool(same and same.get("all_passed"))
                            and bool(timing and timing.get("evidence_valid")))
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    final_source = current_source_snapshot(ROOT)
    result["source_provenance"] = source_provenance(frozen, final_source)
    result["passed"] = bool(result["passed"] and result["source_provenance"]["source_unchanged"]
                            and not final_source["tracked_worktree_dirty"])
    _write_json(output_dir / "report.json", result)
    _write_json(output_dir / "manifest.json", {str(path.relative_to(output_dir)): sha(path)
                for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"})
    print(json.dumps({"passed": result["passed"], "complete": result["complete"],
                      "report": str(output_dir / "report.json"), "wall_continuation": "NOT_MET"}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir, args.reference_dir)["passed"] else 1)


if __name__ == "__main__":
    main()
