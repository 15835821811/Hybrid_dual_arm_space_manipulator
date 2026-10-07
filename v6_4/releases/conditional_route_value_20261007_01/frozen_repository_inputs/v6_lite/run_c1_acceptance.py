"""Predeclare three complete five-scene trials and retain every outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from v6_lite.audit_c1_trace_parity import run as parity
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_v6_lite import V6LiteRunConfig, run_suite
from v6_lite.runtime_timing import latency_summary


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir, reference_dir, *, repeats=3):
    if repeats < 1:
        raise ValueError("repeat count must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = V6LiteRunConfig(pcc_mode="bounded_interval_pcc", dispatch_clock_policy="offline_replay")
    qp = HierarchicalQPConfig(enable_capsule_cbf=True)
    plan = {"schema": "c1_complete_scene_repeat_protocol_v1", "repeats": repeats,
            "run_config": asdict(config), "qp_config": asdict(qp),
            "run_order": list(range(1, repeats + 1)), "scenario_order": list(range(5)),
            "algorithm_p95_deadline_ms": 20, "dispatch_p95_deadline_ms": 20,
            "engineering_goals": {"p95_ms": 16, "p99_ms": 20},
            "load_condition": "single sequential simulation on uncontrolled local Windows desktop",
            "selection_policy": "all predeclared rounds and startup samples retained",
            "dispatch_policy": "offline_replay measures wall latency without claiming wall-valid execution",
            "functional_validation_policy": "independent native/contract/interval validation of first predeclared round; exact non-timing parity checked for every round",
            "reference_dir": str(reference_dir)}
    write(output_dir / "plan.json", plan)
    result = {"schema": "c1_complete_scene_repeat_result_v1", "trials": [], "complete": False,
              "independent_validation_complete": False}
    write(output_dir / "result.json", result)
    for repeat in plan["run_order"]:
        trial_dir = output_dir / f"round_{repeat:02d}"
        print(f"[C.1] complete five-scene round {repeat}/{repeats}", flush=True)
        trial = {"repeat": repeat, "run_dir": str(trial_dir)}
        try:
            payload = run_suite(config, qp, trial_dir)
            report = parity(reference_dir, trial_dir, output_dir / f"parity_{repeat:02d}")
            trial.update({"completed": True, "original_acceptance_passed": payload["passed"],
                "exact_trace_parity": report["all_passed"],
                "scenes": [{"scenario_id": item["scenario"]["scenario_id"],
                    "algorithm": item["metrics"]["rates_and_latency"]["algorithm"],
                    "dispatch": item["metrics"]["rates_and_latency"]["dispatch"],
                    "initialization_latency_s": item["metrics"]["rates_and_latency"]["initialization_latency_s"],
                    "failed_checks": [k for k, v in item["checks"].items() if not v]}
                    for item in payload["scenarios"]]})
            trial["algorithm_timing_passed"] = all(r["algorithm"]["passed"] for r in trial["scenes"])
            trial["dispatch_timing_passed"] = all(r["dispatch"]["passed"] for r in trial["scenes"])
        except Exception as error:
            trial.update({"completed": False, "error_type": type(error).__name__, "message": str(error),
                          "traceback": traceback.format_exc(), "algorithm_timing_passed": False,
                          "dispatch_timing_passed": False, "exact_trace_parity": False})
        result["trials"].append(trial)
        write(output_dir / "result.json", result)
    result["complete"] = True
    for key in ("algorithm_timing_passed", "dispatch_timing_passed", "exact_trace_parity"):
        result["all_" + key] = all(r.get(key, False) for r in result["trials"])
    write(output_dir / "result.json", result)
    first = output_dir / "round_01"
    if (first / "v6_lite_metrics.json").exists():
        print("[C.1] independent native torque replay and eleven contract checks", flush=True)
        from v6_lite.validate_v6_lite import validate_delivery, validate_execution_contract, CONTRACT_VERSION
        native = validate_delivery(Path("v6_lite"), first, CONTRACT_VERSION)
        contract = validate_execution_contract(first)
        print("[C.1] independent interval row reconstruction", flush=True)
        from v6_lite.audit_b2_online_interval_recompute import run as recompute
        interval = recompute(first, output_dir / "independent_interval_recompute")
        result["first_round_validation"] = {
            "native": {k: native[k] for k in ("passed", "passed_count", "total_count", "failures")},
            "execution": {k: contract[k] for k in ("passed", "passed_count", "total_count", "failures")},
            "interval": interval,
        }
        result["independent_validation_complete"] = True
        write(output_dir / "result.json", result)
    inputs = [path for path in output_dir.rglob("*") if path.is_file()]
    manifest = {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    write(output_dir / "manifest.json", manifest)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    result = run(args.output_dir, args.reference_dir, repeats=args.repeats)
    print(json.dumps({k: v for k, v in result.items() if k.startswith("all_")}, indent=2))
