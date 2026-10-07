"""Predeclare five default wall-deadline continuation probes; retain rejects."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_evidence import fail_run, finish_run, start_run
from v6_lite.run_v6_lite import (V6LiteRunConfig, UncertifiedExecutionError, build_scenarios,
                                  default_v6_lite_robot_spec, run_scenario)


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    config = V6LiteRunConfig(pcc_mode="bounded_interval_pcc", dispatch_clock_policy="wall_deadline")
    qp = HierarchicalQPConfig(enable_capsule_cbf=True)
    robot = default_v6_lite_robot_spec()
    scenarios = build_scenarios(robot, config)
    plan = {"schema": "c1_wall_continuation_protocol_v1", "run_config": asdict(config),
            "scenario_order": [s.scenario_id for s in scenarios],
            "policy": "all five probes retained; rejects are unavailable continuation, not safe recovery"}
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    report = {"schema": "c1_wall_continuation_result_v1", "probes": [], "complete": False}
    for scenario in scenarios:
        folder = output_dir / scenario.scenario_id
        metadata = start_run(folder, run_config=config, qp_config=qp, spec=robot, scenarios=(scenario,))
        try:
            payload = run_scenario(robot, config, qp, scenario, folder / "traces")
            finish_run(folder, metadata, passed=payload["passed"], summary=payload["checks"])
            record = {"scenario": scenario.scenario_id, "completed_horizon": True,
                      "original_acceptance_passed": payload["passed"]}
        except UncertifiedExecutionError as error:
            fail_run(folder, metadata, error, scenario_id=scenario.scenario_id)
            failure = json.loads((folder / "failures" / f"{scenario.scenario_id}_interval_failure.json").read_text(encoding="utf-8"))
            record = {"scenario": scenario.scenario_id, "completed_horizon": False,
                      "reject_reason": failure["failure_reason"], "simulation_stop_s": failure["time_s"],
                      "next_servo_step_executed": failure["next_servo_step_executed"],
                      "continuation_guaranteed": False}
        report["probes"].append(record)
        print(json.dumps(record), flush=True)
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["complete"] = True
    report["all_horizons_completed"] = all(p["completed_horizon"] for p in report["probes"])
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run(args.output_dir)
