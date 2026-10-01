"""Predeclared three-repeat timing on the opt-in B.2 online controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_v6_lite import (
    V6LiteRunConfig, build_scenarios, default_v6_lite_robot_spec,
    run_scenario,
)


REPEATS = 3
DURATION_S = 6.0
SCENE_ID = "v6_lite_scenario_01"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    spec = default_v6_lite_robot_spec()
    cfg = V6LiteRunConfig(duration_s=DURATION_S,
                          pcc_mode="bounded_interval_pcc")
    qp_cfg = HierarchicalQPConfig(enable_capsule_cbf=True)
    scenario = next(item for item in build_scenarios(spec, cfg)
                    if item.scenario_id == SCENE_ID)
    records = []
    for repeat in range(REPEATS):
        root = output_dir / f"repeat_{repeat + 1:02d}"
        result = run_scenario(spec, cfg, qp_cfg, scenario, root / "traces")
        trace_path = Path(result["trace"]["path"])
        with np.load(trace_path, allow_pickle=False) as trace:
            full = trace["task_interval_full_control_latency_s"] * 1000
            qp = trace["task_interval_qp_only_latency_s"] * 1000
            torque = trace["torque_latency"] * 1000
            selected = trace["task_selected_command"].copy()
            saved_torque = trace["torque"].copy()
            success = trace["task_success"].copy()
            ticks = len(full)
        over = full > 20
        longest = 0
        current = 0
        for flag in over:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        records.append({
            "repeat": repeat + 1,
            "task_ticks": ticks,
            "torque_steps": len(saved_torque),
            "full_control_p95_ms": float(np.percentile(full, 95)),
            "full_control_p99_ms": float(np.percentile(full, 99)),
            "full_control_max_ms": float(np.max(full)),
            "full_control_over_20ms_count": int(np.count_nonzero(over)),
            "full_control_longest_over_20ms_run": longest,
            "qp_only_p95_ms": float(np.percentile(qp, 95)),
            "torque_p95_ms": float(np.percentile(torque, 95)),
            "qp_failures": int(np.count_nonzero(~success)),
            "trace_sha256": _sha(trace_path),
            "selected_command_sha256": hashlib.sha256(
                selected.tobytes()).hexdigest(),
            "torque_values_sha256": hashlib.sha256(
                saved_torque.tobytes()).hexdigest(),
            "original_task_acceptance_on_6s_subset": result["passed"],
        })
        print(f"[b2-online-repeat] {repeat + 1}/{REPEATS}: "
              f"p95={records[-1]['full_control_p95_ms']:.3f} ms",
              flush=True)
    report = {
        "schema": "v6_2_b2_online_repeated_timing_v1",
        "predeclared_repeats": REPEATS,
        "predeclared_duration_s": DURATION_S,
        "predeclared_scenario_id": SCENE_ID,
        "run_config": asdict(cfg),
        "qp_config": asdict(qp_cfg),
        "all_traces_saved": len(records) == REPEATS,
        "all_full_control_p95_within_20ms": all(
            item["full_control_p95_ms"] <= 20 for item in records),
        "records": records,
    }
    (output_dir / "online_repeated_timing_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({"all_traces_saved": report["all_traces_saved"],
                      "full_control_p95_ms": [item["full_control_p95_ms"]
                                              for item in report["records"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
