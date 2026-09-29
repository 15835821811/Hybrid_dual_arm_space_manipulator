"""Time the private domain-QP decision components on a complete scenario.

The estimate includes current preflight+QP, the ten-step MuJoCo branch and
next-start reconstruction. It excludes the diagnostic independent row parity
run between those calls and does not claim an online hardware deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from v6_lite import audit_b2_discrete_private_rollout as discrete_loop
from v6_lite import audit_b2_strict_domain_private_rollout as strict_loop
from v6_lite.audit_b2_work_domain_qp_private_rollout import run as domain_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: np.ndarray) -> dict:
    return {
        "count": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "over_20ms_count": int(np.count_nonzero(values > 20.0)),
    }


def run(output_dir: Path, baseline_dir: Path, a1_root: Path,
        scenario_id: str, max_ticks: int) -> dict:
    branch_times = []
    next_start_times = []
    original_branch = discrete_loop._branch
    original_next_start = strict_loop._next_start

    def timed_branch(*args, **kwargs):
        started = time.perf_counter()
        result = original_branch(*args, **kwargs)
        branch_times.append((time.perf_counter() - started) * 1000.0)
        return result

    def timed_next_start(*args, **kwargs):
        started = time.perf_counter()
        result = original_next_start(*args, **kwargs)
        next_start_times.append((time.perf_counter() - started) * 1000.0)
        return result

    try:
        discrete_loop._branch = timed_branch
        strict_loop._next_start = timed_next_start
        private = domain_run(output_dir, a1_root, scenario_id, max_ticks)
    finally:
        discrete_loop._branch = original_branch
        strict_loop._next_start = original_next_start
    records_path = output_dir / "private_rollout_records.jsonl"
    trace_path = output_dir / "private_rollout_trace.npz"
    summary_path = output_dir / "private_rollout_summary.json"
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    if (private["stop_reason"] != "HORIZON_COMPLETE"
            or private["executed_ticks"] != max_ticks
            or len(branch_times) != max_ticks
            or len(next_start_times) != max_ticks
            or len(records) != max_ticks):
        raise ValueError("complete private decision timing did not cover every tick")
    baseline_trace_path = baseline_dir / "private_rollout_trace.npz"
    baseline_summary = json.loads((baseline_dir /
        "private_rollout_summary.json").read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as trial, np.load(
            baseline_trace_path, allow_pickle=False) as baseline:
        trace_exact = (np.array_equal(trial["torque"], baseline["torque"])
                       and np.array_equal(trial["qpos_states"],
                                          baseline["qpos_states"]))
    if (not trace_exact or baseline_summary["trace_sha256"]
            != _sha(baseline_trace_path)):
        raise ValueError("timed trial changed the private trajectory")
    preflight_qp = np.asarray([item["preflight_plus_qp_ms"]
                               for item in records], dtype=np.float64)
    branch = np.asarray(branch_times, dtype=np.float64)
    next_start = np.asarray(next_start_times, dtype=np.float64)
    diagnostic_parity = np.asarray([item["independent_row_recompute_ms"]
                                    for item in records], dtype=np.float64)
    component_sum = preflight_qp + branch + next_start
    rows = [{
        "tick": tick,
        "preflight_plus_qp_ms": float(preflight_qp[tick]),
        "ten_step_branch_ms": float(branch[tick]),
        "next_start_ms": float(next_start[tick]),
        "necessary_decision_components_ms": float(component_sum[tick]),
        "excluded_diagnostic_row_parity_ms": float(diagnostic_parity[tick]),
    } for tick in range(max_ticks)]
    rows_path = output_dir / "full_decision_timing_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_private_domain_full_decision_components_v1",
        "scenario_id": scenario_id,
        "task_ticks": max_ticks,
        "preflight_plus_qp": _stats(preflight_qp),
        "ten_step_mujoco_branch": _stats(branch),
        "next_start_reconstruction": _stats(next_start),
        "necessary_decision_component_sum": _stats(component_sum),
        "excluded_diagnostic_row_parity": _stats(diagnostic_parity),
        "component_sum_is_full_online_latency": False,
        "component_sum_is_lower_bound_for_current_private_architecture": True,
        "trajectory_identical_to_prior_private_run": True,
        "online_20ms_p95_admitted": False,
        "source_sha256": _source_sha(Path(__file__)),
        "inputs_sha256": {
            "timed_private_summary": _sha(summary_path),
            "timed_private_records": _sha(records_path),
            "timed_private_trace": _sha(trace_path),
            "baseline_private_summary": _sha(
                baseline_dir / "private_rollout_summary.json"),
            "baseline_private_trace": _sha(baseline_trace_path),
        },
        "rows_sha256": _sha(rows_path),
    }
    report_path = output_dir / "full_decision_timing_summary.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    lines = [
        "# B.2 私有新模式的必要决策步骤计时", "",
        f"`{scenario_id}` 的 {max_ticks} 个规划周期逐项计时；"
        "保存的 67 路力矩与先前新模式完整轨迹逐值相同。", "",
        "| 阶段 | p95 ms | p99 ms | 最大 ms | 超 20 ms 周期 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for title, key in (
            ("当前预检＋QP", "preflight_plus_qp"),
            ("十步 MuJoCo 候选分支", "ten_step_mujoco_branch"),
            ("下一起点重算", "next_start_reconstruction"),
            ("三项之和", "necessary_decision_component_sum")):
        timing = report[key]
        lines.append(f"| {title} | {timing['p95_ms']:.3f} | "
                     f"{timing['p99_ms']:.3f} | {timing['max_ms']:.3f} | "
                     f"{timing['over_20ms_count']} |")
    lines += [
        "", "三项之和排除了独立约束行诊断重算，但未计 Python 循环中其他开销、"
        "真实调度或硬件 I/O；因此它是当前私有结构必要步骤的计时下界，"
        "不是完整在线时延。单周期超时是性能阻断证据，p95 仍不能当作"
        "逐周期截止期限证明。", "",
    ]
    (output_dir / "FULL_DECISION_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--max-ticks", type=int, default=1350)
    args = parser.parse_args()
    report = run(args.output_dir, args.baseline_dir, args.a1_root,
                 args.scenario_id, args.max_ticks)
    print(json.dumps({
        "scenario_id": report["scenario_id"],
        "preflight_plus_qp": report["preflight_plus_qp"],
        "ten_step_mujoco_branch": report["ten_step_mujoco_branch"],
        "next_start_reconstruction": report["next_start_reconstruction"],
        "necessary_decision_component_sum": report[
            "necessary_decision_component_sum"],
    }, indent=2))


if __name__ == "__main__":
    main()
