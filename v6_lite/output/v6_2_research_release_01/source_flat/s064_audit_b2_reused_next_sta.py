"""Rerun the private controller with cached model-constant next-start rows."""

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
from v6_lite.b2_reused_next_start import ReusedNextStart


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: np.ndarray) -> dict:
    return {"p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(np.max(values)),
            "over_20ms_count": int(np.count_nonzero(values > 20.0))}


def run(output_dir: Path, baseline_dir: Path, a1_root: Path,
        scenario_id: str, ticks: int) -> dict:
    builder = ReusedNextStart()
    branch_times = []
    next_times = []
    old_branch = discrete_loop._branch
    old_next = strict_loop._next_start

    def timed_branch(*args, **kwargs):
        started = time.perf_counter()
        result = old_branch(*args, **kwargs)
        branch_times.append((time.perf_counter() - started) * 1000.0)
        return result

    def timed_next(*args, **kwargs):
        started = time.perf_counter()
        result = builder(*args, **kwargs)
        next_times.append((time.perf_counter() - started) * 1000.0)
        return result

    try:
        discrete_loop._branch = timed_branch
        strict_loop._next_start = timed_next
        private = domain_run(output_dir, a1_root, scenario_id, ticks)
    finally:
        discrete_loop._branch = old_branch
        strict_loop._next_start = old_next
    records_path = output_dir / "private_rollout_records.jsonl"
    trace_path = output_dir / "private_rollout_trace.npz"
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    baseline_records = [json.loads(line) for line in (
        baseline_dir / "private_rollout_records.jsonl").read_text(
            encoding="utf-8").splitlines()]
    with np.load(trace_path, allow_pickle=False) as actual, np.load(
            baseline_dir / "private_rollout_trace.npz", allow_pickle=False) as base:
        trajectory_exact = (np.array_equal(actual["torque"], base["torque"])
                            and np.array_equal(actual["qpos_states"],
                                               base["qpos_states"]))
    parity = all(row.get("realized_next_start")
                 == old.get("realized_next_start")
                 for row, old in zip(records, baseline_records))
    if (private["stop_reason"] != "HORIZON_COMPLETE"
            or private["executed_ticks"] != ticks
            or builder.builder_constructions != 1
            or len(branch_times) != ticks or len(next_times) != ticks
            or len(records) != ticks or len(baseline_records) != ticks
            or not trajectory_exact or not parity):
        raise ValueError("reused next-start changed a row or trajectory")
    preflight = np.asarray([row["preflight_plus_qp_ms"] for row in records])
    branch = np.asarray(branch_times)
    next_start = np.asarray(next_times)
    combined = preflight + branch + next_start
    timing_rows = [{"tick": i, "preflight_plus_qp_ms": float(preflight[i]),
                    "ten_step_branch_ms": float(branch[i]),
                    "reused_next_start_ms": float(next_start[i]),
                    "necessary_component_sum_ms": float(combined[i])}
                   for i in range(ticks)]
    rows_path = output_dir / "reused_next_start_timing_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in timing_rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    report = {
        "schema": "v6_2_b2_reused_next_start_private_timing_v1",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "builder_constructions": builder.builder_constructions,
        "trajectory_exact_to_baseline": True,
        "next_start_records_exact_to_baseline": True,
        "preflight_plus_qp": _stats(preflight),
        "ten_step_branch": _stats(branch),
        "reused_next_start": _stats(next_start),
        "necessary_component_sum": _stats(combined),
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in ("b2_reused_next_start.py",
                                       "audit_b2_reused_next_start_timing.py")},
        "inputs_sha256": {
            "new_private_summary": _sha(output_dir / "private_rollout_summary.json"),
            "new_private_trace": _sha(trace_path),
            "baseline_private_summary": _sha(
                baseline_dir / "private_rollout_summary.json"),
            "baseline_private_records": _sha(
                baseline_dir / "private_rollout_records.jsonl"),
            "baseline_private_trace": _sha(
                baseline_dir / "private_rollout_trace.npz"),
        },
        "rows_sha256": _sha(rows_path),
    }
    report_path = output_dir / "reused_next_start_timing_summary.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    lines = ["# B.2 下一起点模型常量复用计时", "",
             "复用独立行构造器中的模型常量及 61 个实际链胶囊；每周期仍重算"
             "当前距离、Jacobian、反作用映射和全部约束行。新旧轨迹与下一起点"
             "记录逐值相同。", "",
             "| 阶段 | p95 ms | 最大 ms | 超 20 ms |",
             "| --- | ---: | ---: | ---: |"]
    for label, key in (("预检＋QP", "preflight_plus_qp"),
                       ("十步候选分支", "ten_step_branch"),
                       ("复用后下一起点", "reused_next_start"),
                       ("三项之和", "necessary_component_sum")):
        timing = report[key]
        lines.append(f"| {label} | {timing['p95_ms']:.3f} | "
                     f"{timing['max_ms']:.3f} | {timing['over_20ms_count']} |")
    lines += ["", "三项之和仍不等于完整在线时延；当前阶段不接入生产模式。", ""]
    (output_dir / "REUSED_NEXT_START_TIMING.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    result = run(args.output_dir, args.baseline_dir, args.a1_root,
                 args.scenario_id, args.ticks)
    print(json.dumps({key: result[key] for key in (
        "builder_constructions", "trajectory_exact_to_baseline",
        "next_start_records_exact_to_baseline", "reused_next_start",
        "necessary_component_sum")}, indent=2))


if __name__ == "__main__":
    main()
