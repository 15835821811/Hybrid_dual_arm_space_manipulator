"""Time solver and non-solver work on the same private B.2 trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from v6_lite.audit_b2_prepared_torque_timing import run as prepared_run
from v6_lite.b2_work_domain_velocity_box import WorkDomainBoundedCholeskyQP
from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator


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


def run(output_dir: Path, reference_dir: Path, screened_dir: Path,
        fixed_dir: Path, baseline_dir: Path, a1_root: Path,
        scenario_id: str, ticks: int) -> dict:
    original_solve = WorkDomainBoundedCholeskyQP.solve
    stages = {
        "reaction_map_ms": (WorkDomainBoundedCholeskyQP,
                            "reaction_velocity_map"),
        "task_jacobians_ms": (WorkDomainBoundedCholeskyQP,
                              "_task_jacobians"),
        "clearance_assembly_ms": (WorkDomainBoundedCholeskyQP,
                                   "_build_all_clearance_constraints"),
        "mujoco_rows_ms": (WorkDomainBoundedCholeskyQP,
                           "_mujoco_clearance_constraint_set"),
        "capsule_rows_ms": (WorkDomainBoundedCholeskyQP,
                            "_capsule_clearance_kinematics"),
        "interval_evaluation_ms": (FixedIntervalCBFEvaluator,
                                   "evaluate_state"),
    }
    originals = {(owner, name): getattr(owner, name)
                 for owner, name in stages.values()}
    measurements = []
    active = [None]

    def timed_stage(metric, original):
        def wrapper(self, *args, **kwargs):
            if active[0] is None:
                return original(self, *args, **kwargs)
            started = time.perf_counter()
            value = original(self, *args, **kwargs)
            active[0][metric] += (time.perf_counter() - started) * 1000.0
            return value
        return wrapper

    def measured_solve(self, *args, **kwargs):
        measurement = {metric: 0.0 for metric in stages}
        active[0] = measurement
        started = time.perf_counter()
        try:
            result = original_solve(self, *args, **kwargs)
        finally:
            active[0] = None
        measurement.update({
            "qp_solve_wall_ms": (time.perf_counter() - started) * 1000.0,
            "solver_latency_ms": result.solver_latency_s * 1000.0,
            "qp_iterations": result.solver_iterations,
        })
        measurements.append(measurement)
        return result

    try:
        for metric, (owner, name) in stages.items():
            setattr(owner, name, timed_stage(metric, originals[(owner, name)]))
        WorkDomainBoundedCholeskyQP.solve = measured_solve
        private = prepared_run(output_dir, screened_dir, fixed_dir,
                               baseline_dir, a1_root, scenario_id, ticks)
    finally:
        WorkDomainBoundedCholeskyQP.solve = original_solve
        for owner, name in stages.values():
            setattr(owner, name, originals[(owner, name)])
    records_path = output_dir / "private_rollout_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    reference = [json.loads(line) for line in (
        reference_dir / "private_rollout_records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if (len(measurements) != ticks or len(records) != ticks
            or len(reference) != ticks
            or not private["all_non_timing_records_exact"]):
        raise ValueError("QP breakdown did not preserve full private trajectory")
    rows = []
    for tick, (measure, record, old) in enumerate(zip(
            measurements, records, reference)):
        if measure["qp_iterations"] != record["qp_iterations"] \
                or record["qp_iterations"] != old["qp_iterations"]:
            raise ValueError(f"QP iterations changed at {tick}")
        before = record["preflight_plus_qp_ms"] - measure["qp_solve_wall_ms"]
        other = measure["qp_solve_wall_ms"] - measure["solver_latency_ms"]
        if before < -1e-6 or other < -1e-6:
            raise ValueError(f"invalid QP timing decomposition at {tick}")
        rows.append({"tick": tick,
                     "preflight_before_solve_ms": before,
                     "qp_solve_wall_ms": measure["qp_solve_wall_ms"],
                     "solver_latency_ms": measure["solver_latency_ms"],
                     "other_in_solve_ms": other,
                     "preflight_plus_qp_ms": record["preflight_plus_qp_ms"],
                     "qp_iterations": measure["qp_iterations"],
                     **{metric: measure[metric] for metric in stages}})
    rows_path = output_dir / "qp_breakdown_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    metrics = {key: _stats(np.asarray([row[key] for row in rows]))
               for key in ("preflight_before_solve_ms", "qp_solve_wall_ms",
                           "solver_latency_ms", "other_in_solve_ms",
                           "preflight_plus_qp_ms", *stages)}
    iterations = np.asarray([row["qp_iterations"] for row in rows])
    solver = np.asarray([row["solver_latency_ms"] for row in rows])
    report = {
        "schema": "v6_2_b2_current_private_qp_breakdown_v2",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "trajectory_exact_to_reference": True,
        "all_non_timing_records_exact": True,
        "qp_iterations_p50": float(np.percentile(iterations, 50)),
        "qp_iterations_p95": float(np.percentile(iterations, 95)),
        "qp_iterations_p99": float(np.percentile(iterations, 99)),
        "qp_iterations_max": int(np.max(iterations)),
        "solver_time_iteration_correlation": float(np.corrcoef(
            iterations, solver)[0, 1]),
        "metrics": metrics,
        "online_20ms_admitted": False,
        "full_online_cycle_evaluated": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_current_qp_breakdown.py",
                              "audit_b2_prepared_torque_timing.py",
                              "hierarchical_qp.py",
                              "pcc_interval_cbf.py",
                              "b2_work_domain_velocity_box.py",
                              "b2_cholesky_unconstrained.py")},
        "inputs_sha256": {
            "reference_records": _sha(
                reference_dir / "private_rollout_records.jsonl"),
            "new_records": _sha(records_path),
            "new_trace": _sha(output_dir / "private_rollout_trace.npz"),
        },
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "qp_breakdown_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# B.2 私有 QP 时间分解", "",
             "场景 01 的 1350 个周期重放同一力矩轨迹；在 QP `solve`"
             "及若干方法边界记录时间，不改变候选和迭代。", "",
             "| 部件 | p50 ms | p95 ms | p99 ms | 最大 ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, key in (("QP 前预检", "preflight_before_solve_ms"),
                       ("QP 内求解循环", "solver_latency_ms"),
                       ("反作用映射", "reaction_map_ms"),
                       ("任务 Jacobian", "task_jacobians_ms"),
                       ("约束行装配总计", "clearance_assembly_ms"),
                       ("其中 MuJoCo 行", "mujoco_rows_ms"),
                       ("其中胶囊行", "capsule_rows_ms"),
                       ("其中区间求值", "interval_evaluation_ms"),
                       ("QP 内其他", "other_in_solve_ms"),
                       ("QP solve 全部", "qp_solve_wall_ms"),
                       ("预检＋QP", "preflight_plus_qp_ms")):
        item = metrics[key]
        lines.append(f"| {label} | {item['p50_ms']:.3f} | "
                     f"{item['p95_ms']:.3f} | {item['p99_ms']:.3f} | "
                     f"{item['max_ms']:.3f} |")
    lines += ["", f"QP 迭代 p95={report['qp_iterations_p95']:.0f}，"
              f"最大={report['qp_iterations_max']}；"
              f"求解循环时间与迭代相关系数="
              f"{report['solver_time_iteration_correlation']:.3f}。", "",
              "约束行装配包含 MuJoCo、胶囊和区间子项；QP 内其他也包含行装配。"
              "各列单独计算分位数，不能相加当作总 p95。"
              "独立约束行诊断不计入表内；这不是完整在线周期。", ""]
    (output_dir / "QP_BREAKDOWN.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--screened-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scenario-id", default="v6_lite_scenario_01")
    parser.add_argument("--ticks", type=int, default=1350)
    args = parser.parse_args()
    report = run(args.output_dir, args.reference_dir, args.screened_dir,
                 args.fixed_dir, args.baseline_dir, args.a1_root,
                 args.scenario_id, args.ticks)
    print(json.dumps({"iterations_p95": report["qp_iterations_p95"],
                      "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
