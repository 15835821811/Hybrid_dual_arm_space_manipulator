"""Time the unchanged ten-step private B.2 candidate branch by operation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from v6_lite import audit_b2_discrete_private_rollout as discrete
from v6_lite import audit_b2_prepared_torque_timing as prepared
from v6_lite.pcc_fixed_arc_positions import FixedArcPCCPositions
from v6_lite.pcc_fixed_arc_state_envelope import FixedArcStateLocalPCCEnvelopeAudit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"p50_ms": float(np.percentile(data, 50)),
            "p95_ms": float(np.percentile(data, 95)),
            "p99_ms": float(np.percentile(data, 99)),
            "max_ms": float(np.max(data))}


def run(output_dir: Path, reference_dir: Path, screened_dir: Path,
        fixed_dir: Path, baseline_dir: Path, a1_root: Path,
        scenario_id: str, ticks: int) -> dict:
    originals = {
        "branch": discrete._branch,
        "forward": mujoco.mj_forward,
        "step": mujoco.mj_step,
        "envelope": FixedArcStateLocalPCCEnvelopeAudit.evaluate,
        "positions": FixedArcPCCPositions.evaluate,
        "torque": prepared.prepared_compensated_torque,
        "reference": discrete.advance_reference,
    }
    active: list[dict | None] = [None]
    measurements: list[dict] = []

    def timed(name, original):
        def wrapper(*args, **kwargs):
            if active[0] is None:
                return original(*args, **kwargs)
            started = time.perf_counter()
            value = original(*args, **kwargs)
            active[0][name + "_ms"] += 1000.0 * (time.perf_counter() - started)
            active[0][name + "_calls"] += 1
            return value
        return wrapper

    def timed_branch(*args, **kwargs):
        row = {name + suffix: 0.0 if suffix == "_ms" else 0
               for name in ("forward", "step", "envelope", "positions",
                            "torque", "reference")
               for suffix in ("_ms", "_calls")}
        active[0] = row
        started = time.perf_counter()
        try:
            result = originals["branch"](*args, **kwargs)
        finally:
            active[0] = None
        row["branch_wall_ms"] = 1000.0 * (time.perf_counter() - started)
        measurements.append(row)
        return result

    try:
        discrete._branch = timed_branch
        mujoco.mj_forward = timed("forward", originals["forward"])
        mujoco.mj_step = timed("step", originals["step"])
        FixedArcStateLocalPCCEnvelopeAudit.evaluate = timed(
            "envelope", originals["envelope"])
        FixedArcPCCPositions.evaluate = timed("positions", originals["positions"])
        prepared.prepared_compensated_torque = timed(
            "torque", originals["torque"])
        discrete.advance_reference = timed("reference", originals["reference"])
        private = prepared.run(output_dir, screened_dir, fixed_dir,
                               baseline_dir, a1_root, scenario_id, ticks)
    finally:
        discrete._branch = originals["branch"]
        mujoco.mj_forward = originals["forward"]
        mujoco.mj_step = originals["step"]
        FixedArcStateLocalPCCEnvelopeAudit.evaluate = originals["envelope"]
        FixedArcPCCPositions.evaluate = originals["positions"]
        prepared.prepared_compensated_torque = originals["torque"]
        discrete.advance_reference = originals["reference"]

    if (len(measurements) != ticks or not private["all_non_timing_records_exact"]
            or not private["trajectory_exact_to_original_baseline"]):
        raise ValueError("branch probe changed the private trajectory")
    for tick, row in enumerate(measurements):
        row["tick"] = tick
        for name, expected in (("forward", 11), ("step", 10),
                               ("envelope", 11), ("positions", 11),
                               ("torque", 10),
                               ("reference", 10)):
            if row[name + "_calls"] != expected:
                raise ValueError(f"unexpected {name} call count at tick {tick}")
        row["unattributed_ms"] = row["branch_wall_ms"] - sum(
            row[name + "_ms"] for name in (
                "forward", "step", "envelope", "torque", "reference"))
        if row["unattributed_ms"] < -1e-6:
            raise ValueError(f"overlapping branch timers at tick {tick}")

    rows_path = output_dir / "branch_breakdown_rows.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in measurements:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    fields = ("branch_wall_ms", "forward_ms", "step_ms", "envelope_ms",
              "positions_ms",
              "torque_ms", "reference_ms", "unattributed_ms")
    metrics = {field: _stats([row[field] for row in measurements])
               for field in fields}
    report = {
        "schema": "v6_2_b2_current_private_branch_breakdown_v1",
        "scenario_id": scenario_id,
        "task_ticks": ticks,
        "trajectory_exact_to_reference": True,
        "all_non_timing_records_exact": True,
        "metrics": metrics,
        "full_online_cycle_evaluated": False,
        "online_20ms_admitted": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in (
                              "audit_b2_current_branch_breakdown.py",
                              "audit_b2_discrete_private_rollout.py",
                              "audit_b2_prepared_torque_timing.py",
                              "b2_prepared_compensated_torque.py",
                              "pcc_fixed_arc_positions.py",
                              "pcc_fixed_arc_state_envelope.py")},
        "inputs_sha256": {
            "reference_records": _sha(reference_dir / "private_rollout_records.jsonl"),
            "new_records": _sha(output_dir / "private_rollout_records.jsonl"),
            "new_trace": _sha(output_dir / "private_rollout_trace.npz"),
        },
        "rows_sha256": _sha(rows_path),
    }
    (output_dir / "branch_breakdown_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# B.2 私有候选分支耗时分解", "",
             "场景 01 的 1350 个完整 20 ms 周期；十步候选力矩分支保持原状态、"
             "67 路力矩和包络检查，原生重放与参考逐值相同。", "",
             "| 部件 | p50 ms | p95 ms | p99 ms | 最大 ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in fields:
        item = metrics[name]
        lines.append(f"| {name} | {item['p50_ms']:.3f} | "
                     f"{item['p95_ms']:.3f} | {item['p99_ms']:.3f} | "
                     f"{item['max_ms']:.3f} |")
    lines += ["", "positions_ms 已包含在 envelope_ms 中。"
              "每项分位数分别计算，不能相加得到总 p95。"
              "仅为私有计时诊断，不是完整在线周期或硬实时保证。", ""]
    (output_dir / "BRANCH_BREAKDOWN.md").write_text(
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
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
