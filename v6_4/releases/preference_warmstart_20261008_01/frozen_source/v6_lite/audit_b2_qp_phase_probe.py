"""Private single-call QP phase timing on a compensated interval rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from v6_lite import audit_b2_discrete_screened_private_rollout as screened
from v6_lite.audit_b2_prepared_screened_private_rollout import run as private_run
from v6_lite.b2_optimized_admm import OptimizedScreenQP


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(data), "p50_ms": float(np.percentile(data, 50)),
            "p95_ms": float(np.percentile(data, 95)),
            "p99_ms": float(np.percentile(data, 99)),
            "max_ms": float(np.max(data))}


PHASES: list[dict] = []


class PhaseTimingQP(OptimizedScreenQP):
    def reaction_velocity_map(self, *args, **kwargs):
        started = time.perf_counter()
        result = super().reaction_velocity_map(*args, **kwargs)
        if hasattr(self, "_phase_active") and self._phase_active:
            self._phase_reaction_ms += 1000.0 * (time.perf_counter() - started)
        return result

    def _build_all_clearance_constraints(self, *args, **kwargs):
        started = time.perf_counter()
        result = super()._build_all_clearance_constraints(*args, **kwargs)
        self._phase_clearance_ms = 1000.0 * (time.perf_counter() - started)
        return result

    def _solve_qp_admm(self, *args, **kwargs):
        started = time.perf_counter()
        result = super()._solve_qp_admm(*args, **kwargs)
        self._phase_admm_ms = 1000.0 * (time.perf_counter() - started)
        return result

    def solve(self, *args, **kwargs):
        self._phase_active = True
        self._phase_reaction_ms = 0.0
        self._phase_clearance_ms = 0.0
        self._phase_admm_ms = 0.0
        self._phase_dense_solve_ms = 0.0
        self._phase_dense_solve_calls = 0
        self._phase_dense_solve_details = []
        original_dense_solve = np.linalg.solve

        def measured_dense_solve(*solve_args, **solve_kwargs):
            started = time.perf_counter()
            value = original_dense_solve(*solve_args, **solve_kwargs)
            elapsed = 1000.0 * (time.perf_counter() - started)
            self._phase_dense_solve_ms += elapsed
            self._phase_dense_solve_calls += 1
            matrix = np.asarray(solve_args[0])
            rhs = np.asarray(solve_args[1])
            self._phase_dense_solve_details.append({
                "matrix_shape": list(matrix.shape),
                "rhs_shape": list(rhs.shape),
                "matrix_dtype": str(matrix.dtype),
                "matrix_contiguous": bool(matrix.flags.c_contiguous),
                "elapsed_ms": elapsed,
            })
            return value

        try:
            np.linalg.solve = measured_dense_solve
            result = super().solve(*args, **kwargs)
        finally:
            np.linalg.solve = original_dense_solve
            self._phase_active = False
        PHASES.append({
            "tick": len(PHASES),
            "qp_full_ms": 1000.0 * result.full_latency_s,
            "reaction_map_ms": self._phase_reaction_ms,
            "clearance_assembly_ms": self._phase_clearance_ms,
            "admm_ms": self._phase_admm_ms,
            "dense_solve_ms": self._phase_dense_solve_ms,
            "dense_solve_calls": self._phase_dense_solve_calls,
            "dense_solve_details": self._phase_dense_solve_details,
            "solver_result_ms": 1000.0 * result.solver_latency_s,
            "qp_iterations": result.solver_iterations,
        })
        return result


def run(output_dir: Path, baseline_root: Path, a1_root: Path,
        scene_index: int, ticks: int) -> dict:
    if not 0 <= scene_index < 5 or ticks != 400:
        raise ValueError("phase protocol requires one full 400-tick private scene")
    PHASES.clear()
    previous = screened._ScreenQP
    try:
        screened._ScreenQP = PhaseTimingQP
        report = private_run(output_dir, a1_root,
                             f"v6_lite_scenario_{scene_index:02d}", ticks)
    finally:
        screened._ScreenQP = previous
    if len(PHASES) != ticks or report["executed_ticks"] != ticks:
        raise ValueError("QP phase telemetry did not cover every private tick")
    baseline_trace = (baseline_root / f"scene_{scene_index:02d}" /
                      "private_rollout_trace.npz")
    if report["trace_sha256"] != _sha(baseline_trace):
        raise ValueError("QP phase probe changed the executed trajectory")
    rollout_records = [json.loads(line) for line in (
        output_dir / "private_rollout_records.jsonl").read_text(
            encoding="utf-8").splitlines()]
    records = []
    for phase, row in zip(PHASES, rollout_records, strict=True):
        full = phase["qp_full_ms"]
        phase["other_qp_ms"] = max(0.0, full - phase["reaction_map_ms"]
                                   - phase["clearance_assembly_ms"]
                                   - phase["admm_ms"]
                                   - phase["dense_solve_ms"])
        phase["external_preflight_ms"] = max(
            0.0, row["preflight_plus_qp_ms"] - full)
        phase["preflight_plus_qp_ms"] = row["preflight_plus_qp_ms"]
        records.append(phase)
    records_path = output_dir / "qp_phase_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    fields = ("external_preflight_ms", "reaction_map_ms",
              "clearance_assembly_ms", "admm_ms", "dense_solve_ms",
              "other_qp_ms",
              "qp_full_ms", "preflight_plus_qp_ms")
    summary = {
        "schema": "v6_2_b2_private_qp_phase_probe_v1",
        "status": "PRIVATE_TIMING_DIAGNOSTIC_NOT_ONLINE",
        "scenario_id": report["scenario_id"],
        "ticks": ticks,
        "baseline_trace_sha256": _sha(baseline_trace),
        "probe_trace_sha256": report["trace_sha256"],
        "private_rollout_summary_sha256": _sha(output_dir / "private_rollout_summary.json"),
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in ("audit_b2_qp_phase_probe.py",
                                       "b2_optimized_admm.py")},
        "phase_statistics": {name: _stats([row[name] for row in records])
                             for name in fields},
        "dense_solve_call_counts": sorted({row["dense_solve_calls"]
                                           for row in records}),
        "records_sha256": _sha(records_path),
        "production_online_controller_changed": False,
        "stage3_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
    }
    summary_path = output_dir / "qp_phase_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")
    lines = ["# B.2 私有 QP 阶段计时", "",
             "单次 400 周期诊断；同一完整私有闭环中的一个 QP 调用，"
             "按反作用映射、碰撞/区间行装配、ADMM、稠密线性求解与其余 QP 计算拆分。"
             "计时包装会增加少量开销，p95 分项不能相加推断总体 p95。", "",
             "| 阶段 | p50 ms | p95 ms | p99 ms | 最大 ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in fields:
        item = summary["phase_statistics"][name]
        lines.append(f"| {name} | {item['p50_ms']:.3f} | {item['p95_ms']:.3f} | "
                     f"{item['p99_ms']:.3f} | {item['max_ms']:.3f} |")
    lines += ["", "力矩轨迹哈希与已独立重放的补偿基线相同。"
              "这不是正式全链实时验收或连续时间保证。", ""]
    document_path = output_dir / "QP_PHASE_PROBE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest_path = output_dir / "qp_phase_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "v6_2_b2_private_qp_phase_probe_manifest_v1",
        "summary_sha256": _sha(summary_path),
        "records_sha256": _sha(records_path),
        "document_sha256": _sha(document_path),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/discrete_private_rollout_400"))
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--scene-index", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=400)
    args = parser.parse_args()
    result = run(args.output_dir, args.baseline_root, args.a1_root,
                 args.scene_index, args.ticks)
    print(json.dumps(result["phase_statistics"], indent=2))


if __name__ == "__main__":
    main()
