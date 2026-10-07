"""Pair original and product-reuse ADMM on identical private QP inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from v6_lite import audit_b2_private_full_qp_sphere as full_trial
from v6_lite.audit_b2_private_qp_sphere import _ScreenQP
from v6_lite.b2_optimized_admm import OptimizedScreenQP


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(data), "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data))}


def run(output_dir: Path, private_root: Path, a1_root: Path,
        *, scene_limit: int = 5, max_ticks: int = 400) -> dict:
    if not 1 <= scene_limit <= 5 or not 1 <= max_ticks <= 400:
        raise ValueError("solver trial outside saved private trace")
    output_dir.mkdir(parents=True, exist_ok=False)
    original = _ScreenQP._solve_qp_admm
    optimized = OptimizedScreenQP._solve_qp_admm
    rows: list[dict] = []

    def measured_solver(self, *args):
        index = len(rows)
        order = (("original", "optimized", "optimized", "original")
                 if index % 2 == 0 else
                 ("optimized", "original", "original", "optimized"))
        timing: dict[str, list[float]] = {"original": [], "optimized": []}
        results = {}
        for name in order:
            implementation = original if name == "original" else optimized
            started = time.perf_counter()
            result = implementation(self, *args)
            timing[name].append((time.perf_counter() - started) * 1000)
            results[name] = result
        left, right = results["original"], results["optimized"]
        candidate_error = float(np.max(np.abs(left[0] - right[0])))
        dual_error = float(np.max(np.abs(left[4] - right[4])))
        same_status = (left[1] == right[1]
                       and left[2] == right[2]
                       and left[3] == right[3])
        if candidate_error > 1e-10 or dual_error > 1e-8 or not same_status:
            raise ValueError(f"ADMM parity failed at screened QP {index}")
        rows.append({
            "screened_qp_index": index,
            "constraint_row_count": int(args[2].shape[0]),
            "iterations": int(left[3]),
            "candidate_max_abs_error": candidate_error,
            "dual_max_abs_error": dual_error,
            "same_status_and_iterations": same_status,
            "original_median_ms": float(np.median(timing["original"])),
            "optimized_median_ms": float(np.median(timing["optimized"])),
        })
        return left

    try:
        _ScreenQP._solve_qp_admm = measured_solver
        instrumented = full_trial.run(
            output_dir / "instrumented_full_qp", private_root, a1_root,
            max_ticks=max_ticks, scene_limit=scene_limit)
    finally:
        _ScreenQP._solve_qp_admm = original
    if (len(rows) != scene_limit * max_ticks
            or instrumented["record_count"] != 2 * scene_limit * max_ticks
            or instrumented["failure_count"]):
        raise ValueError("instrumented full-QP parity failed")
    records_path = output_dir / "admm_solver_trial_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for item in rows:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    report = {
        "schema": "v6_2_b2_admm_product_reuse_trial_v1",
        "scope": "screened_qp_solver_only_identical_saved_private_inputs",
        "scene_limit": scene_limit,
        "ticks_per_scene": max_ticks,
        "screened_qp_inputs": len(rows),
        "instrumented_full_qp_record_count": instrumented["record_count"],
        "instrumented_full_qp_failure_count": instrumented["failure_count"],
        "instrumented_full_qp_timing_valid": False,
        "maximum_candidate_error": max(x["candidate_max_abs_error"]
                                        for x in rows),
        "maximum_dual_error": max(x["dual_max_abs_error"] for x in rows),
        "all_status_and_iterations_identical": all(
            x["same_status_and_iterations"] for x in rows),
        "original_solver_median_ms": _stats(
            [x["original_median_ms"] for x in rows]),
        "optimized_solver_median_ms": _stats(
            [x["optimized_median_ms"] for x in rows]),
        "input_private_hashes": instrumented["input_private_hashes"],
        "instrumented_full_qp_summary_sha256": _sha(
            output_dir / "instrumented_full_qp"
            / "private_full_qp_sphere_summary.json"),
        "production_online_controller_changed": False,
        "online_admission": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "source_sha256": {name: _source_sha(Path("v6_lite") / name)
                          for name in ("audit_b2_admm_solver_trial.py",
                                       "b2_optimized_admm.py",
                                       "hierarchical_qp.py")},
        "records_sha256": _sha(records_path),
    }
    summary_path = output_dir / "admm_solver_trial_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 同一 17 维加权 QP 的 ADMM 乘积复用诊断", "",
        f"在 {len(rows)} 个相同筛选 QP 输入上按交替 ABBA 顺序单独计时。"
        "原算法与乘积复用版本的候选、对偶、状态和迭代次数逐项比较。"
        "外层 QP 总耗时因四次重复求解被污染，不能用于 20 ms 验收。", "",
        "| 求解循环 | 中位数 p50 ms | p95 ms | p99 ms | 最大 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, key in (("原循环", "original_solver_median_ms"),
                      ("乘积复用", "optimized_solver_median_ms")):
        timing = report[key]
        lines.append(
            f"| {name} | {timing['p50']:.3f} | {timing['p95']:.3f} | "
            f"{timing['p99']:.3f} | {timing['max']:.3f} |"
        )
    lines += [
        "", f"候选最大差 {report['maximum_candidate_error']:.3e}，"
        f"对偶最大差 {report['maximum_dual_error']:.3e}；"
        f"迭代与状态全同 {report['all_status_and_iterations_identical']}。"
        "这是离线求解器局部计时，尚未进入正式控制器。", "",
    ]
    doc_path = output_dir / "ADMM_SOLVER_TRIAL.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_admm_product_reuse_manifest_v1",
        "summary_sha256": _sha(summary_path),
        "records_sha256": _sha(records_path),
        "document_sha256": _sha(doc_path),
    }
    with (output_dir / "admm_solver_trial_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/discrete_private_rollout_400"))
    parser.add_argument("--a1-root", type=Path, default=Path(
        "v6_lite/output/v6_2_a1"))
    parser.add_argument("--scene-limit", type=int, default=5)
    parser.add_argument("--max-ticks", type=int, default=400)
    args = parser.parse_args()
    report = run(args.output_dir, args.private_root, args.a1_root,
                 scene_limit=args.scene_limit, max_ticks=args.max_ticks)
    print(json.dumps({"inputs": report["screened_qp_inputs"],
                      "max_candidate_error": report["maximum_candidate_error"],
                      "original_p95_ms": report[
                          "original_solver_median_ms"]["p95"],
                      "optimized_p95_ms": report[
                          "optimized_solver_median_ms"]["p95"]}, indent=2))


if __name__ == "__main__":
    main()
