"""ABBA read-only probe of requested OpenBLAS thread counts.

This changes only a child process environment variable. Each child performs
the same native A.1 torque replay and frozen 17-D QP probe. It does not time
the complete control loop or authorize interval-mode execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


GROUPS = (("a1", 2), ("b1", 1), ("b2", 1), ("a2", 2))
PROBE_TICKS = (50, 100, 150)
POINT_BUDGET = 63
PERIOD_MS = 20.0


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
    if not len(data):
        raise ValueError("timing group is empty")
    return {
        "count": len(data), "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
        "over_20ms_count": int(np.count_nonzero(data > PERIOD_MS)),
    }


def run(output_dir: Path, original_probe_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    original = json.loads(original_probe_path.read_text(encoding="utf-8"))
    if (original["probe_ticks"] != list(PROBE_TICKS)
            or original["point_budget"] != POINT_BUDGET
            or len(original["records"]) != 30
            or not original["passed_as_read_only_integrity"]):
        raise ValueError("reference read-only QP probe does not match protocol")
    reference = {(row["mode"], row["scenario_id"], row["tick"]): row
                 for row in original["records"]}
    if len(reference) != 30:
        raise ValueError("reference probe keys are not unique")
    report = {
        "schema": "v6_2_b2_openblas_threads_abba_v1",
        "scope": "read_only_partial_chain_on_30_frozen_old_A1_states_per_group",
        "groups_predeclared": [{"name": name, "requested_openblas_threads": threads}
                               for name, threads in GROUPS],
        "probe_ticks": list(PROBE_TICKS), "point_budget": POINT_BUDGET,
        "period_reference_ms": PERIOD_MS,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_blas_thread_trial.py", "audit_b2_weighted_qp_probe.py",
            "hierarchical_qp.py", "pcc_interval_cbf.py", "safety_contract.py",
        )},
        "input_original_probe_sha256": _sha(original_probe_path),
        "new_interval_mode_executed": False,
        "full_control_cycle_measured": False,
        "groups": [], "modes": {},
    }
    records = []
    for name, threads in GROUPS:
        folder = output_dir / name
        environment = os.environ.copy()
        environment["OPENBLAS_NUM_THREADS"] = str(threads)
        command = [sys.executable, "-m", "v6_lite.audit_b2_weighted_qp_probe",
                   "--output-dir", str(folder), "--probe-ticks",
                   *(str(tick) for tick in PROBE_TICKS),
                   "--point-budget", str(POINT_BUDGET)]
        started = time.perf_counter()
        completed = subprocess.run(command, env=environment,
                                   capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   check=False)
        stdout_path = output_dir / f"{name}_stdout.txt"
        stderr_path = output_dir / f"{name}_stderr.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
        stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
        group = {
            "name": name, "requested_openblas_threads": threads,
            "environment_variable": "OPENBLAS_NUM_THREADS",
            "command": command,
            "exit_code": completed.returncode,
            "wall_seconds": time.perf_counter() - started,
            "stdout_sha256": _sha(stdout_path),
            "stderr_sha256": _sha(stderr_path),
            "status": "ERROR", "record_count": 0,
        }
        try:
            if completed.returncode:
                raise RuntimeError(f"child probe exited {completed.returncode}")
            path = folder / "weighted_qp_probe.json"
            current = json.loads(path.read_text(encoding="utf-8"))
            if (not current["passed_as_read_only_integrity"]
                    or current["probe_ticks"] != list(PROBE_TICKS)
                    or current["point_budget"] != POINT_BUDGET
                    or current["inputs"] != original["inputs"]
                    or len(current["records"]) != 30):
                raise ValueError("child probe integrity or frozen inputs changed")
            mismatches = 0
            for row in current["records"]:
                key = row["mode"], row["scenario_id"], row["tick"]
                if key not in reference:
                    raise ValueError(f"unexpected child probe key {key}")
                expected = reference[key]
                parity = (row["query_status"] == expected["query_status"]
                          and row["selected_interval_count"]
                          == expected["selected_interval_count"]
                          and row["selected_command"] is not None
                          and np.allclose(row["selected_command"],
                                          expected["selected_command"],
                                          atol=1e-8, rtol=0.0))
                mismatches += not parity
                records.append({
                    "group": name, "requested_openblas_threads": threads,
                    "mode": row["mode"], "scenario_id": row["scenario_id"],
                    "tick": row["tick"], "candidate_parity_ok": bool(parity),
                    "query_plus_qp_probe_ms": row["query_plus_qp_probe_ms"],
                    "qp_full_ms": row["qp_full_ms"],
                    "qp_solver_ms": row["qp_solver_ms"],
                })
            group.update({
                "status": "COMPLETE", "record_count": 30,
                "candidate_mismatch_count": mismatches,
                "report_path": path.as_posix(), "report_sha256": _sha(path),
                "document_sha256": _sha(folder / "WEIGHTED_QP_PROBE.md"),
                "manifest_sha256": _sha(folder / "weighted_qp_probe_manifest.json"),
            })
        except Exception as exc:
            group["exception"] = f"{type(exc).__name__}: {exc}"
            group["traceback"] = traceback.format_exc()
        report["groups"].append(group)
        print(f"[b2-blas-abba] {name} requested_threads={threads}: "
              f"{group['status']}, candidates={group['record_count']}", flush=True)
    for mode in ("baseline", "enabled"):
        report["modes"][mode] = {}
        for threads in (1, 2):
            own = [row for row in records if row["mode"] == mode
                   and row["requested_openblas_threads"] == threads]
            if not own:
                continue
            report["modes"][mode][str(threads)] = {
                "record_count": len(own),
                "query_plus_qp_probe_ms": _stats([
                    row["query_plus_qp_probe_ms"] for row in own]),
                "qp_full_ms": _stats([row["qp_full_ms"] for row in own]),
                "qp_solver_ms": _stats([row["qp_solver_ms"] for row in own]),
            }
    report["complete_group_count"] = sum(
        group["status"] == "COMPLETE" for group in report["groups"])
    report["candidate_mismatch_count"] = sum(
        group.get("candidate_mismatch_count", 0) for group in report["groups"])
    records_path = output_dir / "abba_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report["records_sha256"] = _sha(records_path)
    summary_path = output_dir / "abba_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    lines = [
        "# B.2 只读 QP 的 OpenBLAS 线程数交叉对照", "",
        "固定 A(请求 2 线程)–B(请求 1 线程)–B–A 顺序；每组对两种模式"
        "五场景 tick 50/100/150 从旧 A.1 力矩原生重放、重建模型并清空"
        "对偶热启动。每组的 30 条候选与首次探针逐状态核对。", "",
        "| 模式 | 请求线程 | 记录 | 查询加 QP p95 / p99 / 最大 ms | 超 20 ms | QP 求解 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        for threads in (2, 1):
            item = report["modes"][mode].get(str(threads))
            if item is None:
                continue
            timing = item["query_plus_qp_probe_ms"]
            lines.append(
                f"| {mode} | {threads} | {item['record_count']} | "
                f"{timing['p95']:.3f} / {timing['p99']:.3f} / {timing['max']:.3f} | "
                f"{timing['over_20ms_count']} | "
                f"{item['qp_solver_ms']['p95']:.3f} |"
            )
    lines += [
        "", f"完整组 {report['complete_group_count']}/4，候选不一致 "
        f"{report['candidate_mismatch_count']}。每组原始报告、控制台输出、"
        "失败信息和哈希保留。",
        "只改变子进程 `OPENBLAS_NUM_THREADS` 请求值；并未测量运行时的"
        "实际线程数或固定操作系统调度。因此统计差异属于本机环境相关"
        "的性能诊断，不是算法加速证明。",
        "查询加 QP 仍是旧轨迹冻结状态的只读部分链路；没有新模式闭环、"
        "完整控制周期、在线漏期或硬实时保证。", "",
    ]
    document_path = output_dir / "BLAS_THREAD_TRIAL.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_openblas_threads_abba_manifest_v1",
        "summary_sha256": _sha(summary_path),
        "records_sha256": _sha(records_path),
        "document_sha256": _sha(document_path),
    }
    manifest_path = output_dir / "abba_manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-probe", type=Path, default=Path(
        "v6_lite/output/v6_2_b2/qp_probe_early/weighted_qp_probe.json"))
    args = parser.parse_args()
    result = run(args.output_dir, args.original_probe)
    print(json.dumps({"complete_group_count": result["complete_group_count"],
                      "candidate_mismatch_count": result["candidate_mismatch_count"],
                      "modes": result["modes"]}, indent=2))
    if result["complete_group_count"] != 4 or result["candidate_mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
