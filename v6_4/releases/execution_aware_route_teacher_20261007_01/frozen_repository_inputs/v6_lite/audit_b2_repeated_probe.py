"""Five predeclared repeats of the read-only early weighted-QP probe.

Every round replays the frozen old torques and solves the same 30 states with
fresh models and cleared duals. Raw timings and results are retained by round.
This is query-plus-QP timing, not a new-mode full control-cycle measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from v6_lite.audit_b2_weighted_qp_probe import run as run_probe


ROUNDS = 5
TICKS = (50, 100, 150)
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
    return {
        "count": len(values), "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
        "over_20ms_count": int(np.count_nonzero(data > PERIOD_MS)),
    }


def run(output_dir: Path, a1_root: Path, original_probe_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    original = json.loads(original_probe_path.read_text(encoding="utf-8"))
    if (original["probe_ticks"] != list(TICKS)
            or original["point_budget"] != POINT_BUDGET
            or len(original["records"]) != 30
            or not original["passed_as_read_only_integrity"]):
        raise ValueError("original early probe protocol changed")
    reference = {(r["mode"], r["scenario_id"], r["tick"]): r
                 for r in original["records"]}
    if len(reference) != 30:
        raise ValueError("original early probe keys changed")
    report = {
        "schema": "v6_2_b2_repeated_read_only_weighted_qp_probe_v1",
        "scope": "five_predeclared_repeats_of_thirty_frozen_old_A1_states",
        "round_count": ROUNDS, "probe_ticks": list(TICKS),
        "point_budget": POINT_BUDGET, "period_reference_ms": PERIOD_MS,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "source_sha256": {name: _source_sha(Path("v6_lite") / name) for name in (
            "audit_b2_repeated_probe.py", "audit_b2_weighted_qp_probe.py",
            "hierarchical_qp.py", "pcc_interval_cbf.py",
            "pcc_state_local_envelope.py", "safety_contract.py",
        )},
        "input_original_probe_sha256": _sha(original_probe_path),
        "new_interval_mode_executed": False,
        "full_control_cycle_measured": False,
        "rounds": [], "modes": {},
    }
    all_records = []
    for round_index in range(1, ROUNDS + 1):
        folder = output_dir / f"round_{round_index:02d}"
        try:
            current = run_probe(folder, a1_root, probe_ticks=TICKS,
                                point_budget=POINT_BUDGET)
            result_path = folder / "weighted_qp_probe.json"
            document_path = folder / "WEIGHTED_QP_PROBE.md"
            manifest_path = folder / "weighted_qp_probe_manifest.json"
            if (not current["passed_as_read_only_integrity"]
                    or len(current["records"]) != 30
                    or current["inputs"] != original["inputs"]
                    or current["probe_ticks"] != original["probe_ticks"]):
                raise ValueError("repeated read-only probe integrity changed")
            mismatches = 0
            for row in current["records"]:
                key = row["mode"], row["scenario_id"], row["tick"]
                if key not in reference:
                    raise ValueError(f"unexpected repeated probe key: {key}")
                expected = reference[key]
                if (row["query_status"] != expected["query_status"]
                        or row["selected_interval_count"]
                        != expected["selected_interval_count"]
                        or row["selected_command"] is None
                        or not np.allclose(row["selected_command"],
                                           expected["selected_command"],
                                           atol=1e-8, rtol=0.0)):
                    mismatches += 1
                all_records.append({
                    "round": round_index, "mode": row["mode"],
                    "scenario_id": row["scenario_id"], "tick": row["tick"],
                    "query_plus_qp_probe_ms": row["query_plus_qp_probe_ms"],
                    "qp_full_ms": row["qp_full_ms"],
                    "qp_solver_ms": row["qp_solver_ms"],
                    "state_local_capsule_envelope_check_ms": row[
                        "state_local_capsule_envelope_check_ms"],
                    "probe_plus_envelope_ms": (row["query_plus_qp_probe_ms"]
                                               + row[
                                                   "state_local_capsule_envelope_check_ms"]),
                    "candidate_parity_ok": mismatches == 0,
                })
            round_result = {
                "round": round_index, "status": "COMPLETE",
                "report_path": result_path.as_posix(),
                "report_sha256": _sha(result_path),
                "document_sha256": _sha(document_path),
                "manifest_sha256": _sha(manifest_path),
                "candidate_mismatch_count": mismatches,
                "record_count": 30,
            }
            report["rounds"].append(round_result)
            print(f"[b2-repeated-probe] round {round_index}/{ROUNDS}: "
                  f"30 records, candidate mismatches={mismatches}", flush=True)
        except Exception as exc:
            failure = {
                "round": round_index, "status": "ERROR",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            report["rounds"].append(failure)
            failure_path = output_dir / f"round_{round_index:02d}_failure.json"
            with failure_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(failure, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"[b2-repeated-probe] round {round_index}/{ROUNDS}: "
                  f"ERROR {failure['exception']}", flush=True)
    for mode in ("baseline", "enabled"):
        own = [row for row in all_records if row["mode"] == mode]
        if not own:
            continue
        report["modes"][mode] = {
            "record_count": len(own),
            "query_plus_qp_probe_ms": _stats([
                row["query_plus_qp_probe_ms"] for row in own]),
            "probe_plus_envelope_ms": _stats([
                row["probe_plus_envelope_ms"] for row in own]),
            "qp_solver_ms": _stats([row["qp_solver_ms"] for row in own]),
            "first_round_query_plus_qp_ms": _stats([
                row["query_plus_qp_probe_ms"] for row in own
                if row["round"] == 1]),
            "later_rounds_query_plus_qp_ms": _stats([
                row["query_plus_qp_probe_ms"] for row in own
                if row["round"] > 1]),
        }
    report["complete_round_count"] = sum(
        item["status"] == "COMPLETE" for item in report["rounds"])
    report["candidate_mismatch_count"] = sum(
        item.get("candidate_mismatch_count", 0) for item in report["rounds"])
    records_path = output_dir / "repeated_probe_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for item in all_records:
            stream.write(json.dumps(item, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    report["records_sha256"] = _sha(records_path)
    summary_path = output_dir / "repeated_probe_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    lines = [
        "# B.2 早期冻结状态只读 QP 的五轮重复计时", "",
        "固定五轮、每轮两组五场景 tick 50/100/150；每轮都从旧力矩轨迹"
        "原生重放，独立建立模型、清空 QP 对偶热启动。完整逐轮报告保留，"
        "候选命令、区间选择和代理判定与首次探针逐状态对照。", "",
        "| 模式 | 记录 | 查询加 QP p95 / p99 / 最大 ms | 超 20 ms | 再加当前状态包络检查 p95 / 最大 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = report["modes"].get(mode)
        if not item:
            continue
        q = item["query_plus_qp_probe_ms"]
        e = item["probe_plus_envelope_ms"]
        lines.append(
            f"| {mode} | {item['record_count']} | "
            f"{q['p95']:.3f} / {q['p99']:.3f} / {q['max']:.3f} | "
            f"{q['over_20ms_count']} | {e['p95']:.3f} / {e['max']:.3f} |"
        )
    lines += [
        "", f"完整轮次 {report['complete_round_count']}/{ROUNDS}；"
        f"候选判定/命令不一致 {report['candidate_mismatch_count']}。",
        "部分链路查询加 QP 不包含全部状态采样、监控、500 Hz 力矩计算"
        "和执行；另加包络时间只是同冻结状态的顺序计时之和。"
        "它们不能作为新区间模式全链 20 ms 验收，也不能推导硬实时保证。"
        "超过 20 ms 的探针记录是超参考周期诊断，不等同真实在线漏期。", "",
    ]
    doc_path = output_dir / "REPEATED_QP_PROBE.md"
    with doc_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_repeated_read_only_probe_manifest_v1",
        **{f"{name}_sha256": _sha(path) for name, path in (
            ("summary", summary_path), ("records", records_path),
            ("document", doc_path),
        )},
        "rounds": [
            {key: value for key, value in item.items() if key.endswith("sha256")
             or key in ("round", "status")}
            for item in report["rounds"]
        ],
    }
    with (output_dir / "repeated_probe_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path,
                        default=Path("v6_lite/output/v6_2_a1"))
    parser.add_argument("--original-probe", type=Path,
                        default=Path("v6_lite/output/v6_2_b2/qp_probe_early/weighted_qp_probe.json"))
    args = parser.parse_args()
    report = run(args.output_dir, args.a1_root, args.original_probe)
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))
    if report["complete_round_count"] != ROUNDS or report["candidate_mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
