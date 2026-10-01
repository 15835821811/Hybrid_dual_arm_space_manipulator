"""Bind five complete private B.2 traces without granting online admission.

The private controller records domain diagnostics but does not reject a command
solely because the declared B.1 work domain is exceeded. This report treats
such executed commands as gate blockers, even when replay and budgets pass.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path


SCENES = tuple(f"v6_lite_scenario_{index:02d}" for index in range(5))
TICKS = 1350


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_files(folder: Path, manifest_name: str,
                 files: dict[str, str]) -> None:
    manifest = _read(folder / manifest_name)
    for label, filename in files.items():
        if manifest[f"{label}_sha256"] != _sha(folder / filename):
            raise ValueError(f"changed {label}: {folder / filename}")


def _check_sources(source_hashes: dict[str, str]) -> None:
    for name, expected in source_hashes.items():
        path = Path("v6_lite") / name
        if _source_sha(path) != expected:
            raise ValueError(f"changed source: {path}")


def run(root: Path, output_dir: Path) -> dict:
    root = Path(root)
    scene_reports = []
    failure_rows = []
    total_recomputed_rows = 0
    for scene_id in SCENES:
        suffix = scene_id[-2:]
        prefix = f"cholesky_private_full_scene{suffix}"
        private_dir = root / f"{prefix}_1350"
        replay_dir = root / f"{prefix}_recompute"
        budget_dir = root / f"{prefix}_budget"
        private_paths = {
            "summary": "private_rollout_summary.json",
            "records": "private_rollout_records.jsonl",
            "trace": "private_rollout_trace.npz",
            "document": "PRIVATE_ROLLOUT.md",
        }
        replay_paths = {
            "summary": "private_recompute_summary.json",
            "checks": "private_recompute_checks.jsonl",
            "interval_rows": "private_recompute_interval_rows.jsonl",
            "failures": "private_recompute_failures.jsonl",
            "document": "PRIVATE_RECOMPUTE.md",
        }
        budget_paths = {
            "summary": "unified_budget_summary.json",
            "rows": "unified_budget_rows.jsonl",
            "document": "UNIFIED_BUDGET.md",
        }
        _check_files(private_dir, "private_rollout_manifest.json", private_paths)
        _check_files(replay_dir, "private_recompute_manifest.json", replay_paths)
        _check_files(budget_dir, "unified_budget_manifest.json", budget_paths)
        private_manifest = _read(private_dir / "private_rollout_manifest.json")
        if private_manifest["failure_sha256"] is not None:
            raise ValueError(f"private run did not complete: {scene_id}")
        private = _read(private_dir / private_paths["summary"])
        replay = _read(replay_dir / replay_paths["summary"])
        budget = _read(budget_dir / budget_paths["summary"])
        for report in (private, replay, budget):
            _check_sources(report["source_sha256"])
        if (private["scenario_id"] != scene_id
                or private["predeclared_horizon_ticks"] != TICKS
                or private["executed_ticks"] != TICKS
                or private["stop_reason"] != "HORIZON_COMPLETE"
                or private["records_sha256"] != private_manifest["records_sha256"]
                or private["trace_sha256"] != private_manifest["trace_sha256"]
                or replay["checked_task_states"] != TICKS + 1
                or replay["failure_count"] != 0
                or not replay["pass_recompute"]
                or budget["scenario_id"] != scene_id
                or budget["task_ticks"] != TICKS
                or budget["parity_failure_count"] != 0
                or budget["query_budget_exhausted_count"] != 0
                or budget["budget_overrun_tick_count"] != 0):
            raise ValueError(f"incomplete private evidence: {scene_id}")
        source_hashes = budget["inputs"]
        if (source_hashes["private_summary_sha256"]
                != _sha(private_dir / private_paths["summary"])
                or source_hashes["private_records_sha256"]
                != _sha(private_dir / private_paths["records"])
                or source_hashes["private_trace_sha256"]
                != _sha(private_dir / private_paths["trace"])
                or source_hashes["independent_recompute_sha256"]
                != _sha(replay_dir / replay_paths["summary"])):
            raise ValueError(f"budget provenance changed: {scene_id}")
        replay_input = replay["inputs"][0]
        if (replay_input["summary_sha256"]
                != _sha(private_dir / private_paths["summary"])
                or replay_input["records_sha256"]
                != _sha(private_dir / private_paths["records"])
                or replay_input["trace_sha256"]
                != _sha(private_dir / private_paths["trace"])):
            raise ValueError(f"torque replay provenance changed: {scene_id}")
        records = [json.loads(line) for line in
                   (private_dir / private_paths["records"])
                   .read_text(encoding="utf-8").splitlines()]
        if len(records) != TICKS or [row["tick"] for row in records] != list(range(TICKS)):
            raise ValueError(f"private record cadence changed: {scene_id}")
        unsupported = [row for row in records
                       if not row["strict_online_domain_met"]]
        if (budget["strict_domain_violation_ticks"]
                != [row["tick"] for row in unsupported]
                or budget["strict_domain_violation_tick_count"]
                != len(unsupported)
                or private["strict_online_domain_all_executed_ticks"]
                != (not unsupported)
                or replay["independent_strict_online_domain_all_executed_ticks"]
                != (not unsupported)
                or private["first_tick_outside_strict_online_domain"]
                != (unsupported[0]["tick"] if unsupported else None)):
            raise ValueError(f"work-domain evidence disagrees: {scene_id}")
        for row in unsupported:
            failure_rows.append({
                "scenario_id": scene_id, "tick": row["tick"],
                "time_s": row["time_s"],
                "geometry_domain_status": row["geometry_domain_status"],
                "envelope_evidence_status": row["envelope_evidence_status"],
                "action_mode": row["action_mode"],
                "failure_reason": row["failure_reason"],
                "selected_command": row["selected_command"],
                "private_records_sha256": private_manifest["records_sha256"],
                "private_trace_sha256": private_manifest["trace_sha256"],
            })
        timing = private["private_preflight_plus_qp_timing"]
        if timing["count"] != TICKS:
            raise ValueError(f"private timing cadence changed: {scene_id}")
        scene_reports.append({
            "scenario_id": scene_id, "executed_ticks": TICKS,
            "recomputed_task_states": replay["checked_task_states"],
            "recomputed_interval_rows": replay["checked_interval_rows"],
            "recompute_failures": replay["failure_count"],
            "budget_overrun_ticks": budget["budget_overrun_tick_count"],
            "budget_parity_failures": budget["parity_failure_count"],
            "strict_domain_violation_ticks": len(unsupported),
            "first_strict_domain_violation_tick": (
                unsupported[0]["tick"] if unsupported else None),
            "last_strict_domain_violation_tick": (
                unsupported[-1]["tick"] if unsupported else None),
            "action_modes_outside_domain": dict(Counter(
                row["action_mode"] for row in unsupported)),
            "preflight_plus_qp_timing": timing,
            "inputs": {
                "private_summary_sha256": _sha(private_dir / private_paths["summary"]),
                "private_records_sha256": private_manifest["records_sha256"],
                "private_trace_sha256": private_manifest["trace_sha256"],
                "recompute_summary_sha256": _sha(replay_dir / replay_paths["summary"]),
                "budget_summary_sha256": _sha(budget_dir / budget_paths["summary"]),
            },
        })
        total_recomputed_rows += replay["checked_interval_rows"]
    if len(failure_rows) != 63 or any(row["action_mode"] != "TRACK"
                                   for row in failure_rows):
        raise ValueError("frozen work-domain counterexample changed")
    report = {
        "schema": "v6_2_b2_full_private_five_scene_gate_audit_v1",
        "scope": "five_complete_private_closed_loops_not_production_online",
        "scene_count": len(scene_reports),
        "executed_task_ticks": TICKS * len(scene_reports),
        "recomputed_task_states": (TICKS + 1) * len(scene_reports),
        "recomputed_interval_rows": total_recomputed_rows,
        "strict_domain_violation_tick_count": len(failure_rows),
        "track_commands_issued_outside_declared_domain": len(failure_rows),
        "budget_overrun_tick_count": sum(
            row["budget_overrun_ticks"] for row in scene_reports),
        "timing_p95_over_20ms_scene_count": sum(
            row["preflight_plus_qp_timing"]["p95_ms"] > 20.0
            for row in scene_reports),
        "timing_single_cycle_over_20ms_count": sum(
            row["preflight_plus_qp_timing"]["over_20ms_count"]
            for row in scene_reports),
        "scenes": scene_reports,
        "stage2_status": "GATE_NOT_MET",
        "stage3_admission": False,
        "production_online_controller_changed": False,
        "full_control_time_evaluated": False,
        "new_mode_26_11_acceptance_status": "NOT_EVALUATED",
        "continuous_time_certified": False,
        "source_hash_newline_policy": "LF_NORMALIZED",
        "generator_source_sha256": _source_sha(
            Path("v6_lite/finalize_b2_full_private_five.py")),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    failures_path = output_dir / "private_five_gate_failures.jsonl"
    with failures_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in failure_rows:
            stream.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
    report["failure_records_sha256"] = _sha(failures_path)
    report_path = output_dir / "private_five_gate_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    lines = ["# B.2 五个完整私有闭环的门禁审计", "",
             "五个场景各 1,350 个规划周期，均从 A.1 已发布初态开始，"
             "之后由私有新区间 QP 与原 67 路力矩伺服推进。"
             "每条新轨迹均独立原生力矩重放并重算区间行；"
             "另逐状态核对冻结影子计算预算。", "",
             "| 场景 | 执行周期 | 独立区间行 | 预算超限 | 工作域外且仍 TRACK | 预检＋QP p95 / p99 / 最大 ms | 超 20 ms 周期 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in scene_reports:
        timing = row["preflight_plus_qp_timing"]
        lines.append(
            f"| {row['scenario_id'][-2:]} | {row['executed_ticks']} | "
            f"{row['recomputed_interval_rows']} | {row['budget_overrun_ticks']} | "
            f"{row['strict_domain_violation_ticks']} | "
            f"{timing['p95_ms']:.3f} / {timing['p99_ms']:.3f} / "
            f"{timing['max_ms']:.3f} | {timing['over_20ms_count']} |")
    lines += [
        "", f"总计 {report['executed_task_ticks']} 个执行周期、"
        f"{report['recomputed_task_states']} 个原生重放规划状态和"
        f"{total_recomputed_rows} 条独立区间行，重算不一致 0。",
        "scene_01 在 tick 888–950 的 63 个周期越出声明工作域，"
        "私有诊断仍发出 TRACK 命令；这些周期的代理几何判定和"
        "当前状态包络余量不能替代声明域内的包络证据。"
        "因此该私有轨迹是门禁反例，不能作为安全在线执行证据。",
        "独立力矩重放通过只证明轨迹与模型内计算一致，不消除工作域越界。"
        "当前五场景的预检＋QP p95 均低于 20 ms，但不含完整在线任务层"
        "或硬实时调度；仍有单周期超 20 ms。",
        "", "阶段状态：**GATE_NOT_MET**。生产在线控制未切换；"
        "旧 26/11 项对新模式 trace 的正式验收尚未执行，"
        "也未得到连续时间安全证明。全部越界周期和原始输入哈希单独保存。", "",
    ]
    document_path = output_dir / "FULL_PRIVATE_FIVE_GATE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_full_private_five_gate_manifest_v1",
        "summary_sha256": _sha(report_path),
        "failures_sha256": _sha(failures_path),
        "document_sha256": _sha(document_path),
    }
    (output_dir / "private_five_gate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "scene_count", "executed_task_ticks", "recomputed_interval_rows",
        "strict_domain_violation_tick_count", "stage2_status",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
