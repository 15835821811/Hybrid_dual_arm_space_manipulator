"""Bind five complete domain-QP private traces to independent audits.

The result is a stage-2 private diagnostic. Production online admission and
the original 26/11 checks require a separate stage-3 run and evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    scenes = []
    failures = []
    sources = {}
    for index in range(5):
        scene_id = f"v6_lite_scenario_{index:02d}"
        prefix = f"domain_endpoint_cholesky_scene{index:02d}_1350"
        private_dir = root / prefix
        recompute_dir = root / f"{prefix}_recompute"
        domain_dir = root / f"{prefix}_domain_recompute"
        task_dir = root / f"{prefix}_task_metrics"
        budget_dir = root / f"{prefix}_domain_budget"
        paths = {
            "private_summary": private_dir / "private_rollout_summary.json",
            "private_records": private_dir / "private_rollout_records.jsonl",
            "private_trace": private_dir / "private_rollout_trace.npz",
            "private_manifest": private_dir / "private_rollout_manifest.json",
            "recompute_summary": recompute_dir / "private_recompute_summary.json",
            "recompute_manifest": recompute_dir / "private_recompute_manifest.json",
            "domain_summary": domain_dir / "domain_endpoint_recompute.json",
            "task_summary": task_dir / "private_task_metrics.json",
            "budget_summary": budget_dir / "unified_budget_summary.json",
            "budget_manifest": budget_dir / "unified_budget_manifest.json",
        }
        for name, path in paths.items():
            sources[f"{scene_id}:{name}"] = {
                "path": path.as_posix(), "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
        private = _load(paths["private_summary"])
        private_manifest = _load(paths["private_manifest"])
        recompute = _load(paths["recompute_summary"])
        recompute_manifest = _load(paths["recompute_manifest"])
        domain = _load(paths["domain_summary"])
        task = _load(paths["task_summary"])
        budget = _load(paths["budget_summary"])
        budget_manifest = _load(paths["budget_manifest"])
        if (private_manifest["summary_sha256"] != _sha(paths["private_summary"])
                or private_manifest["records_sha256"] != _sha(paths["private_records"])
                or private_manifest["trace_sha256"] != _sha(paths["private_trace"])
                or recompute_manifest["summary_sha256"]
                != _sha(paths["recompute_summary"])
                or budget_manifest["summary_sha256"]
                != _sha(paths["budget_summary"])):
            raise ValueError(f"{scene_id} raw manifest hash changed")
        if (domain["inputs_sha256"]["private_summary"]
                != _sha(paths["private_summary"])
                or domain["inputs_sha256"]["private_records"]
                != _sha(paths["private_records"])
                or domain["inputs_sha256"]["private_trace"]
                != _sha(paths["private_trace"])
                or domain["inputs_sha256"]["independent_recompute"]
                != _sha(paths["recompute_summary"])
                or task["inputs_sha256"]["private_summary"]
                != _sha(paths["private_summary"])
                or task["inputs_sha256"]["private_trace"]
                != _sha(paths["private_trace"])
                or budget["inputs"]["private_summary_sha256"]
                != _sha(paths["private_summary"])
                or budget["inputs"]["private_records_sha256"]
                != _sha(paths["private_records"])
                or budget["inputs"]["private_trace_sha256"]
                != _sha(paths["private_trace"])
                or budget["inputs"]["independent_recompute_sha256"]
                != _sha(paths["recompute_summary"])):
            raise ValueError(f"{scene_id} derived audit input hash changed")
        for name, digest in private["source_sha256"].items():
            if _source_sha(Path("v6_lite") / name) != digest:
                raise ValueError(f"{scene_id} controller source changed: {name}")
        for report in (recompute, budget):
            for name, digest in report["source_sha256"].items():
                if _source_sha(Path("v6_lite") / name) != digest:
                    raise ValueError(f"{scene_id} audit source changed: {name}")
        if (domain["source_sha256"] != _source_sha(
                Path("v6_lite/audit_b2_domain_endpoint_recompute.py"))
                or task["source_sha256"] != _source_sha(
                    Path("v6_lite/audit_b2_private_task_metrics.py"))):
            raise ValueError(f"{scene_id} domain or task audit source changed")
        checks = {
            "complete_27s": (private["scenario_id"] == scene_id
                             and private["executed_ticks"] == 1350
                             and private["attempted_ticks"] == 1350
                             and private["stop_reason"] == "HORIZON_COMPLETE"),
            "original_servo_and_interval_replay": (
                recompute["pass_recompute"] and recompute["failure_count"] == 0
                and recompute["maximum_native_state_error"] <= 1e-8),
            "all_microstates_in_work_domain": (
                domain["pass_domain_recompute"]
                and domain["checked_2ms_states"] == 13501
                and domain["minimum_executed_domain_margin_rad"] >= 0),
            "original_task_and_clearance_thresholds": (
                task["original_task_and_clearance_thresholds_passed"]
                and task["servo_states_evaluated"] == 13500),
            "saved_state_budget": (
                budget["budget_overrun_tick_count"] == 0
                and budget["parity_failure_count"] == 0
                and budget["within_frozen_shadow_budget_on_saved_states"]),
        }
        if not all(checks.values()):
            failures.append({"scenario_id": scene_id,
                             "failed_checks": [name for name, value in checks.items()
                                               if not value]})
        scenes.append({
            "scenario_id": scene_id,
            "checks": checks,
            "executed_ticks": private["executed_ticks"],
            "recomputed_task_states": recompute["checked_task_states"],
            "recomputed_interval_rows": recompute["checked_interval_rows"],
            "checked_2ms_states": domain["checked_2ms_states"],
            "minimum_work_domain_margin_rad": domain[
                "minimum_executed_domain_margin_rad"],
            "minimum_frozen_endpoint_margin_rad": domain[
                "minimum_frozen_endpoint_margin_rad"],
            "whole_body_minimum_clearance_m": task["whole_body_clearance"][
                "minimum_clearance"],
            "continuum_path_rmse_m": task["metric"][
                "continuum_active_path_rmse_m"],
            "task_tracking": task["metric"],
            "preflight_plus_qp_timing": private[
                "private_preflight_plus_qp_timing"],
            "budget_overrun_ticks": budget["budget_overrun_tick_count"],
            "source_sha256": {name: sources[f"{scene_id}:{name}"]["sha256"]
                              for name in paths},
        })
    failure_path = output_dir / "domain_private_five_failures.jsonl"
    with failure_path.open("x", encoding="utf-8", newline="\n") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    report = {
        "schema": "v6_2_b2_domain_private_five_gate_v1",
        "private_diagnostic_status": (
            "PASS_PRIVATE_FIVE_WITH_LIMITS" if not failures else "PRIVATE_FIVE_FAILED"),
        "stage2_online_gate_status": "GATE_NOT_MET",
        "stage3_admission": False,
        "production_online_controller_changed": False,
        "original_26_11_online_acceptance_completed": False,
        "full_cycle_20ms_acceptance": False,
        "continuous_time_certified": False,
        "scene_count": len(scenes),
        "executed_task_ticks": sum(item["executed_ticks"] for item in scenes),
        "independent_recomputed_task_states": sum(
            item["recomputed_task_states"] for item in scenes),
        "independent_recomputed_interval_rows": sum(
            item["recomputed_interval_rows"] for item in scenes),
        "independent_checked_2ms_states": sum(
            item["checked_2ms_states"] for item in scenes),
        "minimum_work_domain_margin_rad": min(
            item["minimum_work_domain_margin_rad"] for item in scenes),
        "maximum_preflight_plus_qp_p95_ms": max(
            item["preflight_plus_qp_timing"]["p95_ms"] for item in scenes),
        "maximum_preflight_plus_qp_single_ms": max(
            item["preflight_plus_qp_timing"]["max_ms"] for item in scenes),
        "preflight_plus_qp_over_20ms_tick_count": sum(
            item["preflight_plus_qp_timing"]["over_20ms_count"]
            for item in scenes),
        "failure_count": len(failures),
        "failure_records_sha256": _sha(failure_path),
        "generator_source_sha256": _source_sha(Path(__file__)),
        "scenes": scenes,
    }
    summary_path = output_dir / "domain_private_five_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")
    lines = [
        "# B.2 声明工作域端点约束的五场景私有闭环", "",
        "新私有控制分支保留同一个 17 维加权 QP、67 路力矩伺服、原 MuJoCo／"
        "胶囊约束以及原安全裕度。QP 加入冻结斜坡末端的声明工作域约束；"
        "每个候选仍须通过 11 个 MuJoCo 微状态的执行前检查。", "",
        "| 场景 | 执行周期 | 独立区间行 | 最小工作域余量 rad | 整机最小净空 m | 路径 RMSE m | 预检＋QP p95 / 最大 ms | 超 20 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in scenes:
        timing = item["preflight_plus_qp_timing"]
        lines.append(
            f"| {item['scenario_id'][-2:]} | {item['executed_ticks']} | "
            f"{item['recomputed_interval_rows']} | "
            f"{item['minimum_work_domain_margin_rad']:.6g} | "
            f"{item['whole_body_minimum_clearance_m']:.6g} | "
            f"{item['continuum_path_rmse_m']:.6g} | "
            f"{timing['p95_ms']:.3f} / {timing['max_ms']:.3f} | "
            f"{timing['over_20ms_count']} |"
        )
    lines += [
        "", f"五场景共 {report['executed_task_ticks']} 个执行规划周期，"
        f"{report['independent_recomputed_task_states']} 个独立重放规划状态，"
        f"{report['independent_recomputed_interval_rows']} 条独立区间行，"
        f"{report['independent_checked_2ms_states']} 个逐微步工作域状态。"
        f"失败检查 {len(failures)} 项；历史旧私有轨迹的越界反例继续单独保留。",
        "", "本报告的任务指标与整机净空按原 A.1 阈值复算；统一预算属于保存状态"
        "的事后核对。预检＋QP 计时不包含十步预测执行前检查、完整在线任务层、"
        "力矩伺服或真实调度；因此超过 20 ms 的单周期仍被逐项报告。",
        "", "阶段状态：**GATE_NOT_MET**。这些新轨迹是模型内私有闭环证据，"
        "尚无生产开关式在线模式的原 26/11 项正式验收或连续时间证明。", "",
    ]
    document_path = output_dir / "DOMAIN_PRIVATE_FIVE.md"
    document_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema": "v6_2_b2_domain_private_five_manifest_v1",
        "status": report["private_diagnostic_status"],
        "sources": sources,
        "summary_sha256": _sha(summary_path),
        "failures_sha256": _sha(failure_path),
        "document_sha256": _sha(document_path),
    }
    manifest_path = output_dir / "domain_private_five_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                             + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "private_diagnostic_status", "stage2_online_gate_status",
        "executed_task_ticks", "independent_recomputed_interval_rows",
        "independent_checked_2ms_states", "failure_count",
        "preflight_plus_qp_over_20ms_tick_count",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
