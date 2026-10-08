"""Generate the B.2 switchable online-mode evidence report from saved runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _failed_scenario_checks(metrics: dict) -> dict[str, list[str]]:
    return {item["scenario"]["scenario_id"]: [
        key for key, passed in item["checks"].items() if not passed]
        for item in metrics["scenarios"]}


def run(old_dir: Path, new_dir: Path, unoptimized_dir: Path,
        recompute_dir: Path, injection_dir: Path, repeats_dir: Path,
        visual_dir: Path, aabb_dir: Path, qp_dir: Path, parity_dir: Path,
        shadow_dir: Path, near_gate_dir: Path,
        output_dir: Path, doc_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {
        "old_metrics": old_dir / "v6_lite_metrics.json",
        "old_native_validation": old_dir / "validation.json",
        "old_execution_validation": old_dir / "execution_validation.json",
        "new_metrics": new_dir / "v6_lite_metrics.json",
        "new_native_validation": new_dir / "validation.json",
        "new_execution_validation": new_dir / "execution_validation.json",
        "unoptimized_metrics": unoptimized_dir / "v6_lite_metrics.json",
        "interval_recompute": recompute_dir / "online_recompute_summary.json",
        "failure_injection": injection_dir / "failure_injection_summary.json",
        "repeated_timing": repeats_dir / "online_repeated_timing_summary.json",
        "online_visual": visual_dir / "online_progress_summary.json",
        "aabb_negative": aabb_dir / "aabb_screen_visual_summary.json",
        "qp_breakdown": qp_dir / "latest_qp_breakdown_summary.json",
        "trace_parity": parity_dir / "trace_parity_summary.json",
        "historical_a1_shadow": shadow_dir / "shadow_report.json",
        "near_gate_holdout": near_gate_dir / "near_gate_audit.json",
    }
    evidence = {key: _read(path) for key, path in inputs.items()}
    old = evidence["old_metrics"]
    new = evidence["new_metrics"]
    unoptimized = evidence["unoptimized_metrics"]
    native = evidence["new_native_validation"]
    execution = evidence["new_execution_validation"]
    recompute = evidence["interval_recompute"]
    injection = evidence["failure_injection"]
    repeats = evidence["repeated_timing"]
    visual = evidence["online_visual"]
    trace_parity = evidence["trace_parity"]
    shadow = evidence["historical_a1_shadow"]
    near_gate = evidence["near_gate_holdout"]
    if (len(old["scenarios"]) != 5 or len(new["scenarios"]) != 5
            or old["run_config"]["pcc_mode"] != "legacy_pcc"
            or new["run_config"]["pcc_mode"] != "bounded_interval_pcc"
            or new["qp_config"]["enable_pcc_cbf"]
            or not new["qp_config"]["enable_capsule_cbf"]
            or native["evidence_type"] != "native_mujoco_torque_replay"
            or execution["evidence_type"]
            != "independent_execution_contract_trace_validation"
            or not execution["passed"] or not recompute["passed"]
            or not injection["passed"] or not repeats["all_traces_saved"]
            or not visual["full_online_cycle_timed"]
            or not trace_parity["all_passed"]
            or not shadow["passed_as_read_only_audit"]
            or not near_gate["passed"]):
        raise ValueError("saved B.2 online evidence is incomplete")
    common_qp = set(old["qp_config"]) & set(new["qp_config"])
    changed_qp = sorted(key for key in common_qp
                        if old["qp_config"][key] != new["qp_config"][key])
    if changed_qp != ["enable_pcc_cbf"]:
        raise ValueError(f"unexpected old/new QP configuration change: {changed_qp}")
    changed_run = sorted(key for key in old["run_config"]
                         if old["run_config"][key] != new["run_config"][key])
    if changed_run != ["pcc_mode"]:
        raise ValueError(f"unexpected old/new run configuration change: {changed_run}")
    new_failures = _failed_scenario_checks(new)
    original_gate_only_failed = all(
        values == ["task_controller_runs_within_50hz_p95"]
        for values in new_failures.values())
    all_new_scenario_checks_pass = all(not values for values in
                                       new_failures.values())
    trace_hashes = {}
    for item in new["scenarios"]:
        path = Path(item["trace"]["path"])
        trace_hashes[item["scenario"]["scenario_id"]] = _sha(path)
        if trace_hashes[item["scenario"]["scenario_id"]] != item["trace"]["sha256"]:
            raise ValueError(f"new trace changed: {path}")
    old_p95 = old["aggregate_metrics"]["task_full_latency_p95_ms_max"]
    unoptimized_p95 = unoptimized["aggregate_metrics"][
        "task_full_latency_p95_ms_max"]
    optimized_p95 = new["aggregate_metrics"][
        "task_full_latency_p95_ms_max"]
    repeat_p95 = [item["full_control_p95_ms"] for item in repeats["records"]]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    labels = ["Legacy QP", "First new QP", "New full control", "3 repeats\n(6 s)"]
    values = [old_p95, unoptimized_p95, optimized_p95,
              float(np.mean(repeat_p95))]
    bars = ax.bar(labels, values, color=["#5379a7", "#ae6264", "#dc923a",
                                         "#789f6d"])
    ax.errorbar(3, values[3], yerr=[[values[3]-min(repeat_p95)],
                                     [max(repeat_p95)-values[3]]],
                fmt="none", ecolor="black", capsize=5)
    ax.axhline(20, color="#ab3941", linestyle="--", linewidth=1.5,
               label="Original 20 ms p95 threshold")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .3,
                f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(values) + 4)
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("B.2 online performance and unchanged 20 ms criterion")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=.2)
    figure = output_dir / "b2-online-evidence.png"
    fig.savefig(figure, dpi=170, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    report = {
        "schema": "v6_2_b2_online_evidence_report_v1",
        "stage3_functional_five_scenes_complete": True,
        "original_20ms_gate_met": optimized_p95 <= 20,
        "functional_performance_waiver_used": optimized_p95 > 20,
        "original_five_scene_acceptance_met": all_new_scenario_checks_pass,
        "new_mode_native_26_checks_passed": (
            native["passed_count"] == 26 and native["total_count"] == 26),
        "all_predeclared_repeats_within_20ms": repeats[
            "all_full_control_p95_within_20ms"],
        "new_mode_original_scenario_checks_fail_only_latency":
            original_gate_only_failed,
        "new_mode_native_replay": {
            "passed_count": native["passed_count"],
            "total_count": native["total_count"],
            "failures": native["failures"],
        },
        "new_mode_execution_contract": {
            "passed_count": execution["passed_count"],
            "total_count": execution["total_count"],
            "passed": execution["passed"],
        },
        "new_mode_interval_recompute": {
            "task_state_checks": recompute["task_state_checks"],
            "interval_rows_recomputed": recompute["interval_rows_recomputed"],
            "failure_count": recompute["failure_count"],
        },
        "new_mode_failure_injection_passed": sum(
            item["passed"] for item in injection["records"]),
        "new_mode_nontiming_trace_parity_scenarios": sum(
            item["passed"] for item in trace_parity["records"]),
        "historical_a1_shadow_gate": shadow["online_admission_gate"]["status"],
        "user_authorized_stage_order_waiver_used": (
            shadow["online_admission_gate"]["status"] != "MET"),
        "near_gate_holdout_counts": near_gate["counts"],
        "legacy_native_replay": {
            "passed_count": evidence["old_native_validation"]["passed_count"],
            "total_count": evidence["old_native_validation"]["total_count"],
            "failures": evidence["old_native_validation"]["failures"],
        },
        "legacy_execution_contract": {
            "passed_count": evidence["old_execution_validation"]["passed_count"],
            "total_count": evidence["old_execution_validation"]["total_count"],
        },
        "metrics": {
            "legacy_qp_p95_max_ms": old_p95,
            "first_new_qp_p95_max_ms": unoptimized_p95,
            "optimized_new_full_control_p95_max_ms": optimized_p95,
            "repeat_full_control_p95_ms": repeat_p95,
            "new_rigid_final_error_m_max": new["aggregate_metrics"][
                "rigid_final_error_m_max"],
            "new_continuum_path_rmse_m_max": new["aggregate_metrics"][
                "continuum_irregular_path_rmse_m_max"],
            "new_whole_body_minimum_clearance_m": new["aggregate_metrics"][
                "whole_body_minimum_clearance_m"],
            "new_qp_failures": new["aggregate_metrics"]["total_qp_failures"],
        },
        "configuration_changes": {
            "qp": changed_qp, "run": changed_run,
        },
        "input_sha256": {key: _sha(path) for key, path in inputs.items()},
        "new_trace_sha256": trace_hashes,
        "figure_sha256": _sha(figure),
        "historical_a1_artifact_reused_as_new_replay": False,
        "continuous_time_certified": False,
    }
    summary_path = output_dir / "b2_online_evidence_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False,
                                       indent=2, allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")
    image_rel = Path("../v6_lite/output/v6_2_b2") / output_dir.name / figure.name
    lines = [
        "# V6.2-B.2 开关式区间 PCC 在线证据", "",
        "本报告由已保存的运行、重放和故障注入文件自动生成。"
        "`legacy_pcc` 与 `bounded_interval_pcc` 都从同一五场景配置运行；"
        "后者关闭旧 PCC 查询，保留原 MuJoCo 与实际链胶囊约束、17 维加权"
        "速度 QP、67 路力矩执行和 0.45/0.55 十步斜坡。", "",
        "## 功能与原门槛", "",
        "| 项目 | 旧 PCC 新运行 | 区间 PCC 新运行 |", "| --- | ---: | ---: |",
        f"| 五场景完整 trace | {len(old['scenarios'])}/5 | {len(new['scenarios'])}/5 |",
        f"| 原 26 项真实 MuJoCo 力矩重放 | "
        f"{report['legacy_native_replay']['passed_count']}/"
        f"{report['legacy_native_replay']['total_count']} | "
        f"{native['passed_count']}/{native['total_count']} |",
        f"| 原 11 项执行合同 | "
        f"{report['legacy_execution_contract']['passed_count']}/"
        f"{report['legacy_execution_contract']['total_count']} | "
        f"{execution['passed_count']}/{execution['total_count']} |",
        f"| 完整控制或旧 QP p95 最大值 | {old_p95:.3f} ms | "
        f"{optimized_p95:.3f} ms |",
        f"| 整机最小 signed clearance | "
        f"{1000*old['aggregate_metrics']['whole_body_minimum_clearance_m']:.3f} mm | "
        f"{1000*new['aggregate_metrics']['whole_body_minimum_clearance_m']:.3f} mm |",
        "", f"![B.2 在线证据性能图]({image_rel.as_posix()})", "",
        "[最新完整五场景图与视频](V6_2_B2_LATEST_VISUALIZATION.md)", "",
        f"新区间模式五场景完整调用 p95 最大值 {optimized_p95:.3f} ms，"
        + ("本次低于原 20 ms 门槛，原 26 项真实力矩重放全部通过。"
           if report["original_20ms_gate_met"] and
           report["new_mode_native_26_checks_passed"]
           else "本次仍未通过原 20 ms 与 26 项完整验收门槛。")
        + "三轮预声明的 6 s 计时 p95 为 "
        + " / ".join(f"{value:.3f}" for value in repeat_p95)
        + " ms；重复计时的超时结果按原值保留，不能据单次五场景通过"
          "推断稳定的硬实时保证。", "",
        "## 独立与故障证据", "",
        f"新区间行从保存的真实力矩轨迹独立重算："
        f"{recompute['task_state_checks']} 个规划边界、"
        f"{recompute['interval_rows_recomputed']} 条区间行、"
        f"{recompute['failure_count']} 个不一致。"
        "此重算不调用在线 QP 求解或约束装配，但共享声明的几何与动力学模型。", "",
        f"故障注入 {sum(item['passed'] for item in injection['records'])}/"
        f"{len(injection['records'])}：查询未知、QP 故障、斜坡包络失效、"
        "命令过期均在下一力矩步前拒绝。", "",
        "新旧在线实现的五场景非计时 trace 逐项相同，包括全部选中命令和"
        "67 路力矩；性能优化只改变计算路径与实测耗时。", "",
        "## 历史影子与门槛附近边界", "",
        f"旧 A.1 轨迹的只读影子审计通过，但其当时的在线门禁为 "
        f"`{shadow['online_admission_gate']['status']}`；旧轨迹存在代理未知、"
        "形状子空间外与冻结起点违例。此次新区间闭环在自身新轨迹上重新"
        "完成完整执行及重放。依据用户后续明确的有限放宽授权，放宽的是"
        "先后阶段门禁；安全距离、PCC 半径、数值容差和执行器限制未放宽。"
        "旧影子结果仍为未通过，不能改写成当时已通过门禁。", "",
        f"冻结门槛附近留出集 {near_gate['counts']['checked_count']} 例，"
        f"经验代理假安全 {near_gate['counts']['empirical_proxy_false_safe']} 例，"
        f"经验误拒绝 {near_gate['counts']['empirical_proxy_false_reject']} 例；"
        "这是有限样本几何对照，不代表全状态域结论。", "",
        "A.1 历史产物仅是开发基线；本报告的 26 项检查从本次保存的 67 路"
        "力矩重新运行 MuJoCo。失败轨迹和性能反例保留在各自独立目录。"
        "数值区间实现没有形式化浮点认证，代理包络与真实机器人全域安全"
        "不能等同，本报告不宣称连续时间安全证明。", "",
        "## 输入 SHA-256", "",
    ]
    lines += [f"- `{key}`: `{value}`" for key, value in report[
        "input_sha256"].items()]
    lines += [""]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-dir", "new-dir", "unoptimized-dir", "recompute-dir",
                 "injection-dir", "repeats-dir", "visual-dir", "aabb-dir",
                 "qp-dir", "parity-dir", "shadow-dir", "near-gate-dir",
                 "output-dir", "doc-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.old_dir, args.new_dir, args.unoptimized_dir,
                 args.recompute_dir, args.injection_dir, args.repeats_dir,
                 args.visual_dir, args.aabb_dir, args.qp_dir, args.parity_dir,
                 args.shadow_dir, args.near_gate_dir,
                 args.output_dir,
                 args.doc_path)
    print(json.dumps({"stage3_functional_five_scenes_complete": result[
        "stage3_functional_five_scenes_complete"],
        "original_20ms_gate_met": result["original_20ms_gate_met"],
        "metrics": result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
