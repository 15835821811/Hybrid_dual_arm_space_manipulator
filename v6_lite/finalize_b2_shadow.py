"""Generate a hash-bound B.2 stage-2 gate report from immutable audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_blockers(frontier: dict, refined: dict) -> dict[str, dict]:
    """Separate witnessed proxy penetration from old-velocity CBF failures."""

    if frontier["budgets"][-1] != 511 or frontier["gate_m"] != .005:
        raise ValueError("the frozen high-budget witness protocol changed")
    blockers = {}
    for mode in ("baseline", "enabled"):
        entries = [item for item in frontier["entries"] if item["mode"] == mode]
        records = [item for item in refined["records"] if item["mode"] == mode]
        witnessed_below = []
        for entry in entries:
            last = entry["budget_ladder"][-1]
            if last["point_budget"] != 511:
                raise ValueError("missing high-budget witness result")
            if last["proxy_status"] == "PROXY_CLEARANCE_BELOW_GATE":
                if not last["upper_m"] < frontier["gate_m"]:
                    raise ValueError("below-gate status lacks a point witness")
                witnessed_below.append(entry)
        start_bad = [item for item in records if item["feasibility"]["status"]
                     == "START_CLEARANCE_VIOLATION"]
        initial = [item for item in records if item["tick"] == 0]
        positive_h_bad = [item for item in start_bad
                          if item["worst_interval_start_terms"] is not None
                          and item["worst_interval_start_terms"]["h_m"] >= 0.0]
        if (len(witnessed_below) != frontier["modes"][mode]["status_by_budget"]
                ["511"].get("PROXY_CLEARANCE_BELOW_GATE", 0)
                or len(start_bad) != refined["modes"][mode]
                ["diagnostic_refined_start_violation_count"]
                or len(positive_h_bad) != refined["modes"][mode]
                ["start_violations_with_positive_interval_h"]
                or len(initial) != 5
                or len({item["scenario_id"] for item in initial}) != 5):
            raise ValueError(f"gate blocker totals disagree with source: {mode}")

        def earliest_by_scenario(items: list[dict]) -> dict[str, int]:
            first = {}
            for item in items:
                name = item["scenario_id"]
                first[name] = min(first.get(name, item["tick"]), item["tick"])
            return dict(sorted(first.items()))

        blockers[mode] = {
            "initial_proxy_safe_and_frozen_rows_feasible": sum(
                item["proxy_status"] == "PROXY_CLEARANCE_AT_LEAST_GATE"
                and item["feasibility"]["status"] == "FROZEN_ROWS_FEASIBLE"
                for item in initial),
            "sampled_proxy_below_gate_with_point_witness": len(witnessed_below),
            "first_witnessed_below_tick_by_scenario": earliest_by_scenario(
                witnessed_below),
            "sampled_old_start_clearance_violations_after_refinement": len(start_bad),
            "first_old_start_violation_tick_by_scenario": earliest_by_scenario(
                start_bad),
            "old_start_violations_with_nonnegative_interval_h": len(positive_h_bad),
        }
    return blockers


def finalize(root: Path, output_dir: Path) -> dict:
    root = Path(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = {
        "cold_shadow": root / "shadow_a1" / "shadow_report.json",
        "warm_shadow": root / "shadow_warm" / "shadow_report.json",
        "near_gate": root / "near_gate_audit" / "near_gate_audit.json",
        "b1_holdout_geometry": root / "heldout_geometry" / "heldout_geometry_report.json",
        "budget_frontier": root / "budget_frontier" / "budget_frontier.json",
        "refined_start": root / "refined_start" / "refined_start_report.json",
        "repartition_counterfactual": root / "repartition_counterfactual" / "repartition_counterfactual.json",
        "repartition_handoff": root / "repartition_handoff" / "repartition_handoff.json",
    }
    values = {name: json.loads(path.read_text(encoding="utf-8"))
              for name, path in sources.items()}
    cold, warm, near, heldout, frontier, refined, repartition, handoff = (
        values[name] for name in sources
    )
    if not cold["passed_as_read_only_audit"] or not warm["passed_as_read_only_audit"]:
        raise ValueError("native replay shadow integrity failed")
    if not near["passed"] or not heldout["passed"]:
        raise ValueError("frozen geometry audit failed")
    if frontier["input_shadow"]["sha256"] != _sha(sources["warm_shadow"]):
        raise ValueError("budget frontier is not tied to the current warm shadow")
    if refined["input_frontier"]["sha256"] != _sha(sources["budget_frontier"]):
        raise ValueError("refined start is not tied to the current budget frontier")
    if repartition["input_shadow"]["sha256"] != _sha(sources["warm_shadow"]):
        raise ValueError("repartition counterfactual is not tied to the current warm shadow")
    if (repartition["point_budget"] != 64 or repartition["gate_m"] != .005
            or repartition["repartition_applied_to_execution"]):
        raise ValueError("repartition counterfactual changed the frozen comparison")
    for mode in ("baseline", "enabled"):
        counts = repartition["modes"][mode]["counts"]
        expected = warm["modes"][mode]["counts"].get("warm_all_task_unknown", 0)
        expected_ticks = warm["modes"][mode]["summaries"]["warm_all_task_query_ms"]["count"]
        if (counts.get("warm_UNKNOWN_CROSSES_GATE", 0) != expected
                or counts.get("task_ticks", 0) != expected_ticks):
            raise ValueError(f"repartition counterfactual warm count changed: {mode}")
    if handoff["input_repartition"]["sha256"] != _sha(sources["repartition_counterfactual"]):
        raise ValueError("repartition handoff is not tied to the current counterfactual")
    if (handoff["repartition_admitted_online"]
            or len(handoff["native_replay_checks"]) != 10
            or any(item["max_state_error"] > 1e-8
                   for item in handoff["native_replay_checks"])):
        raise ValueError("repartition handoff native replay integrity failed")
    for mode in ("baseline", "enabled"):
        expected_samples = warm["modes"][mode]["counts"].get("UNKNOWN_CROSSES_GATE", 0)
        if handoff["modes"][mode]["sample_count"] != expected_samples:
            raise ValueError(f"repartition handoff sampled count changed: {mode}")
    if (len(refined["native_replay_checks"]) != 10
            or any(x["max_state_error"] > 1e-8
                   for x in refined["native_replay_checks"])):
        raise ValueError("refined native torque replay integrity failed")
    gate = warm["online_admission_gate"]
    blockers = _gate_blockers(frontier, refined)
    remaining = sum(refined["modes"][mode]["old_bad_remains_unexecutable"]
                    for mode in ("baseline", "enabled"))
    status = ("GATE_NOT_MET" if gate["status"] == "NOT_MET" or remaining
              else "NEEDS_TRUE_ONLINE_TIMING")
    lines = [
        "# V6.2-B.2 第二阶段：影子评估与在线接入门禁", "",
        f"当前门禁：**{status}**。因此本证据不作为新区间模式的五场景闭环验收。",
        "", "## A.1 原生力矩重放上的只读几何", "",
        "两组旧控制 trace 均按 500 Hz 力矩原生重放；每个规划边界先调用 `mj_forward`"
        " 更新空间几何量，再与保存的 50 Hz 状态及哈希核对。"
        "每组五场景，每场景预定抽取 27 个规划状态；持久查询则在每组 "
        f"{warm['modes']['baseline']['summaries']['warm_all_task_query_ms']['count']:,} "
        "个规划 tick 上更新。", "",
        "| 模式 | 冷查询 p95 ms | 持久查询采样 p95 ms | 持久查询全部 tick 未知 | 最大叶数 | 配对容量估计 p95 ms | 真实几何误拒绝 | 子空间外状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        c, w = cold["modes"][mode], warm["modes"][mode]
        lines.append(
            f"| {mode} | {c['summaries']['query_time_ms']['p95']:.3f} | "
            f"{w['summaries']['query_time_ms']['p95']:.3f} | "
            f"{w['counts'].get('warm_all_task_unknown', 0)} | "
            f"{w['summaries']['warm_all_task_partition_size']['max']:.0f} | "
            f"{w['summaries']['paired_replacement_estimate_ms']['p95']:.3f} | "
            f"{w['counts'].get('proxy_false_reject_vs_mujoco', 0)} | "
            f"{w['counts'].get('off_shape_subspace', 0)} |"
        )
    lines += [
        "", "持久区间未合并时可增长到预算边界，许多贴近 5 mm 门槛的状态保持未知。"
        "排除区间在所测下一 tick 进入激活距离的计数为 0，但此有限重放结果不是一般动态保证。",
        "配对容量估计使用旧全链耗时减旧形状查询耗时再加影子查询耗时；"
        "它不是新模式实测，也不能单独证明 20 ms 全链 p95。", "",
        "区间导数对每个状态的基座和目标刚体原点各调用一次 MuJoCo Jacobian，"
        "再将刚体速度精确平移至各中点及目标 witness；"
        "下面的计数包含形状 Jacobian 与这两次 MuJoCo 调用。", "",
        "| 模式 | 抽样 Jacobian 次数 p95 | 统一预算超限 | 持久热查询 p95 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = warm["modes"][mode]
        lines.append(
            f"| {mode} | {item['summaries']['jacobian_evaluations']['p95']:.0f} | "
            f"{item['counts'].get('unified_budget_exceeded', 0)} | "
            f"{item['summaries']['query_time_ms']['p95']:.3f} |"
        )
    lines += [
        "", "调用次数下降不等于墙钟时延同比例下降；这里仍无新区间模式的实测全链时延。", "",
        "持久未知状态另以保存的 A.1 规划位置做同状态对照：保持 64 点预算不变，"
        "只从五段根区间重新查询，未重放力矩，也未将新分区用于动作。", "",
        "| 模式 | 持久未知 | 从根区间后确定安全 | 确定低于门槛 | 仍未知 | 额外冷查询 p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = repartition["modes"][mode]
        c = item["counts"]
        lines.append(
            f"| {mode} | {c.get('warm_UNKNOWN_CROSSES_GATE', 0)} | "
            f"{c.get('cold_PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{c.get('cold_PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{c.get('cold_UNKNOWN_CROSSES_GATE', 0)} | "
            f"{item['counterfactual_cold_query_time_ms'].get('p95', float('nan')):.3f} |"
        )
    lines += [
        "", "重分区能解释部分持久未知，但会改变区间安全函数集合。"
        "这项额外耗时不能与热查询并列当作新模式全链计时；"
        "分区切换后的斜坡起点、必要约束和预算仍须单独验证。", "",
        "对持久未知的抽样 tick，又以 500 Hz 旧力矩原生重放，从根分区重建区间行，"
        "加入独立 MuJoCo／胶囊行，按原速度盒与十步斜坡做冻结 LP。", "",
        "| 模式 | 抽样未知 | 根分区确定安全 | 安全且旧起点可执行 | 安全但旧起点不可执行 | Jacobian 超预算 | 下一 tick 漏选激活 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        item = handoff["modes"][mode]
        lines.append(
            f"| {mode} | {item['sample_count']} | "
            f"{item['cold_status'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)} | "
            f"{item['safe_repartition_executable_candidate_count']} | "
            f"{item['safe_repartition_not_executable_count']} | "
            f"{item['jacobian_budget_exceeded']} | "
            f"{item['excluded_reached_activation_next_tick']} |"
        )
    lines += [
        "", "根分区代理安全只完成几何判定；旧斜坡起点仍可能不合格，"
        "且此计算时间不含独立原约束装配、LP 或新控制器全链。"
        "这个有限旧轨迹审计不能批准在线重分区或证明新控制轨迹可行。", "",
        "区间激活筛选从现有速度、加速度和关节限位得到候选盒，"
        "并将十步斜坡起点速度纳入逐轴绝对上界；"
        "空盒退回全局速度上界并单独使门禁失败。"
        "以下计数仅比较两种冻结模型筛选，均未修改安全距离或真实执行动作。", "",
        "| 查询 | 模式 | 比全局速度筛选少选区间 | 多选区间 | 空候选盒 | 排除区间下一 tick 激活 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for query_name, report in (("cold", cold), ("persistent", warm)):
        for mode in ("baseline", "enabled"):
            counts = report["modes"][mode]["counts"]
            lines.append(
                f"| {query_name} | {mode} | "
                f"{counts.get('ramp_box_newly_excluded', 0)} | "
                f"{counts.get('ramp_box_newly_selected', 0)} | "
                f"{counts.get('empty_candidate_velocity_box', 0)} | "
                f"{counts.get('excluded_reached_activation_next_tick', 0)} |"
            )
    lines += [
        "", "速度盒只约束声明的斜坡起点和候选终点；反作用映射、目标漂移"
        "和几何灵敏度在筛选中仍冻结。零漏选是有限重放观察，不是跨周期证明。", "",
        "## 同一冻结行的动作可行性", "",
        "在抽样 A.1 状态上，独立重算 MuJoCo 和实际链胶囊行后加入必要的区间行；"
        "只读 LP 使用原 17 维速度盒、十步斜坡起点、原容差和冻结前瞻约束。"
        "LP 可行仅表示存在终点，不表示历史斜坡起点或任何实际命令通过。", "",
        "| 模式 | 冷查询起点违反 | 持久查询起点违反 | 持久查询无可行终点 | 持久查询历史动作被拒 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        c = cold["modes"][mode]["counts"]
        w = warm["modes"][mode]["counts"]
        lines.append(
            f"| {mode} | {c.get('frozen_START_CLEARANCE_VIOLATION', 0) + c.get('frozen_START_VELOCITY_VIOLATION', 0)} | "
            f"{w.get('frozen_START_CLEARANCE_VIOLATION', 0) + w.get('frozen_START_VELOCITY_VIOLATION', 0)} | "
            f"{w.get('frozen_NO_FEASIBLE_ENDPOINT', 0)} | "
            f"{w.get('historical_endpoint_rejected_by_new_rows', 0)} |"
        )
    failures = [
        record for report in (cold, warm) for mode in ("baseline", "enabled")
        for record in report["modes"][mode]["frozen_feasibility_records"]
        if record["feasibility"]["status"].startswith("START_")
    ]
    if failures and all(
        record["feasibility"].get("worst_start_source", "").startswith("pcc_interval:")
        for record in failures
    ):
        lines.append(
            "四组所列起点反例的最差行均来自 PCC 区间；该代理残差不能解释为实际链碰撞。"
        )
    lines += [
        "", "逐状态失败、区间 ID、起点残差和 LP 结果保留在两份影子 JSON 中。"
        "这些是旧轨迹的反例，不可引用为新控制器已经失败或已完成闭环。", "",
        "## 预算阶梯与细分后起点", "",
        "固定预算阶梯 31/63/127/255/511 从完整根分区重新查询保存状态。"
        "随后在原生力矩重放中，以 255 点诊断预算重建新区间行与斜坡起点。"
        "高预算只用于辨别原因，不是在线接入配置。", "",
        "| 模式 | 511 点确定代理低于 5 mm | 511 点仍未知 | 旧起点违例经细分消除 | 细分后旧违例仍不可执行 | 细分计算超过 20 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        front = frontier["modes"][mode]
        ref = refined["modes"][mode]
        lines.append(
            f"| {mode} | {front['status_by_budget']['511'].get('PROXY_CLEARANCE_BELOW_GATE', 0)} | "
            f"{front['status_by_budget']['511'].get('UNKNOWN_CROSSES_GATE', 0)} | "
            f"{ref['old_bad_cured_by_diagnostic_refinement']} | "
            f"{ref['old_bad_remains_unexecutable']} | "
            f"{ref['interval_20ms_exceeded']} |"
        )
    lines += [
        "", "代理低于门槛不是实际链碰撞；细分后区间函数与梯度必须重新计算，"
        "且分区切换还可能产生新的起点违例。逐状态预算、失败原因及运行耗时均保留在"
        "预算阶梯和细分起点 JSON 中。", "",
        "最差 PCC 起点行进一步拆为静态区间 h、形变速度、基座反作用、目标漂移"
        "和屏障项，各项和须重组为保存的起点残差：", "",
        "| 模式 | 细分后起点违反且最差区间 h >= 0 | 细分后起点违反且最差区间 h < 0 |",
        "| --- | ---: | ---: |",
    ]
    for mode in ("baseline", "enabled"):
        ref = refined["modes"][mode]
        lines.append(
            f"| {mode} | {ref['start_violations_with_positive_interval_h']} | "
            f"{ref['start_violations_with_negative_interval_h']} |"
        )
    enabled_bad_terms = [
        item["worst_interval_start_terms"] for item in refined["records"]
        if item["mode"] == "enabled"
        and item["feasibility"]["status"] == "START_CLEARANCE_VIOLATION"
        and item["worst_interval_start_terms"] is not None
    ]
    if enabled_bad_terms:
        def mean_rate_mm_s(name: str) -> float:
            return (1000.0 * sum(item[name] for item in enabled_bad_terms)
                    / len(enabled_bad_terms))

        lines.append(
            f"启用组这 {len(enabled_bad_terms)} 条最差行的平均形变速率 "
            f"{mean_rate_mm_s('shape_rate_m_s'):.2f} mm/s、基座反作用 "
            f"{mean_rate_mm_s('base_reaction_rate_m_s'):.2f} mm/s、目标漂移 "
            f"{mean_rate_mm_s('target_drift_m_s'):.2f} mm/s、屏障项 "
            f"{mean_rate_mm_s('barrier_rate_m_s'):.2f} mm/s。"
        )
    lines += [
        "", "启用组静态下界仍达到 5 mm 门槛的起点也可能因旧动作的相对逼近速率"
        "违反新区间 CBF；进一步提高静态查询预算本身不能使这些旧斜坡起点合格。"
        "这不证明新区间模式一定无法找到不同轨迹，也不批准在当前门禁下接入。", "",
        "## 门禁障碍的首次出现", "",
        "511 点诊断中的代理低于门槛状态都有 PCC 曲线中点 witness：该点的"
        " PCC 净空上界小于 5 mm。继续细分不能把同一旧状态变成代理安全；"
        "这仍不等于实际离散链碰撞。下表的 tick 是 50 Hz 规划 tick，"
        "来自每 50 tick 抽样，"
        "只是所测首次出现位置。", "",
        "| 模式 | 五场景初始可行 | 代理低于门槛的 witness 状态 | 细分后旧起点违反 | 其中最差行 h >= 0 | 各场景首次 witness tick | 各场景首次起点违反 tick |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for mode in ("baseline", "enabled"):
        item = blockers[mode]

        def ticks(name: str) -> str:
            values = item[name]
            return ", ".join(f"{scenario[-2:]}:{tick}"
                             for scenario, tick in values.items()) or "无"

        lines.append(
            f"| {mode} | "
            f"{item['initial_proxy_safe_and_frozen_rows_feasible']}/5 | "
            f"{item['sampled_proxy_below_gate_with_point_witness']} | "
            f"{item['sampled_old_start_clearance_violations_after_refinement']} | "
            f"{item['old_start_violations_with_nonnegative_interval_h']} | "
            f"{ticks('first_witnessed_below_tick_by_scenario')} | "
            f"{ticks('first_old_start_violation_tick_by_scenario')} |"
        )
    lines += [
        "", "开启组的静态 witness 未发现低于门槛，但旧起点仍可因相对逼近过快"
        "而失效。要验证不同的控制轨迹，必须先解决在线接入门禁；"
        "旧 trace 的继续重放不能代替新区间闭环。", "",
        "## 冻结留出与真实几何", "",
        f"旧 B.1 独立留出 {heldout['counts']['checked_count']} 例完成新补的 MuJoCo 离散链对照："
        f"代理假安全 {heldout['counts']['empirical_proxy_false_safe']}、"
        f"代理误拒绝 {heldout['counts']['empirical_proxy_false_reject']}。",
        f"新冻结的 5 mm 附近 {near['counts']['checked_count']} 例分布于门槛两侧："
        f"代理确定低于门槛 {near['counts'].get('PROXY_CLEARANCE_BELOW_GATE', 0)}、"
        f"确定达到门槛 {near['counts'].get('PROXY_CLEARANCE_AT_LEAST_GATE', 0)}、"
        f"未知 {near['counts'].get('UNKNOWN_CROSSES_GATE', 0)}；"
        f"独立 MuJoCo 对照中的代理假安全 {near['counts']['empirical_proxy_false_safe']}。",
        "样本由 B.1 区间查询定位到代理门槛附近，冻结后才计算真实几何；"
        "密采样或 B.1 区间中点均不被当作连续真值。", "",
        "## 证据边界与下一动作", "",
        "`interval_well_formed`、完整覆盖、精确模型下 K=1 假设、双精度认证状态、"
        "实际链子空间残差与有限样本包络证据仍分开报告。"
        "当前在线接入门禁未通过，应继续处理持久分区未知、预算与保守性；"
        "不接入新区间 QP，也不引用旧 trace 冒充新控制运行。", "",
    ]
    doc = output_dir / "STAGE2_EVIDENCE.md"
    with doc.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    manifest = {
        "schema": "v6_2_b2_stage2_hash_manifest_v1",
        "status": status,
        "gate_blockers": blockers,
        "sources": {name: {"path": path.as_posix(), "sha256": _sha(path),
                           "bytes": path.stat().st_size}
                    for name, path in sources.items()},
        "generated_document": {"path": doc.as_posix(),
                               "sha256": _sha(doc), "bytes": doc.stat().st_size},
    }
    with (output_dir / "stage2_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("v6_lite/output/v6_2_b2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.root, args.output_dir)
    print(json.dumps({"status": result["status"],
                      "document": result["generated_document"]}, indent=2))


if __name__ == "__main__":
    main()
