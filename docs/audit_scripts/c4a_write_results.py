"""Render complete DEV accounting as Markdown without any new experiment work."""
import argparse
import csv
import json
from pathlib import Path


def csv_read(path):
    with path.open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def num(value, digits=6, scale=1):
    return "N/A" if value in (None, "", "None") else f"{float(value)*scale:.{digits}f}"


def table(headers, rows):
    return "|" + "|".join(headers) + "|\n|" + "|".join("---" for _ in headers) + "|\n" + "".join("|" + "|".join(str(x).replace("|", "/") for x in row) + "|\n" for row in rows) + "\n"


def hit(value):
    d = json.loads(value)
    return str(d["slot"]) if d["status"] == "HIT" else d["status"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summary = json.loads((a.results / "summary.json").read_text(encoding="utf8"))
    if summary["status"] != "COMPLETED":
        raise RuntimeError("Final result document requires completed execution")
    ss = csv_read(a.results / "dev_streams.csv")
    ee = csv_read(a.results / "dev_endpoints.csv")
    cc = csv_read(a.results / "dev_candidates.csv")
    ag, b = summary["strategies"], summary["budget"]
    lines = ["# C4-A 两个新 DEV Task 的受控验证结果\n\n",
             "## 范围与证据状态\n\n",
             "2026-10-10；状态 COMPLETED / 实际物理执行已完成。固定两个独立母场景的 DEV：`c4a_dev0_plus` (seed 2028300010) 与 `c4a_dev1_minus` (seed 2028404739)。P0=Rule8，P1=Diffusion8-Replacement，P2=Diffusion8-Rule-Preserving Portfolio。六个搜索流全部封存后才执行最终 Actual；原27秒任务和五门禁未改变。\n\n",
             "完整机器表：`v6_4/c4a_evidence/results_01/{summary.json,dev_candidates.csv,dev_streams.csv,dev_endpoints.csv}`。原始日志与轨迹：`E:/v64c4a/v6_4/c4a_evidence/dev_01/`。紧凑发布包：`v6_4/releases/c4a_architecture_audit_20261010_01/`，保留全部原始文件 SHA 和未复制的大轨迹文件路径。\n\n",
             "只有两个母场景；这些是逐任务先导结果，不报告显著性或总体泛化。相同 Actual 的严格 alias 不增加独立样本。\n\n",
             "## 先评价任务能力与安全\n\n"]
    lines.append(table(["策略", "完整 Actual + 五门禁", "A", "B", "B端点实际B30", "NO_PLAN", "NOT_RUN", "TOOL_ERROR"],
                       [[k, f'{v["full_actual_passed"]}/4', f'{v["A_passed"]}/2', f'{v["B_passed"]}/2', f'{v["B_actual_B30"]}/2', v["NO_PLAN"], v["NOT_RUN"], v["TOOL_ERROR"]] for k, v in ag.items()]))
    lines.append("分母包括失败和 NO_PLAN。下表是逐逻辑端点结果，五门禁保留原字段；是否完整达成任务另列，B30不是五门禁的替代品。\n\n")
    lines.append(table(["Task", "策略/偏好", "状态", "完整+五门禁", "五门禁", "独立执行", "alias"],
                       [[e["task_id"], e["strategy"]+"/"+e["preference"], e["status"], e["full_task_and_five_gates"], e["five_gates"], e["unique_run"], e["alias_of_slot"] or "—"] for e in ee]))
    lines.append("## 再评价轨迹质量\n\n")
    lines.append("新DEV没有R12参考：**near_R12=N/A_NOT_RUN_BUDGET**。没有借用其他Task的R12，也没有追加物理预算。辅助 near_P0 在结果产生前声明：A要求I≤P0+0.001 rad/s且L≤P0+0.005m；B要求L≤P0+0.005m且d≥0.030m。near_P0不冒充near_R12。\n\n")
    lines.append(table(["策略", "near_P0 A", "near_P0 B", "near_R12"], [[k, f'{v["near_P0_A"]}/2', f'{v["near_P0_B"]}/2', v["near_R12"]] for k, v in ag.items()]))
    lines.append(table(["Task", "策略/偏好", "I_support rad/s", "L_full m", "d_support mm", "基座平移峰 mm", "基座旋转峰 rad", "near_P0"],
                       [[e["task_id"], e["strategy"]+"/"+e["preference"], num(e["actual_I_support"],9), num(e["actual_L_full"],9), num(e["actual_d_support"],6,1000), num(e["base_translation_peak_m"],6,1000), num(e["base_rotation_peak_rad"],9), e["near_P0"]] for e in ee]))
    lines.append("## 候选覆盖、拒绝和来源\n\n")
    lines.append(table(["Task", "策略", "合法/生成raw", "完整名义合格/槽", "B30候选", "首次完整槽", "首次B30槽", "缓存"],
                       [[s["task_id"], s["strategy"], s["raw_legal"]+"/"+s["raw_generated"] if s["strategy"]!="P0" else "N/A(无模型)", s["full_nominal_qualified"]+"/"+s["candidate_slots"], s["B30_candidates"], hit(s["first_complete"]), hit(s["first_B30"]), s["cache_hits"]] for s in ss]))
    lines.append("首次命中位置使用1基消耗槽号；未命中应保留RIGHT_CENSORED，不用0或删除任务。生成提案与物理预测分开计数。下表列出全部Diffusion raw，包括原样拒绝。P1/P2为相同噪声配对重复，不能当成8个独立噪声。\n\n")
    lines.append(table(["Task", "策略/槽", "族", "raw合法", "拒绝原因", "名义完整", "B30", "d_support mm"],
                       [[c["task_id"], c["strategy"]+"/"+c["slot_1based"], c["family"], c["raw_legal"], c["rejection"] or "—", c["prediction_admissible"], c["B30"], num(c["d_support"],6,1000)] for c in cc if c["source"]=="diffusion"]))
    lines.append(table(["Task", "策略/偏好", "最终候选", "来源分类", "原始来源", "完整谱系"],
                       [[e["task_id"], e["strategy"]+"/"+e["preference"], e["selected_candidate_id"], e["source_attribution"], e["selected_origin"], e["selected_lineage"]] for e in ee]))
    lines.append("## 最后评价规划成本\n\n")
    lines.append("每行成本只计一次共享A/B搜索。cold service含初值准备及搜索服务，outer process还含进程启动/导入/验证；不得混用口径。计时为此CPU单线程单次串行运行，微小差异不构成加速证据。\n\n")
    lines.append(table(["Task", "策略", "槽", "真实预测", "cold service s", "cold outer s", "主预测积分", "private preview", "几何查询", "QP", "独立重放"],
                       [[s["task_id"], s["strategy"], s["candidate_slots"], s["prediction_calls"], num(s["cold_service_s"],3), num(s["cold_outer_process_s"],3), s["prediction_physics_steps"], s["private_preview_physics_steps"], s["native_geometry_query_calls"], s["qp_solve_calls"], s["independent_saved_torque_replay_steps"]] for s in ss]))
    lines.append(table(["策略", "总cold service s", "总cold outer s", "主预测积分", "private preview", "几何查询"],
                       [[k, num(v["cold_service_s"],3), num(v["cold_outer_process_s"],3), v["prediction_physics_steps"], v["private_preview_physics_steps"], v["native_geometry_query_calls"]] for k,v in ag.items()]))
    lines.append("模型加载、条件编码和两次DDIM生成的独立计时保留在 `dev_streams.csv` 的 `initializer_setup` 与原 `planning_cost.json`。raw资格检验、提案构造和缓存查找包含在optimizer/cold墙钟中，但没有独立计时字段，记为INCLUDED_NOT_SEPARATELY_TIMED，不能填成零。每候选物理成本保留在 `dev_candidates.csv`。非法raw少做一次物理预测造成的时长下降属于工作量减少，不是已证明的有效搜索加速。\n\n")
    lines.append("## 预算实账与独立证据\n\n")
    lines.append("```json\n"+json.dumps(b,ensure_ascii=False,indent=2)+"\n```\n\n")
    actual_keys=("actual_physics_steps", "private_preview_physics_steps", "native_geometry_query_calls", "independent_saved_torque_replay_steps", "qp_solve_calls")
    totals={k:sum(int(e[k] or 0) for e in ee) for k in actual_keys}
    lines.append("全部Actual计费合计（alias没有新物理证据）：\n\n```json\n"+json.dumps(totals,ensure_ascii=False,indent=2)+"\n```\n\n")
    lines.append("离线账另列：阶段A为128个诊断输出+16次复现=144次冻结D采样和16次S forward；数值单测另有2次冻结D确定性采样。DEV为8次DDIM（4组唯一Task/条件噪声在P1/P2重复），每次20网络forward。所有阶段训练更新为0；没有新C.3 TEST物理运行。\n\n")
    lines.append("## 验证和复现边界\n\n")
    lines.append("137个不同测试节点全部通过；计入复用测试共176次通过执行。新增核心数值15项、Portfolio9项、DEV合同4项。最早一次旧测试收集因namespace相对导入失败，没有执行测试；改为 `--import-mode=importlib` 后72项通过。原失败回执与修正命令完整保留，详见 `review_01/test_coverage.json` 和 `docs/audit_receipts/c4a/`。\n\n")
    lines.append("最终 `final_integrity.json` 验证原C.3冻结字节、控制/执行配置、DEV输入与源码身份、候选封印、独立初态、P0/P2四规则一致、P1/P2 raw一致及全部Actual封印。该核验没有再次运行物理或模型。P0/P1/R12默认调度和输出的回归为固定mock环境下对旧实现的字节比较；真实DEV性能仅来自本次实际日志。\n\n")
    lines.append("物理producer为 `dd834783847bfbcc4f8ec624b7bce0d22777b6de`；最终交付commit仅追加报告/证据/报告脚本，不改变已封存物理源码。确切命令、UTC时间、退出码、stdout/stderr及SHA见回执。新视频未生成。解释和下一阶段门禁见 `V6_4_C4A_DECISION.md`。\n")
    with a.output.open("x",encoding="utf8",newline="\n") as f:
        f.write("".join(lines))
    print(json.dumps(dict(status="PASS",output=str(a.output),endpoints=len(ee),candidate_rows=len(cc),streams=len(ss))))


if __name__ == "__main__":
    main()
