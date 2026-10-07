"""Build a static B.3 negative-pilot dashboard from sealed collected evidence.

This standalone delivery tool imports no experiment, simulator, optimizer or
sampler. It adds only its own HTML/table files and dashboard_manifest.json to
an existing directory containing the two verified paired route figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
from urllib.parse import quote


REPORT_SCHEMA = "v64_b3_conditional_route_value_report_v1"
PAIRED_SCHEMA = "v64_b3_paired_metrics_v1"
FIGURE_SCHEMA = "v64_b3_paired_route_figures_v1"
METHODS = ("Z0", "R0", "U0", "D_true", "D_swap")
FIGURES = ("01_pilot_c_plus_routes", "02_pilot_c_minus_routes")
NEW_FILES = ("index.html", "method_table.csv", "method_table.md", "dashboard_manifest.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, relative):
    path = Path(path)
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def escape(value):
    return html.escape(str(value), quote=True)


def link(path):
    return quote(str(path).replace("\\", "/"), safe="/.#:-_")


def number(value, digits=6, factor=1.):
    if value is None:
        return "—"
    value = float(value) * factor
    if not math.isfinite(value):
        raise ValueError("dashboard refuses nonfinite numeric evidence")
    return f"{value:.{digits}f}"


def check_named_artifacts(root, artifacts):
    seen = set()
    for item in artifacts:
        relative = item["path"]
        path = (root / relative).resolve()
        if path in seen or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("missing, duplicate or escaping visualization artifact: " + relative)
        if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("visualization artifact digest/size differs: " + relative)
        seen.add(path)


def validate_sources(source, visualization):
    report, paired = read(source / "report.json"), read(source / "paired_metrics.json")
    plan, decision = read(source / "plan.json"), read(source / "pilot/decision.json")
    if report.get("schema") != REPORT_SCHEMA or paired.get("schema") != PAIRED_SCHEMA:
        raise ValueError("dashboard needs the collected B.3 report schemas")
    if (decision.get("route_value_identifiable") is not False
            or decision.get("training_authorized_by_pilot") is not False
            or report.get("route_value_identifiable") is not False
            or report.get("training_executed") is not False
            or report.get("status") != "ROUTE_VALUE_NOT_IDENTIFIABLE_WITHIN_CURRENT_REPRESENTATION"):
        raise ValueError("this delivery tool is restricted to the complete negative pilot")
    if report.get("deployment") != "NOT_MET" or report.get("research_execution_complete") is not True:
        raise ValueError("terminal research/deployment status is inconsistent")
    if paired["pilot"]["decision"] != decision or report["decision"] != decision:
        raise ValueError("pilot decision differs between collected records")
    if report["plan_sha256"] != sha(source / "plan.json") or paired["plan_sha256"] != report["plan_sha256"]:
        raise ValueError("collected plan identity differs")
    if report["source_identity_sha256"] != sha(source / "source_identity.json"):
        raise ValueError("collected source identity differs")
    for relative, digest in report["sources"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or sha(path) != digest:
            raise ValueError("collected report source differs: " + relative)
    rows = paired["pilot"]["slots"]
    declared = plan["pilot_slots"]
    if len(rows) != 6 or len(declared) != 6 or len({r["slot_id"] for r in rows}) != 6:
        raise ValueError("all six terminal pilot slots are required")
    for row, fixed in zip(rows, declared):
        if (row["slot_id"], row["task_id"], row["method"]) != (fixed["slot_id"], fixed["task_id"], fixed["candidate_name"]):
            raise ValueError("collected slot differs from fixed pilot order")
        if row["full_task_success"] is True and row["safety"]["full_task_and_original_safety_passed"] is not True:
            raise ValueError("complete success label contradicts independent safety eligibility")
    safe = sum(r["full_task_success"] is True for r in rows)
    if report["pilot_full_task_success"] != {"numerator": safe, "denominator": 6}:
        raise ValueError("pilot success counter differs from its six records")
    if paired["costs"] != report["costs"]:
        raise ValueError("collected cost ledgers differ")
    for key in ("teacher_actual_slots", "new_model_ddim_calls", "new_training_runs", "optimizer_updates", "new_TEST_actual_slots"):
        if report["costs"][key] != 0:
            raise ValueError("negative pilot stop contradicts later-stage execution costs")
    statuses = report["independent_test_task_success_by_method"]
    if set(statuses) != set(METHODS) or any(v != "NOT_RUN_PILOT_STOP" for v in statuses.values()):
        raise ValueError("unexecuted formal TEST methods must remain explicitly NOT_RUN")
    if paired["new_independent_TEST"]["success_by_method"] != {m: None for m in METHODS}:
        raise ValueError("unexecuted TEST success counts cannot be fabricated")
    if len(report["five_questions"]) != 5:
        raise ValueError("the report must answer all five requested questions")
    figure_manifest = read(visualization / "manifest.json")
    plot_data = read(visualization / "plot_data.json")
    if figure_manifest.get("schema") != FIGURE_SCHEMA or plot_data.get("schema") != FIGURE_SCHEMA:
        raise ValueError("paired figure schema differs")
    if figure_manifest["source_run_id"] != source.name or plot_data["pilot_decision"] != decision:
        raise ValueError("figures belong to a different study or decision")
    check_named_artifacts(visualization, figure_manifest["artifacts"])
    panels = plot_data["panels"]
    if len(panels) != 2 or {p["id"] for p in panels} != set(FIGURES):
        raise ValueError("exactly the two declared paired route figures are required")
    artifacts = {a["path"] for a in figure_manifest["artifacts"]}
    for panel in panels:
        if panel["png"] != panel["id"] + ".png" or panel["pdf"] != panel["id"] + ".pdf":
            raise ValueError("unexpected paired figure filename")
        if not {panel["png"], panel["pdf"]}.issubset(artifacts):
            raise ValueError("paired figure lacks manifest bindings")
    csv_bytes = (source / "method_table.csv").read_bytes()
    csv_rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    if [(r["slot_id"], r["task_id"], r["method"]) for r in csv_rows] != [(r["slot_id"], r["task_id"], r["method"]) for r in rows]:
        raise ValueError("method CSV differs from six collected slots")
    for tabular, row in zip(csv_rows, rows):
        if any(value != ("" if row[key] is None else str(row[key])) for key, value in tabular.items()):
            raise ValueError("method CSV values differ from paired metrics")
    return report, paired, plan, panels, csv_bytes, csv_rows


def method_markdown(csv_rows):
    columns = ("slot_id", "task_id", "method", "status", "saved_horizon_s", "full_task_success",
               "I_route_rad_s", "I_full_rad_s", "continuum_path_length_m", "route_clearance_m", "full_clearance_m")
    def cell(value):
        return str(value or "—").replace("|", "\\|").replace("\n", " ")
    lines = ["# V6.4-B.3 方法表", "", "以下为固定开发 pilot 的六个实际槽位；净空单位为 m，干预单位为 rad/s。空值未填入完整质量比较。", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in csv_rows]
    lines += ["", "## 正式新 TEST", "", "| 方法 | 状态 | 完整成功数 |", "|---|---|---|"]
    lines += [f"| {method} | NOT_RUN_PILOT_STOP | 未评价 |" for method in METHODS]
    lines += ["", "预声明 pilot 停止后，teacher、训练、新模型采样及正式 TEST 均未运行。NOT_RUN 不等于 0/4 失败。", ""]
    return "\n".join(lines)


STYLE = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#1b2733;font:16px/1.65 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}a{color:#155c8b;text-underline-offset:3px}main{max-width:1280px;margin:auto;padding:30px 24px 70px}header,section{background:white;border:1px solid #dde3e8;border-radius:14px;padding:26px;margin:18px 0;box-shadow:0 2px 12px #152d3a05}h1{font-size:clamp(25px,4vw,38px);line-height:1.25;margin:8px 0 18px}h2{font-size:23px;margin:0 0 14px}h3{font-size:19px;margin:8px 0}p{margin:10px 0}.eyebrow,.muted{color:#617180}.eyebrow{font-weight:700;letter-spacing:.08em;font-size:13px}.stop{background:#fff6df;border-left:5px solid #c48b24;padding:15px 18px;border-radius:6px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0 0}.card{background:#f2f5f8;border-radius:9px;padding:16px}.value{font-size:27px;display:block;font-weight:750}.small{font-size:13px}.tag{padding:3px 9px;border-radius:20px;background:#e9eff5;white-space:nowrap;font-size:12px}.pass{background:#e3f1e9;color:#1d6440}.fail{background:#f8e9e8;color:#8c3330}nav{display:flex;flex-wrap:wrap;gap:18px;margin-top:18px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #e3e8ec;text-align:left;padding:11px 10px;vertical-align:top}th{background:#f1f5f8;color:#344859;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}code{font:13px/1.5 ui-monospace,Consolas,monospace;background:#edf1f4;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{font:12px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:15px;border-radius:8px}figure{margin:22px 0 32px}figure img{display:block;width:100%;height:auto;border:1px solid #e3e8ec;border-radius:8px}figcaption{font-size:14px;color:#526572;margin-top:10px}.questions article{padding:14px 0;border-bottom:1px solid #e3e8ec}.questions article:last-child{border:0}.costs{display:grid;grid-template-columns:1fr 1fr;gap:24px}details{margin:12px 0}summary{cursor:pointer;color:#466578}.identity{font-size:13px;overflow-wrap:anywhere}footer{font-size:13px;color:#617180;margin:20px 4px}@media(max-width:780px){main{padding:14px 10px 40px}header,section{padding:18px}.cards{grid-template-columns:1fr 1fr}.costs{grid-template-columns:1fr}table{font-size:12px}th,td{padding:9px 7px}}"""


def render(report, paired, plan, panels):
    safe = report["pilot_full_task_success"]
    probe = paired["old_checkpoint_condition_probe"]
    table_rows = []
    failure_details = []
    for row in paired["pilot"]["slots"]:
        good = row["full_task_success"] is True
        state = '<span class="tag pass">完整合格</span>' if good else '<span class="tag fail">未完整合格</span>'
        table_rows.append("<tr>" + f"<td><code>{escape(row['slot_id'])}</code><br><span class='small'>{escape(row['task_id'])}</span></td>"
                          + f"<td>{escape(row['method'])}</td><td>{state}<br><span class='small'>{escape(row['status'])}</span></td>"
                          + f"<td class='num'>{number(row['saved_horizon_s'],3)}</td>"
                          + "".join(f"<td class='num'>{number(row[key], digits, factor)}</td>" for key, digits, factor in (
                              ("I_route_rad_s",6,1), ("I_full_rad_s",6,1), ("continuum_path_length_m",4,1),
                              ("route_clearance_m",3,1000), ("full_clearance_m",3,1000))) + "</tr>")
        detail = {key: row.get(key) for key in ("safety", "execution_failure", "evaluation_errors", "metric_unavailable", "failed_prefix_metrics")}
        failure_details.append(f"<details><summary>{escape(row['slot_id'])}：独立门禁与前缀证据</summary><pre>{escape(json.dumps(detail,ensure_ascii=False,indent=2))}</pre></details>")
    figures = []
    for panel in panels:
        figures.append(f"<figure><h3>{'c+ 障碍侧' if panel['side']=='c_plus' else 'c− 障碍侧'} · {escape(panel['task_id'])}</h3>"
                       + f"<a href='{link(panel['png'])}'><img src='{link(panel['png'])}' alt='{escape(panel['task_id'])} 固定路线参考与独立保存实际路径'></a>"
                       + f"<figcaption>{escape(panel['caption'])} <a href='{link(panel['pdf'])}'>PDF</a> · <a href='{link(panel['png'])}'>PNG</a></figcaption></figure>")
    question_html = "".join(f"<article><h3>{i}. {escape(q['question'])}</h3><p>{escape(q['answer'])}</p></article>" for i,q in enumerate(report["five_questions"],1))
    method_labels = {"Z0":"零残差", "R0":"TRAIN 高质量检索", "U0":"TRAIN 高质量经验抽样", "D_true":"正确条件 Diffusion", "D_swap":"错配障碍条件 Diffusion"}
    formal = "".join(f"<tr><td>{method}</td><td>{method_labels[method]}</td><td><code>NOT_RUN_PILOT_STOP</code></td><td>未评价</td></tr>" for method in METHODS)
    cost_labels = {"actual_slots_consumed":"消耗的 pilot 槽位", "actual_runner_entries":"进入 actual runner 的槽位", "slots_with_physics":"发生物理步的槽位", "actual_physics_steps":"actual 物理步", "private_preview_physics_steps":"私有预演物理步", "independent_saved_torque_replay_steps":"独立保存力矩重放步", "preview_calls":"私有预演调用", "native_geometry_query_calls":"执行与证据流程原生几何查询", "additional_route_quality_geometry_queries":"额外路线质量几何查询", "input_precheck_geometry_queries":"输入几何预检查询", "preflight_geometry_binding_check_queries":"旧状态场景绑定预检查询", "old_checkpoint_ddim_calls":"旧权重诊断 DDIM", "teacher_actual_slots":"teacher actual 槽位", "new_model_ddim_calls":"新模型 DDIM", "new_training_runs":"新模型主训练次数", "optimizer_updates":"新模型 optimizer 更新", "new_TEST_actual_slots":"正式新 TEST actual 槽位"}
    cost_rows = "".join(f"<tr><td>{escape(label)}</td><td class='num'>{int(report['costs'][key]):,}</td></tr>" for key,label in cost_labels.items() if key in report["costs"])
    status_keys = ("research_execution_complete", "research_delivery_complete", "route_value_identifiable", "training_executed", "condition_response_observed", "conditional_value_supported_in_pilot", "advantage_over_retrieval", "deployment")
    status_rows = "".join(f"<tr><td><code>{key}</code></td><td>{escape(report.get(key))}</td></tr>" for key in status_keys)
    release = "../../releases/" + quote(report["run_id"], safe="-_")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V6.4-B.3 · 固定配对路线价值</title><style>{STYLE}</style></head>
<body><main><header><div class="eyebrow">V6.4-B.3 / CONDITIONAL ROUTE VALUE</div><h1>障碍换边后的路线价值：固定预算负结果</h1>
<div class="stop"><strong>未建立满足预声明门槛的双侧路线价值。</strong><br>六个开发 pilot 槽位完成后按计划停止；teacher、主训练和正式新 TEST 未运行。有限负结果不否定 Diffusion。</div>
<div class="cards"><div class="card"><span class="muted small">pilot 完整 Task + 原独立安全门禁</span><span class="value">{safe['numerator']} / {safe['denominator']}</span></div><div class="card"><span class="muted small">旧权重诊断</span><span class="value">{probe['ddim_calls']} DDIM</span></div><div class="card"><span class="muted small">新训练 / 正式 TEST</span><span class="value">未运行</span></div><div class="card"><span class="muted small">部署</span><span class="value">NOT_MET</span></div></div>
<nav><a href="#routes">两侧路线图</a><a href="#pilot">六槽方法表</a><a href="#formal">五组新 TEST</a><a href="#answers">五问回答</a><a href="#cost">成本与证据</a></nav></header>
<section id="routes"><h2>两张固定配对路线图</h2><p>仅一个开发母场景、两个障碍侧。参考与保存的实际路径分别标示，拒绝前缀止于保存末态；图中投影不替代全连续体碰撞检查。</p>{''.join(figures)}</section>
<section id="pilot"><h2>六槽实际记录与相关障碍净空</h2><p>仅完整 27 s、Task 与原独立门禁全部通过的记录进入完整质量比较。破折号表示未取得相应完整质量值，失败前缀没有补入完整均值。</p>
<div class="scroll"><table><thead><tr><th>槽位 / 任务</th><th>候选</th><th>终态</th><th>保存时长/s</th><th>I_route<br>rad/s</th><th>全任务干预<br>rad/s</th><th>连续体路径/m</th><th>相关球窗口<br>最小净空/mm</th><th>相关球全时域<br>最小净空/mm</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p class="small">I_route = sqrt(mean(task_avoidance_intervention²))，使用完整预声明关键区间的闭区间 50 Hz 样本。原字段是 17 维 QP 名义速度与实际选中速度差的范数；名义值经过原速度边界裁剪，两个源向量未保存。相关净空仅计算连续体与移动球的几何对，不用 rigid-target 主导的整机最小值替代。</p>
<p><a href="method_table.csv">原始方法表 CSV</a> · <a href="method_table.md">方法表 Markdown</a></p>{''.join(failure_details)}</section>
<section id="formal"><h2>正式新 TEST：五组均未运行</h2><p>P1 未通过路线价值门槛，因此 P2/P3/P4 按停止规则保持未运行。没有把预留四个任务记为 0/4 失败，也没有依据未运行对照宣布检索或 Diffusion 胜出。</p><div class="scroll"><table><thead><tr><th>组别</th><th>预声明输入</th><th>状态</th><th>完整任务成功数</th></tr></thead><tbody>{formal}</tbody></table></div></section>
<section><h2>P0：条件响应与条件价值分别报告</h2><p>旧 B.2 update250 权重在固定噪声下仅交换障碍条件：{probe['same_noise_pairs_observed']} / 16 对观察到条件响应，原始幅值合法 {probe['amplitude_legal_raw_outputs']} / 32。没有裁剪、替换、重训或新增物理执行。</p><p><strong>响应变化不证明路线选择正确。</strong>公开旧 TEST 仅用于诊断；本轮没有新模型的真条件、错条件、无条件经验抽样或检索 TEST 比较。</p></section>
<section id="answers" class="questions"><h2>五个问题的直接回答</h2>{question_html}</section>
<section id="cost"><h2>成本分账与结论边界</h2><div class="costs"><div><table><thead><tr><th>成本项</th><th>实测计数</th></tr></thead><tbody>{cost_rows}</tbody></table><p class="small">actual、私有预演、独立保存力矩重放分别计费；几何查询不算独立实验。异常前缀成本采用实际 ledger，不用缺失报告字段推定零成本。</p></div><div><table><thead><tr><th>独立状态</th><th>报告值</th></tr></thead><tbody>{status_rows}</tbody></table><p>20 ms 规划、2 ms 物理、27 s 任务、20 mm 残差范围与原控制/安全标准保持。墙钟 20 ms 不作研究门禁；无硬实时、连续时间或模型失配保证。</p><p>robot-target 原生 500 Hz；whole-body 50 Hz 边界加配置空间 subdivisions4，不声称所有几何对均在 500 Hz 验证。</p></div></div>
<details><summary>完整成本字典与冻结指标定义</summary><pre>{escape(json.dumps({'costs':report['costs'],'I_route_scope':report['I_route_scope']},ensure_ascii=False,indent=2))}</pre></details></section>
<section><h2>证据与历史版本</h2><p><a href="{release}/report.md">可移植研究报告</a> · <a href="{release}/snapshot/report.json">机器可读 report</a> · <a href="{release}/snapshot/paired_metrics.json">配对指标</a> · <a href="dashboard_manifest.json">本页与方法表哈希</a> · <a href="manifest.json">原两图 manifest</a> · <a href="plot_data.json">两图数据</a></p><p><a href="../task_anchored_residual_20261007_01/index.html">B.2 历史可视化</a> · <a href="../../../v6_lite/visualization/latest/index.html">V6.2 历史可视化</a>。历史媒体保留各自实验范围。</p>
<p class="identity">本轮实际 producer：<code>{escape(report['source_producer_commit'])}</code><br>B.2 发布基点：<code>{escape(report['published_B2_base'])}</code><br>B.2 原算法 producer：<code>{escape(report['B2_algorithm_producer'])}</code><br>plan SHA-256：<code>{escape(report['plan_sha256'])}</code><br>source identity SHA-256：<code>{escape(report['source_identity_sha256'])}</code></p></section>
<footer>{escape(report['run_id'])} · 静态证据页，读取已保存结果生成，新增物理步 / DDIM / optimizer / 几何查询均为 0。</footer></main></body></html>
"""


def build(source, visualization):
    source, visualization = Path(source).resolve(), Path(visualization).resolve()
    if source == visualization or source in visualization.parents:
        raise ValueError("dashboard output must be separate from source evidence")
    if not visualization.is_dir() or any((visualization / name).exists() for name in NEW_FILES):
        raise FileExistsError("need existing two-figure directory with no prior dashboard payload")
    report, paired, plan, panels, csv_bytes, csv_rows = validate_sources(source, visualization)
    input_names = ("report.json", "paired_metrics.json", "plan.json", "source_identity.json", "method_table.csv",
                   "pilot/decision.json", "pilot/quality_records.json", "old_checkpoint_probe/report.json")
    inputs = [record(source / name, name) for name in input_names]
    visual_inputs = [record(visualization / name, name) for name in ("manifest.json", "plot_data.json")]
    page, markdown = render(report, paired, plan, panels), method_markdown(csv_rows)
    for path, data in ((visualization / "index.html", page.encode("utf8")),
                       (visualization / "method_table.csv", csv_bytes),
                       (visualization / "method_table.md", markdown.encode("utf8"))):
        with path.open("xb") as stream:
            stream.write(data)
    check_named_artifacts(source, inputs)
    check_named_artifacts(visualization, visual_inputs)
    manifest = {"schema":"v64_b3_conditional_dashboard_v1", "source_run_id":source.name,
                "generator":record(Path(__file__).resolve(), Path(__file__).name),
                "inputs":inputs, "visualization_inputs":visual_inputs,
                "artifacts":[record(visualization / name, name) for name in NEW_FILES[:-1]],
                "figure_manifest_overwritten":False, "source_evidence_modified":False,
                "physics_steps":0, "DDIM_calls":0, "optimizer_updates":0, "geometry_queries":0,
                "formal_TEST_status":"NOT_RUN_PILOT_STOP", "deployment":"NOT_MET"}
    with (visualization / "dashboard_manifest.json").open("x", encoding="utf8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.visualization)
    print(json.dumps({"status":"COMPLETED", "source_run_id":result["source_run_id"],
                      "artifacts":len(result["artifacts"]), "physics_steps":0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
