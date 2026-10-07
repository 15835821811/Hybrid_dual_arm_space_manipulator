"""Publication-only export: zero physics/geometry/QP/network/model calls."""
from pathlib import Path
import csv
import html
import json
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from v6_4.route_optimizer_protocol import read, write, sha, verify_frozen
from v6_4.route_candidate_evaluator import verify_seal
from v6_4.continuous_route_optimizer import rank_candidates

RUN = Path(__file__).resolve().parent
RELEASE = ROOT / "v6_4/releases" / RUN.name
VIEW = ROOT / "v6_4/visualization" / RUN.name


def export():
    verify_frozen(RUN)
    verify_seal(RUN)
    if RELEASE.exists() or VIEW.exists():
        raise FileExistsError("exclusive publication destination required")
    summary = read(RUN / "summary.json")
    identity = read(RUN / "source_identity.json")
    frozen_source = {}
    for name in ("v6_4/route_optimizer_protocol.py", "v6_4/continuous_route_optimizer.py",
                 "v6_4/route_candidate_evaluator.py", "v6_4/evaluate_route_optimizer.py",
                 "v6_4/tests/test_continuous_route_optimizer.py"):
        if sha(ROOT / name) != identity["source_sha256"][name]:
            raise ValueError("formal producer source changed before publication")
        destination = RELEASE / "frozen_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        frozen_source[name] = sha(destination)
    write(RELEASE / "frozen_source_manifest.json", {"algorithm_producer_commit": summary["algorithm_producer_commit"],
        "source_sha256": frozen_source, "inherited_controller_model_and_safety_sources": "unchanged from recorded fixed base/producer; complete hashes in snapshot/source_identity.json"})
    inventory = []
    for relative, digest in read(RUN / "manifest.json").items():
        source = RUN / relative
        suffix = source.suffix.lower()
        # Reports, budgets, selections and exact bindings are portable. Native
        # states/torques and large timing/interval logs stay in the raw archive.
        copied = suffix in (".json", ".md", ".py", ".csv")
        copied |= relative == "teacher_records.jsonl" or relative.endswith("candidate_registry.jsonl") or relative == "command_receipts.jsonl"
        copied |= relative in ("figures/routes.png", "figures/quality_cost.png")
        if copied:
            destination = RELEASE / "snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        inventory.append({"path": relative, "sha256": digest, "size": source.stat().st_size,
                          "copied": copied, "availability": "portable_copy" if copied else "raw_local_archive_only"})
    shutil.copyfile(RUN / "manifest.json", RELEASE / "raw_manifest.json")
    write(RELEASE / "release_manifest.json", {"schema": "v64_c1_portable_release_v1", "source_run": str(RUN),
        "source_manifest_sha256": sha(RUN / "manifest.json"), "inventory": inventory,
        "algorithm_producer_commit": summary["algorithm_producer_commit"],
        "publication_script_sha256": sha(__file__), "export_utc": datetime.now(timezone.utc).isoformat(),
        "new_physics_steps": 0, "new_geometry_queries": 0, "new_QP_calls": 0,
        "native_replay_archive_complete_in_clone": False,
        "verification_scope": "copied bytes and complete raw omission ledger; not physical replay"})
    for row in inventory:
        if row["copied"] and sha(RELEASE / "snapshot" / row["path"]) != row["sha256"]:
            raise ValueError("export changed copied bytes")
    VIEW.mkdir(parents=True)
    for name in ("routes.png", "quality_cost.png"):
        shutil.copyfile(RUN / "figures" / name, VIEW / name)
    # Failed-prefix quality is isolated in the portable presentation; it never
    # populates complete-task metrics or successful pair means.
    slots = [read(p) for p in sorted((RUN / "actual").glob("*/*/slot.json"))]
    columns = ("task_id", "method", "status", "full_task_success", "original_independent_gates_passed", "clearance_30mm_met", "alias_of_method", "actual_steps")
    metrics = ("I_support", "I_route_key_legacy", "I_full", "L_full", "L_support", "d_support", "base_translation_peak_m", "base_rotation_peak_rad", "torque_saturation_count")
    rows = []; prefixes = []
    for s in slots:
        row = {k: s.get(k) for k in columns}; m = s.get("quality") or {}
        row.update({k: m.get(k) if s["full_task_success"] else None for k in metrics})
        row["quality_scope"] = "complete27s" if s["full_task_success"] else "NO_COMPLETE_QUALITY"
        row["prefix_horizon_s"] = m.get("saved_horizon_s") if not s["full_task_success"] else None
        rows.append(row)
        if not s["full_task_success"] and m:
            prefixes.append({"task_id": s["task_id"], "method": s["method"], "status": s["status"],
                "saved_horizon_s": m["saved_horizon_s"], "prefix_metrics": m, "eligible_for_full_quality_comparison": False})
    with (VIEW / "actual_methods.csv").open("x", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(VIEW / "failed_prefix_quality.json", prefixes)
    write(VIEW / "summary.json", summary)
    search = []
    for task in read(RUN / "plan.json")["tasks"]:
        tid = task["task_id"]
        pool = read(RUN / "planning" / tid / "candidate_registry.json")
        seed_rank, full_rank = rank_candidates(pool[:4]), rank_candidates(pool)
        for preference, key, metric in (("A", "strict", "I_support"), ("B", "B", "L_full")):
            seed, full = seed_rank[key], full_rank[key]
            selected = full_rank["tie_selected"] if preference == "A" else full
            value = lambda row: row["prediction_metrics"][metric] if row else None
            search.append({"task_id": tid, "preference": preference, "objective": metric,
                "seed_best_id": seed["candidate_id"] if seed else None, "seed_best_value": value(seed),
                "full_strict_id": full["candidate_id"] if full else None, "full_strict_value": value(full),
                "strict_gain_vs_seeds": value(seed) - value(full) if seed and full else None,
                "selected_id": selected["candidate_id"] if selected else None,
                "selected_source": selected["source"] if selected else "NO_PLAN",
                "selected_budget_position": int(selected["candidate_id"][1:]) + 1 if selected else None,
                "selected_value": value(selected),
                "new_admissible_capability_vs_seeds": seed is None and full is not None})
    with (VIEW / "search_contribution.csv").open("x", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(search[0])); writer.writeheader(); writer.writerows(search)
    tasks = {p.parent.name: read(p) for p in (RUN / "frozen_tasks").glob("*/task.json")}
    with (VIEW / "teacher_records.jsonl").open("x", encoding="utf8") as stream:
        for line in (RUN / "teacher_records.jsonl").read_text(encoding="utf8").splitlines():
            row = json.loads(line); task = tasks[row["task_id"]]
            row["source_teacher_sha256"] = sha(RUN / "teacher_records.jsonl")
            row["environment"]["scenario"] = task["scenario"]
            row["environment"]["initial_state_identity"] = {
                "qpos": task["initial_qpos"], "qvel": task["initial_qvel"],
                "task_model_contract_sha256": task["model_contract_sha256"]}
            for binding in row["actual_bindings"]:
                q = binding.pop("quality")
                success = binding["status"] == "TASK_COMPLETED" and binding["original_gates"]
                binding["full_quality"] = q if success else None
                binding["failed_prefix_quality"] = q if not success else None
                binding["quality_scope"] = "complete27s" if success else "failed_prefix_or_missing"
            predicted = row.get("prediction_metrics") or {}
            row["preference_condition_prediction"] = bool(row["prediction_eligible"] and
                (row["preference"] == "A" or predicted.get("d_support") is not None and predicted["d_support"] >= .030))
            row["preference_condition_actual"] = bool(row["qualification"] == "validated_execution_example" and any(
                b["full_quality"] and (row["preference"] == "A" or b["full_quality"]["d_support"] >= .030)
                for b in row["actual_bindings"]))
            row["prediction_quality_scope"] = "complete27s" if row["prediction_eligible"] else "failed_prefix_or_missing"
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write(RELEASE / "portable_paths.json", {str(RUN / row["path"]): {
        "available": row["copied"], "path": "snapshot/" + row["path"] if row["copied"] else None,
        "sha256": row["sha256"], "size": row["size"]} for row in inventory})
    portable_report = ["# V6.4-C.1 portable evidence", "",
        "[完整封存原报告](snapshot/REPORT.md) · [机器结论](snapshot/summary.json) · [可读可视化](../../visualization/" + RUN.name + "/index.html)", "",
        "## 本轮实际结论", "",
        "|方法|完整Task与原独立门禁/4|完整且净空≥30mm/4|", "|---|---:|---:|"]
    portable_report += [f"|{m}|{summary['full_task_success_by_method'][m]}|{summary['preferred_clearance_30mm_met_by_method'][m]}|" for m in ("Z0", "G0", "OI", "OC")]
    rescued = sorted(s["task_id"] for s in slots if s["method"] == "OI" and s["full_task_success"] and all(
        not b["full_task_success"] for b in slots if b["task_id"] == s["task_id"] and b["method"] in ("Z0", "G0")))
    unmet = sorted(s["task_id"] for s in slots if s["method"] == "OC" and s["status"] == "NO_PLAN")
    portable_report += ["", f"48共享候选槽中实际评价{summary['continuous_candidates_evaluated']}个非历史连续点；16逻辑方法槽对应{summary['actual_unique_runs']}个唯一actual。所有选择在任何actual开始前封存。",
        "", "A输出优先采用strict+0.001工程并列规则，其中" + str(sum(r["selected_OI_is_seed"] for r in summary["selected_seed_vs_optimized"])) + "/4来自初值。strict小幅改善不冒充已下发收益。相对两基线新增完整执行的任务：" + (", ".join(rescued) or "无") + "。",
        "", "OC在以下任务预算内未达到偏好并返回NO_PLAN：" + (", ".join(unmet) or "无") + "。其他OC的净空、路径与干预取舍见原报告的同任务完整配对表；不宣称Pareto支配。",
        "", "[两偏好的初值最优、全池strict、实际输出、新点收益与预算位置](../../visualization/" + RUN.name + "/search_contribution.csv)。A收益单位rad/s，B收益单位m；无合格初值时收益为null，能力改变另列。",
        "", "训练、神经采样和学习更新均为0；本轮收益来自非学习模型搜索。总体泛化、Diffusion优势、连续时间及硬件安全未建立，部署NOT_MET。", "",
        "本导出保留所有复制原件的原始字节；完整状态/力矩/大日志留在本地原始归档，缺失项逐文件记录SHA/大小。Git克隆的字节核验不等于物理重放。", "",
        "失败行的完整质量为null；已消费前缀质量单独保存在[failed_prefix_quality.json](../../visualization/" + RUN.name + "/failed_prefix_quality.json)，不参与完整质量均值。原报告表中full_success=false行若显示数字，其范围仅为保存的失败前缀。", "",
        "![路线](../../visualization/" + RUN.name + "/routes.png)", "",
        "![质量与成本](../../visualization/" + RUN.name + "/quality_cost.png)", "",
        "轨迹图的失败方法仅绘制到拒绝时刻；完整质量柱图排除失败前缀。原runner自动生成的PCC监视图保留在本地归档，本次发布只增加两张核心图，无视频或PDF。", "",
        "命令/环境/producer、并行调度与0物理步的包装脚本导入修复见snapshot/RUN_NOTES.md和相应收据。优化算法与逐任务预算未改动。", ""]
    (RELEASE / "report.md").write_text("\n".join(portable_report), encoding="utf8")
    _page(summary, rows, prefixes)
    _navigation(summary)
    write(RELEASE / "verification.json", {"passed": True, "copied_count": sum(r["copied"] for r in inventory),
        "omitted_count": sum(not r["copied"] for r in inventory), "release_manifest_sha256": sha(RELEASE / "release_manifest.json"),
        "physical_replay_performed_by_export": False, "new_physics_steps": 0})
    publication_paths = [p for folder in (RELEASE, VIEW) for p in folder.rglob("*") if p.is_file()]
    publication_paths += [ROOT / "README.md", ROOT / "docs/V6_4_C1_VISUALIZATION.md"]
    publication_paths += [ROOT / p for p in read(RELEASE / "navigation_receipt.json")["refreshed_entrypoints"]]
    write(RELEASE / "publication_manifest.json", {"files": {
        p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(publication_paths))},
        "scope": "published copied evidence, both figures, teacher, and current viewer navigation; zero additional physics"})
    print(json.dumps({"release": str(RELEASE), "view": str(VIEW), "copied": sum(r["copied"] for r in inventory)}))


def _page(s, rows, prefixes):
    status = s["full_task_success_by_method"]; clearance = s["preferred_clearance_30mm_met_by_method"]
    def cell(v):
        if v is None: return '<span class="missing">—</span>'
        if isinstance(v, float): return f"{v:.7g}"
        return html.escape(str(v))
    head = ["任务", "方法", "状态", "完整Task", "原独立门禁", "I_support rad/s", "I_key legacy", "I_full", "L_full m", "d_support mm", "30mm", "actual别名"]
    body = []
    for r in rows:
        vals = [r["task_id"], r["method"], r["status"], r["full_task_success"], r["original_independent_gates_passed"],
                r["I_support"], r["I_route_key_legacy"], r["I_full"], r["L_full"], r["d_support"] * 1000 if r["d_support"] is not None else None,
                r["clearance_30mm_met"], r["alias_of_method"]]
        body.append('<tr>' + ''.join('<td>' + cell(v) + '</td>' for v in vals) + '</tr>')
    cards = ''.join(f'<article><span>{m}</span><b>{status[m]}/4</b><small>完整Task与原门禁 · 30mm {clearance[m]}/4</small></article>' for m in ('Z0','G0','OI','OC'))
    paired = []
    for group in (s["improvement_over_zero"], s["improvement_over_geometric_rule"]):
        for p in group:
            m = p["paired_mean_difference"]
            paired.append('<tr>' + ''.join('<td>'+cell(v)+'</td>' for v in (p['method'], p['baseline'], p['paired_complete_count'], m['I_support'], m['L_full'], m['d_support']*1000 if m['d_support'] is not None else None)) + '</tr>')
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V6.4-C.1 连续路线优化</title>
<style>body{{font:15px/1.65 system-ui,sans-serif;margin:0;color:#22384a;background:#f2f6f8}}main{{max-width:1450px;margin:auto;padding:36px}}h1{{font-size:38px;margin:6px 0}}h2{{margin-top:32px}}.eyebrow{{letter-spacing:.14em;color:#177f80;font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}article,.panel{{background:white;border:1px solid #dce5ea;border-radius:14px;padding:20px}}article span,article b,article small{{display:block}}article b{{font-size:36px;color:#177f80}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums;width:100%;font-size:13px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #e0e8ec}}th{{background:#eaf1f5}}img{{width:100%;height:auto;border-radius:12px}}a{{color:#136f8a}}.missing{{color:#999}}.badge{{display:inline-block;background:#fff0cf;padding:4px 12px;border-radius:20px}}code{{background:#e8eff4;padding:2px 5px}}details pre{{white-space:pre-wrap;word-break:break-word}}@media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:18px}}h1{{font-size:28px}}}}</style><main>
<div class="eyebrow">HYBRID DUAL-ARM SPACE MANIPULATOR · V6.4-C.1</div><h1>连续路线优化与两种质量偏好</h1>
<p>4个冻结任务 · 2个新母场景 · 每任务12候选共享池 · 非学习模型搜索</p><span class="badge">部署 NOT_MET</span>
<p>工程完成、完整执行、偏好达标与相对基线收益分别判定。30mm是新增质量要求；未达到不表示碰撞。没有训练或Diffusion采样。</p>
<p><a href="../../releases/{RUN.name}/report.md">证据说明与原报告</a> · <a href="actual_methods.csv">16槽完整质量CSV</a> · <a href="search_contribution.csv">两偏好搜索贡献</a> · <a href="failed_prefix_quality.json">失败前缀另表</a> · <a href="teacher_records.jsonl">场景与资格完整的有限教师</a> · <a href="summary.json">全部机器结论</a></p>
<div class="cards">{cards}</div><section class="panel"><h2>全部方法槽（每方法固定分母4）</h2><p>Z0零参考；G0冻结v2远离12mm；OI偏好A低干预；OC偏好B净空≥30mm后短路径。NO_PLAN不执行fallback。失败行完整质量留空，前缀另存。</p><div class="scroll"><table><thead><tr>{''.join('<th>'+v+'</th>' for v in head)}</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>
<h2>参考与实际路线</h2><img src="routes.png" alt="四新任务参考与实际路线"><p>固定世界投影；虚线为参考、实线为实际；失败轨迹仅到拒绝时刻。相关连续体几何—球净空由原生碰撞对计算，不由图中点距代替。</p>
<h2>成功、质量与计算成本</h2><img src="quality_cost.png" alt="两偏好质量与计算成本"><p>仅完整合格轨迹进入质量柱图。{s['candidate_slots_consumed']}/48候选槽，{s['continuous_candidates_evaluated']}个非历史连续点评价；16逻辑方法槽、{s['actual_unique_runs']}个唯一actual。十步私有预演与独立力矩重放另计。</p>
<section class="panel"><h2>同任务完整成功子集的配对差值</h2><p>优化−基线，缺失不填惩罚常数；原全4任务成功率同时见上表。额外预测成本属于优化器，不作equal-compute优势声明。</p><div class="scroll"><table><thead><tr><th>方法</th><th>基线</th><th>完整配对数</th><th>ΔI_support</th><th>ΔL_full m</th><th>Δd_support mm</th></tr></thead><tbody>{''.join(paired)}</tbody></table></div></section>
<h2>有限结论</h2><p>I_support使用固定前驱与关键区间并集；旧I_key保留，不能改判旧B.3门槛或把旧1.03%与新窗口直接比较。A严格最低与工程tie-selected分开。B无合格计划返回null；预算耗尽不表示全局最优或物理不可行。</p><p>收益如有，来自本轮非学习搜索。四任务是两母场景成对产生；总体泛化、Diffusion优势、硬实时、模型误差鲁棒性、连续时间及硬件安全均未建立。原全身验收50Hz/subdivisions4、robot-target原生500Hz。</p>
<details><summary>查看预算、搜索贡献与全部机器状态</summary><pre>{html.escape(json.dumps(s,indent=2,ensure_ascii=False))}</pre></details>
<p><a href="../execution_aware_route_teacher_20261007_01/index.html">B.3.1历史</a> · <a href="../conditional_route_value_20261007_01/index.html">B.3历史</a> · <a href="../task_anchored_residual_20261007_01/index.html">B.2历史</a> · <a href="../../../v6_lite/visualization/latest/index.html">V6.2历史</a></p></main></html>'''
    (VIEW / "index.html").write_text(text, encoding="utf8")


def _navigation(s):
    latest = f"v6_4/visualization/{RUN.name}/index.html"
    readme = ROOT / "README.md"
    old = readme.read_text(encoding="utf8")
    counts = s["full_task_success_by_method"]; clearance = s["preferred_clearance_30mm_met_by_method"]
    prefix = f'''# Hybrid Dual-Arm Space Manipulator — V6.4-C.1

已实现可调用的连续Cartesian路线优化器。四个新冻结任务来自两个母场景；两偏好共享每任务12候选槽，未训练或采样网络。工程完成不等于优化优越。

|方法|完整Task与原独立门禁/4|完整且净空≥30mm/4|
|---|---:|---:|
''' + ''.join(f"|{m}|{counts[m]}|{clearance[m]}|\n" for m in ('Z0','G0','OI','OC')) + f'''

[当前两图与16槽可读总表]({latest}) · [portable证据说明](v6_4/releases/{RUN.name}/report.md) · [完整封存报告](v6_4/releases/{RUN.name}/snapshot/REPORT.md) · [机器结论](v6_4/releases/{RUN.name}/snapshot/summary.json) · [有限教师](v6_4/visualization/{RUN.name}/teacher_records.jsonl) · [冻结协议](docs/V6_4_C1_PROTOCOL.md)

内部预测{s['prediction_rollouts_started']}，非历史连续点评价{s['continuous_candidates_evaluated']}；最终逻辑方法槽16，唯一actual {s['actual_unique_runs']}。搜索、十步预演、独立重放、几何与QP分别计账。B未满足时返回NO_PLAN，不替换成基线。完整质量配对只用同任务双方完整成功子集。

Producer `{s['algorithm_producer_commit']}`；固定开发基点`{s['base_publication_commit']}`。I_support是新窗口指标，不改旧B.3结论。30mm是质量偏好，不替换原硬安全距离。Diffusion收益、总体泛化、连续时间与硬件安全仍未建立；deployment=`NOT_MET`。

```sh
python -B -X utf8 -m v6_4.continuous_route_optimizer prepare --output v6_4/output/{RUN.name}
python -B -X utf8 -m v6_4.continuous_route_optimizer run-all --run v6_4/output/{RUN.name}
```

上列命令已执行；完整记录、原冻结环境与路径见发布快照。大型原生状态/力矩日志留在本地；clone提供逐文件遗漏账本，字节核验不冒充物理重放。

<details>
<summary>历史：B.3.1及更早发布记录（原结论保留）</summary>

'''
    readme.write_text(prefix + old + "\n</details>\n", encoding="utf8")
    entries = [ROOT / "v6_4/visualization" / name / "index.html" for name in (
        "execution_aware_route_teacher_20261007_01", "execution_aware_media_20261007_01",
        "conditional_route_value_20261007_01", "task_anchored_residual_20261007_01", "web")]
    entries += [ROOT / "v6_lite/visualization/latest/index.html"]
    refreshed = []
    for path in entries:
        if not path.exists(): continue
        import os
        href = os.path.relpath(VIEW / "index.html", path.parent).replace("\\", "/")
        content = path.read_text(encoding="utf8")
        banner = '<div style="padding:12px 22px;background:#dceef0;color:#184957;font:15px system-ui"><a style="color:inherit" href="' + href + '">最新：V6.4-C.1 连续路线优化 · 两图与全部16方法槽 ↗</a>　此页保留历史实验内容与原结论。</div>'
        import re
        content, count = re.subn(r'(<body[^>]*>)', lambda match: match.group(1) + banner, content, count=1, flags=re.I)
        if count:
            path.write_text(content, encoding="utf8"); refreshed.append(path.relative_to(ROOT).as_posix())
    for path in (ROOT / "v6_4/visualization/index.html", ROOT / "v6_lite/visualization/index.html"):
        import os
        href = os.path.relpath(VIEW / "index.html", path.parent).replace("\\", "/")
        path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0;url=' + href + '"><title>V6.4-C.1 当前可视化</title><body><h1>V6.4-C.1 连续路线优化</h1><p><a href="' + href + '">打开两图与全部16方法槽总表</a></p><p>48槽预算，4任务/2母场景；无训练或神经采样；部署NOT_MET。历史实验、视频和原结论通过当前页历史链接访问。</p></body></html>', encoding="utf8")
        refreshed.append(path.relative_to(ROOT).as_posix())
    doc = ROOT / "docs/V6_4_C1_VISUALIZATION.md"
    doc.write_text(f"# V6.4-C.1 当前可视化\n\n[两图与16槽总表](../{latest})。所有当前查看入口已添加C.1导航，历史图/报告/视频保持原结论与原资产，不重渲染196视频。\n\n成功分母各4；失败prefix与NO_PLAN完整质量留空，prefix另表。当前只有两张核心PNG，无新增视频/PDF。\n\n[完整报告](../v6_4/releases/{RUN.name}/snapshot/REPORT.md) · [原件/遗漏清单](../v6_4/releases/{RUN.name}/release_manifest.json)\n\n刷新入口：\n\n" + ''.join('- '+p+'\n' for p in refreshed), encoding="utf8")
    write(RELEASE / "navigation_receipt.json", {"refreshed_entrypoints": refreshed, "new_core_figures": 2,
        "new_videos": 0, "new_PDFs": 0, "old_reports_figures_and_video_assets_modified": False})


if __name__ == "__main__": export()
