"""Saved-file B.3.1 dashboard, integrity audit and exclusive local seal."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote, urlsplit


FIGURES = ("fig_reference_actual.png", "fig_route_quality.png")
GATES = ("task_requirements", "execution_contract", "independent_interval", "native_geometry", "reference_binding")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def stamp(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}


def write(path, data):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def build(source, visual):
    source, visual = Path(source).resolve(), Path(visual).resolve()
    report = read(source/"report.json")
    matrix = read(source/"quality_matrix.json")
    require(len(matrix["rows"]) == 28, "Dashboard requires all 28 fixed rows")
    require(not visual.exists(), "Exclusive dashboard output already exists")
    visual.mkdir(parents=True)
    copied = []
    for name in (*FIGURES, "all_candidates.csv", "teacher_records.json"):
        original = source/"figures"/name if name in FIGURES else source/name
        shutil.copyfile(original, visual/name)
        copied.append({"source": str(original), **stamp(original), "destination": name})
    data = {"schema": "v64_b31_dashboard_data_v1", "run_id": source.name,
        "source_producer_commit": report["source_producer_commit"],
        "total_slots": 28, "complete_safe_slots": report["complete_safe_slots"],
        "quality_comparable_slots": report["quality_comparable_slots"],
        "reused_slots": 6, "new_slots": 22,
        "policy_comparisons": report["policy_comparisons"],
        "conditional_selection_potential": report["conditional_selection_potential"],
        "new_costs": report["new_costs"], "rows": matrix["rows"],
        "deployment": "NOT_MET", "development_only": True,
        "training_runs": 0, "model_samples": 0}
    write(visual/"dashboard_data.json", data)
    columns = ("slot_id", "task_id", "mode", "status", "quality_stage_status", "eligible", "I_route_rad_s",
        "continuum_path_length_m", "route_clearance_m", "source_role")
    labels = ("Slot", "Task", "Mode", "Actual status", "Quality stage", "Quality eligible", "I_route rad/s",
        "Full path m", "Route sphere clearance m", "Source")
    def text(value):
        return html.escape("unavailable" if value is None else f"{value:.9g}" if isinstance(value, float) else str(value))
    headers = "".join("<th>"+label+"</th>" for label in labels)
    body = "".join("<tr>"+"".join("<td>"+text(row.get(key))+"</td>" for key in columns)+"</tr>" for row in matrix["rows"])
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6.4-B.3.1 控制器感知参考与有限教师</title><style>
body{{margin:0;background:#f4f6f8;color:#172335;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:32px}}h1{{line-height:1.3}}a{{color:#0a6084}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card,section{{background:white;padding:22px;border:1px solid #dae1e7;border-radius:10px;margin:18px 0}}.card strong{{display:block;font-size:27px}}img{{width:100%;height:auto}}.table{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dce4e9;text-align:left;white-space:nowrap}}th{{background:#eef3f6}}code{{font-size:13px}}.scope{{border-left:4px solid #b47716;padding:14px;background:#fff8e9}}
</style></head><body><main><p>HYBRID DUAL-ARM SPACE MANIPULATOR · V6.4-B.3.1</p>
<h1>控制器感知参考与有限质量教师</h1>
<p>固定四个开发任务、七个参考模式。提前平台参考与原单峰参考在原执行和安全门禁下比较。</p>
<div class="cards"><div class="card">完整任务与原安全门禁<strong>{report['complete_safe_slots']} / 28</strong></div><div class="card">完整质量可比<strong>{report['quality_comparable_slots']} / 28</strong></div><div class="card">严格复用旧证据<strong>6</strong></div><div class="card">新增固定槽位<strong>22</strong></div><div class="card">训练 / 模型采样<strong>0 / 0</strong></div><div class="card">部署<strong>NOT_MET</strong></div></div>
<p class="scope">一个母场景上的有限开发研究；失败留在分母中，失败前缀不参加成功路径均值。I_route 是全部约束共同作用的17维指标，不能解释为单球贡献。此处不支持 Diffusion 优势、独立泛化、硬实时或连续时间安全结论。原 B.3 六槽的负结果保持不变。</p>
<p><a href="../../releases/execution_aware_route_teacher_20261007_01/report.md">研究报告</a> · <a href="all_candidates.csv">全部候选 CSV</a> · <a href="teacher_records.json">有限教师与全部对照</a> · <a href="dashboard_data.json">结构化质量与成本</a> · <a href="../../../README.md">分支说明</a></p>
<section><h2>参考与真实执行响应</h2><a href="{FIGURES[0]}"><img src="{FIGURES[0]}" alt="四任务全部候选的reference-base、actual-base及同任务actual-zero差异"></a><p>参考解析采样与保存的实际状态分别标注；缺测和失败不会补齐为完整轨迹。</p></section>
<section><h2>路线质量及代价冲突</h2><a href="{FIGURES[1]}"><img src="{FIGURES[1]}" alt="全部固定候选的17维干预、路径、相关球净空和响应"></a><p>图中的端点响应不能代替全臂碰撞验证。完整质量必须经过 Task、执行合同、区间重算、原生几何和参考消费绑定。</p></section>
<section><h2>全部28个固定槽位</h2><div class="table"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div></section>
<section><h2>证据与复验</h2><p>20 ms规划、2 ms物理、27 s任务与20 mm权限保持。墙钟只作研究诊断。actual、private preview、保存力矩replay、几何查询分别计费。两张PNG的刷新不新增物理、几何、训练、采样或PDF/video。</p><p>当前发布为保留原字节的轻量快照；省略的原始NPZ和大流逐项记录SHA与大小，克隆不包含全部可重放原始证据。</p><p><a href="visualization_manifest.json">可视化清单</a> · <a href="../../releases/execution_aware_route_teacher_20261007_01/release_manifest.json">发布清单</a> · <a href="../../releases/execution_aware_route_teacher_20261007_01/release_verification.json">便携校验</a> · <a href="../conditional_route_value_20261007_01/index.html">B.3历史</a> · <a href="../task_anchored_residual_20261007_01/index.html">B.2历史</a></p></section>
<footer>v6.4-b3-1-execution-aware-route-teacher · producer <code>{report['source_producer_commit']}</code></footer>
</main></body></html>'''
    (visual/"index.html").write_text(document, encoding="utf8", newline="\n")
    inventory = {p.name: stamp(p) for p in sorted(visual.iterdir()) if p.is_file()}
    write(visual/"visualization_manifest.json", {"schema": "v64_b31_visualization_manifest_v1",
        "source_producer_commit": report["source_producer_commit"], "copied_inputs": copied,
        "artifacts": inventory, "new_physics_steps": 0, "new_geometry_queries": 0,
        "new_training_runs": 0, "new_model_samples": 0, "new_video_or_PDF": 0})
    return {"status": "PASS", "visualization": str(visual), "figures": 2, "rows": 28}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.paths=[]; self.rows=0
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.rows += 1
        for key, value in attrs:
            if key in ("href", "src") and value: self.paths.append(value)


def audit(source, repository, visual, review, *, allow_pending_release=False):
    source, repository, visual, review = map(lambda p: Path(p).resolve(), (source, repository, visual, review))
    inputs = {}
    def bind(p, expected=None):
        p=Path(p).resolve(); row=stamp(p)
        require(expected is None or row["sha256"] == expected, "Identity mismatch: "+str(p))
        inputs[str(p)] = row
    plan, identity, report, matrix = (read(source/name) for name in ("plan.json", "source_identity.json", "report.json", "quality_matrix.json"))
    require(report["source_producer_commit"] == identity["algorithm_producer_commit"], "Report actual producer mismatch")
    require(report["postprocessing_source_sha256"] == stamp(repository/"v6_4/evaluate_execution_aware_teacher.py")["sha256"], "Report reducer source mismatch")
    for name, digest in identity["source_sha256"].items(): bind(repository/name, digest)
    for name, digest in identity["protected_artifacts"].items(): bind(name, digest)
    require(report["research_protocol_execution_complete"] and report["deployment"] == "NOT_MET", "Terminal scope mismatch")
    require(report["total_slots"] == 28 and report["reused_slots"] == 6 and report["new_slots"] == 22, "Budget mismatch")
    require(report["training_executed"] is False and report["model_sampling_executed"] is False, "Forbidden learning")
    require(report["old_B3_decision_rewritten"] is False and report["continuous_time_certified"] is False, "Scope claim mismatch")
    rows={r["slot_id"]:r for r in matrix["rows"]}; require(len(rows)==28, "Missing matrix row")
    actual=eligible=full_safe=0; totals={key:0 for key in ("actual_physics_steps", "private_preview_physics_steps", "independent_saved_torque_replay_steps", "native_geometry_query_calls", "additional_route_quality_geometry_queries", "qp_solve_calls", "preview_calls")}
    for slot in plan["slots"]:
        path=source/"slots"/slot["slot_id"]/"slot_result.json"; bind(path); r=read(path)
        require(r["frozen_slot"]==slot, "Frozen slot changed")
        for name,digest in r["evidence_bindings"].items(): bind(name,digest)
        if slot["old_source_slot"]:
            require(r["new_cost"]["actual_physics_steps"]==0, "Reuse counted as new actual")
        else: actual += 1
        q=r["quality"]; qualified=q.get("quality_label_eligible") is True
        full_safe += int(r["source_result"]["full_task_success"] is True)
        require(rows[slot["slot_id"]]["eligible"]==qualified, "Quality row eligibility mismatch")
        if qualified:
            require(r["source_result"]["actual_steps"]==13500 and r["full_task_success"], "Incomplete qualified trajectory")
            require(all(q["safety"]["gates"].get(key) is True for key in GATES), "Original gate missing")
            require(abs(rows[slot["slot_id"]]["I_route_rad_s"]-q["full_metrics"]["I_route_rad_s"])<1e-12, "I_route changed")
            eligible += 1
        for key in totals: totals[key] += int(r["new_cost"].get(key,0))
    require(actual==22 and eligible==report["quality_comparable_slots"], "Slot totals mismatch")
    require(full_safe==report["complete_safe_slots"], "Actual safety totals mismatch")
    require(totals["actual_physics_steps"]<=297000, "Actual physics budget exceeded")
    require(all(report["new_costs"][k]==v for k,v in totals.items()), "Cost totals mismatch")
    with (source/"all_candidates.csv").open(encoding="utf8",newline="") as f: table=list(csv.DictReader(f))
    require(len(table)==28 and {r["slot_id"] for r in table}==set(rows), "CSV omits fixed slots")
    for row in table:
        expected=rows[row["slot_id"]]
        for key,value in row.items():
            require(key in expected, "CSV contains an unbound field: "+key)
            source_value=expected[key]
            require(value == ("" if source_value is None else str(source_value)), "CSV differs from matrix: "+key)
    manifest=read(visual/"visualization_manifest.json")
    for row in manifest["copied_inputs"]:
        expected={key:row[key] for key in ("sha256","bytes")}
        require(stamp(row["source"])==expected, "Dashboard copied source changed")
        require(stamp(visual/row["destination"])==expected, "Dashboard copied destination differs")
    for name,row in manifest["artifacts"].items():
        require(stamp(visual/name)==row, "Visual bytes changed: "+name)
    from PIL import Image
    image_checks=[]
    for name in FIGURES:
        with Image.open(visual/name) as picture:
            picture.verify()
        with Image.open(visual/name) as picture:
            require(picture.width>=2000 and picture.height>=1400, "Undersized figure")
            image_checks.append({"name":name,"width":picture.width,"height":picture.height})
    page=Links(); page.feed((visual/"index.html").read_text(encoding="utf8"))
    require(page.rows==29, "HTML does not contain all 28 rows")
    deferred=[]
    future_release=repository/"v6_4/releases/execution_aware_route_teacher_20261007_01"
    pending_names={future_release/name for name in ("report.md","release_manifest.json","release_verification.json")}
    for value in page.paths:
        parsed=urlsplit(value)
        require(not parsed.scheme and not parsed.netloc, "Unexpected external dashboard dependency")
        if parsed.path:
            target=(visual/unquote(parsed.path)).resolve()
            if not target.exists() and allow_pending_release and target in pending_names:
                deferred.append(value)
            else:
                require(target.exists(), "Dashboard link missing: "+value)
    dashboard=read(visual/"dashboard_data.json")
    for key in ("source_producer_commit","complete_safe_slots","quality_comparable_slots","policy_comparisons","conditional_selection_potential","new_costs"):
        require(dashboard[key]==report[key], "Dashboard report projection differs: "+key)
    require(dashboard["rows"]==matrix["rows"], "Dashboard matrix differs")
    verdict=read(review/"verdict.json"); require(verdict.get("blocking_issues")==[], "Independent review blocks delivery")
    review_inputs=read(review/"inputs.json")
    require(review_inputs.get("schema")=="v64_b31_independent_result_review_inputs_v1", "Wrong review input schema")
    for value in (review_inputs,verdict):
        require(value.get("run_id")==source.name and value.get("source_producer_commit")==identity["algorithm_producer_commit"], "Review run/producer mismatch")
    require(verdict.get("input_manifest_sha256")==stamp(review/"inputs.json")["sha256"], "Review verdict is not bound to current inputs")
    seen=set()
    for row in review_inputs["input_files"]:
        original=Path(row["original_path"]).resolve()
        require(str(original) not in seen, "Duplicate review input")
        seen.add(str(original))
        expected={key:row[key] for key in ("sha256","bytes")}
        snapshot=(review/row["snapshot_path"]).resolve() if row.get("snapshot_path") else None
        if snapshot is not None:
            require(snapshot.is_relative_to(review) and stamp(snapshot)==expected, "Reviewed snapshot changed")
        if original==source/"report.json":
            require(snapshot is not None, "Report requires an exact reviewed snapshot")
            before=read(snapshot); after=dict(report)
            for field in ("research_delivery_complete","independent_review_status"):
                before.pop(field,None); after.pop(field,None)
            require(before==after, "Scientific report payload changed after independent review")
        else:
            require(stamp(original)==expected, "Reviewed evidence changed: "+str(original))
        bind(snapshot or original)
    required_review={str(source/name) for name in ("report.json","quality_matrix.json","teacher_records.json","all_candidates.csv","final_conclusions.json","plan.json","source_identity.json","figures/plot_manifest.json")}
    require(required_review.issubset(seen), "Independent review omits critical current evidence")
    for name in ("prompt.md","response.md","inputs.json","verdict.json","handoff.md"): bind(review/name)
    for name in ("report.json","REPORT.md","quality_matrix.json","teacher_records.json","all_candidates.csv","final_conclusions.json","execution_complete.json","figures/plot_manifest.json"):
        bind(source/name)
    for p in visual.rglob("*"):
        if p.is_file(): bind(p)
    return {"schema":"v64_b31_preseal_checks_v1","passed":True,"allow_pending_delivery":False,
        "report_sha256":stamp(source/"report.json")["sha256"],"fixed_slots":28,"new_slots":22,
        "complete_quality_slots":eligible,"cost_totals":totals,"PNG_checks":image_checks,
        "HTML_rows":28,"local_links_checked":len(page.paths),
        "expected_release_links_deferred_until_export":deferred,
        "browser_render_check":"NOT_VERIFIED_TOOL_POLICY","input_bindings":inputs,
        "new_physics_steps":0,"new_geometry_queries":0,"new_training_runs":0,"new_model_samples":0}


def seal(source, repository, visual, review):
    source, repository, visual, review = map(lambda p: Path(p).resolve(), (source, repository, visual, review))
    require(not any((source/n).exists() for n in ("manifest.json","verification.json")), "Exclusive seal exists")
    checks=audit(source,repository,visual,review,allow_pending_release=True)
    require(read(source/"report.json")["research_delivery_complete"] is True, "Delivery remains pending")
    write(source/"preseal_checks.json",checks)
    frozen=read(source/"source_identity.json"); external={}
    for name,row in checks["input_bindings"].items():
        path=Path(name)
        if not path.is_relative_to(source): external[name]=row
    for relative in ("README.md",".gitattributes","MANIFEST.md","findings.md",
        "paper/EXPERIMENT_PLAN.md","paper/EXPERIMENT_TRACKER.md","docs/V6_4_B31_VISUALIZATION.md",
        "docs/V6_2_LATEST_VISUALIZATION.md","v6_lite/README.md","v6_lite/visualization/index.html",
        "v6_4/visualization/index.html","v6_4/evaluate_execution_aware_teacher.py",
        "v6_4/plot_execution_aware_teacher.py","v6_4/export_execution_aware_release.py",
        "v6_4/deliver_execution_aware_teacher.py"):
        p=repository/relative; external[str(p)]=stamp(p)
    for p in review.rglob("*"):
        if p.is_file(): external[str(p)]=stamp(p)
    payload={p.relative_to(source).as_posix():stamp(p) for p in sorted(source.rglob("*")) if p.is_file()}
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repository).decode().strip()
    manifest={"schema":"v64_b31_final_local_delivery_manifest_v1","status":"FINAL",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_producer_commit":frozen["algorithm_producer_commit"],"algorithm_producer_commit":frozen["algorithm_producer_commit"],
        "base_publication_commit":frozen["base_publication_commit"],"development_git_head":head,"delivery_git_head":head,
        "payload":payload,"external":external,"excluded_self_referential_controls":["manifest.json","verification.json"],
        "new_physics_steps":0,"new_geometry_queries":0,"new_training_runs":0,"new_model_samples":0}
    write(source/"manifest.json",manifest)
    for name,row in payload.items(): require(stamp(source/name)==row,"Payload changed during seal")
    for name,row in external.items(): require(stamp(name)==row,"External input changed during seal")
    result={"schema":"v64_b31_final_local_delivery_verification_v1","status":"PASS",
        "manifest_sha256":stamp(source/"manifest.json")["sha256"],"verified_payload_files":len(payload),
        "verified_external_files":len(external),"mismatches":[],"fixed_28_slots_verified":True,
        "independent_review_blocking_issues":[],"browser_render_check":"NOT_VERIFIED_TOOL_POLICY",
        "new_physics_steps":0,"new_geometry_queries":0,"new_training_runs":0,"new_model_samples":0}
    write(source/"verification.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("build","audit","seal"))
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--visualization",required=True,type=Path)
    parser.add_argument("--repository",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review",type=Path)
    args=parser.parse_args()
    if args.command=="build": result=build(args.source,args.visualization)
    else:
        require(args.review is not None,"Review directory required")
        result=(audit if args.command=="audit" else seal)(args.source,args.repository,args.visualization,args.review)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
