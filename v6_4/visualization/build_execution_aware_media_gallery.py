"""Build an append-only B3.1 media supplement from saved-state outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_PUBLICATION = "e67f8cfee583fb3d42fad69b8372fbb8dd0e8d16"
RUN = "execution_aware_route_teacher_20261007_01"
SLOTS = [f"EA_{n:02d}" for n in range(28)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path):
    path = Path(path)
    return {"path": path.resolve().relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def write(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


def build(output):
    plot_path = output / "figures/plots_final.json"
    if not plot_path.exists():
        plot_path = output / "figures/plots.json"
    videos, plots = read(output / "videos.json"), read(plot_path)
    poster_path = output / "route_posters/poster_manifest.json"
    poster_manifest = read(poster_path)
    posters = poster_manifest["route_posters"]
    if set(posters) != set(SLOTS):
        raise ValueError("Gallery requires route-window posters for all fixed 28 slots")
    vr, pr = videos["records"], plots["records"]
    if [r["slot_id"] for r in vr] != SLOTS or [r["slot_id"] for r in pr] != SLOTS:
        raise ValueError("Gallery requires the complete fixed 28-slot cohort")
    rows = []
    for v, p in zip(vr, pr):
        if v["task_id"] != p["task_id"] or v["mode"] != p["mode"]:
            raise ValueError("Video/plot task and mode differ")
        row = {key: p[key] for key in ("slot_id", "task_id", "mode", "source_slot", "source_role",
                                      "figures", "csv", "summary")}
        row.update({"videos": v["videos"], "preview": v["preview"],
                    "focus_preview": v["focus_preview"],
                    "reference_version": v["reference_version"]})
        rows.append(row)
    data = {"schema": "v64_b31_media_gallery_v1", "base_research_publication_commit": BASE_PUBLICATION,
            "records": rows, "task_figures": plots["task_figures"],
            "all_tasks_figures": plots.get("all_tasks_figures", {}),
            "video_count": sum(len(r["videos"]) for r in rows),
            "source_videos": record(output / "videos.json"),
            "source_plots": record(plot_path),
            "route_posters": posters, "source_route_posters": record(poster_path),
            "physics_steps_executed": 0, "new_acceptance": False}
    write(output / "gallery_data.json", json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    table = []
    md = ["# V6.4-B.3.1 全套保存状态可视化", "",
          "覆盖四任务 × 七参考的全部 28 槽：196 个视频、末端轨迹与误差、姿态、基座漂移及保存的净空诊断。视频读取保存状态并调用 mj_forward，不重跑闭环；轨迹/误差图使用原实际状态绑定的 fresh 运动学及真正消费的参考。", "",
          "[交互式目录](index.html) · [视频来源](videos.json) · [图表与数值来源](figures/plots.json) · [原研究图表](../"+RUN+"/index.html)", "",
          "执行参考误差和独立 Task/base 曲线偏差分开显示。中间绕行导致的 base 偏差不等同于 Task 失败；28/28 原 Task 与门禁结论保持，部署状态仍为 NOT_MET。", "",
          "![全部四任务与七参考的数值总览]("+plots["all_tasks_figures"]["metric_matrix"]+")", "",
          "[EA_00 五视角视频]("+rows[0]["videos"]["five_view_grid"]+") · [EA_00 连续体侧视视频]("+rows[0]["videos"]["continuum_focus"]+") · [EA_00 末端轨迹]("+rows[0]["figures"]["trajectory"]+") · [EA_00 跟踪误差]("+rows[0]["figures"]["tracking"]+")", "",
          "![连续体侧视路线窗口预览；完整28槽均在下表]("+posters[rows[0]["slot_id"]]["continuum_focus"]+")", "",
          "| 槽位 | 任务 / 参考 | 原来源 | 五视角 | 连续体侧视 | 末端轨迹 | 跟踪误差 | 净空 / 基座 | 控制诊断 |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        sid, task, mode = row["slot_id"], row["task_id"], row["mode"]
        links = [("五视角", row["videos"]["five_view_grid"]),
                 ("连续体侧视", row["videos"]["continuum_focus"]),
                 ("末端轨迹", row["figures"]["trajectory"]),
                 ("跟踪误差", row["figures"]["tracking"]),
                 ("净空 / 基座", row["figures"]["safety_base"]),
                 ("控制诊断", row["figures"]["control_diagnostics"])]
        cells = "".join(f'<td><a href="{escape(path, quote=True)}">{label}</a></td>' for label, path in links)
        table.append(f'<tr><th>{sid}</th><td>{escape(task)}<br>{escape(mode)}</td><td>{escape(row["source_slot"])}<br>{escape(row["source_role"])}</td>{cells}</tr>')
        md.append(f'| {sid} | {task} / {mode} | {row["source_slot"]} | '+" | ".join(f"[{label}]({path})" for label, path in links)+" |")
    initial = rows[0]
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", payload).replace("__TABLE__", "\n".join(table))
    page = page.replace("__GRID__", initial["videos"]["five_view_grid"]).replace("__FOCUS__", initial["videos"]["continuum_focus"])
    page = page.replace("__PREVIEW__", posters[initial["slot_id"]]["five_view_grid"]).replace("__FOCUS_PREVIEW__", posters[initial["slot_id"]]["continuum_focus"])
    page = page.replace("__TRAJECTORY__", initial["figures"]["trajectory"]).replace("__TRACKING__", initial["figures"]["tracking"]).replace("__SAFETY__", initial["figures"]["safety_base"])
    page = page.replace("__CONTROL__", initial["figures"]["control_diagnostics"])
    page = page.replace("__MATRIX__", plots["all_tasks_figures"]["metric_matrix"])
    if plot_path.name != "plots.json":
        page = page.replace('href="figures/plots.json"', 'href="figures/plots_final.json"')
    write(output / "index.html", page)
    write(output / "README.md", ("\n".join(md)+"\n").replace("(figures/plots.json)", "(figures/"+plot_path.name+")"))
    files = [p for p in sorted(output.rglob("*")) if p.is_file()]
    manifest = {"schema": "v64_b31_media_supplement_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "base_research_publication_commit": BASE_PUBLICATION,
                "source_manifest": record(ROOT / "v6_4/output" / RUN / "manifest.json"),
                "generator": record(__file__), "record_count": 28,
                "video_count": data["video_count"], "source_research_artifacts_modified": False,
                "physics_steps_executed": 0, "QP_solves": 0, "training_updates": 0, "model_samples": 0,
                "excluded_self_referential_files": ["media_manifest.json", "validation.json"],
                "payload": {p.relative_to(output).as_posix(): {"sha256": record(p)["sha256"], "bytes": p.stat().st_size} for p in files}}
    write(output / "media_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return {"status": "BUILT", "slots": 28, "videos": data["video_count"], "payload_files": len(files)}


TEMPLATE = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6.4-B.3.1 · 五视角、连续体侧视与末端误差</title>
<style>
:root{color-scheme:light;--ink:#17283a;--muted:#526479;--line:#d9e2eb;--blue:#175ca5}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1600px;margin:auto;padding:32px 24px 60px}h1{font-size:30px;line-height:1.3;margin:0 0 12px}h2{font-size:23px;margin:0 0 14px}h3{font-size:17px;margin:0 0 8px}p{margin:8px 0}a{color:var(--blue)}.lead{max-width:1120px;color:var(--muted)}.stats{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}.stat{background:#e5edf7;border-radius:7px;padding:7px 13px}.card{padding:22px;background:white;border:1px solid var(--line);border-radius:10px;margin:20px 0}.controls{display:flex;align-items:end;gap:16px;flex-wrap:wrap;margin:14px 0}.controls label{display:flex;flex-direction:column;font-size:13px;font-weight:600;gap:4px}select,button{font:inherit;border:1px solid #aebfd0;border-radius:6px;background:white;color:var(--ink);padding:8px 12px}button{cursor:pointer}.pair{display:grid;grid-template-columns:3fr 2fr;gap:18px;align-items:start}video{width:100%;background:#0a0e18;border-radius:6px}img{width:100%;height:auto;display:block}.caption{font-size:13px;color:var(--muted);margin:8px 0 16px}.provenance{font-size:14px;background:#f0f5fa;padding:10px 14px;border-radius:6px;overflow-wrap:anywhere}.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}.wide{grid-column:1/-1}details{margin-top:15px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:14px;font-size:13px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}thead{background:#edf3f8}tbody th{white-space:nowrap}.taskfig{display:grid;grid-template-columns:1fr;gap:16px}.foot{color:var(--muted);font-size:13px}@media(max-width:1000px){.pair,.plots,.taskfig{grid-template-columns:1fr}.wide{grid-column:auto}main{padding:20px 12px}.card{padding:16px}h1{font-size:25px}}
</style></head><body><main>
<h1>V6.4-B.3.1 · 全套保存状态可视化</h1>
<p class="lead">四个固定开发任务、七种参考，全部 28 组结果同步展示五视角、连续体侧视、双臂末端轨迹、位置与姿态误差、基座漂移及保存的净空诊断。六条旧 B.3 原件复用与 22 条新增执行保留各自来源。</p>
<div class="stats"><span class="stat">28 / 28 原 Task + 门禁通过</span><span class="stat">196 个视频 · 每条完整 27 s</span><span class="stat">15 fps · 精确保存末帧</span><span class="stat">新增物理 / QP / 训练 / 采样：0</span></div>
<p><a href="README.md">GitHub 可读目录 / 下载链接</a> · <a href="../execution_aware_route_teacher_20261007_01/index.html">原两张研究图、28 候选与有限教师</a> · <a href="../../releases/execution_aware_route_teacher_20261007_01/report.md">研究报告</a> · <a href="media_manifest.json">媒体清单</a> · <a href="validation.json">验证收据</a></p>
<section class="card"><h2>选择任务和参考</h2>
<div class="controls"><label>任务<select id="task"></select></label><label>参考<select id="mode"></select></label><label>左侧视角<select id="view"><option value="five_view_grid">五视角组合</option><option value="overview">总览</option><option value="front">正面</option><option value="side">侧面</option><option value="top">俯视</option><option value="iso">等轴</option></select></label><button id="sync" type="button">对齐两段视频时间</button></div>
<p id="provenance" class="provenance"></p>
<div class="pair"><div><h3 id="left-title">五视角组合</h3><video id="grid" controls preload="metadata" poster="__PREVIEW__" src="__GRID__"></video><p class="caption"><a id="grid-link" href="__GRID__">打开 / 下载当前视角视频</a>。五视角分别为总览、正面、侧面、俯视、等轴。</p></div><div><h3>连续体侧视</h3><video id="focus" controls preload="metadata" poster="__FOCUS_PREVIEW__" src="__FOCUS__"></video><p class="caption"><a id="focus-link" href="__FOCUS__">打开 / 下载连续体侧视视频</a>。独立近景相机，刚性臂半透明；世界坐标 RGB = XYZ。</p></div></div>
<p class="caption">绿色：按保存时钟求值的解析残差参考；红色：独立 Task/base 曲线；青色：保存实际末端轨迹。视频只选择已有状态并调用 mj_forward，不积分、不插值、不续接。呈现帧率为 15 fps，源状态间隔 2 ms；编码包含 t=0 与精确末状态，片长约 27.067 s。默认海报选取冻结路线窗口的中点附近帧（t≈10.2 s）；原视频中点海报仍保留。部署状态保持 NOT_MET。</p>
</section>
<section class="card"><h2>当前槽位：轨迹与误差</h2><div class="plots"><div><h3>双臂末端轨迹</h3><a id="trajectory-link" href="__TRAJECTORY__"><img id="trajectory" src="__TRAJECTORY__" alt="当前槽位双臂末端轨迹"></a></div><div><h3>位置与姿态误差</h3><a id="tracking-link" href="__TRACKING__"><img id="tracking" src="__TRACKING__" alt="当前槽位末端跟踪误差"></a></div><div class="wide"><h3>已有净空诊断与基座漂移</h3><a id="safety_base-link" href="__SAFETY__"><img id="safety_base" src="__SAFETY__" alt="当前槽位净空与基座漂移"></a></div><div class="wide"><h3>保存的命令、力矩与区间诊断</h3><a id="control_diagnostics-link" href="__CONTROL__"><img id="control_diagnostics" src="__CONTROL__" alt="当前槽位控制与力矩诊断"></a></div></div>
<p class="caption">执行参考误差使用真正消费的控制边界参考；独立 Task/base 曲线偏差单列。合法中间绕行会改变 base 偏差，该诊断不代替 Task 锚点与原安全门禁。阴影标记冻结路线区间。净空沿用原查询范围与删失标记，不作连续时间安全证明。</p>
<p><a id="native-csv" href="figures/plots.json">下载状态抽样数据 CSV（20 ms）</a> · <a id="consumed-csv" href="figures/plots.json">下载消费参考数据 CSV</a> · <a id="control-csv" href="figures/plots.json">下载控制诊断 CSV</a> · <a href="figures/plots.json">全部图表、数值摘要与来源</a></p><details><summary>当前槽位数值摘要</summary><pre id="summary"></pre></details></section>
<section class="card"><h2>当前任务：七参考完整比较</h2><div id="task-figures" class="taskfig"></div><p class="caption">固定七参考全部保留，不筛选最好结果。汇总值来自原保存数组；质量教师与学习收益结论见原研究报告。</p></section>
<section class="card"><h2>全部四任务 × 七参考：数值总览</h2><a href="__MATRIX__"><img src="__MATRIX__" loading="lazy" alt="全部28槽跟踪和基座诊断汇总"></a><p class="caption">完整固定开发集的数值比较；原安全与Task结论、有限质量教师结论分别保留。</p></section>
<section class="card"><h2>原研究结论与来源</h2><p>本页是在已发布研究提交 <code>e67f8cf</code> 上新增的媒体补充。原实验目录、两图仪表板、portable release、run03 独立审查及其清单不修改；当前入口随后指向本页。</p><div class="plots"><a href="../execution_aware_route_teacher_20261007_01/fig_reference_actual.png"><img src="../execution_aware_route_teacher_20261007_01/fig_reference_actual.png" loading="lazy" alt="原参考与实际响应研究图"></a><a href="../execution_aware_route_teacher_20261007_01/fig_route_quality.png"><img src="../execution_aware_route_teacher_20261007_01/fig_route_quality.png" loading="lazy" alt="原路线质量研究图"></a></div><p class="caption">有限 teacher 较最佳常量低 3.27%、较冻结几何规则低 1.03%；同幅值 v2 没有一致优势，Diffusion 收益未建立。该结论及部署 NOT_MET 保持。</p></section>
<section class="card"><h2>全部 28 槽：直接链接</h2><p class="caption">GitHub 页面可通过 README 与下表查看图片、打开或下载 MP4。交互目录不依赖外部库，使用本地嵌入数据。</p><div class="scroll"><table><thead><tr><th>槽位</th><th>任务 / 参考</th><th>原来源</th><th>五视角</th><th>连续体侧视</th><th>末端轨迹</th><th>跟踪误差</th><th>净空 / 基座</th><th>控制诊断</th></tr></thead><tbody>__TABLE__</tbody></table></div></section>
<p class="foot">保存状态渲染与数值后处理是本次可视化成本，不计作新增实际闭环、独立实验或安全验收。原研究完整证据与媒体补充分别记账。</p>
</main><script id="gallery-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('gallery-data').textContent);
const task=document.getElementById('task'),mode=document.getElementById('mode'),view=document.getElementById('view');
const grid=document.getElementById('grid'),focus=document.getElementById('focus');
const labels={b3_mother_00_c_plus:'43 mm · c+',b3_mother_00_c_minus:'43 mm · c−',b31_mother_00_d055_c_plus:'55 mm · c+',b31_mother_00_d055_c_minus:'55 mm · c−'};
for(const id of [...new Set(data.records.map(r=>r.task_id))]){const option=document.createElement('option');option.value=id;option.textContent=labels[id]||id;task.append(option)}
function modes(){const old=mode.value;mode.replaceChildren();for(const row of data.records.filter(r=>r.task_id===task.value)){const option=document.createElement('option');option.value=row.mode;option.textContent=row.mode;mode.append(option)}if([...mode.options].some(o=>o.value===old))mode.value=old;update()}
function media(el,src,poster){el.pause();el.src=src;el.poster=poster;el.load()}
function update(){const row=data.records.find(r=>r.task_id===task.value&&r.mode===mode.value);media(grid,row.videos[view.value],view.value==='five_view_grid'?data.route_posters[row.slot_id].five_view_grid:'');media(focus,row.videos.continuum_focus,data.route_posters[row.slot_id].continuum_focus);document.getElementById('left-title').textContent=view.selectedOptions[0].textContent;document.getElementById('grid-link').href=row.videos[view.value];document.getElementById('focus-link').href=row.videos.continuum_focus;document.getElementById('provenance').textContent=`${row.slot_id} · ${row.task_id} · ${row.mode} · ${row.reference_version} · 原来源 ${row.source_slot} / ${row.source_role}`;for(const key of ['trajectory','tracking','safety_base','control_diagnostics']){document.getElementById(key).src=row.figures[key];document.getElementById(key+'-link').href=row.figures[key]}document.getElementById('native-csv').href=row.csv.native;document.getElementById('consumed-csv').href=row.csv.consumed;document.getElementById('control-csv').href=row.csv.control;document.getElementById('summary').textContent=JSON.stringify(row.summary,null,2);const box=document.getElementById('task-figures');box.replaceChildren();const group=data.task_figures.find(r=>r.task_id===row.task_id);if(group){for(const [name,path] of Object.entries(group.figures)){if(typeof path!=='string'||!path.endsWith('.png'))continue;const a=document.createElement('a'),img=document.createElement('img');a.href=path;img.src=path;img.alt=name;img.loading='lazy';a.append(img);box.append(a)}}history.replaceState(null,'','#'+row.slot_id)}
task.addEventListener('change',modes);mode.addEventListener('change',update);view.addEventListener('change',()=>{const row=data.records.find(r=>r.task_id===task.value&&r.mode===mode.value);const time=grid.currentTime,playing=!grid.paused;grid.addEventListener('loadedmetadata',()=>{grid.currentTime=time;if(playing)grid.play().catch(()=>{})},{once:true});media(grid,row.videos[view.value],view.value==='five_view_grid'?data.route_posters[row.slot_id].five_view_grid:'');document.getElementById('left-title').textContent=view.selectedOptions[0].textContent;document.getElementById('grid-link').href=row.videos[view.value]});document.getElementById('sync').addEventListener('click',()=>{focus.currentTime=grid.currentTime;if(!grid.paused)focus.play().catch(()=>{})});const selected=data.records.find(r=>r.slot_id===location.hash.slice(1));if(selected)task.value=selected.task_id;modes();if(selected){mode.value=selected.mode;update()}
</script></body></html>
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False))
