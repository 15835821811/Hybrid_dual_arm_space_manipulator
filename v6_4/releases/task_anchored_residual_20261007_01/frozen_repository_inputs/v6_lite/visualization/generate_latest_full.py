"""Build the complete current research visualization from immutable saved evidence.

This does not run a controller, repeat physics, or reclassify prior acceptance.
Historical visualization folders remain intact; ``latest`` is the current entry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "v6_lite/output/runs"
DEFAULT_OUTPUT = ROOT / "v6_lite/visualization/latest"
DEFAULT_DOCUMENT = ROOT / "docs/V6_2_LATEST_VISUALIZATION.md"
SCHEMA = "v6_2_latest_visualization_bundle_v1"
ACCEPTANCE_PIN = "379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0"
INERTIA_PIN = "ee98249e6ba936278e005943dc942724e57fb5ef2f59d2fe226b0e101d89158e"
VELOCITY_PIN = "1da995a2b611635db5dd05129dceedc9adf27983e28aa8edae08eec6f846dda0"
CATEGORIES = ["nominal", "conservatism", "shape", "velocity", "inertia", "wall"]
TITLES = {"nominal": "名义研究与五场景回放", "conservatism": "区间代理保守性",
          "shape": "形状与包络诊断", "velocity": "2 倍目标速度压力实验",
          "inertia": "惯量模型与执行敏感性", "wall": "计算性能与墙钟部署"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def portable(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def record(path: Path) -> dict:
    return {"path": portable(path), "sha256": sha(path), "bytes": path.stat().st_size}


def convert(value):
    if isinstance(value, Path):
        return portable(value)
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def manifest_path(base: Path, name: str) -> Path:
    posix = PurePosixPath(name.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or ":" in name:
        raise ValueError("nonportable/escaping evidence path: " + name)
    path = base.joinpath(*posix.parts).resolve()
    path.relative_to(base.resolve())
    return path


def bound_input(directory: str, pin: str, count: int) -> tuple[dict, list[dict]]:
    base = RUNS / directory
    path = base / "manifest.json"
    if sha(path) != pin:
        raise ValueError("source manifest changed: " + directory)
    manifest = read(path)
    if len(manifest) != count:
        raise ValueError("unexpected evidence file coverage: " + directory)
    refs = [record(path)]
    for name, expected in manifest.items():
        p = manifest_path(base, name)
        digest = expected if isinstance(expected, str) else expected["sha256"]
        if sha(p) != digest:
            raise ValueError("source raw file changed: " + name)
        refs.append(record(p))
    return {"manifest_record": refs[0], "verified_file_count": count}, refs


def load_inputs() -> tuple[dict, dict, list[dict]]:
    acceptance, a_refs = bound_input("research_acceptance_01", ACCEPTANCE_PIN, 46)
    inertia, i_refs = bound_input("research_inertia_shadow_resumed_01", INERTIA_PIN, 146)
    velocity, v_refs = bound_input("research_velocity_stress_01", VELOCITY_PIN, 34)
    acceptance["source_commit"] = "9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c"
    inertia["source_commit"] = "19ad85c7db81a2e254fd176077b6aae39333a439"
    report = read(RUNS / "research_acceptance_01/report.json")
    r = read(RUNS / "research_inertia_shadow_resumed_01/report.json")
    if not (report["passed"] and report["complete"] and r["complete"] and r["evidence_valid"]):
        raise ValueError("required nominal or finite inertia evidence is incomplete")
    delivery = report["algorithm_simulation"]["delivery"]
    execution = report["algorithm_simulation"]["execution"]
    if (delivery["passed_count"], execution["passed_count"]) != (25, 11):
        raise ValueError("wrong research acceptance profile")
    metrics = read(RUNS / "research_acceptance_01/simulation/v6_lite_metrics.json")
    if len(metrics["scenarios"]) != 5 or metrics["run_config"]["dispatch_clock_policy"] != "research_simulation":
        raise ValueError("source is not the current five-scene non-real-time research run")
    for scene in metrics["scenarios"]:
        # Original JSON is immutable and contains the producer's Windows path.
        # Rebase only in memory so a GitHub clone can render on another host.
        sid = scene["scenario"]["scenario_id"]
        p = RUNS / "research_acceptance_01/simulation/traces" / (sid + ".npz")
        if sha(p) != scene["trace"]["sha256"]:
            raise ValueError("current nominal trace does not match original metadata: " + sid)
        scene["trace"]["path"] = str(p)
    return metrics, {"acceptance": acceptance, "inertia": inertia, "velocity": velocity}, a_refs + i_refs + v_refs


def source_records() -> list[dict]:
    names = ["generate_latest_full.py", "latest_nominal.py", "latest_saved_renderer.py", "latest_studies.py"]
    return [record(Path(__file__).parent / name) for name in names]


def _source_refs(values: list) -> list[dict]:
    out = []
    for value in values:
        if isinstance(value, (str, Path)):
            p = Path(value)
            out.append(record(p if p.is_absolute() else ROOT / p))
        else:
            p = ROOT / value["path"]
            if "sha256" in value and sha(p) != value["sha256"]:
                raise ValueError("helper source changed: " + value["path"])
            out.append(record(p))
    return out


def _local(root_relative: str, output: Path) -> str:
    return (ROOT / root_relative).relative_to(output).as_posix()


def panels_for(nominal: dict, studies: dict, output: Path) -> list[dict]:
    panels = []
    names = {
        "error_curves.png": "五场景六面板跟踪误差与基座漂移",
        "safety_clearance_summary.png": "五场景离散间隙",
        "full_control_timing.png": "全部规划、发布及力矩计时",
        "execution_contract.png": "QP 候选、实际命令与斜坡合同",
        "actual_actions.png": "17 维实际命令与 67 路执行力矩",
        "base_pose_drift.gif": "基座平移与姿态漂移动画",
    }
    for name in nominal["artifacts"]:
        p = ROOT / name
        if p.suffix.lower() not in (".png", ".gif", ".svg") or "preview" in p.name:
            continue
        title = names.get(p.name, p.stem.replace("_", " "))
        category = "wall" if "timing" in p.name else "nominal"
        panels.append({"path": name, "category": category, "title": title,
                       "caption": "最新名义非实时研究；保存数据的有限仿真观察。"})
    for panel in studies.get("panels", []):
        panels.append(panel)
    if not studies.get("panels"):
        for name in studies["artifacts"]:
            p = ROOT / name
            if p.suffix.lower() in (".png", ".svg"):
                category = next((c for c in CATEGORIES[1:] if c in p.stem), "inertia")
                panels.append({"path": name, "category": category, "title": p.stem.replace("_", " "),
                               "caption": "冻结输入的有限研究诊断；失败与中断样本保留。"})
    for panel in panels:
        panel["url"] = _local(panel["path"], output)
    return panels


def write_html(output: Path, data: dict) -> Path:
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html_text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V6.2 最新研究结果</title>
<style>
:root{font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#18263b;background:#f3f6fb;line-height:1.55}
body{margin:0}header{padding:40px max(24px,calc((100vw - 1240px)/2));background:#13273f;color:#fff}
header h1{font-size:32px;margin:0 0 8px}header p{margin:8px 0;color:#d5e1f1;max-width:1000px}
main{max-width:1240px;margin:auto;padding:24px}.statuses{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.status{padding:16px;border-radius:10px;background:#fff;border:1px solid #d8e1ed}.status strong{display:block;font-size:21px;margin:6px 0}
.pass{color:#197356}.fail{color:#b64922}.limited{color:#70521a}nav{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}
button,select{padding:9px 14px;font:inherit;border:1px solid #bdcddd;border-radius:7px;background:#fff;cursor:pointer}
button.active{background:#1d547e;color:#fff;border-color:#1d547e}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
article{background:#fff;padding:18px;border-radius:10px;border:1px solid #d8e1ed}article h3{margin:0 0 10px;font-size:18px}
article img{width:100%;height:auto;display:block}article p{font-size:13px;color:#506078}article a{color:#1d547e}
video{width:100%;background:#13273f;border-radius:8px}#replay{margin:24px 0;background:#fff;padding:20px;border-radius:10px;border:1px solid #d8e1ed}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}details{background:#fff;padding:18px;margin:20px 0;border-radius:10px}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px}.note{font-size:13px;color:#506078}
@media(max-width:800px){.statuses,.grid{grid-template-columns:1fr}header{padding:28px 20px}header h1{font-size:26px}main{padding:16px}}
</style>
<header><h1>V6.2 最新研究结果</h1><p>名义五场景 · 保守性与形状诊断 · 2 倍速度压力实验 · 15 组惯量敏感性</p>
<p>仿真规划 20 ms／物理步 2 ms。墙钟耗时单列性能，失败结果完整展示；离散仿真与开环诊断不建立硬实时或硬件安全保证。</p></header>
<main><section class="statuses"><div class="status">名义研究验收<strong class="pass">PASSED</strong>25 项功能／11 项合同／区间独立重算</div>
<div class="status">2 倍速度压力实验<strong class="fail">FAILED · INCOMPLETE</strong>5 场全尝试，2 场完成、3 场拒绝</div>
<div class="status">惯量模型研究<strong class="limited">DIAGNOSTIC COMPLETE</strong>15 × 27 s，同力矩开环敏感性</div>
<div class="status">墙钟部署<strong class="fail">NOT_MET</strong>管理员支线暂缓；硬件与硬实时未建立</div></section>
<nav id="tabs"></nav><section id="replay"><h2>五场景状态回放</h2><div class="controls"><label>场景 <select id="scene"></select></label>
<label>视角 <select id="view"></select></label></div><video id="player" controls preload="metadata"></video>
<p class="note">读取最新保存的完整 qpos/qvel，仅做 mj_forward 渲染；不重新运行控制器或积分。源区间 0–27 s，30 FPS 共 811 帧（编码时长 27.033 s）。</p>
<p id="videoLinks"></p></section><section class="grid" id="panels"></section>
<details><summary>数据来源、完整清单与结论范围</summary><p><a href="visualization_manifest.json">SHA-256 清单</a> · <a href="visualization_validation.json">独立可视化核验</a> · <a href="result_data.json">结构化结果</a></p>
<p>原始失败、暂停前部分观察及续作记录仍保留。惯量使用 15 组完整合并结果，压力实验包含全部三条 partial trace。未验证扰动反馈闭环、连续时间碰撞保证或安全备份。</p><pre id="provenance"></pre></details></main>
<script id="bundle-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('bundle-data').textContent), panels=document.getElementById('panels');
let active='all';const categories=[['all','全部最新结果'],...Object.entries(data.category_titles)];
function show(){panels.replaceChildren();data.panels.filter(p=>active==='all'||p.category===active).forEach(p=>{const a=document.createElement('article'),h=document.createElement('h3'),link=document.createElement('a'),i=document.createElement('img'),c=document.createElement('p');h.textContent=p.title;i.src=p.url;i.alt=p.title;i.loading='lazy';link.href=p.url;link.target='_blank';link.append(i);c.textContent=p.caption;a.append(h,link,c);panels.append(a)});document.getElementById('replay').hidden=!['all','nominal'].includes(active)}
categories.forEach(([key,title])=>{const b=document.createElement('button');b.textContent=title;b.dataset.key=key;if(key==='all')b.classList.add('active');b.onclick=()=>{active=key;document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x.dataset.key===key));show()};document.getElementById('tabs').append(b)});
const scene=document.getElementById('scene'),view=document.getElementById('view'),player=document.getElementById('player');
data.replays.forEach((s,i)=>scene.add(new Option(s.scenario_id,String(i))));
[['five_view_grid','五视角组合'],['overview','总览'],['front','正面'],['side','侧面'],['top','俯视'],['iso','斜视'],['continuum_focus','连续体单侧']].forEach(([k,t])=>view.add(new Option(t,k)));
function play(){const s=data.replays[Number(scene.value)],file=s.videos[view.value];player.src=file;player.poster=s.preview;const box=document.getElementById('videoLinks');box.replaceChildren();const a=document.createElement('a');a.href=file;a.textContent='打开当前 MP4';box.append(a)}scene.onchange=play;view.onchange=play;
document.getElementById('provenance').textContent=JSON.stringify({inputs:data.inputs,claim_scope:data.claim_scope},null,2);show();play();
</script></html>"""
    path = output / "index.html"
    path.write_text(html_text.replace("__DATA__", payload), encoding="utf-8", newline="\n")
    return path


def replays_for(nominal: dict, output: Path) -> list[dict]:
    replays = []
    for item in nominal["videos"]:
        sid = item["scenario_id"]
        files = item["files"]
        mapping = {}
        preview = None
        for name in files:
            p = ROOT / name
            if p.suffix.lower() == ".mp4":
                view = p.stem.removeprefix(sid + "_")
                mapping[view] = _local(name, output)
            elif "five_view_preview" in p.name:
                preview = _local(name, output)
        if set(mapping) != {"overview", "front", "side", "top", "iso", "five_view_grid", "continuum_focus"}:
            raise ValueError("incomplete latest scene views: " + sid)
        replays.append({"scenario_id": sid, "videos": mapping, "preview": preview})
    return replays


def write_document(path: Path, manifest: dict, data: dict) -> None:
    def link(name):
        # The canonical document is in docs; pathlib cannot express ../ via relative_to.
        import os
        return Path(os.path.relpath(ROOT / name, path.parent)).as_posix()
    lines = ["# V6.2 最新结果全套可视化", "",
             "本页是当前展示入口，覆盖最新名义研究五场景、保守性与形状诊断、2 倍速度全部失败/成功尝试和 15 组惯量敏感性。历史 B.2/V6.1 图与原始失败仍在原目录保留。", "",
             "名义验收 **PASSED（25/25 功能、11/11 合同、6,755 边界与 12,350 区间行重算）**；压力实验 **FAILED / INCOMPLETE**；惯量为 **DIAGNOSTIC COMPLETE**；墙钟部署 **NOT_MET**。模拟周期仍为 20 ms / 2 ms，墙钟统计单列性能。", "",
             f"[完整清单]({link(manifest['output_directory'] + '/visualization_manifest.json')}) · [独立校验]({link(manifest['output_directory'] + '/visualization_validation.json')}) · [交互式本地总览]({link(manifest['output_directory'] + '/index.html')}) · [结构化结果]({link(manifest['output_directory'] + '/result_data.json')})", "",
             "GitHub 页面可直接查看下方 PNG/GIF；MP4 点击打开或下载。交互式 HTML 在本地浏览器打开或由仓库内的本地 HTTP 服务查看，不依赖外部网站。", "",
             "## 名义五场景图、计时与执行", ""]
    for panel in data["panels"]:
        if panel["category"] not in ("nominal", "wall") or "studies" in panel["path"]:
            continue
        if "tracking_paths" in panel["path"] or "/pcc_monitor/" in panel["path"]:
            continue
        lines.extend([f"### {panel['title']}", "", f"![{panel['title']}]({link(panel['path'])})", "", panel["caption"], ""])
    lines.extend(["## 五场景完整视频与路径", "",
                  "视频从完整保存状态渲染，不重新积分。每场 5 个视角、1 个组合、1 个连续体单侧，共 35 个视频；初始与终态包含在 811 帧中。末端坐标系、七个航点与基座漂移逐帧展示。", ""])
    for item in manifest["nominal"]["videos"]:
        sid = item["scenario_id"]
        lines.extend([f"### {sid}", ""])
        scene_files = [name for name in item["files"] if name.endswith(".png")]
        for name in scene_files:
            lines.extend([f"![{sid} 预览]({link(name)})", ""])
        links = []
        for name in item["files"]:
            if name.endswith(".mp4"):
                label = Path(name).stem.removeprefix(sid + "_")
                links.append(f"[{label}]({link(name)})")
        lines.extend([" · ".join(links), ""])
        for panel in data["panels"]:
            if sid in panel["path"] and panel["category"] == "nominal":
                lines.extend([f"![{panel['title']}]({link(panel['path'])})", ""])
    for category in CATEGORIES[1:]:
        lines.extend([f"## {TITLES[category]}", ""])
        for panel in data["panels"]:
            if panel["category"] == category and (category != "wall" or "/studies/" in panel["path"]):
                lines.extend([f"### {panel['title']}", "", f"![{panel['title']}]({link(panel['path'])})", "", panel["caption"], ""])
    lines.extend(["## 来源与范围", "",
                  "名义运行：`research_acceptance_01`，生成源码 `9cd1831`；惯量完整合并：`research_inertia_shadow_resumed_01`，生成源码 `19ad85c`。精确输入 SHA 见清单；14 组观察复用、1 组续作及原中断文件保留。", "",
                  "新鲜当前状态曲线与原缓存日志统计分别标注，原名义验收数字不被图表改写。压力图包含完整 01/02 和部分 00/03/04，拒绝后的轨迹没有延伸。惯量只改变 body inertia，使用同一 67 路力矩；没有重新求解反馈闭环，也未建立模型误差界、连续时间安全或硬件保证。", "",
                  "PCC 三类监控图从本次名义运行逐字节复制；旧监控绑定占位字段不作为区间 CBF 活跃性证据，区间有效性使用原独立重算报告。", "",
                  "重新生成须使用新目录，原始实验文件不覆盖：", "", "```powershell",
                  "python -m v6_lite.visualization.generate_latest_full --output-dir v6_lite/visualization/latest_rebuild",
                  "python -m v6_lite.visualization.validate_latest_full --output-dir v6_lite/visualization/latest_rebuild", "```", "",
                  "本地交互展示：", "", "```powershell", "python -m http.server 8765 --bind 127.0.0.1",
                  "# http://127.0.0.1:8765/v6_lite/visualization/latest/index.html", "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate(output: Path, document: Path | None = None) -> dict:
    from v6_lite.visualization.latest_nominal import generate_nominal
    from v6_lite.visualization.latest_studies import generate_studies
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new visualization directory: " + str(output))
    metrics, inputs, refs = load_inputs()
    generator_source = source_records()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "generation_plan.json", {"schema": SCHEMA, "started_utc": datetime.now(timezone.utc).isoformat(),
          "inputs": inputs, "generator_source": generator_source, "expected_scenarios": 5, "expected_videos": 35,
          "physics_recomputed": False, "old_outputs_overwritten": False})
    try:
        studies = convert(generate_studies(output))
        write(output / "studies_metadata.json", studies)
        print("latest finite-research plots completed", flush=True)
        nominal = convert(generate_nominal(metrics, output))
        write(output / "nominal_metadata.json", nominal)
        panels = panels_for(nominal, studies, output)
        categories = {p["category"] for p in panels}
        if categories != set(CATEGORIES):
            raise ValueError("latest-result categories missing: " + repr(set(CATEGORIES) - categories))
        claim_scope = {"research_acceptance": "PASSED", "functional_checks": "25/25", "execution_checks": "11/11",
                       "interval_state_checks": 6755, "interval_rows": 12350, "timing_is_research_gate": False,
                       "velocity_stress": "FAILED_INCOMPLETE", "model_sensitivity": "DIAGNOSTIC_COMPLETE",
                       "wall_deployment": "NOT_MET", "hardware": "NOT_ESTABLISHED", "hard_real_time": False,
                       "shadow_closed_loop_robustness": False, "continuous_time_collision_guarantee": False,
                       "visualization_is_new_functional_acceptance": False}
        data = {"schema": SCHEMA, "category_titles": TITLES, "panels": panels,
                "replays": replays_for(nominal, output), "inputs": inputs, "claim_scope": claim_scope,
                "nominal": nominal.get("scenes", []), "studies": studies.get("data", {})}
        write(output / "result_data.json", data)
        write_html(output, data)
        refs.extend(_source_refs(nominal.get("source_refs", [])))
        refs.extend(_source_refs(studies.get("source_refs", [])))
        # Verify helper inputs and immutable raw files again after rendering.
        unique_refs = {r["path"]: r for r in refs}
        for expected in [*unique_refs.values(), *generator_source]:
            if sha(ROOT / expected["path"]) != expected["sha256"]:
                raise ValueError("source changed while rendering: " + expected["path"])
        artifacts = [record(p) for p in sorted(output.rglob("*")) if p.is_file()]
        manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "generator_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "generator_source": generator_source, "output_directory": portable(output), "inputs": inputs,
                    "source_inputs": list(unique_refs.values()), "artifacts": artifacts, "nominal": nominal,
                    "studies": studies, "claim_scope": claim_scope,
                    "coverage": {"nominal_scenarios": 5, "nominal_duration_s": 27.,
                                 "views": ["overview", "front", "side", "top", "iso"], "video_count": 35,
                                 "research_categories": CATEGORIES, "all_pressure_scenarios_included": True,
                                 "all_inertia_runs_included": 15, "old_b2_used_as_current_source": False},
                    "complete": True, "physics_recomputed": False}
        write(output / "visualization_manifest.json", manifest)
        if document is not None:
            write_document(document.resolve(), manifest, data)
        return manifest
    except BaseException as error:
        write(output / "generation_failure.json", {"complete": False, "type": type(error).__name__,
              "message": str(error), "partial_outputs_preserved": True, "original_experiment_files_modified": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        document = args.document or (DEFAULT_DOCUMENT if args.output_dir.resolve() == DEFAULT_OUTPUT else None)
        manifest = generate(args.output_dir, document)
        print(json.dumps({"complete": manifest["complete"], "artifacts": len(manifest["artifacts"]),
                          "video_count": manifest["coverage"]["video_count"]}), flush=True)
    from v6_lite.visualization.validate_latest_full import validate
    result = validate(args.output_dir.resolve())
    print(json.dumps({k: result[k] for k in ("passed", "check_count", "passed_check_count", "error")}, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
