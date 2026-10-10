"""C4-A portable dashboard and seven-view videos from sealed DEV actual states.

The C.2 render and tracking implementations are reused with an explicit C4-A
loader, schema and annotation namespace. Their module globals are never changed.
No physics, collision queries, optimizer updates or neural samples are permitted.
All aliases share one saved actual's media; NO_PLAN has no invented video.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from types import FunctionType

import numpy as np
from PIL import Image, ImageDraw

from . import build_preference_warmstart_media as c2
from ..route_candidate_evaluator import verify_seal
from ..route_optimizer_protocol import read, sha
from ..task_anchored_reference import TaskAnchoredResidualPlan, TaskAnchoredResidualReferenceProvider
from ..task_protocol import TaskSpec

ENDPOINTS = ("P0", "P1", "P2")
VIEWS = ("five_view_grid", "continuum_focus", "overview", "front", "side", "top", "iso")
SCHEMA = "v64_c4a_saved_actual_media_v1"
CHARTS = ("coverage_and_near_quality.png", "actual_quality_and_cost.png")
write, record, unchanged = c2.write, c2.record, c2.unchanged


def _portable_path(run, value, original_root=None):
    """Rebind a copied complete run by its frozen relative path, never basename."""
    run = Path(run).resolve(); path = Path(value)
    if not path.is_absolute():
        candidate = (run / path).resolve()
        if run not in candidate.parents:
            raise ValueError("scientific source path escapes the complete run")
        return candidate
    if path.is_relative_to(run):
        return path
    if original_root is not None and path.is_relative_to(original_root):
        return run / path.relative_to(original_root)
    # Genuine external scientific files remain external; no filename guessing.
    return path


def terminal_slots(run):
    """Require the committed C4-A DEV inventory, original selection and slot seals."""
    run = Path(run).resolve()
    if read(run / "execution_complete.json")["status"] != "COMPLETED":
        raise ValueError("C4-A DEV must have completed before media generation")
    inventory_path = Path(__file__).resolve().parents[1] / "releases/c4a_architecture_audit_20261010_01/raw_inventory_sha256.json"
    inventory = {r["path"]: r for r in read(inventory_path)["all_files"]}
    def bound(path):
        rel = path.relative_to(run).as_posix()
        if sha(path) != inventory[rel]["sha256"]:
            raise ValueError("source differs from original committed raw inventory: " + rel)
        return read(path)
    protocol = bound(run / "plan.json")
    bound(run / "execution_complete.json")
    manifest = bound(run / "validation/sealed_selections/all_selections.json")
    if manifest["phase"] != "DEV" or manifest["endpoints"] != list(ENDPOINTS) or manifest["logical_actual_slots"] != 12:
        raise ValueError("expected twelve C4-A DEV slots across P0/P1/P2")
    if manifest["bindings"]["protocol_sha256"] != sha(run / "plan.json"):
        raise ValueError("DEV protocol binding differs")
    original_root = Path(manifest["entries"][0]["selection_path"]).parents[3]
    for value, expected in manifest["files"].items():
        path = _portable_path(run, value, original_root)
        if sha(path) != expected:
            raise ValueError("sealed DEV search or selection changed")
    tasks = protocol["tasks"]
    if len(tasks) != 2 or any(t["split"] != "dev" for t in tasks):
        raise ValueError("expected two independent DEV tasks")
    slots = []
    for task in tasks:
        for endpoint in ENDPOINTS:
            selection_path = run / "validation/sealed_selections" / task["task_id"] / (endpoint + ".json")
            selection = bound(selection_path)
            for preference in ("A", "B"):
                path = run / "validation/actual" / task["task_id"] / (endpoint + "_" + preference) / "slot.json"
                verify_seal(path.parent); slot = bound(path)
                if (slot["phase"], slot["task_id"], slot["task_sha256"], slot["endpoint"], slot["preference"], slot["selection_sha256"]) != (
                        "DEV", task["task_id"], task["task_sha256"], endpoint, preference, sha(selection_path)):
                    raise ValueError("actual slot differs from frozen DEV selection")
                expected = TaskAnchoredResidualPlan.from_dict(selection["preferences"][preference]["selected_plan"]).sha256()
                if slot["plan_sha256"] != expected or slot["actual_steps"] != 13500 or not slot["original_independent_gates_passed"]:
                    raise ValueError("C4-A frozen full actual binding differs")
                if slot.get("alias_of_slot"):
                    source = _portable_path(run, slot["alias_of_slot"], original_root)
                    if sha(source) != slot["alias_of_slot_sha256"]:
                        raise ValueError("strict alias source changed")
                    verify_seal(source.parent); original = bound(source)
                    if original.get("alias_of_slot") or any(slot[k] != original[k] for k in ("alias_identity", "task_sha256", "plan_sha256")):
                        raise ValueError("alias identity differs")
                slots.append(slot)
    if len(slots) != 12 or sum(bool(s.get("alias_of_slot")) for s in slots) != 5:
        raise ValueError("expected seven unique actuals and five aliases")
    return slots


def resolve_unique(slots, task_id, method):
    by_key = {(row["task_id"], row["method"]): row for row in slots}
    slot = by_key[(task_id, method)]
    if not slot.get("alias_of_slot"):
        return slot
    original_method = Path(slot["alias_of_slot"]).parent.name
    source = by_key[(task_id, original_method)]
    if source.get("alias_of_slot") or any(source[k] != slot[k] for k in ("alias_identity", "task_sha256", "plan_sha256")):
        raise ValueError("media alias is not a strict original same-task actual")
    return source


def load_actual(run, slot):
    """C.2 saved-state/replay binding adapted only for C4-A directory layout."""
    run = Path(run).resolve()
    directory = run / "validation/actual" / slot["task_id"] / slot["method"]
    attempt = directory / "attempt"
    task_path = run / "frozen_tasks" / slot["task_id"] / "task.json"
    plan_path, result_path = directory / "selected_plan.json", attempt / "attempt_result.json"
    evaluation_path = attempt / "actual/evaluation/report.json"
    replay_path = evaluation_path.parent / "fresh_replay.npz"
    task = TaskSpec.from_dict(read(task_path)); plan = TaskAnchoredResidualPlan.from_dict(read(plan_path))
    executed_plan = TaskAnchoredResidualPlan.from_dict(read(attempt / "plan.json"))
    result, evaluation = read(result_path), read(evaluation_path)
    # Rebind by the original frozen run root; an old absolute path still being
    # readable must not divert a portable copy back to the previous checkout.
    original_root = Path(result["qp_config_path"]).parent
    trace_path = _portable_path(run, result["trace_path"], original_root)
    if not trace_path.is_relative_to(attempt) or not trace_path.is_file():
        raise ValueError("actual trace is not the exact local attempt subpath")
    if (task.sha256() != slot["task_sha256"] or plan.sha256() != slot["plan_sha256"]
            or executed_plan.sha256() != plan.sha256() or result["plan_content_sha256"] != plan.sha256()
            or result["task_sha256"] != task.sha256() or result["plan_file_sha256"] != sha(attempt / "plan.json")
            or result["trace_sha256"] != sha(trace_path) or evaluation["fresh_replay_sha256"] != sha(replay_path)
            or result["evaluation_sha256"] != sha(evaluation_path)):
        raise ValueError("actual rendering evidence bindings differ")
    with np.load(trace_path, allow_pickle=False) as archive:
        actual = {k: archive[k].copy() for k in ("initial_qpos", "initial_qvel", "actual_full_qpos",
                  "actual_full_qvel", "time", "generated_continuum_position")}
    needed = ("time", "qpos", "qvel", "base_pose", "target_position", "target_rotation",
        "rigid_position", "rigid_rotation", "continuum_position", "continuum_rotation")
    with np.load(replay_path, allow_pickle=False) as archive:
        replay = {k: archive[k].copy() for k in needed}
    n = int(evaluation.get("actual_physics_steps", slot["actual_steps"]))
    qpos = np.vstack((actual["initial_qpos"], actual["actual_full_qpos"][:n]))
    qvel = np.vstack((actual["initial_qvel"], actual["actual_full_qvel"][:n]))
    times = np.r_[0., actual["time"][:n]]
    shapes = {"time": (n + 1,), "base_pose": (n + 1, 7),
        **{k: (n + 1, 3) for k in ("target_position", "rigid_position", "continuum_position")},
        **{k: (n + 1, 3, 3) for k in ("target_rotation", "rigid_rotation", "continuum_rotation")}}
    if (n <= 0 or n != slot["actual_steps"] or (slot.get("full_task_success") and n != 13500)
            or qpos.shape != (n + 1, 81) or qvel.shape != (n + 1, 79)
            or replay["qpos"].shape != qpos.shape or replay["qvel"].shape != qvel.shape
            or any(replay[k].shape != shape for k, shape in shapes.items())
            or not np.array_equal(qpos[0], task.initial_qpos) or not np.array_equal(qvel[0], task.initial_qvel)
            or not all(np.isfinite(v).all() for v in (*actual.values(), *replay.values()))
            or not np.allclose(times, np.arange(n + 1) * .002, atol=1e-9, rtol=0.)
            or not np.allclose(times, replay["time"], atol=1e-9, rtol=0.)):
        raise ValueError("saved actual state dimension/clock/initial/finite check failed")
    qpos_error = float(np.max(np.abs(qpos - replay["qpos"])))
    qvel_error = float(np.max(np.abs(qvel - replay["qvel"])))
    if qpos_error > 1e-9 or qvel_error > 1e-8:
        raise ValueError("actual state and independent replay parity failed")
    parity = {"qpos_max_abs_difference": qpos_error, "qvel_max_abs_difference": qvel_error,
        "qpos_bit_exact": bool(np.array_equal(qpos, replay["qpos"])),
        "qvel_bit_exact": bool(np.array_equal(qvel, replay["qvel"])), "initial_state_matches_Task": True}
    replay["qpos"], replay["qvel"], replay["time"] = qpos, qvel, times
    reference, _, _ = TaskAnchoredResidualReferenceProvider(task, plan).continuum_kinematics(times)
    generated = actual["generated_continuum_position"][:n]
    if generated.shape != (n, 3):
        raise ValueError("saved generated continuum reference shape differs")
    reference_error = float(np.max(np.abs(reference[1:] - generated)))
    if reference_error > 1e-12:
        raise ValueError("frozen reference differs from saved actual generated reference")
    parity["generated_continuum_reference_max_abs_difference_m"] = reference_error
    offset, _, _ = plan.offset_kinematics(times); base = reference - offset
    sources = {name: record(path, run) for name, path in (
        ("slot", directory / "slot.json"), ("task", task_path), ("plan", plan_path),
        ("executed_plan", attempt / "plan.json"), ("attempt_result", result_path),
        ("evaluation", evaluation_path), ("actual_trace", trace_path), ("fresh_replay", replay_path))}
    parity["saved_execution_compiled_model_sha256"] = c2.execution_model_binding(run, attempt, sources)
    return task, plan, result, evaluation, replay, reference, base, parity, sources


def _annotation(frame, slot, task, replay, state, view, reference, strip=120):
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + strip), (10, 14, 24))
    image.paste(Image.fromarray(frame), (0, strip)); draw = ImageDraw.Draw(image)
    complete = slot["full_task_success"] and slot["original_independent_gates_passed"]
    lines = [f"C4-A {slot['method']} | {view.upper()}", f"Task {task.task_id}",
        f"{'FULL ACTUAL + 5 GATES PASS' if complete else 'FAILED ACTUAL PREFIX'} | saved {replay['time'][state]:.3f}/{replay['time'][-1]:.3f}s",
        f"Continuum frozen reference error {1000*np.linalg.norm(replay['continuum_position'][state]-reference[state]):.2f}mm",
        "green=frozen reference; red=Task/base; cyan=actual; RGB=XYZ",
        "Saved actual states + forward only | deployment NOT_MET"]
    for i, line in enumerate(lines):
        draw.text((7, 4 + i * 19), line, fill=(245, 193, 130) if i == 2 and not complete else (225, 234, 245),
            font=c2.legacy._font(12 if image.width < 900 else 16))
    return np.asarray(image)


def _reuse(function):
    namespace = {**function.__globals__, "load_actual": load_actual, "_annotation": _annotation, "SCHEMA": SCHEMA}
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


_render_impl, _figure_impl = _reuse(c2.render_actual), _reuse(c2.trajectory_figures)


def render_actual(run, output, slot, fps=15., width=640, height=480, focus_width=960, focus_height=720):
    return _render_impl(run, output, slot, fps, width, height, focus_width, focus_height)


def trajectory_figures(run, output, slot):
    return _figure_impl(run, output, slot)


def core_charts(run, output, slots):
    from .c4a_media_charts import build_charts
    return build_charts(run, output, slots)


def dashboard(output, payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C4-A · 五视角、连续体侧视与轨迹误差</title><style>
body{font:15px system-ui,sans-serif;background:#f4f6fa;color:#253249;max-width:1440px;margin:auto;padding:24px}h1{font-size:28px}p{line-height:1.6}section{background:white;border:1px solid #dce2eb;border-radius:12px;padding:20px;margin:20px 0}.charts,.figures{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{width:100%;height:auto}video{width:100%;max-height:760px;background:#111827}select{padding:8px;margin:5px;max-width:40vw;border:1px solid #aabbcc;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e5eaf1}.pass{color:#148067}.fail{color:#b25c25}.badge{background:#e7edf5;padding:3px 8px;border-radius:6px}.scroll{overflow:auto}#status{font-size:16px;padding:12px 0}@media(max-width:800px){.charts,.figures{grid-template-columns:1fr}body{padding:12px}}
</style><h1>V6.4-C4-A · 当前执行与可视化</h1>
<p>P0: Rule8. P1: Diffusion8 replacement. P2: rule-preserving portfolio (4 rule + 2 frozen Diffusion + 2 original polls). Two independent DEV tasks; eight candidate slots per shared A/B search. Twelve logical actuals, seven unique executions and five strict aliases. No new training or C.3 TEST execution.</p>
<p>每个结果都可切换总览、正面、侧面、俯视、等轴测、五视角拼接和连续体侧视特写。下方同步显示双臂末端轨迹与原生 2 ms 跟踪误差，视频和 CSV 可直接下载。</p><p id="conclusion"></p><p><span class="badge">DATA_LIMITED pilot</span> <span class="badge">deployment NOT_MET</span> <span class="badge">continuous-time safety NOT_ESTABLISHED</span> <span class="badge">hardware safety NOT_ESTABLISHED</span></p>
<section><details><summary>展开：候选覆盖、执行质量、成本与 P0/P1/P2 误差对比（6 张图）</summary><div class="charts" id="charts"></div><p>Charts use the sealed C4-A DEV metrics. Planning cost is measured separately from offline teacher, training, VAL and saved-state rendering. A budget reduction alone does not establish speedup. Missing measurements remain unavailable.</p></details></section>
<section><label>任务 Task <select id="task"></select></label><label>策略 <select id="endpoint"></select></label><label>偏好 <select id="preference"></select></label><label>视角 <select id="view"></select></label>
<div id="status"></div><video id="video" controls preload="metadata"></video><p id="videoNote"></p><div id="downloads"></div><div class="figures"><img id="trajectory" alt="Saved actual tip trajectories"><img id="tracking" alt="Native tracking error"></div></section>
<section><h2>全部 12 个结果槽 · 7 次独立执行</h2><p>Aliases share one unique actual and its media. NO_PLAN has zero execution steps and no video. Failed prefixes stop at the exact saved endpoint. Prefix metrics are descriptive and cannot win against a full task. Original independent five gates determine acceptance.</p><div class="scroll"><table><thead><tr><th>Task</th><th>Endpoint / pref</th><th>Status</th><th>Saved steps</th><th>Task + five gates</th><th>B ≥30mm</th><th>I [rad/s]</th><th>L [m]</th><th>d [mm]</th><th>Media source</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<script>const D=__DATA__;const $=id=>document.getElementById(id);function esc(v){const p=document.createElement('span');p.textContent=String(v);return p.innerHTML}function fmt(x,d=4){return x==null?'—':Number(x).toFixed(d)}function options(id,values){$(id).innerHTML=values.map(v=>'<option>'+esc(v)+'</option>').join('')}
$('conclusion').textContent='Decision: MODIFY. Diffusion architecture: NOT_JUSTIFIED. Local proposal usefulness does not establish superiority over simpler proposal methods.';$('charts').innerHTML=Object.entries(D.charts).map(([name,path])=>'<img alt="'+esc(name)+'" src="'+esc(path)+'">').join('');options('task',D.tasks);options('endpoint',D.endpoints);options('preference',['A','B']);options('view',D.views);
function show(){const r=D.slots.find(r=>r.task_id===$('task').value&&r.endpoint===$('endpoint').value&&r.preference===$('preference').value);if(!r)return;const v=r.media?.videos?.[$('view').value];$('video').style.display=v?'block':'none';if(v){if(!$('video').src.endsWith(v)){$('video').src=v;$('video').poster=($('view').value==='continuum_focus'?r.media.focus_preview:r.media.preview)||''}}else{$('video').removeAttribute('src');$('video').load()}
$('status').textContent=r.status+' | '+r.actual_steps+' saved steps | '+(r.alias_of_slot?'alias of '+r.media_source_method:r.unique_run?'unique actual':'no physical run');$('status').className=r.full_task_success?'pass':'fail';$('videoNote').textContent=r.media?'Saved endpoint '+fmt(r.media.source_end_s,3)+' s; exact endpoint included; saved actual states only.':r.actual_steps?'Video not yet rendered for this actual.':'No saved execution states: '+r.status;for(const name of ['trajectory','tracking']){$(name).style.display=r.figures?'block':'none';if(r.figures)$(name).src=r.figures[name]}
$('downloads').innerHTML=Object.entries(r.media?.videos||{}).map(([name,path])=>'<a href="'+esc(path)+'">'+esc(name)+'</a>').join(' · ')+(r.figures?' · <a href="'+esc(r.figures.tracking_csv)+'">native tracking CSV</a>':'')}
$('rows').innerHTML=D.slots.map(r=>{const q=r.full_task_success?r.quality||{}:{};return '<tr><td>'+esc(r.task_id)+'</td><td>'+esc(r.endpoint+' / '+r.preference)+'</td><td class="'+(r.full_task_success?'pass':'fail')+'">'+esc(r.status)+'</td><td>'+r.actual_steps+'</td><td>'+((r.full_task_success&&r.original_independent_gates_passed)?'PASS':'FAIL')+'</td><td>'+(r.preference==='A'?'N/A':r.clearance_30mm_met===true?'YES':r.clearance_30mm_met===false?'NO':'—')+'</td><td>'+fmt(q.I_support)+'</td><td>'+fmt(q.L_full)+'</td><td>'+fmt(q.d_support==null?null:q.d_support*1000,2)+'</td><td>'+esc(r.media_source_method)+'</td></tr>'}).join('');for(const id of ['task','endpoint','preference','view'])$(id).addEventListener('change',show);show();</script></html>'''
    (Path(output) / "index.html").write_text(page.replace("__DATA__", encoded), encoding="utf8")


def build(run, output, *, render=False, check_inputs=False, workers=2, fps=15., width=640,
          height=480, focus_width=960, focus_height=720):
    started = time.perf_counter(); run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("media output must be separate from frozen scientific run")
    slots = terminal_slots(run); output.mkdir(parents=True, exist_ok=True)
    request = {"schema": SCHEMA, "plan_sha256": sha(run / "plan.json"),
        "execution_complete_sha256": sha(run / "execution_complete.json"),
        "sealed_dev_selection_sha256": sha(run / "validation/sealed_selections/all_selections.json"),
        "fps": fps, "width": width, "height": height, "focus_width": focus_width, "focus_height": focus_height}
    request_path = output / "build_request.json"
    if request_path.exists() and read(request_path) != request:
        raise ValueError("media output belongs to a different run or render configuration")
    if not request_path.exists():
        if any(output.iterdir()):
            raise FileExistsError("exclusive new C4-A media output required")
        write(request_path, request)
    unique = [s for s in slots if not s.get("alias_of_slot") and s["actual_steps"] > 0]
    checks = []
    for slot in unique:
        loaded = load_actual(run, slot)
        checks.append({"task_id": slot["task_id"], "method": slot["method"], "actual_fresh_parity": loaded[-2],
                       "source_end_s": float(loaded[4]["time"][-1]), "sources": loaded[-1]})
    write(output / "input_checks.json", {"schema": SCHEMA, "status": "PASS", "checks": checks,
        "logical_slots": len(slots), "unique_saved_actuals": len(unique), "physics_steps": 0,
        "geometry_queries": 0, "QP_solves": 0})
    if check_inputs:
        return {"status": "PASS", "unique_saved_actuals": len(unique), "logical_slots": len(slots)}
    figures = {(s["task_id"], s["method"]): trajectory_figures(run, output, s) for s in unique}
    charts = core_charts(run, output, slots)
    videos = {}
    if render:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ValueError("ffmpeg and ffprobe required for seven-view refresh")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(render_actual, run, output, s, fps, width, height, focus_width, focus_height): s for s in unique}
            for future in as_completed(pending):
                row = future.result(); videos[(row["task_id"], row["method"])] = row
                print(json.dumps({"event": "C4A_MEDIA", "task": row["task_id"], "method": row["method"],
                    "videos": len(row["videos"]), "saved_endpoint_s": row["source_end_s"]}), flush=True)
    else:
        for slot in unique:
            path = output / "replays" / (slot["task_id"] + "_" + slot["method"]) / "replay_metadata.json"
            if path.exists():
                videos[(slot["task_id"], slot["method"])] = read(path)
    display = []
    for slot in slots:
        source = resolve_unique(slots, slot["task_id"], slot["method"]); key = (source["task_id"], source["method"])
        display.append({**slot, "figures": figures.get(key), "media": videos.get(key), "media_source_method": source["method"]})
    payload = {"schema": SCHEMA, "tasks": list(dict.fromkeys(s["task_id"] for s in slots)),
        "endpoints": list(ENDPOINTS), "views": list(VIEWS), "charts": charts, "slots": display,
        "summary": read(run.parent / 'results_01/summary.json')}
    write(output / "dashboard_data.json", payload); dashboard(output, payload)
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "run_identity": request,
        "generator": record(__file__), "reused_renderer": record(c2.__file__), "source_producer_commit": "dd834783847bfbcc4f8ec624b7bce0d22777b6de",
        "logical_actual_slots": len(slots), "unique_actual_media": len(videos),
        "video_count": sum(len(row["videos"]) for row in videos.values()),
        "alias_slots": sum(bool(s.get("alias_of_slot")) for s in slots),
        "no_plan_slots": sum(s["status"] == "NO_PLAN" for s in slots), "wall_s": time.perf_counter() - started,
        "files": {p.relative_to(output).as_posix(): record(p, output) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name != "visualization_manifest.json"},
        "physics_steps_executed": 0, "geometry_queries": 0, "QP_solves": 0, "optimizer_updates": 0,
        "new_model_samples": 0, "historical_outputs_overwritten": False, "visualization_is_new_acceptance": False}
    write(output / "visualization_manifest.json", manifest)
    return {"status": "PASS", "output": str(output), "logical_slots": len(slots),
            "unique_actual_media": len(videos), "video_count": manifest["video_count"], "physics_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true"); parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2); parser.add_argument("--fps", type=float, default=15.)
    parser.add_argument("--width", type=int, default=640); parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--focus-width", type=int, default=960); parser.add_argument("--focus-height", type=int, default=720)
    args = parser.parse_args()
    if args.check_inputs and args.render:
        parser.error("check-inputs cannot render")
    if not np.isfinite(args.fps) or not 0. < args.fps <= 60.:
        parser.error("FPS must be in (0,60]")
    if min(args.width, args.focus_width) < 640 or min(args.height, args.focus_height) < 480 or any(v % 2 for v in (args.width, args.height, args.focus_width, args.focus_height)):
        parser.error("even dimensions of at least 640x480 required")
    print(json.dumps(build(args.run, args.output, render=args.render, check_inputs=args.check_inputs,
        workers=args.workers, fps=args.fps, width=args.width, height=args.height,
        focus_width=args.focus_width, focus_height=args.focus_height)), flush=True)


if __name__ == "__main__":
    main()
