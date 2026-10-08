"""C.1 saved-state media. No integration, solver, search or new clearance query.

Run with the original sealed local experiment directory. Reuse the established
camera/encoder implementation through an explicit C.1 source adapter, while
leaving historical media and the experiment seal untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from v6_4.task_protocol import TaskSpec
from v6_4.task_anchored_reference import TaskAnchoredResidualPlan, TaskAnchoredResidualReferenceProvider
from v6_4.visualization import render_execution_aware_replays as engine
from v6_4.visualization import render_residual_replays as legacy

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("Z0", "G0", "OI", "OC")
SCHEMA = "v64_c1_saved_actual_media_v1"
read, sha, require = engine.read, engine.sha, engine.require


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")


class Sources:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.manifest = read(self.source/"manifest.json")
        self.paths = {"source_manifest": engine.record(self.source/"manifest.json")}

    def bind(self, name, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(self.source), "input outside sealed C.1 experiment")
        record = engine.record(path)
        require(self.manifest[path.relative_to(self.source).as_posix()] == record["sha256"], "sealed input changed")
        require(expected is None or expected == record["sha256"], "independent input digest differs")
        self.paths[name] = record
        return path

    def unchanged(self):
        for record in self.paths.values():
            require(engine.record(record["path"]) == record, "render input changed")


def slots(source):
    source = Path(source)
    return [read(source/"actual"/task.name/method/"slot.json")
            for task in sorted((source/"actual").iterdir()) for method in METHODS]


def load_slot(source, slot_id):
    sources = Sources(source)
    task_id, method = slot_id.rsplit("_", 1)
    directory = sources.source/"actual"/task_id/method
    slot = read(sources.bind("logical_slot", directory/"slot.json"))
    require(slot["unique_run"] and slot["entered_actual"], "render unique actual evidence only")
    attempt = directory/"attempt"
    result = read(sources.bind("attempt_result", attempt/"attempt_result.json"))
    task = TaskSpec.from_dict(read(sources.bind("task", attempt/"task.json")))
    plan = TaskAnchoredResidualPlan.from_dict(read(sources.bind("plan", attempt/"plan.json")))
    require(task.sha256() == slot["task_sha256"] == result["task_sha256"], "task binding differs")
    require(plan.sha256() == slot["plan_sha256"] == result["plan_content_sha256"], "plan binding differs")
    trace_path = sources.bind("actual_trace", result["trace_path"], result["trace_sha256"])
    evaluation_path = sources.bind("evaluation", result["evaluation_path"], result["evaluation_sha256"])
    evaluation = read(evaluation_path)
    fresh_path = sources.bind("fresh_replay", evaluation_path.parent/"fresh_replay.npz", evaluation["fresh_replay_sha256"])
    if (attempt/"actual/historical_metric_observations.json").exists():
        sources.bind("historical_model_observation", attempt/"actual/historical_metric_observations.json")
    with np.load(trace_path, allow_pickle=False) as archive:
        actual = {k: archive[k] for k in ("initial_qpos", "initial_qvel", "actual_full_qpos", "actual_full_qvel", "time",
                                         "generated_continuum_position", "generated_rigid_position")}
    with np.load(fresh_path, allow_pickle=False) as archive:
        replay = {k: archive[k] for k in archive.files}
    n = slot["actual_steps"]
    qpos = np.vstack((actual["initial_qpos"], actual["actual_full_qpos"]))
    qvel = np.vstack((actual["initial_qvel"], actual["actual_full_qvel"]))
    times = np.r_[0., actual["time"]]
    require(qpos.shape == (n+1, 81) and qvel.shape == (n+1, 79), "saved actual shape differs")
    require(np.array_equal(qpos[0], task.initial_qpos) and np.array_equal(qvel[0], task.initial_qvel), "initial state differs")
    require(np.array_equal(qpos, replay["qpos"]) and np.array_equal(qvel, replay["qvel"]), "actual/replay parity failed")
    delta = float(np.max(np.abs(times-replay["time"])))
    require(delta <= 1e-9 and np.allclose(times, np.arange(n+1)*.002, rtol=0, atol=1e-9), "clock parity failed")
    require(all(np.isfinite(v).all() for v in (*actual.values(), *replay.values())), "nonfinite saved states")
    replay["qpos"], replay["qvel"], replay["time"] = qpos, qvel, times
    provider = TaskAnchoredResidualReferenceProvider(task, plan)
    reference = provider.continuum_kinematics(times)[0]
    base = reference-plan.offset_kinematics(times)[0]
    rigid = replay["target_position"]+np.einsum("nij,j->ni", replay["target_rotation"], task.scenario["grasp_point_target_frame_m"])
    c = float(np.max(np.abs(reference[1:]-actual["generated_continuum_position"])))
    r = float(np.max(np.abs(rigid[:-1]-actual["generated_rigid_position"])))
    require(c <= 1e-12 and r <= 1e-12, "generated reference binding differs")
    parity = {"state_count": n+1, "qpos_bit_exact": True, "qvel_bit_exact": True,
              "time_max_abs_difference_s": delta, "fresh_replay_is_actual_state_source": False}
    binding = {"continuum_poststep_max_abs_m": c, "rigid_cached_prestep_max_abs_m": r,
               "poststep_saved_reference_samples": n, "scope": "generated reference; consumed QP reference is separate"}
    frozen = {"slot_id": slot_id, "mode": method, "source_role": "C1_UNIQUE_ACTUAL",
              "reference_version": plan.representation_version,
              "amplitude_m": float(np.max(np.linalg.norm(plan.z_m, axis=1)))}
    return sources, frozen, task, plan, result, {"quality_label_eligible": slot["full_task_success"]}, evaluation, replay, reference, base, parity, binding


def annotate(frame, frozen, task, plan, result, replay, state, view, reference, base, *, smoke, strip):
    image = Image.new("RGB", (frame.shape[1], frame.shape[0]+strip), (10, 14, 24))
    image.paste(Image.fromarray(frame), (0, strip))
    draw = ImageDraw.Draw(image)
    c = np.linalg.norm(replay["continuum_position"][state]-reference[state])*1000
    b = np.linalg.norm(replay["continuum_position"][state]-base[state])*1000
    z = np.asarray(plan.z_m)[[1, 2]].ravel()*1000
    status = "COMPLETE ACTUAL" if result["full_task_success"] else "FAILED PREFIX | UNEXECUTED TAIL OMITTED"
    lines = [f"C.1 {task.task_id} | {frozen['mode']} | {view.upper()}",
             f"{plan.representation_version.rsplit('_',1)[-1]} | active z(mm)={np.round(z,3).tolist()}",
             f"{status} | saved t={replay['time'][state]:.3f}/{replay['time'][-1]:.3f}s",
             f"Generated C reference error={c:.2f}mm | independent Task deviation={b:.2f}mm",
             "green=generated ref; red=Task route; cyan=actual; yellow/blue=Task/actual rigid",
             "Actual saved states + forward | 0 physics | deployment NOT_MET"]
    for i, line in enumerate(lines):
        size = 12 if image.width < 900 else 16
        font = legacy._font(size)
        while draw.textlength(line, font=font) > image.width-14 and size > 8:
            size -= 1
            font = legacy._font(size)
        draw.text((7, 3+i*((strip-6)//len(lines))), line, font=font, fill=(225,234,245))
    return np.asarray(image)


def producer_records():
    return [engine.record(p) for p in (Path(__file__), Path(engine.__file__), Path(legacy.__file__),
            ROOT/"v6_lite/visualization/latest_saved_renderer.py", ROOT/"v6_lite/visualization/generate_visualizations.py")]


def render_one(source, output, slot_id):
    engine.load_slot, engine.annotate, engine.generator_records, engine.SCHEMA = load_slot, annotate, producer_records, SCHEMA
    original = legacy.make_model
    def checked_model(task, evaluation, directory, slot):
        values = original(task, evaluation, directory, slot)
        attempt = Path(source)/"actual"/task.task_id/slot_id.rsplit("_",1)[1]/"attempt"
        slot = read(attempt.parent/"slot.json")
        require(values[3] == slot["quality"]["source_compiled_model_sha256"], "compiled actual model differs")
        if (attempt/"actual/historical_metric_observations.json").exists():
            historical = read(attempt/"actual/historical_metric_observations.json")
            require(values[3] == historical["execution_contract"]["source_compiled_model_sha256"], "historical compiled model differs")
        return values
    legacy.make_model = checked_model
    try:
        metadata = engine.render_slot(source, output, slot_id)
    finally:
        legacy.make_model = original
    plan = read(Path(source)/"actual"/metadata["task_id"]/metadata["mode"]/"attempt/plan.json")
    metadata.update({"schema": SCHEMA, "z_m": plan["z_m"], "active_coefficients": [1,2],
                     "experiment_manifest_sha256": sha(Path(source)/"manifest.json"),
                     "failed_prefix": not metadata["full_task_success"],
                     "amplitude_scope": "maximum interval vector norm; full z_m is authoritative"})
    write(Path(output)/"replays"/slot_id/"replay_metadata.json", metadata)
    print(json.dumps({"rendered": slot_id, "frames": metadata["frame_count"], "end_s": metadata["source_end_s"]}), flush=True)
    return metadata


def angle(actual, reference):
    trace = np.einsum("nij,nij->n", actual, reference)
    return np.rad2deg(np.arccos(np.clip((trace-1)/2, -1, 1)))


def csv_save(path, columns):
    with Path(path).open("w", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def plot_one(source, output, slot_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sources, frozen, task, plan, result, quality, evaluation, f, reference, base, parity, binding = load_slot(source, slot_id)
    trace_path = sources.paths["actual_trace"]["path"]
    with np.load(trace_path, allow_pickle=False) as archive:
        tr = {k: archive[k] for k in archive.files if k.startswith("task_") and not archive[k].dtype.hasobject}
        torque = archive["torque"]
    count = result["actual_steps"]//10
    tt = tr["task_time"][:count]
    indices = np.rint(tt/.002).astype(int)
    require(len(tt) == count and np.max(np.abs(f["time"][indices]-tt)) <= 1e-9, "consumed tick clock differs")
    require(np.max(np.abs(f["continuum_position"][indices]-tr["task_qp_continuum_actual_position_m"][:count])) <= 1e-12,
            "saved consumed actual kinematics differ")
    ref = {arm: tr["task_input_reference_"+arm+"_target_position"][:count] for arm in ("continuum", "rigid")}
    errors = {arm: np.linalg.norm(f[arm+"_position"][indices]-ref[arm],axis=1) for arm in ref}
    rotations = {arm: angle(f[arm+"_rotation"][indices], tr["task_input_reference_"+arm+"_target_rotation"][:count]) for arm in ref}
    prefs = read(sources.bind("preferences", Path(source)/"frozen_tasks"/task.task_id/"preferences.json"))[0]
    t = f["time"]
    keep = np.unique(np.r_[np.arange(0,len(t),10),len(t)-1])
    folder = Path(output)/"figures"/slot_id
    folder.mkdir(parents=True, exist_ok=False)
    status = "COMPLETE" if result["full_task_success"] else "FAILED PREFIX (tail unexecuted)"
    title = f"{task.task_id} | {frozen['mode']} | {status} | saved end {t[-1]:.3f} s"
    figures = {}
    def finish(fig, stem, foot):
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0,.07,1,.95))
        fig.text(.02,.015,foot,fontsize=8)
        path = folder/(stem+".png")
        fig.savefig(path,dpi=145)
        plt.close(fig)
        figures[stem] = path.relative_to(output).as_posix()
    def windows(ax):
        for low, high in prefs["W_support"]:
            ax.axvspan(low,high,color="#e7d8ef",alpha=.45)
        ax.set_xlim(0,27)
        if not result["full_task_success"]:
            ax.axvspan(t[-1],27,color="#ddd",alpha=.45,hatch="//")
        ax.set_xlabel("Physical time (s)")
        ax.grid(alpha=.2)
    fig, axes = plt.subplots(2,3,figsize=(13,7))
    for row, arm in enumerate(ref):
        for col, (a,b) in enumerate(((0,1),(1,2),(0,2))):
            ax = axes[row,col]
            ax.plot(f[arm+"_position"][keep,a],f[arm+"_position"][keep,b],label="Actual",color="#087cad")
            ax.plot(ref[arm][:,a],ref[arm][:,b],"--",label="Consumed QP reference",color="#e87c30")
            if arm == "continuum":
                ax.plot(base[keep,a],base[keep,b],":",label="Independent Task route",color="#555")
                points = np.asarray(task.scenario["continuum_target"]["waypoint_points_m"])
                ax.scatter(points[:,a],points[:,b],marker="x",s=15,color="#555")
            ax.scatter(f[arm+"_position"][-1,a],f[arm+"_position"][-1,b],marker="s",s=20,color="#087cad")
            ax.set(xlabel=f"World {'XYZ'[a]} (m)",ylabel=f"{arm} {'XYZ'[b]} (m)")
            ax.set_aspect("equal",adjustable="datalim"); ax.grid(alpha=.2)
            if col == 0: ax.legend(fontsize=8)
    finish(fig,"trajectory","Saved actual/replay kinematics have exact full-state parity. Blue square = last executed state. No continuation beyond a failed prefix.")
    fig, axes = plt.subplots(2,2,figsize=(12,7))
    for col, arm in enumerate(ref):
        axes[0,col].plot(tt,errors[arm]*1000,label="Actual - consumed QP reference",color="#087cad")
        axes[1,col].plot(tt,rotations[arm],color="#087cad")
        if arm == "continuum":
            axes[0,col].plot(t[keep],np.linalg.norm(f[arm+"_position"][keep]-base[keep],axis=1)*1000,":",label="Actual - independent Task route",color="#555")
        axes[0,col].set_ylabel(arm+" position error (mm)")
        axes[1,col].set_ylabel(arm+" orientation error (deg)")
        axes[0,col].legend(fontsize=8)
        for row in range(2): windows(axes[row,col])
    finish(fig,"tracking","Errors use exact consumed pre-step QP ticks. Purple = frozen W_support union. Gray hatched tail = unexecuted. Task-route deviation is separate from tracking error.")
    qdir = Path(source)/"actual"/task.task_id/frozen["mode"]/"quality"
    clearance_path = sources.bind("support_clearance",qdir/"support_clearance.npz")
    with np.load(clearance_path,allow_pickle=False) as z:
        ct, cd = z["time"], z["signed_distance_m"]
    initial = np.asarray(task.base_pose)
    translation = np.linalg.norm(f["base_pose"][:,:3]-initial[:3],axis=1)
    quat = f["base_pose"][:,3:]/np.linalg.norm(f["base_pose"][:,3:],axis=1)[:,None]
    initial_q = initial[3:]/np.linalg.norm(initial[3:])
    rotation = np.rad2deg(2*np.arccos(np.clip(np.abs(quat@initial_q),0,1)))
    fig, axes = plt.subplots(2,2,figsize=(12,7))
    for ax, x, y, label in ((axes[0,0],t[keep],translation[keep]*1000,"Base translation drift (mm)"),
             (axes[0,1],t[keep],rotation[keep],"Base rotation drift (deg)"),
             (axes[1,0],t[keep],f["target_minimum_m"][keep]*1000,"Robot-target clearance (mm)"),
             (axes[1,1],ct,cd*1000,"Related-sphere clearance in W_support (mm)")):
        # Support windows are disjoint: do not draw a fictitious bridge.
        if ax is axes[1,1]:
            for low, high in prefs["W_support"]:
                use = (x >= low-1e-9)&(x <= high+1e-9)
                ax.plot(x[use],y[use],color="#087cad",lw=1)
            ax.axhline(30,color="#e87c30",ls="--",label="C.1 30 mm preference"); ax.legend(fontsize=8)
        else: ax.plot(x,y,color="#087cad",lw=1)
        ax.set_ylabel(label); windows(ax)
    axes[1,0].axhline(5,color="#e87c30",ls="--",label="Original 5 mm gate"); axes[1,0].legend(fontsize=8)
    finish(fig,"safety_base","Original saved discrete geometry evidence only. Related-sphere and robot-target pair sets differ. No new clearance query or safety acceptance.")
    intervention = tr["task_qp_selected_velocity"][:count]-tr["task_qp_box_nominal_velocity"][:count]
    require(np.max(np.abs(np.linalg.norm(intervention,axis=1)-tr["task_avoidance_intervention"][:count])) <= 1e-12,"intervention scalar differs")
    fig, axes = plt.subplots(2,2,figsize=(12,7))
    axes[0,0].plot(tt,np.linalg.norm(intervention,axis=1),label="17D intervention")
    axes[0,0].set_ylabel("Velocity difference norm (rad/s)")
    axes[0,1].plot(t[1:][::10],np.linalg.norm(torque,axis=1)[::10],label="67-channel torque L2")
    axes[0,1].plot(t[1:][::10],np.max(np.abs(torque),axis=1)[::10],label="Peak absolute channel")
    axes[0,1].set_ylabel("Applied torque (N m)")
    for ax, keys in ((axes[1,0],("task_pcc_clearance","task_capsule_clearance")),
                     (axes[1,1],("task_interval_current_envelope_margin_m","task_interval_ramp_minimum_envelope_margin_m"))):
        for key in keys: ax.plot(tt,tr[key][:count]*1000,label=key.removeprefix("task_").replace("_"," "),lw=1)
        ax.set_ylabel("Saved original diagnostic (mm)")
    for ax in axes.flat: windows(ax); ax.legend(fontsize=7)
    finish(fig,"control","Only consumed QP inputs and applied torques. Rejected unconsumed reference rows are excluded. Existing PCC, capsule and interval scopes remain distinct.")
    columns = {"time_s":tt,"actual_saved_state_index":indices}
    for arm in ref:
        columns[arm+"_position_error_m"] = errors[arm]
        columns[arm+"_orientation_error_deg"] = rotations[arm]
        for j, axis in enumerate("xyz"):
            columns[arm+"_actual_"+axis+"_m"] = f[arm+"_position"][indices,j]
            columns[arm+"_consumed_reference_"+axis+"_m"] = ref[arm][:,j]
    columns["intervention_17D_rad_s"] = np.linalg.norm(intervention,axis=1)
    csv_save(folder/"consumed_tracking.csv",columns)
    csv_save(folder/"native_diagnostics.csv",{"time_s":t[keep],"base_translation_m":translation[keep],"base_rotation_deg":rotation[keep],
             "robot_target_minimum_m":f["target_minimum_m"][keep],"robot_target_censored":f["target_minimum_censored"][keep]})
    csv_save(folder/"support_clearance.csv",{"time_s":ct,"signed_distance_m":cd})
    sources.unchanged()
    return {"slot_id":slot_id,"figures":figures,"consumed_ticks":count,"source_end_s":float(t[-1]),
            "tracking_scope":"actual versus consumed reference, exact pre-step QP ticks, whole executed horizon; prefixes not full-task metrics",
            "tracking":{arm:{"position_rms_m":float(np.sqrt(np.mean(errors[arm]**2))),"position_peak_m":float(errors[arm].max()),
                         "orientation_rms_deg":float(np.sqrt(np.mean(rotations[arm]**2)))} for arm in ref},
            "sources":sources.paths,"csv":[p.relative_to(output).as_posix() for p in folder.glob("*.csv")]}


def gallery(output, rows, media, plots):
    media = {m["slot_id"]:m for m in media}
    plots = {p["slot_id"]:p for p in plots}
    pages, logical = [], []
    for row in rows:
        slot = row["task_id"]+"_"+row["method"]
        evidence = row.get("evidence_method")
        source_slot = row["task_id"]+"_"+evidence if evidence else None
        m, p = media.get(source_slot), plots.get(source_slot)
        alias = row.get("alias_of_method")
        title = f"{row['task_id']} · {row['method']}"
        detail = f"{row['status']} · "+(f"复用 {alias} 的同一实际证据" if alias else "独立实际轨迹" if m else "无计划，不执行 fallback，无视频或误差轨迹")
        content = f"<h2>{html.escape(title)}</h2><p>{html.escape(detail)}</p>"
        if m:
            content += f"<p>保存终点 {m['source_end_s']:.3f}s；{'完整27秒' if m['full_task_success'] else '失败前缀，后续未执行'}。<a href='../{m['slot_id']}/replay_metadata.json'>来源与帧时序</a></p>"
            for key, label, poster in (("five_view_grid","五视角合成",m["preview"]),("continuum_focus","连续体侧视 / 聚焦",m["focus_preview"])):
                content += f"<h3>{label}</h3><video controls preload='none' poster='../../{poster}' src='../../{m['videos'][key]}'></video>"
            content += "<p>单独视角："+" · ".join(f"<a href='../../{m['videos'][key]}'>{key}</a>" for key in legacy.VIEWS)+"</p>"
            for key,label in (("trajectory","双臂末端轨迹"),("tracking","位置与姿态跟踪误差"),("safety_base","净空及基座漂移"),("control","干预、力矩及原有诊断")):
                content += f"<h3>{label}</h3><a href='../../{p['figures'][key]}'><img loading='lazy' src='../../{p['figures'][key]}'></a>"
            content += "<p>数值下载："+" · ".join(f"<a href='../../{path}'>{Path(path).name}</a>" for path in p["csv"])+"</p>"
        page = Path(output)/"replays"/slot/"index.html"
        page.parent.mkdir(parents=True,exist_ok=True)
        page.write_text(shell(title,"<p><a href='../../index.html'>返回全部16槽</a></p>"+content),encoding="utf8",newline="\n")
        logical.append({"task_id":row["task_id"],"method":row["method"],"status":row["status"],"alias_of_method":alias,
                        "evidence_slot":source_slot,"page":page.relative_to(output).as_posix(),"media_available":m is not None})
        poster = f"<img loading='lazy' src='{m['focus_preview']}'>" if m else "<div class='empty'>NO_PLAN</div>"
        pages.append(f"<article><a href='{page.relative_to(output).as_posix()}'>{poster}<h3>{title}</h3></a><p>{detail}</p></article>")
    body = "<h1>V6.4-C.1 视频与跟踪诊断</h1><p>4任务 × Z0 / G0 / OI / OC；16逻辑槽，12条独立实际轨迹，3个复用槽，1个 NO_PLAN。共84段视频、48张诊断图。</p><p><a href='../continuous_route_optimizer_20261007_01/index.html'>结果总览</a> · <a href='collection.json'>来源与验证记录</a></p><p>视频仅读取保存的 actual 状态，五个视角、合成画面和连续体侧视使用同一帧时序。两条失败前缀分别止于11.08s和10.80s；未执行部分不补齐。跟踪误差在实际消耗的QP输入时刻计算，另显示原始任务路线偏差。部署 NOT_MET；可视化不增加安全验收。</p><div class='grid'>"+"".join(pages)+"</div>"
    (Path(output)/"index.html").write_text(shell("C.1 视频与跟踪诊断",body),encoding="utf8",newline="\n")
    return logical


def shell(title, body):
    return "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>"+html.escape(title)+"</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#edf3f6;color:#203b50;margin:0}main{max-width:1440px;margin:auto;padding:28px}a{color:#007984}h1{font-size:34px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{background:white;padding:18px;border-radius:14px;border:1px solid #d6e3eb}img,video{width:100%;height:auto;border-radius:10px}video{background:#090e18}.empty{height:180px;background:#dce3e9;display:grid;place-items:center;font-size:28px;color:#5c6e7b}h2{margin-top:28px}h3{margin-bottom:8px}p{max-width:1200px}</style><main>"+body+"</main></html>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--check-inputs",action="store_true")
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    parser.add_argument("--only",help="Render one unique slot; normal production resumes verified completed slots")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output),"output must be separate from raw evidence")
    manifest_sha = sha(source/"manifest.json")
    rows = slots(source)
    unique = [r["task_id"]+"_"+r["method"] for r in rows if r["unique_run"] and r["entered_actual"]]
    require(len(rows)==16 and len(unique)==12 and sum(r["status"]=="NO_PLAN" for r in rows)==1,"C.1 cohort differs")
    for slot in unique:
        loaded = load_slot(source,slot)
        loaded[0].unchanged()
        print(json.dumps({"preflight":slot,"parity":loaded[-2]}),flush=True)
    if args.check_inputs: return
    output.mkdir(parents=True,exist_ok=True)
    candidates = [args.only] if args.only else unique
    require(all(s in unique for s in candidates),"only unique actual slots can be rendered")
    todo = [s for s in candidates if not (output/"replays"/s/"replay_metadata.json").exists()]
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(render_one,source,output,s) for s in todo]
            for future in as_completed(futures): future.result()
    if args.only: return
    media = [read(output/"replays"/s/"replay_metadata.json") for s in unique]
    plots = []
    for slot in unique:
        path = output/"figures"/slot/"plots.json"
        p = read(path) if path.exists() else plot_one(source,output,slot)
        write(path,p)
        plots.append(p)
        print(json.dumps({"plots":slot,"consumed_ticks":p["consumed_ticks"]}),flush=True)
    for m in media:
        require(m["schema"]==SCHEMA and m["experiment_manifest_sha256"]==manifest_sha,"wrong media source")
        require(m["actual_fresh_parity"]["qpos_bit_exact"] and m["actual_fresh_parity"]["qvel_bit_exact"],"state parity missing")
        require(m["includes_exact_saved_endpoint"] and not m["smoke_truncated"],"truncated media")
        require(m["render_cost"]["physics_steps"]==m["render_cost"]["geometry_queries"]==m["render_cost"]["QP_solves"]==0,"render-only contract failed")
        for v in m["video_records"].values():
            require(sha(output/v["relative_path"])==v["sha256"],"media digest differs")
            legacy.probe(output/v["relative_path"],m["frame_count"],v["width"],v["height"],m["fps"])
    logical = gallery(output,rows,media,plots)
    require(sha(source/"manifest.json")==manifest_sha,"experiment seal changed")
    collection = {"schema":SCHEMA,"experiment_manifest_sha256":manifest_sha,
                  "producer_bindings":producer_records(),"slots":logical,"unique_actual":12,"video_count":84,"figure_count":48,
                  "physics_steps_added":0,"geometry_queries_added":0,"QP_solves_added":0,
                  "deployment":"NOT_MET","media":media,"plots":plots,
                  "verification":{"all_84_videos_ffprobe":"PASS","actual_saved_state_parity":"PASS","saved_endpoint_included":"PASS",
                                  "browser_live_preview":"NOT_VERIFIED","experiment_seal_unchanged":"PASS"}}
    write(output/"collection.json",collection)
    files = {p.relative_to(output).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file() and p.name!="media_manifest.json"}
    write(output/"media_manifest.json",{"schema":SCHEMA,"source_manifest_sha256":manifest_sha,"files":files})
    print(json.dumps({"complete":True,"videos":84,"figures":48,"unique_actual":12}),flush=True)


if __name__ == "__main__":
    main()
