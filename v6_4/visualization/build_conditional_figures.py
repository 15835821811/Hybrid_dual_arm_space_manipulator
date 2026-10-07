"""Two B.3 pilot route figures from frozen declarations and fresh saved states.

No repository imports, torch, MuJoCo, inference, dynamics or geometry query.
Missing independent fresh states remain missing. Failed prefixes never receive
full-trajectory quality values. The compact plot_data file is clone-portable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
import numpy as np

SCHEMA = "v64_b3_paired_route_figures_v1"
MODES = ("z0", "z+", "z-")
COLORS = {"z0":"#0072B2", "z+":"#D55E00", "z-":"#009E73"}
LABELS = {"z0":"z0", "z+":"z+ (+12 mm)", "z-":"z- (-12 mm)"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    text=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path=Path(path)
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")


class Inputs:
    def __init__(self,source):
        self.source=Path(source).resolve()
        self.records={}
        self.frozen={}

    def bind(self,path,expected=None):
        path=Path(path)
        if not path.is_absolute(): path=self.source/path
        path=path.resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        if not path.is_relative_to(self.source):
            raise ValueError("plot input escapes the exclusive B.3 run: "+str(path))
        relative=path.relative_to(self.source).as_posix()
        digest=sha(path)
        if expected is not None and expected != digest:
            raise ValueError("source digest differs: "+str(path))
        if relative in self.frozen and digest != self.frozen[relative]:
            raise ValueError("consumed frozen input differs from plan: "+relative)
        record={"path":relative,"sha256":digest,"bytes":path.stat().st_size}
        if relative in self.records and self.records[relative] != record:
            raise ValueError("plot source changed during read: "+relative)
        self.records[relative]=record
        return path

    def read(self,path,expected=None):
        return json.loads(self.bind(path,expected).read_text(encoding="utf-8-sig"))

    def verify_unchanged(self):
        for record in list(self.records.values()): self.bind(record["path"],record["sha256"])


def minimum_jerk_reference(task,times):
    """Same declared C2 polynomial, with no provider or model invocation."""
    source=task["scenario"]["continuum_target"]
    if source.get("mode")!="irregular_waypoints" or source.get("reference_profile")!="minimum_jerk_c2":
        raise ValueError("unsupported base reference")
    times=np.asarray(times,dtype=float)
    points=np.asarray(source["waypoint_points_m"],dtype=float)
    durations=np.asarray(source["segment_durations_s"],dtype=float)
    transition=float(source["transition_duration_s"])
    cumulative=np.r_[0.,np.cumsum(durations)]
    initial=np.asarray(source["initial_position_w"],dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(durations)!=len(points)-1
            or not np.isfinite(points).all() or np.any(durations<=0)
            or abs(durations.sum()-source["path_duration_s"])>1e-9
            or times.ndim!=1 or not np.isfinite(times).all()
            or np.any(times<0) or np.any(times>27)):
        raise ValueError("invalid declared minimum jerk path")
    result=np.empty((len(times),3))
    for i,t in enumerate(times):
        if t<transition:
            lower,upper=initial,points[0]
            u=t/transition
        elif t-transition>=source["path_duration_s"]:
            result[i]=points[-1]
            continue
        else:
            elapsed=t-transition
            segment=min(max(int(np.searchsorted(cumulative,elapsed,side="right")-1),0),len(durations)-1)
            lower,upper=points[segment],points[segment+1]
            u=(elapsed-cumulative[segment])/durations[segment]
        progress=10*u**3-15*u**4+6*u**5
        result[i]=lower+progress*(upper-lower)
    return result


def residual_offsets(definition,z,times):
    z=np.asarray(z,dtype=float)
    if z.shape!=(6,2) or not np.isfinite(z).all(): raise ValueError("invalid fixed raw z")
    output=np.zeros((len(times),3))
    for interval,basis,coefficients,enabled in zip(definition["intervals_s"],definition["transverse_bases"],z,definition["interval_mask"]):
        if not enabled:
            if np.any(coefficients): raise ValueError("nonzero inactive fixed candidate")
            continue
        lower,upper=interval
        inside=(times>lower)&(times<upper)
        u=np.where(inside,(times-lower)/(upper-lower),0.)
        output+=np.where(inside,64*u**3*(1-u)**3,0.)[:,None]*(np.asarray(basis)@coefficients)
    return output


def sample_indices(times,max_points=1100,extra_times=()):
    indices=np.linspace(0,len(times)-1,min(len(times),max_points)).round().astype(int)
    if len(times):
        indices=np.r_[indices,0,len(times)-1,*[int(np.argmin(np.abs(times-t))) for t in extra_times]]
    return np.unique(indices)


def load_actual(inputs,declared,quality):
    """Only the independently produced fresh_replay.npz supplies plotted states."""
    slot=declared["slot_id"]
    directory=Path("pilot/attempts")/slot
    attempt=inputs.read(directory/"attempt_result.json")
    if (attempt["task_id"],attempt["slot_id"])!=(declared["task_id"],slot):
        raise ValueError("pilot attempt identity differs from declared slot")
    if attempt["task_sha256"]!=quality["task_sha256"]:
        raise ValueError("quality/attempt task binding differs")
    attempted_task=inputs.read(directory/"task.json")
    if canonical_sha(attempted_task)!=attempt["task_sha256"]:
        raise ValueError("saved attempted TaskSpec does not match source identity")
    if quality["candidate_name"]!=declared["candidate_name"]:
        raise ValueError("quality/declared candidate differs")
    if quality["route_intervals_s"]!=[declared["T_route_s"]]:
        raise ValueError("quality route interval changed")
    for name,digest in quality["sources"].items(): inputs.bind(name,digest)
    eligible=bool(quality.get("quality_label_eligible") is True and
                  quality.get("safety",{}).get("full_task_and_original_safety_passed") is True)
    metrics=quality.get("full_metrics") if eligible else None
    if eligible and not metrics: raise ValueError("eligible quality lacks full metrics")
    if not eligible and quality.get("full_metrics"):
        raise ValueError("ineligible run was given full trajectory metrics")
    # Retain portable numbers, rather than copying absolute source paths or
    # large collision-pair records into a plot payload.
    if metrics:
        metrics={k:metrics[k] for k in ("I_route_rad_s","I_full_rad_s","continuum_path_length_m",
            "continuum_route_window_path_length_m","base_translation_peak_m","base_rotation_peak_rad",
            "saved_horizon_s","saved_native_state_count") if k in metrics} | {
            "continuum_route_obstacle_clearance":{k:quality["full_metrics"]["continuum_route_obstacle_clearance"][k]
            for k in ("obstacle_name","pair_count","full_saved_horizon_minimum_m","route_window_minimum_m",
                      "native_period_s","state_count","source_compiled_model_sha256")
            if k in quality["full_metrics"]["continuum_route_obstacle_clearance"]}}
    def compact_failure(failure):
        return {k:failure[k] for k in ("type","message") if k in failure} if isinstance(failure,dict) else failure
    unavailable=quality.get("metric_unavailable")
    record={"slot_id":slot,"task_id":declared["task_id"],"candidate_name":declared["candidate_name"],
        "status":attempt["status"],"actual_physics_steps":attempt.get("actual_steps",0),
        "full_task_success":attempt.get("full_task_success") is True,
        "complete_quality_eligible":eligible,"full_metrics":metrics,
        "execution_failure":compact_failure(attempt.get("execution_failure")),
        "pipeline_failure":compact_failure(attempt.get("pipeline_failure")),
        "metric_unavailable_reason":unavailable.get("reason") if isinstance(unavailable,dict) else unavailable,
        "fresh_states_available":False,"saved_horizon_s":None,"time_s":[],"continuum_position_world_m":[]}
    if not attempt.get("evaluation_path"):
        if eligible: raise ValueError("eligible run lacks independent evaluation")
        record["fresh_state_status"]="NO_INDEPENDENT_FRESH_STATES"
        return record
    evaluation_path=inputs.bind(attempt["evaluation_path"],attempt["evaluation_sha256"])
    evaluation=inputs.read(evaluation_path)
    if evaluation.get("task_sha256",attempt["task_sha256"])!=attempt["task_sha256"]:
        raise ValueError("independent evaluation belongs to another task")
    evaluation_manifest=inputs.read(evaluation_path.parent/"manifest.json")
    fresh=evaluation_path.parent/"fresh_replay.npz"
    if not fresh.exists():
        if eligible: raise ValueError("eligible run lacks independent fresh states")
        if "fresh_replay.npz" in evaluation_manifest: raise ValueError("declared independent fresh file missing")
        record["fresh_state_status"]="NO_INDEPENDENT_FRESH_STATES"
        record["independent_evaluation_failed"]=bool(evaluation.get("errors"))
        return record
    if "fresh_replay.npz" not in evaluation_manifest:
        raise ValueError("independent fresh file is not hash-bound")
    inputs.bind(fresh,evaluation_manifest["fresh_replay.npz"])
    with np.load(fresh,allow_pickle=False) as data:
        times=np.asarray(data["time"],dtype=float)
        position=np.asarray(data["continuum_position"],dtype=float)
    if (times.ndim!=1 or not len(times) or position.shape!=(len(times),3)
            or not np.isfinite(times).all() or not np.isfinite(position).all()
            or abs(times[0])>1e-9 or np.any(np.diff(times)<=0)
            or times[-1]>27+1e-9):
        raise ValueError("independent fresh state times/positions invalid")
    if len(times)!=attempt.get("actual_steps",0)+1 or abs(times[-1]-.002*attempt.get("actual_steps",0))>1e-8:
        raise ValueError("fresh state prefix differs from consumed attempted steps")
    if eligible and (len(times)!=13501 or abs(times[-1]-27)>1e-9):
        raise ValueError("eligible quality must bind all 27 seconds of fresh states")
    indices=sample_indices(times,extra_times=declared["T_route_s"])
    record.update({"fresh_states_available":True,"saved_horizon_s":float(times[-1]),
        "fresh_state_status":"INDEPENDENT_FRESH_FULL" if eligible else "INDEPENDENT_FRESH_PREFIX_OR_INELIGIBLE",
        "time_s":times[indices].tolist(),"continuum_position_world_m":position[indices].tolist(),
        "source_state_count":len(times),"selected_source_state_indices":indices.tolist(),
        "exact_saved_endpoint_included":int(indices[-1])==len(times)-1,
        "state_source":fresh.relative_to(inputs.source).as_posix(),
        "source_fresh_sha256":sha(fresh)})
    return record


def build_plot_data(source):
    inputs=Inputs(source)
    plan=inputs.read("plan.json")
    inputs.frozen={r["path"]:r["sha256"] for r in plan["frozen_task_artifacts"]}
    pairs=inputs.read("route_pair_inputs.json",plan["route_pair_inputs_sha256"])
    decision=inputs.read("pilot/decision.json")
    qualities=inputs.read("pilot/quality_records.json")
    if len(qualities)!=6 or len(plan["pilot_slots"])!=6:
        raise ValueError("all six terminal pilot slots required before figures")
    if len({r["slot_id"] for r in qualities})!=6: raise ValueError("duplicate quality slot")
    by_slot={r["slot_id"]:r for r in qualities}
    pilot_pair=[p for p in pairs["pair_records"] if p["role"]=="pilot"]
    if len(pilot_pair)!=1: raise ValueError("exactly one fixed pilot pair required")
    pair=pilot_pair[0]
    scenes=[]
    for ti,task_id in enumerate(pair["task_ids"]):
        task=inputs.read(f"tasks/{task_id}/task.json")
        definition=inputs.read(f"tasks/{task_id}/definition.json")
        if canonical_sha(task)!=pair["task_sha256"][ti] or definition["task_sha256"]!=canonical_sha(task):
            raise ValueError("frozen pair task/definition digest mismatch")
        declared=[d for d in plan["pilot_slots"] if d["task_id"]==task_id]
        if {d["candidate_name"] for d in declared}!=set(MODES) or len(declared)!=3:
            raise ValueError("every side requires the fixed z0/z+/z- slots")
        slot=int(pair["key_interval_slot"])
        route=pair["T_route_s"]
        if route!=definition["intervals_s"][slot] or any(d["T_route_s"]!=route for d in declared):
            raise ValueError("frozen route window mismatch")
        times=np.unique(np.r_[np.arange(1351)*.02,route,np.mean(route)])
        base=minimum_jerk_reference(task,times)
        candidates=[]
        actual=[]
        for mode in MODES:
            d=next(d for d in declared if d["candidate_name"]==mode)
            fixed=inputs.read(d["plan_path"])
            if fixed["definition"]!=definition: raise ValueError("fixed candidate definition changed")
            z=np.asarray(fixed["z_m"],dtype=float)
            expected=np.zeros((6,2))
            if mode!="z0": expected[slot,0]=.012 if mode=="z+" else -.012
            if not np.array_equal(z,expected): raise ValueError("pilot fixed coefficient changed")
            candidates.append({"mode":mode,"z_m":z.tolist(),"reference_position_world_m":(base+residual_offsets(definition,z,times)).tolist()})
            if by_slot[d["slot_id"]]["task_sha256"]!=canonical_sha(task):
                raise ValueError("route quality belongs to another frozen task")
            actual.append(load_actual(inputs,d,by_slot[d["slot_id"]]))
        sphere=task["scenario"]["workspace_obstacles"][1]
        basis=np.asarray(definition["transverse_bases"][slot])
        tangent=np.cross(basis[:,0],basis[:,1])
        tangent/=np.linalg.norm(tangent)
        center=minimum_jerk_reference(task,[np.mean(route)])[0]
        requirements=[p for p in task["requirements"] if p["arm"]=="continuum" and p["frame"]=="world"]
        scenes.append({"task_id":task_id,"side":"c_plus" if ti==0 else "c_minus",
            "task_sha256":canonical_sha(task),"T_route_s":route,"key_interval_slot":slot,
            "reference_times_s":times.tolist(),"base_reference_position_world_m":base.tolist(),
            "reference_candidates":candidates,"actual_records":actual,
            "moved_sphere":{"name":sphere["name"],"center_world_m":sphere["center_w"],"radius_m":sphere["radius_m"]},
            "projection":{"origin_world_m":center.tolist(),"horizontal_world_direction":tangent.tolist(),
                          "vertical_world_direction":basis[:,0].tolist(),"units":"mm"},
            "continuum_world_requirements":requirements})
    inputs.verify_unchanged()
    return {"schema":SCHEMA,"run_id":Path(source).name,"pilot_decision":decision,"scenes":scenes,
            "input_sources":list(inputs.records.values()),"physics_steps":0,"inference_calls":0,
            "optimizer_updates":0,"geometry_queries":0,"actual_trace_substitution":False,
            "paired_scene_mirroring_used":False,"full_quality_requires_all_independent_gates":True,
            "deployment":"NOT_MET","figure_scope":"two development pilot task sides; not independent TEST or global route optimum"},inputs


def _project(scene,positions):
    p=np.asarray(positions)-scene["projection"]["origin_world_m"]
    return np.c_[p@np.asarray(scene["projection"]["horizontal_world_direction"]),
                 p@np.asarray(scene["projection"]["vertical_world_direction"])]*1000


def draw_scene(data,scene,output):
    fig=plt.figure(figsize=(16,10),layout="constrained")
    gs=fig.add_gridspec(2,2,width_ratios=(1.13,1),height_ratios=(1.2,.88))
    local=fig.add_subplot(gs[0,0]); world=fig.add_subplot(gs[0,1],projection="3d")
    time_ax=fig.add_subplot(gs[1,0]); table_ax=fig.add_subplot(gs[1,1]); table_ax.axis("off")
    times=np.asarray(scene["reference_times_s"])
    base=np.asarray(scene["base_reference_position_world_m"])
    low,high=scene["T_route_s"]
    route=(times>=low)&(times<=high)
    mid=float((low+high)/2)
    base_xy=_project(scene,base)
    sphere=scene["moved_sphere"]
    sphere_xy=_project(scene,[sphere["center_world_m"]])[0]
    local.add_patch(Circle(sphere_xy,sphere["radius_m"]*1000,facecolor="#D9A441",edgecolor="#8B6515",alpha=.25,lw=1.3,label="moved sphere (projection)"))
    local.plot(base_xy[route,0],base_xy[route,1],color="#2F3437",lw=3,alpha=.55,label="base reference in T_route")
    world.plot(*base.T,color="#9AA2AA",lw=1.8,label="base reference, 0–27 s")
    world.plot(*base[route].T,color="#2F3437",lw=3,label="base T_route")
    world.scatter(*sphere["center_world_m"],color="#D9A441",s=85,edgecolor="#8B6515",label="moved sphere center")
    # Wire sphere at its real world radius; an endpoint projection is never
    # presented as collision geometry for the whole continuum arm.
    u=np.linspace(0,2*np.pi,30);v=np.linspace(0,np.pi,15)
    c=np.asarray(sphere["center_world_m"]);r=sphere["radius_m"]
    world.plot_wireframe(c[0]+r*np.outer(np.cos(u),np.sin(v)),c[1]+r*np.outer(np.sin(u),np.sin(v)),
                         c[2]+r*np.outer(np.ones_like(u),np.cos(v)),color="#B58423",alpha=.25,lw=.5,rstride=3,cstride=3)
    for cand in scene["reference_candidates"]:
        mode=cand["mode"];p=np.asarray(cand["reference_position_world_m"]);xy=_project(scene,p)
        local.plot(xy[route,0],xy[route,1],ls="--",color=COLORS[mode],lw=1.7,label=LABELS[mode]+" reference")
        direction=np.asarray(scene["projection"]["vertical_world_direction"])
        displacement=(p-base)@direction*1000
        time_ax.plot(times,displacement,ls="--",lw=1.3,color=COLORS[mode])
    rows=[];missing=[]
    for rec in scene["actual_records"]:
        mode=rec["candidate_name"];color=COLORS[mode]
        horizon=rec["saved_horizon_s"]
        state_label="full safe 27.000 s" if rec["complete_quality_eligible"] else (
            f"prefix/ineligible {horizon:.3f} s" if horizon is not None else f"{rec['actual_physics_steps']*.002:.3f} s; no fresh")
        status_cell="PASS / 27.000 s" if rec["complete_quality_eligible"] else (
            textwrap.fill(rec["status"].replace("_"," "),28)+"\n"+state_label)
        if rec["fresh_states_available"]:
            t=np.asarray(rec["time_s"]);p=np.asarray(rec["continuum_position_world_m"])
            xy=_project(scene,p);window=(t>=low)&(t<=high)
            if window.any():
                local.plot(xy[window,0],xy[window,1],color=color,lw=2.2,label=mode+" actual: "+state_label)
            else:
                local.plot([],[],color=color,lw=2.2,label=mode+" actual ends before T_route")
            world.plot(*p.T,color=color,lw=1.6,label=mode+" actual")
            time_ax.plot(t,(p-minimum_jerk_reference_from_scene(scene,t))@np.asarray(scene["projection"]["vertical_world_direction"])*1000,color=color,lw=1.7,label=mode+" fresh actual")
            if not rec["complete_quality_eligible"]:
                world.scatter(*p[-1],marker="x",color=color,s=45,lw=1.7)
                if window.any() and t[-1]<=high: local.scatter(*xy[-1],marker="x",color=color,s=48,lw=1.8)
                time_ax.scatter(t[-1],((p[-1]-minimum_jerk_reference_from_scene(scene,[t[-1]])[0])@np.asarray(scene["projection"]["vertical_world_direction"]))*1000,marker="x",color=color,s=45)
        else: missing.append(mode+": no independent fresh states; no actual curve")
        metrics=rec["full_metrics"]
        if metrics:
            clearance=metrics["continuum_route_obstacle_clearance"]
            rows.append([mode,"PASS / 27.000 s",f"{metrics['I_route_rad_s']:.6f}",
                f"{clearance['route_window_minimum_m']*1000:.3f}",f"{clearance['full_saved_horizon_minimum_m']*1000:.3f}"])
        else: rows.append([mode,status_cell,"—","—","—"])
    local.set_title("Route window: orthographic endpoint-path view",loc="left",weight="bold")
    local.set_xlabel("Along declared route tangent (mm)");local.set_ylabel("First world transverse direction (mm)")
    local.set_aspect("equal",adjustable="datalim")
    local.legend(fontsize=8,loc="upper left",framealpha=.9)
    local.grid(alpha=.18)
    if missing: local.text(.02,.02,"\n".join(missing),transform=local.transAxes,fontsize=8,color="#8D302D",va="bottom",bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
    world.set_title("Saved actual continuum endpoint in world XYZ",loc="left",weight="bold")
    world.set_xlabel("World X (m)");world.set_ylabel("World Y (m)");world.set_zlabel("World Z (m)")
    for axis in (world.xaxis,world.yaxis,world.zaxis): axis.set_major_locator(MaxNLocator(4))
    points=np.vstack([base,c-r,c+r])
    for rec in scene["actual_records"]:
        if rec["fresh_states_available"]: points=np.vstack([points,np.asarray(rec["continuum_position_world_m"])])
    center=(points.max(0)+points.min(0))/2;extent=np.maximum(points.max(0)-points.min(0),.05)
    for dimension,set_lim in enumerate((world.set_xlim,world.set_ylim,world.set_zlim)):
        set_lim(center[dimension]-.55*extent[dimension],center[dimension]+.55*extent[dimension])
    world.set_box_aspect(extent);world.view_init(elev=24,azim=-65);world.legend(fontsize=8,loc="upper left")
    time_ax.axvspan(low,high,color="#F0E2BA",alpha=.65,label="fixed T_route")
    time_ax.axvline(mid,color="#8B6515",lw=.8,alpha=.5)
    time_ax.set_xlim(0,27);time_ax.grid(alpha=.18)
    time_ax.set_xlabel("Physical task time (s)")
    time_ax.set_ylabel("First transverse offset from base reference (mm)")
    time_ax.set_title(f"Frozen route window [{low:.3f}, {high:.3f}] s; dashed=reference",loc="left",weight="bold")
    time_ax.legend(fontsize=8,loc="upper left",ncol=2)
    table_ax.set_title("Complete Task + original independent gates required",loc="left",weight="bold",pad=12)
    table=table_ax.table(cellText=rows,colLabels=["Input","Outcome / saved horizon","I_route\n(rad/s)","Related sphere\nT_route min (mm)","Related sphere\nfull min (mm)"],
        colWidths=[.10,.34,.18,.20,.18],cellLoc="center",bbox=[0,.46,1,.48])
    table.auto_set_font_size(False);table.set_fontsize(8.6)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("#D3D9DE")
        cell.set_facecolor("#E9EFF3" if row==0 else "white")
        if row==0:cell.set_text_props(weight="bold")
    side=next((r for r in data["pilot_decision"]["sides"] if r["task_id"]==scene["task_id"]),{})
    preferred=side.get("preferred_qualifying_direction") or "none established"
    note=("I_route = RMS of the saved 17-D QP nominal-to-selected command norm over the frozen window. "
          "The nominal diagnostic includes the original velocity-bound clipping; source vectors were not logged. "
          "Clearance covers continuum collision geoms versus this moved sphere, at independently saved 2 ms states. "
          "No full quality value is credited to failed/ineligible prefixes. A projected endpoint path is not whole-arm clearance. "
          f"Qualifying preferred direction for this side: {preferred}. Deployment: NOT_MET.")
    table_ax.text(0,.38,textwrap.fill(note,98),ha="left",va="top",fontsize=9,color="#38424C",linespacing=1.45)
    decision_label=("route value identifiable" if data["pilot_decision"].get("route_value_identifiable") is True
                    else "route value not identifiable")
    if data["pilot_decision"]["status"].startswith("SYNTHETIC_TEST"):
        decision_label="SYNTHETIC TEST ONLY — no research result"
    fig.suptitle(f"B.3 development pilot — {scene['side'].replace('_',' ')} — {scene['task_id']}\n"
                 f"One sphere moved; fixed z0 / ±12 mm routes; pair decision: {decision_label}",fontsize=15,weight="bold")
    name="01_pilot_c_plus_routes" if scene["side"]=="c_plus" else "02_pilot_c_minus_routes"
    png=output/(name+".png");pdf=output/(name+".pdf")
    fig.savefig(png,dpi=170,facecolor="white")
    fig.savefig(pdf,facecolor="white")
    plt.close(fig)
    return {"id":name,"task_id":scene["task_id"],"side":scene["side"],"png":png.name,"pdf":pdf.name,
            "caption":note,"T_route_s":scene["T_route_s"]}


def minimum_jerk_reference_from_scene(scene,times):
    """Interpolate plotting baseline only; actual state positions never filled."""
    # reference_times_s is a declared dense 20ms grid, not an actual trajectory.
    # For the time panel an exact analytic formula is preferable: retain the
    # declaration rather than treating the displayed reference as ground truth.
    return minimum_jerk_reference(scene["base_reference_task_declaration"],times)


def build(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output: raise ValueError("derived visualization cannot replace source")
    if output.exists(): raise FileExistsError("exclusive output directory already exists")
    data,inputs=build_plot_data(source)
    # Preserve the small analytic source so exported plot_data remains useful.
    for scene in data["scenes"]:
        task=inputs.read(f"tasks/{scene['task_id']}/task.json")
        scene["base_reference_task_declaration"]={"scenario":{"continuum_target":task["scenario"]["continuum_target"]}}
    data["input_sources"]=list(inputs.records.values())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
    output.mkdir(parents=True)
    panels=[draw_scene(data,scene,output) for scene in data["scenes"]]
    data["panels"]=panels
    write(output/"plot_data.json",data)
    inputs.verify_unchanged()
    artifacts=[{"path":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]
    generator=Path(__file__).resolve()
    write(output/"manifest.json",{"schema":SCHEMA,"source_run_id":source.name,
        "generator":{"name":generator.name,"sha256":sha(generator),"bytes":generator.stat().st_size},
        "inputs":list(inputs.records.values()),"artifacts":artifacts,
        "all_sources_unchanged_after_render":True,"physics_steps":0,"DDIM_calls":0,
        "new_samples":0,"optimizer_updates":0,"geometry_queries":0,
        "failed_prefix_substitution_or_mirroring":False})
    # Structural and image validation is local-only and never an acceptance gate.
    from PIL import Image
    for panel in panels:
        with Image.open(output/panel["png"]) as im:
            if im.width<2000 or im.height<1400 or np.asarray(im.convert("RGB")).std()<5:
                raise ValueError("figure appears blank or is undersized")
        if not (output/panel["pdf"]).read_bytes().startswith(b"%PDF"):
            raise ValueError("PDF output invalid")
    return {"status":"COMPLETED","panels":panels,"input_count":len(inputs.records),
            "output":str(output),"physics_steps":0,"DDIM_calls":0,"actual_curves_use_independent_fresh_states_only":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.output),sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
