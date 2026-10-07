"""Rebuild the B.2 publication views from sealed files, without executing a model.

Only JSON/NPZ reads and analytic evaluation of the declared compact-support basis
are permitted here. No runtime, MuJoCo, torch, optimizer or inference import occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


METHODS = ("E0", "E1", "E2")
COLORS = {"E0": "#0072B2", "E1": "#009E73", "E2": "#D55E00"}
LABELS = {"E0": "E0 zero residual", "E1": "E1 TRAIN retrieval", "E2": "E2 diffusion K1"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def array_json(value):
    if isinstance(value, np.ndarray):
        return array_json(value.tolist())
    if isinstance(value, dict):
        return {str(k): array_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [array_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual(t, definition, plan):
    """Closed-form position offset in world metres; no controller call."""
    z = np.asarray(plan["z_m"], dtype=float)
    out = np.zeros((len(t), 3))
    for i, ((start, end), enabled) in enumerate(zip(definition["intervals_s"], definition["interval_mask"])):
        if not enabled:
            continue
        u = (t - start) / (end - start)
        active = (u > 0) & (u < 1)
        phi = 64 * u[active] ** 3 * (1 - u[active]) ** 3
        out[active] += phi[:, None] * (np.asarray(definition["transverse_bases"][i]) @ z[i])[None, :]
    return out


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in keys}


def sampled_xy(x, y, max_points=1100):
    indices = np.unique(np.linspace(0, len(x) - 1, min(len(x), max_points)).astype(int))
    return x[indices], y[indices]


def shade(ax, definition):
    for low, high in definition["protected_time_intervals_s"]:
        if high > low:
            ax.axvspan(low, high, color="#eeeeee", zorder=0)
    ax.set_xlim(0, 27)


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside sealed source.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "axes.grid": True,
                         "grid.alpha": .18, "figure.dpi": 120, "savefig.dpi": 210,
                         "savefig.bbox": "tight", "axes.axisbelow": True})
    panels, source_files = [], set()

    def save(fig, name, category, title, caption, paths):
        fig.tight_layout(pad=1.3)
        if name.endswith("_actual_paths"):
            fig.subplots_adjust(left=.01,right=.94,wspace=.2,top=.94,bottom=.08)
        relative = f"figures/{len(panels)+1:02d}_{name}"
        fig.savefig(output / (relative + ".png"), facecolor="white")
        fig.savefig(output / (relative + ".pdf"), facecolor="white")
        plt.close(fig)
        paths = sorted(set(str(Path(p)).replace("\\", "/") for p in paths))
        source_files.update(paths)
        panels.append({"id": name, "path": relative + ".png", "pdf_path": relative + ".pdf",
                       "category": category, "title": title, "caption": caption, "source_paths": paths})
        print(f"{len(panels):02d} {name}", flush=True)

    summary = read(source / "summary.json")
    teacher = read(source / "teacher_results.json")
    train = read(source / "training/model/training_report.json")
    curves = list(json_lines(source / "training/model/curves.jsonl"))
    raw = read(source / "raw_reference_diagnostics.json")
    delivery = read(source / "delivery_analysis.json")
    budget = read(source / "budget_ledger.json")
    task_suite = read(source / "tasks.json")
    task_specs = {t["task_id"]: t for t in task_suite["tasks"]}
    all_attempts = [read(p) for p in sorted((source / "attempts").glob("*/attempt_result.json"))]
    test = {}
    fresh_keys = ["time", "q", "dq", "base_pose", "rigid_position", "continuum_position", "target_minimum_m", "target_minimum_censored"]
    trace_keys = ["time", "command_velocity", "torque",
                  "continuum_error", "rigid_error", "continuum_orientation_error_deg", "rigid_orientation_error_deg",
                  "generated_continuum_position", "task_time", "task_selected_command", "task_solver_candidate",
                  "task_interval_selected_rows", "task_interval_proxy_lower_m", "task_interval_current_envelope_margin_m",
                  "task_interval_realized_next_start_minimum_slack_m_s", "task_pcc_clearance", "task_capsule_clearance",
                  "task_mujoco_continuum_target_clearance", "task_full_latency", "task_solver_latency", "torque_latency",
                  "task_interval_preflight_latency_s", "task_interval_qp_only_latency_s", "task_interval_preview_latency_s",
                  "task_interval_next_start_latency_s", "task_interval_full_control_latency_s"]
    for ti in range(4):
        for method in METHODS:
            slot = f"TEST_{ti:02d}_{method}"
            directory = source / "attempts" / slot
            attempt = read(directory / "attempt_result.json")
            tid = attempt["task_id"]
            saved_trace = directory / "actual/traces" / f"{tid}.npz"
            if not saved_trace.is_file():
                candidates = list((directory / "actual/failures").glob("*_partial_trace.npz"))
                if len(candidates) != 1:
                    raise ValueError(f"Expected one saved trace for {slot}")
                saved_trace = candidates[0]
            timing = list(json_lines(directory / "actual/timing" / f"{tid}.jsonl"))
            test[(ti, method)] = {"attempt": attempt, "evaluation": attempt["evaluation"],
                "definition": read(source / "definitions" / f"{tid}.json"),
                "plan": read(directory / "plan.json"),
                "fresh": load_npz(directory / "actual/evaluation/fresh_replay.npz", fresh_keys),
                "trace": load_npz(saved_trace, trace_keys),
                "trace_source_path":saved_trace.relative_to(source).as_posix(),
                "dispatch_ms": np.array([x["dispatch_latency_s"] * 1000 for x in timing]),
                "dispatch_t": np.array([x["source_simulation_time_s"] for x in timing]),
                "timing_phases": {name: np.array([sum(p["wall_s"] for p in x["phases"] if p["name"] == name) * 1000 for x in timing])
                    for name in sorted({p["name"] for x in timing for p in x["phases"]})}}
            del timing

    def sources(ti=None, method=None, kind="attempt_result.json"):
        tis = range(4) if ti is None else [ti]
        ms = METHODS if method is None else [method]
        return [f"attempts/TEST_{i:02d}_{m}/{kind}" for i in tis for m in ms]

    def traces(ti=None):
        return [test[(i,m)]["trace_source_path"]
                for i in (range(4) if ti is None else [ti]) for m in METHODS]

    def fresh_sources(ti=None):
        return sources(ti, kind="actual/evaluation/fresh_replay.npz")

    # 1: fixed-denominator outcomes.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    rows = summary["table_B"]
    values = [r["full_task_success_count"] for r in rows]
    axes[0].bar(METHODS, values, color=[COLORS[m] for m in METHODS])
    axes[0].set(ylim=(0, 4.65), ylabel="Full Task successes / fixed 4 TEST Tasks")
    for i, v in enumerate(values): axes[0].text(i, v+.1, f"{v}/4", ha="center", fontweight="bold")
    matrix = np.array([[test[(i,m)]["attempt"]["full_task_success"] for m in METHODS] for i in range(4)])
    axes[1].imshow(matrix, cmap=ListedColormap(["#f6b8a1", "#b9dccd"]), vmin=0, vmax=1, aspect="auto")
    axes[1].set(xticks=range(3), xticklabels=METHODS, yticks=range(4), yticklabels=[f"TEST {i:03d}" for i in range(4)])
    axes[1].grid(False)
    for i in range(4):
        for j,m in enumerate(METHODS): axes[1].text(j,i,"PASS 27 s" if matrix[i,j] else "REFUSED\n16.620 s",ha="center",va="center")
    save(fig,"fixed_test_outcomes","Outcome","Fixed TEST outcomes", "All 12 predeclared attempts are shown. E2 TEST 001 stopped at 16.620 s under the unchanged ramp-domain guard; no replacement, fallback or K4 actual run. Full Task success includes original independent execution, interval, native geometry and consumed-reference gates. Learning advantage over E1 is not established; deployment remains NOT_MET.",["summary.json","table_B.json"]+sources())

    # 2: teacher matrix.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    matrix = np.array([r["full_task_success"] for r in teacher["records"]]).reshape(8,3)
    ax.imshow(matrix,cmap=ListedColormap(["#f6b8a1","#b9dccd"]),vmin=0,vmax=1,aspect="auto")
    ax.set(xticks=range(3),xticklabels=["+ transverse 1", "- transverse 1", "+ transverse 2"],yticks=range(8),
           yticklabels=[f"TRAIN {i:03d}" for i in range(6)]+[f"VAL {i:03d}" for i in range(2)])
    ax.grid(False); ax.axhline(5.5,color="white",lw=4)
    for i in range(8):
        for j in range(3):ax.text(j,i,"PASS" if matrix[i,j] else "REFUSED 14.900 s\nexcluded from labels",ha="center",va="center")
    save(fig,"teacher_fixed_candidates","Teacher","All 24 fixed teacher outcomes","Exactly three 10 mm patterns per TRAIN/VAL Task were executed. 23/24 completed: TRAIN 17 references from 6 Tasks and VAL 6 references from 2 Tasks. teacher_17 failed the unchanged guard and was excluded from labels; its saved prefix remains available.",["teacher_results.json","dataset/manifest.json"])

    # 3: all actual horizons; failure prefixes never padded.
    order = [next(a for a in all_attempts if a["slot_id"]=="zero_interface")]+[next(a for a in all_attempts if a["slot_id"]==f"teacher_{i:02d}") for i in range(24)]+[test[(i,m)]["attempt"] for i in range(4) for m in METHODS]
    fig,ax=plt.subplots(figsize=(12,4.5)); horizons=[a["actual_steps"]*.002 for a in order]
    ax.bar(range(37),horizons,color=["#009E73" if a["full_task_success"] else "#D55E00" for a in order])
    ax.axhline(27,color="#333333",ls="--",lw=1);ax.set(ylim=(0,30),ylabel="Executed physical horizon (s)",xticks=range(37),xticklabels=["zero"]+[f"T{i:02d}" for i in range(24)]+[f"{i}:{m}" for i in range(4) for m in METHODS]);ax.tick_params(axis="x",rotation=90)
    for i,a in enumerate(order):
        if not a["full_task_success"]:ax.text(i,horizons[i]+.3,f"{horizons[i]:.3f}s",ha="center",fontsize=8)
    save(fig,"all_37_attempt_horizons","Outcome","All 37 actual attempt horizons","One zero interface, 24 teachers, and 12 TEST attempts exhaust the fixed 37-attempt budget. Only saved physical time is plotted; the two guard-refused prefixes are neither padded nor treated as complete successes.",["budget_ledger.json"]+[f"attempts/{a['slot_id']}/attempt_result.json" for a in order])

    # 4/5: training and exposure.
    tc=[c for c in curves if c["kind"]=="train"];vc=[c for c in curves if c["kind"]=="validation"]
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    axes[0].plot([c["update"] for c in tc],[c["loss"] for c in tc],color=COLORS["E0"],label="TRAIN batch loss (every 50)")
    axes[0].plot([c["update"] for c in vc],[c["loss"] for c in vc],"o-",color=COLORS["E2"],label="VAL fixed draws (every 250)")
    axes[0].axvline(train["selected_update"],color="#333333",ls="--",label=f"selected update {train['selected_update']}");axes[0].set(xlabel="Optimizer update",ylabel="v mean squared error");axes[0].legend(fontsize=8)
    for tid in sorted(vc[0]["per_task"]):axes[1].plot([c["update"] for c in vc],[c["per_task"][tid] for c in vc],"o-",label=tid.replace("b2_continuum_route_",""))
    axes[1].axvline(train["selected_update"],color="#333333",ls="--");axes[1].set(xlabel="Optimizer update",ylabel="VAL per-Task v MSE");axes[1].legend()
    save(fig,"training_and_validation","Learning","Training and fixed VAL curves","One fresh model completed 4,000 AdamW updates with 128,000 sample exposures. The selected checkpoint is update 250 (8,000 exposures), chosen by the minimum predeclared fixed VAL loss. TRAIN batch and VAL evaluation estimates use different samples and are not like-for-like test performance.",["training/model/curves.jsonl","training/model/training_report.json"])
    fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    axes[0].plot([c["update"] for c in tc],[c["gradient_norm_before_clip"] for c in tc],color=COLORS["E0"]);axes[0].axhline(1,color="#333333",ls="--",label="clip threshold 1");axes[0].set(xlabel="Optimizer update",ylabel="Gradient norm before clipping");axes[0].legend()
    exposures=train["sample_exposures"];names=list(exposures);axes[1].bar(range(len(names)),list(exposures.values()),color=[COLORS["E0"] if "train" in n else "#999999" for n in names]);axes[1].set(xticks=range(len(names)),xticklabels=[n.replace("b2_continuum_route_","").replace("__candidate_",":") for n in names],ylabel="Training sample exposures");axes[1].tick_params(axis="x",rotation=90,labelsize=7)
    save(fig,"gradient_and_training_exposures","Learning","Gradient and TRAIN-only exposure audit","Gradient clipping was the predeclared optimizer operation. All VAL references have zero training exposures. Unequal TRAIN exposure reflects the frozen Task-balanced draw schedule; it does not create extra independent references.",["training/model/curves.jsonl","training/model/training_report.json"])

    # 6/7: all raw candidate bounds and literal coefficients.
    raw_records=[]
    for ti in range(4):
        tid=test[(ti,"E0")]["attempt"]["task_id"]
        for slot in range(4):
            path=f"learned_candidates/{tid}/E2/slot_{slot:02d}/raw.json";obj=read(source/path);z=np.asarray(obj["raw_z_m"])
            raw_records.append({"task_index":ti,"task_id":tid,"slot":slot,"seed":obj["seed"],"z_m":z.tolist(),"maximum_row_norm_m":float(np.linalg.norm(z,axis=1).max()),"accepted":bool(np.linalg.norm(z,axis=1).max()<=.020),"source_path":path})
    fig,ax=plt.subplots(figsize=(11,3.8));peaks=np.array([r["maximum_row_norm_m"] for r in raw_records])*1000
    ax.bar(range(16),peaks,color=[COLORS["E2"] if r["accepted"] else "#7b3294" for r in raw_records]);ax.axhline(20,color="#333333",ls="--",label="20 mm bound");ax.set(xticks=range(16),xticklabels=[f"{r['task_index']:03d}:{r['slot']}" for r in raw_records],ylabel="Maximum raw 2D row norm (mm)",xlabel="TEST Task : fixed candidate slot");ax.legend()
    for i,r in enumerate(raw_records):
        if not r["accepted"]:ax.text(i,peaks[i]+.2,"REJECT",ha="center",fontsize=8)
    save(fig,"raw_candidate_amplitudes","Reference","All 16 raw amplitude checks","All 16 raw outputs are finite, nonzero and distinct. Fourteen satisfy the unchanged 20 mm row-norm bound; TEST 000 slot 1 and TEST 001 slot 1 are rejected with original out-of-bound values preserved. K1 always uses slot 0; K4 actual was NOT_RUN.",["raw_reference_diagnostics.json"]+[r["source_path"] for r in raw_records])
    fig,axes=plt.subplots(1,4,figsize=(12,3.5));angle=np.linspace(0,2*np.pi,200)
    for ti,ax in enumerate(axes):
        ax.plot(20*np.cos(angle),20*np.sin(angle),color="#444444",ls="--",lw=1)
        for r in raw_records[ti*4:(ti+1)*4]:
            z=np.array(r["z_m"])*1000;ax.scatter(z[:,0],z[:,1],label=f"slot {r['slot']}"+(" reject" if not r["accepted"] else ""),marker=["o","x","^","s"][r["slot"]],s=30)
        ax.set(xlim=(-23,23),ylim=(-23,23),xlabel=f"TEST {ti:03d}: z first (mm)",ylabel="z second (mm)");ax.set_aspect("equal");ax.legend(fontsize=7,loc="lower left")
    save(fig,"raw_coefficient_planes","Reference","Literal raw transverse coefficients","Each point is one of six 2D residual coefficients, before legality rejection. Coefficients are expressed in each interval's frozen transverse basis, so points from different intervals are parameter coordinates rather than a shared world direction. No clipping, projection or re-sampling was used.",[r["source_path"] for r in raw_records])

    # 8: basis and protected windows.
    fig,axes=plt.subplots(3,2,figsize=(12,8));u=np.linspace(0,1,501)
    phi=64*u**3*(1-u)**3;dp=192*u**2-768*u**3+960*u**4-384*u**5;ddp=384*u-2304*u**2+3840*u**3-1920*u**4
    for ax,y,label in zip(axes[0],[phi,dp], ["phi(u)", "d phi / du"]):ax.plot(u,y,color=COLORS["E0"]);ax.set(xlabel="Normalized interval u",ylabel=label)
    for ti,ax in enumerate(axes[1:].flat):
        d=test[(ti,"E0")]["definition"];shade(ax,d)
        for j,(lo,hi) in enumerate(d["intervals_s"]):ax.broken_barh([(lo,hi-lo)],(j-.3,.6),facecolor=COLORS["E1"])
        ax.set(xlabel=f"TEST {ti:03d}: physical time (s)",ylabel="Enabled interval",yticks=range(6),ylim=(-.7,5.7))
    save(fig,"basis_and_protected_windows","Reference","C2 compact support and protected windows","The analytic basis phi(u)=64 u^3 (1-u)^3 has zero position, velocity and acceleration at both ends. Green bars are the deterministic earliest six legal gaps; gray spans protect all declared Task windows and the final reference freeze. No residual is applied in those protected intervals. The second derivative is computed analytically by this script and its endpoint zeros are verified.",["plan.json"]+[f"definitions/{test[(i,'E0')]['attempt']['task_id']}.json" for i in range(4)])
    assert np.allclose([phi[0],phi[-1],dp[0],dp[-1],ddp[0],ddp[-1]],0)

    # 9-12: all fixed paths, both arms, including refused prefix.
    for ti in range(4):
        fig=plt.figure(figsize=(12,5))
        for col,arm in enumerate(["continuum","rigid"]):
            ax=fig.add_subplot(1,2,col+1,projection="3d")
            for m in METHODS:
                rec=test[(ti,m)];f=rec["fresh"];p=f[f"{arm}_position"];complete=rec["attempt"]["full_task_success"]
                plotted=sampled_xy(f["time"],p,max_points=1200)[1]
                ax.plot(*plotted.T,color=COLORS[m],label=m+(" (prefix refused)" if not complete else ""),lw=1.3,ls="--" if not complete else "-")
                if not complete:ax.scatter(*p[-1],color=COLORS[m],marker="x",s=60)
            if arm=="continuum":
                pts=np.array([r["position_m"] for r in task_specs[test[(ti,"E0")]["attempt"]["task_id"]]["requirements"] if r["arm"]==arm]);ax.scatter(*pts.T,color="#222222",s=22,marker="o",label="Task waypoint/terminal")
            ax.set(xlabel="World x (m)",ylabel="World y (m)",zlabel="");ax.text2D(1.04,.52,"World z (m)",rotation=90,va="center",fontsize=9,transform=ax.transAxes);ax.text2D(.03,.97,f"TEST {ti:03d}: {arm}",transform=ax.transAxes);ax.legend(fontsize=7,loc="upper right",bbox_to_anchor=(1.02,.95));ax.view_init(elev=24,azim=-64)
        save(fig,f"test_{ti:03d}_actual_paths","Actual paths",f"TEST {ti:03d}: both actual arm paths", "Fresh replay of the saved applied torques shows the actual continuum and rigid paths. Every predeclared method is included. The refused E2 TEST 001 is drawn only through its saved 16.620 s prefix and marked at the refusal endpoint. Spatial path change alone does not establish quality or learning advantage.",fresh_sources(ti)+sources(ti)+[f"tasks/{test[(ti,'E0')]['attempt']['task_id']}.json"])

    # 13-16: commanded analytic residual vs actual change relative to E0.
    for ti in range(4):
        fig,axes=plt.subplots(2,1,figsize=(11,5.7),sharex=True);base=test[(ti,"E0")]["fresh"]
        for m in METHODS:
            rec=test[(ti,m)];f=rec["fresh"];t=f["time"];delta=residual(t,rec["definition"],rec["plan"])
            style="--" if not rec["attempt"]["full_task_success"] else "-"
            axes[0].plot(*sampled_xy(t,np.linalg.norm(delta,axis=1)*1000),color=COLORS[m],ls=style,label=m)
            samebase=np.column_stack([np.interp(t,base["time"],base["continuum_position"][:,j]) for j in range(3)])
            axes[1].plot(*sampled_xy(t,np.linalg.norm(f["continuum_position"]-samebase,axis=1)*1000),color=COLORS[m],ls=style,label=m+(" prefix only" if style=="--" else ""))
        for ax in axes:shade(ax,test[(ti,"E0")]["definition"]);ax.legend(ncol=3,loc="upper right")
        axes[0].set(ylabel="Analytic reference offset (mm)");axes[1].set(xlabel="Physical time (s)",ylabel="Actual path difference vs E0 (mm)")
        save(fig,f"test_{ti:03d}_residual_and_actual_change","Actual paths",f"TEST {ti:03d}: residual and actual path change","Top: exact analytic residual of the frozen executed plan. Bottom: same-physical-time actual continuum distance from the E0 witness, using fresh saved-torque replay. Gray spans are protected Task windows. The failed E2 prefix is shown as a diagnostic only, not a complete-path quality comparison; no extrapolation or padding occurs.",fresh_sources(ti)+sources(ti,kind="plan.json")+[f"definitions/{test[(ti,'E0')]['attempt']['task_id']}.json"])

    # 17: native Task errors use the chosen same-time pose.
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));error_tables=[]
    for key,tolkey,ax,label in [("position_error_m","position_tolerance_m",axes[0],"position error / tolerance"),("orientation_error_rad","orientation_tolerance_rad",axes[1],"orientation error / tolerance")]:
        table=np.array([[np.nan if r[key] is None else r[key]/r[tolkey] for r in test[(i,m)]["evaluation"]["task_requirements"]["requirements"]] for i in range(4) for m in METHODS]);error_tables.append(table)
        im=ax.imshow(np.ma.masked_invalid(np.log10(np.maximum(table,1e-6))),cmap="viridis",aspect="auto",vmin=-6,vmax=0)
        ax.set(xticks=range(8),xticklabels=[f"C{j}" for j in range(7)]+["R end"],yticks=range(12),yticklabels=[f"{i:03d} {m}" for i in range(4) for m in METHODS],xlabel="Frozen Task requirement");ax.grid(False);fig.colorbar(im,ax=ax,label="log10 "+label)
        for y,x in zip(*np.where(~np.isfinite(table))):ax.text(x,y,"N/R",ha="center",va="center",fontsize=7)
    save(fig,"task_requirement_errors","Task acceptance","Task best-time position and orientation errors","Ratios below 1 meet the original tolerance. Position and orientation are evaluated together at each recorded best_time_s within the frozen Task window; terminal requirements use the actual final state. N/R means the refused prefix never reached that requirement and is not a pass. These are Task errors, distinct from whole-curve tracking RMSE.",sources())

    # 18: original runtime curve diagnostics remain visible.
    fig,axes=plt.subplots(1,3,figsize=(13,4));xs=np.arange(4)
    for mi,m in enumerate(METHODS):
        entries=[]
        for i in range(4):
            tr=test[(i,m)]["trace"];entries.append([np.sqrt(np.mean(tr["continuum_error"]**2))*1000,np.sqrt(np.mean(tr["rigid_error"]**2))*1000])
        entries=np.array(entries)
        for j,ax in enumerate(axes[:2]):
            bars=ax.bar(xs+(mi-1)*.24,entries[:,j],width=.23,color=COLORS[m],label=m)
            if m=="E2":
                bars[1].set_hatch("//");bars[1].set_edgecolor("#333333")
                ax.annotate("prefix",xy=(1+(mi-1)*.24,entries[1,j]),xytext=(0,7),textcoords="offset points",ha="center",fontsize=7)
    for j,arm in enumerate(["continuum","rigid"]):axes[j].set(xticks=xs,xticklabels=[f"{i:03d}" for i in range(4)],ylabel=f"{arm} whole-curve RMSE (mm)",xlabel="TEST Task");axes[j].legend()
    flags=np.array([[1 if test[(i,m)]["evaluation"]["runtime_reported_passed"] is True else 0 if test[(i,m)]["evaluation"]["runtime_reported_passed"] is False else np.nan for m in METHODS] for i in range(4)])
    axes[2].imshow(np.ma.masked_invalid(flags),cmap=ListedColormap(["#f6b8a1","#b9dccd"]),vmin=0,vmax=1,aspect="auto");axes[2].set(xticks=range(3),xticklabels=METHODS,yticks=range(4),yticklabels=[f"{i:03d}" for i in range(4)]);axes[2].grid(False)
    for i in range(4):
        for j in range(3):axes[2].text(j,i,"not reported" if np.isnan(flags[i,j]) else "runtime PASS" if flags[i,j] else "runtime FALSE",ha="center",va="center",fontsize=8)
    save(fig,"legacy_curve_diagnostics","Task acceptance","Whole-curve diagnostics and historical runtime flags","The original logged whole-curve tracking errors and runtime passed flags are retained. E1 and completed E2 can have runtime FALSE because intentional intermediate detours differ from the old base curve; these flags are explicitly not this B2 Task gate. TEST 001 E2 RMSE uses only its saved prefix and is not comparable to a full 27 s curve.",traces()+sources())

    # 19: native target curves, finite sampled evidence only.
    fig,axes=plt.subplots(2,2,figsize=(12,6.8),sharex=True)
    for ti,ax in enumerate(axes.flat):
        for m in METHODS:
            rec=test[(ti,m)];f=rec["fresh"];ax.plot(f["time"],f["target_minimum_m"]*1000,color=COLORS[m],label=m,ls="--" if not rec["attempt"]["full_task_success"] else "-")
        ax.axhline(5,color="#333333",ls=":",label="5 mm gate");ax.set(xlabel=f"TEST {ti:03d}: physical time (s)",ylabel="Native robot-target minimum (mm)",xlim=(0,27));ax.legend(ncol=4,fontsize=8)
    save(fig,"native_robot_target_clearance","Geometry","Native robot-target clearance at every saved 2 ms state","The original independent replay queried all saved native 2 ms states, including the initial state. Curves show the minimum over policy-eligible robot-target pairs. The failed E2 prefix stops at refusal; prefix clearance does not establish a successful Task. This is dense discrete evidence, not continuous-time certification.",fresh_sources()+sources())

    # 20: whole-body policy classes.
    classes=sorted(next(iter(test.values()))["evaluation"]["native_geometry"]["whole_body"]["minimum_by_class"])
    table=np.array([[test[(i,m)]["evaluation"]["native_geometry"]["whole_body"]["minimum_by_class"][c]*1000 for c in classes] for i in range(4) for m in METHODS])
    fig,ax=plt.subplots(figsize=(13,5.4));im=ax.imshow(np.log10(table),aspect="auto",cmap="viridis");ax.set(xticks=range(len(classes)),xticklabels=[c.replace("_","\n") for c in classes],yticks=range(12),yticklabels=[f"{i:03d} {m}"+(" prefix" if not test[(i,m)]["attempt"]["full_task_success"] else "") for i in range(4) for m in METHODS]);ax.grid(False)
    for i in range(12):
        for j in range(len(classes)):ax.text(j,i,f"{table[i,j]:.1f}",ha="center",va="center",fontsize=8,color="white" if np.log10(table[i,j])<1.8 else "black")
    fig.colorbar(im,ax=ax,label="log10 minimum clearance (mm)")
    save(fig,"whole_body_class_minima","Geometry","Whole-body minimum by original pair class","Numbers are original independent native signed-distance minima by pair class, in mm. Whole-body verification uses saved 50 Hz boundary states plus four configuration-space subdivisions under the original pair policy. The refused E2 row covers only its prefix; terminal grasp-contact exemptions remain the original declared policy. No continuous-time guarantee is claimed.",sources())

    # 21: base drift, independently replayed position and orientation.
    fig,axes=plt.subplots(4,2,figsize=(12,9),sharex=True)
    for ti in range(4):
        for m in METHODS:
            rec=test[(ti,m)];f=rec["fresh"];pose=f["base_pose"];disp=np.linalg.norm(pose[:,:3]-pose[0,:3],axis=1)*1000
            quat=pose[:,3:];cos=np.clip(np.abs(np.sum(quat*quat[0],axis=1)),0,1);angle=np.rad2deg(2*np.arccos(cos))
            for ax,y in zip(axes[ti],[disp,angle]):ax.plot(*sampled_xy(f["time"],y),color=COLORS[m],label=m,ls="--" if not rec["attempt"]["full_task_success"] else "-");ax.set_xlim(0,27)
        axes[ti,0].set(ylabel=f"TEST {ti:03d}\ntranslation (mm)");axes[ti,1].set(ylabel="orientation drift (deg)")
    for ax in axes[-1]:ax.set_xlabel("Physical time (s)")
    axes[0,1].legend(ncol=3)
    save(fig,"base_translation_and_orientation","Dynamics","Free-base translation and orientation drift","Base motion is derived from fresh saved-torque replay relative to the declared initial base pose. Quaternion orientation uses the sign-invariant shortest rotation angle. All methods and the refused prefix are shown; this diagnostic is not a new acceptance gate.",fresh_sources())

    # 22/23: actual 17 velocity channels and 67 applied torque channels.
    command_max=np.array([np.max(np.abs(test[(i,m)]["trace"]["command_velocity"]),axis=0) for i in range(4) for m in METHODS])
    torque_max=np.array([np.max(np.abs(test[(i,m)]["trace"]["torque"]),axis=0) for i in range(4) for m in METHODS])
    for name,table,unit in [("command_17_channel_peaks",command_max,"Absolute peak command (rad/s)"),("torque_67_channel_peaks",torque_max,"Absolute peak applied torque (N m)")]:
        fig,ax=plt.subplots(figsize=(13,5));im=ax.imshow(table,aspect="auto",cmap="viridis");ticks=list(range(17)) if table.shape[1]==17 else list(range(0,65,5))+[66]
        ax.set(xticks=ticks,xticklabels=[str(i+1) for i in ticks],yticks=range(12),yticklabels=[f"{i:03d} {m}"+(" prefix" if not test[(i,m)]["attempt"]["full_task_success"] else "") for i in range(4) for m in METHODS],xlabel="Actual channel index (one based)");ax.grid(False);fig.colorbar(im,ax=ax,label=unit)
        save(fig,name,"Dynamics",unit,"Absolute per-channel peaks are aggregated from the original consumed trace: 17 planner command coordinates or 67 applied actuator torques. The failed E2 row covers only its prefix. No channels are removed and no new controller or torque run is performed; original execution-contract saturation checks remain authoritative.",traces()+sources())

    # 24: interval diagnostics, no substitution for native geometry.
    fig,axes=plt.subplots(4,2,figsize=(12,9),sharex=True)
    for ti in range(4):
        for m in METHODS:
            rec=test[(ti,m)];tr=rec["trace"];t=tr["task_time"][:len(tr["task_interval_selected_rows"])]
            axes[ti,0].plot(*sampled_xy(t,tr["task_interval_selected_rows"]),color=COLORS[m],label=m)
            proxy=tr["task_interval_proxy_lower_m"];axes[ti,1].plot(*sampled_xy(t,proxy*1000),color=COLORS[m],label=m)
        axes[ti,0].set(ylabel=f"TEST {ti:03d}\nselected interval rows",xlim=(0,27));axes[ti,1].set(ylabel="PCC proxy lower bound (mm)",xlim=(0,27));axes[ti,1].axhline(5,color="#333333",ls=":")
    axes[0,1].legend(ncol=3)
    for ax in axes[-1]:ax.set_xlabel("Physical time (s)")
    save(fig,"pcc_interval_rows_and_bounds","PCC / contract","PCC interval rows and proxy lower bounds","These are the logged original bounded-interval PCC diagnostics at planning ticks. Proxy lower bounds and selected rows are not native whole-body distances. Independent interval recomputation reports zero mismatches on every executed prefix. Unavailable/nonfinite proxy values remain gaps; failed traces end at the real refusal.",traces()+sources())

    # 25: all independent gates, preserve prefix/full distinction.
    gate_names=["evidence","contract","interval","native","binding","full Task"]
    table=[]
    for a in order:
        e=a["evaluation"];table.append([e["evidence_valid"],e["execution_contract"]["passed"],e["independent_interval"]["passed"],e["native_geometry"]["passed"],e["reference_binding"]["passed"],a["full_task_success"]])
    fig,ax=plt.subplots(figsize=(11,9));ax.imshow(table,cmap=ListedColormap(["#f6b8a1","#b9dccd"]),vmin=0,vmax=1,aspect="auto");ax.set(xticks=range(6),xticklabels=gate_names,yticks=range(37),yticklabels=[a["slot_id"] for a in order]);ax.tick_params(axis="y",labelsize=7);ax.grid(False)
    for y,a in enumerate(order):
        if not a["full_task_success"]:ax.text(5,y,"FAIL",ha="center",va="center",fontsize=7)
    save(fig,"all_independent_gate_matrix","PCC / contract","All 37 attempts: independent gates and full Task","Executed-prefix evidence, execution contract, interval recomputation, native geometry, and consumed-reference binding are distinct from completing the full 27 s Task. The two refused prefixes pass their prefix checks but fail the full Task column. Prefix pass is never relabeled as overall safety or Task success.",[f"attempts/{a['slot_id']}/attempt_result.json" for a in order])

    # 26/27: dispatch distributions and temporal long tails.
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));dispatch_arrays=[test[(i,m)]["dispatch_ms"] for i in range(4) for m in METHODS]
    box=axes[0].boxplot(dispatch_arrays,patch_artist=True,whis=(5,95),showfliers=False)
    for patch,m in zip(box["boxes"],METHODS*4):patch.set_facecolor(COLORS[m]);patch.set_alpha(.65)
    axes[0].set(xticks=np.arange(1,13),xticklabels=[f"{i}:{m}" for i in range(4) for m in METHODS],ylabel="Dispatch latency (ms)");axes[0].tick_params(axis="x",rotation=60)
    over=[int(np.sum(x>20)) for x in dispatch_arrays];axes[1].bar(range(12),over,color=[COLORS[m] for m in METHODS*4]);axes[1].set(xticks=range(12),xticklabels=[f"{i}:{m}" for i in range(4) for m in METHODS],ylabel="Recorded dispatch cycles above 20 ms");axes[1].tick_params(axis="x",rotation=60)
    axes[0].axhline(20,color="#333333",ls="--")
    save(fig,"dispatch_distribution_and_overruns","Timing","Dispatch distributions and deadline overruns","Left boxes show medians and interquartile ranges with 5th/95th percentile whiskers (long tails are shown in the next panel). Right counts all >20 ms dispatch cycles, using the existing state-acquisition-to-torque-publication clock. Different prefix lengths imply different cycle counts. Wall 20 ms is not this research gate; delayed-state deployment validity is NOT_VERIFIED and deployment remains NOT_MET.",[f"attempts/TEST_{i:02d}_{m}/actual/timing/{test[(i,m)]['attempt']['task_id']}.jsonl" for i in range(4) for m in METHODS])
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for ti,ax in enumerate(axes):
        for m in METHODS:
            rec=test[(ti,m)];ax.plot(rec["dispatch_t"],rec["dispatch_ms"],color=COLORS[m],lw=.65,alpha=.8,label=m)
        ax.axhline(20,color="#333333",ls="--");ax.set(ylabel=f"TEST {ti:03d}\nlatency (ms)",xlim=(0,27));ax.legend(ncol=3,fontsize=8)
    axes[-1].set_xlabel("Physical source time (s)")
    save(fig,"dispatch_temporal_long_tails","Timing","Every dispatch cycle and its long tail","Every saved planning dispatch is plotted without downsampling, including first-cycle cost and maxima. The E2 TEST 001 line ends where the original guard refuses further physics at 16.620 s. The 20 ms line is diagnostic only and does not certify hard real-time operation.",[f"attempts/TEST_{i:02d}_{m}/actual/timing/{test[(i,m)]['attempt']['task_id']}.jsonl" for i in range(4) for m in METHODS])

    # 28: quality is conditioned on complete Tasks.
    fig,axes=plt.subplots(1,3,figsize=(13,4.2));quality=delivery["TEST_rows"]
    metrics=[("continuum path (m)",lambda r:r["actual_path_length_m"]["continuum"]),("joint path (rad)",lambda r:r["actual_joint_path_length_rad"]),("command intervention RMS (rad/s)",lambda r:r["command_intervention_rad_s"]["rms"])]
    for ax,(label,getter) in zip(axes,metrics):
        for m in METHODS:
            rr=[r for r in quality if r["method"]==m and r["path_quality_comparison_eligible"]]
            ax.scatter([int(r["task_id"].split("_")[-1]) for r in rr],[getter(r) for r in rr],color=COLORS[m],s=45,label=f"{m} (n={len(rr)})",marker={"E0":"o","E1":"s","E2":"^"}[m])
        ax.set(xticks=range(4),xticklabels=[f"{i:03d}" for i in range(4)],xlabel="TEST Task",ylabel=label);ax.legend(fontsize=8)
    save(fig,"complete_task_conditional_quality","Quality","Path quality conditional on full Task success","Only independently verified complete 27 s Tasks are eligible: E0 n=4, E1 n=4, E2 n=3. E2 TEST 001 is excluded rather than assigned an artificially short successful path. Different conditional sample sizes preclude a success-independent learning advantage claim. Actual path changes and these diagnostics do not overturn the 4/4 versus 3/4 success result.",["delivery_analysis.json"])

    # 29: comparison witnesses and cost separation.
    fig,axes=plt.subplots(1,2,figsize=(12,4));paired=read(source/"paired_path_comparison.json")
    for mi,m in enumerate(["E1","E2"]):
        rr=[r for r in paired if r["comparison"]==f"{m}_vs_E0" and r["eligible"]]
        axes[0].scatter([int(r["task_id"].split("_")[-1]) for r in rr],[r["actual_continuum_path_difference_peak_m"]*1000 for r in rr],color=COLORS[m],marker=["s","^"][mi],label=m)
    axes[0].set(xticks=range(4),xticklabels=[f"{i:03d}" for i in range(4)],xlabel="Complete paired TEST Task",ylabel="Actual continuum difference peak vs E0 (mm)");axes[0].legend()
    cost=delivery["cost"];axes[1].bar(["actual physics", "independent replay"],[cost["actual_physics_steps"],cost["independent_torque_replay_physics_steps"]],color=[COLORS["E0"],COLORS["E1"]]);axes[1].set_ylabel("Physics steps (separate cost scopes)")
    save(fig,"paired_path_change_and_cost","Quality","Paired actual path changes and separate replay cost","Only full verified pairs contribute to peak actual path-difference comparisons. E0 and E1 traces are comparison witnesses and were not inputs to E2. The right panel keeps the 488,260 executed physics steps separate from the equally sized independent saved-torque replay; these are not additional actual attempt slots.",["paired_path_comparison.json","delivery_analysis.json","budget_ledger.json"])

    # 30: per-cycle components; totals are independently measured.
    fig,axes=plt.subplots(1,2,figsize=(13,5));phase_names=sorted(next(iter(test.values()))["timing_phases"])
    phase_table=np.array([[np.percentile(test[(i,m)]["timing_phases"][phase],95) for phase in phase_names] for i in range(4) for m in METHODS])
    im=axes[0].imshow(phase_table,aspect="auto",cmap="viridis");axes[0].set(xticks=range(len(phase_names)),xticklabels=[p.replace("_","\n") for p in phase_names],yticks=range(12),yticklabels=[f"{i:03d} {m}" for i in range(4) for m in METHODS]);axes[0].tick_params(axis="x",labelsize=6);axes[0].grid(False);fig.colorbar(im,ax=axes[0],label="phase p95 (ms)")
    dd=[test[(i,m)]["evaluation"]["dispatch_timing"]["records"][0]["dispatch_latency"] for i in range(4) for m in METHODS]
    for key,label,marker in [("p95_ms","p95","o"),("p99_ms","p99","s"),("max_ms","maximum","^")]:axes[1].plot(range(12),[d[key] for d in dd],marker+"-",label=label)
    axes[1].axhline(20,color="#333333",ls="--");axes[1].set(xticks=range(12),xticklabels=[f"{i}:{m}" for i in range(4) for m in METHODS],ylabel="Whole dispatch latency (ms)");axes[1].tick_params(axis="x",rotation=60);axes[1].legend()
    save(fig,"dispatch_phase_and_tail_summary","Timing","Measured dispatch components and tail statistics","Component p95 values are marginal quantiles and must not be summed to estimate total p95. Whole dispatch p95, p99 and maxima use the original independently measured timeline, including reference/first-torque preparation and publication. All 12 runs are shown; the refused prefix remains a shorter clock sample.",[f"attempts/TEST_{i:02d}_{m}/actual/timing/{test[(i,m)]['attempt']['task_id']}.jsonl" for i in range(4) for m in METHODS]+sources())

    # Compact dashboard data, with authoritative paths and explicit comparability.
    test_rows=[]
    for i in range(4):
        for m in METHODS:
            rec=test[(i,m)];a=rec["attempt"];e=rec["evaluation"];tr=rec["trace"];f=rec["fresh"]
            row=next(r for r in delivery["TEST_rows"] if r["slot_id"]==a["slot_id"]).copy()
            row.update({"task_index":i,"dispatch":e["dispatch_timing"]["records"][0]["dispatch_latency"],
                        "whole_curve_rmse_mm":{arm:float(np.sqrt(np.mean(tr[f"{arm}_error"]**2))*1000) for arm in ["continuum","rigid"]},
                        "whole_curve_scope":"full 27 s" if a["full_task_success"] else "saved refused prefix only; not comparable with full curves",
                        "command_17_absolute_peak_rad_s":np.max(np.abs(tr["command_velocity"]),axis=0).tolist(),
                        "torque_67_absolute_peak_nm":np.max(np.abs(tr["torque"]),axis=0).tolist(),
                        "independent_interval":e["independent_interval"],"execution_contract":e["execution_contract"],
                        "native_geometry":e["native_geometry"],"plot_paths":[p["path"] for p in panels if p["id"].startswith(f"test_{i:03d}")]})
            for key in ["execution_failure","pipeline_failure"]:
                if isinstance(row.get(key),dict):row[key]={k:v for k,v in row[key].items() if k!="traceback"}
            test_rows.append(row)
    data={"schema":"v64_b2_publication_plot_data_v1","source_run":"task_anchored_residual_20261007_01",
          "source_producer":"7d0a3fd4bd17f41b99b388c30c5ac4897ac4d31d","source_manifest_sha256":sha(source/"manifest.json"),
          "source_verification":read(source/"verification.json"),"scope":"Read-only JSON/NPZ visualization; zero new physics, inference, optimizer updates or samples.",
          "critical_limits":["Dense discrete evidence only; no continuous-time guarantee.","Wall 20 ms is a research diagnostic; deployment NOT_MET.","E2 TEST 001 is a refused prefix, never padded or counted as a complete path.","Whole-curve RMSE and original runtime FALSE remain separate from the B2 Task gate.","Learning advantage over retrieval is not established.","K4 actual NOT_RUN; K1 uses fixed slot 0."],
          "panels":panels,"verdict":summary["verdict"],"table_B":[{k:v for k,v in r.items() if k!="records"} for r in summary["table_B"]],
          "teachers":teacher,"training":train,"training_curves":curves,"raw_candidates":raw_records,"raw_diagnostics":{k:v for k,v in raw.items() if k!="rows"},
          "test_runs":test_rows,"paired_path_comparison":paired,"cost":cost,
          "tasks":[{"task_id":t["task_id"],"split":t["split"],"seed":t["seed"],"family":t["family"],"requirements":t["requirements"]} for t in task_suite["tasks"]],
          "source_files":[{"path":p,"sha256":sha(source/p),"bytes":(source/p).stat().st_size} for p in sorted(source_files)],
          "additional_physics_steps":0,"additional_optimizer_updates":0,"additional_inference_calls":0,"additional_samples":0}
    encoded=json.dumps(array_json(data),ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if len(encoded.encode("utf-8"))>5_000_000:raise ValueError("Dashboard plot_data exceeds 5 MB")
    (output/"plot_data.json").write_text(encoded,encoding="utf-8")
    for panel in panels:
        for key in ["path","pdf_path"]:
            if not (output/panel[key]).is_file() or (output/panel[key]).stat().st_size<1000:raise RuntimeError(panel[key])
    print(json.dumps({"panels":len(panels),"plot_data_bytes":len(encoded.encode("utf-8")),"output":str(output),"read_only_derivation":True}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    build(args.source,args.output)
