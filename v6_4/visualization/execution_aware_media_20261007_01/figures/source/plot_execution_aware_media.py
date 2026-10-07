"""Append-only B.3.1 trajectory and tracking media from sealed saved arrays.

Uses NumPy/Matplotlib only. No controller, reference provider, MuJoCo, model,
geometry query, rollout, inference, interpolation, or experimental gate update.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MODES = ("z0", "v1_plus12", "v1_minus12", "v2_plus12", "v2_minus12", "v2_plus20", "v2_minus20")
LABELS = ("z0", "v1 +12 mm", "v1 -12 mm", "v2 +12 mm", "v2 -12 mm", "v2 +20 mm", "v2 -20 mm")
COLORS = ("#333333", "#D55E00", "#0072B2", "#CC79A7", "#009E73", "#E69F00", "#56B4E9")
STYLES = ("-", "--", "--", "-", "-", ":", ":")
FRESH_KEYS = ("time", "base_pose", "target_position", "target_rotation", "rigid_position", "rigid_rotation", "continuum_position", "continuum_rotation", "target_minimum_m", "target_minimum_censored")
TRACE_KEYS = ("time", "task_time", "task_input_reference_rigid_target_position", "task_input_reference_rigid_target_rotation", "task_input_reference_continuum_target_position", "task_input_reference_continuum_target_rotation", "task_qp_continuum_actual_position_m")
TRACE_KEYS += ("torque", "torque_saturation_count", "task_avoidance_intervention", "task_qp_box_nominal_velocity", "task_qp_selected_velocity",
    "task_pcc_clearance", "task_capsule_clearance", "task_interval_current_envelope_margin_m", "task_interval_ramp_minimum_envelope_margin_m")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42, "path.simplify": False})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


class Sources:
    def __init__(self, source):
        self.source = source.resolve()
        self.files = {}
        self.seal = json.loads((source/"manifest.json").read_text(encoding="utf-8-sig"))
        self.bind(source/"manifest.json")

    def bind(self, path, expected=None):
        path = Path(path)
        path = (self.source/path).resolve() if not path.is_absolute() else path.resolve()
        key = str(path)
        relative = path.relative_to(self.source).as_posix() if path.is_relative_to(self.source) else None
        entry = self.seal["payload"].get(relative) if relative is not None else self.seal["external"].get(key)
        if entry:
            if expected is not None:
                require(expected == entry["sha256"], "declared binding disagrees with sealed manifest: "+key)
            expected = entry["sha256"]
        require(expected is not None or path == self.source/"manifest.json", "unsealed source: "+key)
        if key in self.files:
            require(expected is None or self.files[key]["sha256"] == expected, "conflicting source binding: "+key)
            return path
        digest = sha(path)
        require(expected is None or digest == expected, "source digest mismatch: "+key)
        self.files[key] = {"path": key, "source_relative_path": relative, "sha256": digest,
                           "bytes": path.stat().st_size, "external_reused_source": relative is None}
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text(encoding="utf-8-sig"))

    def arrays(self, path, keys, expected=None):
        with np.load(self.bind(path, expected), allow_pickle=False) as data:
            return {key: data[key].copy() for key in keys if key in data.files}

    def verify_unchanged(self):
        for record in self.files.values():
            require(sha(record["path"]) == record["sha256"], "source changed during plotting: "+record["path"])


def mask(times, interval):
    return (times >= interval[0]-1e-9) & (times <= interval[1]+1e-9)


def rotation_error_deg(actual, reference):
    product = np.einsum("...ji,...jk->...ik", reference, actual)
    cosine = (np.trace(product, axis1=-2, axis2=-1)-1.)/2.
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))


def norm(value):
    return np.linalg.norm(value, axis=-1)


def stats(value):
    value = np.asarray(value)
    if value.size == 0:
        return {"status": "UNAVAILABLE_NO_SAVED_SAMPLES", "samples": 0}
    return {"samples": int(value.size), "rms": float(np.sqrt(np.mean(value**2))),
            "mean": float(np.mean(value)), "maximum": float(np.max(value)), "minimum": float(np.min(value)),
            "last": float(value[-1])}


def base_path(task, times):
    """Original independent Task/base minimum-jerk path, not consumed routing."""
    target = task["scenario"]["continuum_target"]
    require(target["mode"] == "irregular_waypoints" and target["reference_profile"] == "minimum_jerk_c2", "unsupported Task base path")
    points = np.asarray(target["waypoint_points_m"])
    durations = np.asarray(target["segment_durations_s"])
    transition = target["transition_duration_s"]
    knots = np.r_[0., np.cumsum(durations)]
    elapsed = np.clip(times-transition, 0., durations.sum())
    index = np.minimum(np.searchsorted(knots, elapsed, side="right")-1, len(durations)-1)
    u = (elapsed-knots[index])/durations[index]
    path = points[index]+(10*u**3-15*u**4+6*u**5)[:, None]*(points[index+1]-points[index])
    before = times < transition
    v = np.clip(times[before]/transition, 0., 1.)
    path[before] = np.asarray(target["initial_position_w"])+(10*v**3-15*v**4+6*v**5)[:, None]*(points[0]-target["initial_position_w"])
    return path


def source_records(sources, paths):
    return [dict(sources.files[str(Path(path).resolve())]) for path in sorted(set(map(str, paths)))]


def load_slot(sources, slot, task, declaration):
    terminal_path = sources.bind(Path("slots")/slot["slot_id"]/"slot_result.json")
    terminal = sources.read(terminal_path)
    require(terminal["frozen_slot"] == slot, "fixed-slot identity changed")
    result, quality = terminal["source_result"], terminal["quality"]
    require(result["actual_steps"] == 13500 and result["full_task_success"], "media requires the sealed complete actual slot")
    require(quality["safety"]["full_task_and_original_safety_passed"] and quality["quality_label_eligible"], "slot is not complete safe qualified evidence")
    evidence = terminal["evidence_bindings"]
    evaluation_path = sources.bind(result["evaluation_path"], result["evaluation_sha256"])
    evaluation = sources.read(evaluation_path)
    require(evaluation == result["evaluation"], "embedded original evaluation changed")
    fresh_path = evaluation_path.parent/"fresh_replay.npz"
    require(str(fresh_path) in evidence, "fresh states missing terminal binding")
    require(evidence[str(fresh_path)] == evaluation["fresh_replay_sha256"], "original evaluator and terminal fresh digests differ")
    fresh = sources.arrays(fresh_path, FRESH_KEYS, evaluation["fresh_replay_sha256"])
    trace_path = sources.bind(result["trace_path"], result["trace_sha256"])
    trace = sources.arrays(trace_path, TRACE_KEYS, result["trace_sha256"])
    t, tt = fresh["time"], trace["task_time"][:1350]
    require(t.shape == (13501,) and np.allclose(t, np.arange(13501)*.002, atol=1e-8, rtol=0.), "fresh native clock invalid")
    require(trace["time"].shape == (13500,) and np.allclose(trace["time"], t[1:], atol=1e-9, rtol=0.), "post-step trace time invalid")
    require(tt.shape == (1350,) and np.allclose(tt, np.arange(1350)*.02, atol=1e-8, rtol=0.), "consumed QP clock invalid")
    indices = np.rint(tt/.002).astype(int)
    require(np.allclose(t[indices], tt, atol=1e-9, rtol=0.), "QP/fresh clocks do not align exactly")
    if "task_qp_continuum_actual_position_m" in trace:
        require(np.allclose(trace["task_qp_continuum_actual_position_m"][:1350], fresh["continuum_position"][indices], atol=1e-9, rtol=0.), "saved QP actual position disagrees with fresh state")
    for key in FRESH_KEYS:
        require(key in fresh and (fresh[key].dtype.kind == "b" or np.isfinite(fresh[key]).all()), "invalid fresh field: "+key)
    for arm in ("rigid", "continuum"):
        for field in ("position", "rotation"):
            key = "task_input_reference_"+arm+"_target_"+field
            require(key in trace and len(trace[key]) >= 1350 and np.isfinite(trace[key][:1350]).all(), "missing consumed reference: "+key)
    qpath = Path(slot["old_quality_path"]) if slot["old_quality_path"] else sources.source/"quality"/slot["slot_id"]/"route_quality.json"
    qpath = sources.bind(qpath, evidence[str(qpath)])
    require(sources.read(qpath) == quality, "terminal quality differs from original object")
    qmanifest_path = sources.bind(qpath.parent/"manifest.json")
    qmanifest = sources.read(qmanifest_path)
    route_path = qpath.parent/"route_clearance.npz"
    clearance = sources.arrays(route_path, ("time", "continuum_obstacle_minimum_m"), qmanifest["route_clearance.npz"])
    require(np.allclose(clearance["time"], t, atol=1e-9, rtol=0.), "saved sphere clearance clock differs")
    paths = [terminal_path, evaluation_path, fresh_path, trace_path, qpath, qmanifest_path, route_path,
             sources.bind(slot["task_path"], declaration["task_file_sha256"]), sources.bind(slot["plan_path"], slot["plan_file_sha256"])]
    base = base_path(task, t)
    grasp = np.asarray(task["scenario"]["grasp_point_target_frame_m"])
    grasp_rotation = np.asarray(task["scenario"]["grasp_rotation_target_frame"])
    task_grasp_p = fresh["target_position"]+np.einsum("nij,j->ni", fresh["target_rotation"], grasp)
    task_grasp_r = fresh["target_rotation"]@grasp_rotation
    task_cont_r = np.broadcast_to(np.asarray(task["scenario"]["continuum_target_rotation_world"]), (len(t), 3, 3))
    native = {"time_s": t, "fresh_state_index": np.arange(len(t)), "base_translation_drift_m": norm(fresh["base_pose"][:, :3]-task["base_pose"][:3]),
        "robot_target_minimum_m": fresh["target_minimum_m"], "robot_target_minimum_censored": fresh["target_minimum_censored"],
        "continuum_sphere_minimum_m": clearance["continuum_obstacle_minimum_m"],
        "continuum_actual_minus_task_base_m": norm(fresh["continuum_position"]-base),
        "continuum_orientation_to_Task_deg": rotation_error_deg(fresh["continuum_rotation"], task_cont_r),
        "rigid_position_to_terminal_Task_grasp_m": norm(fresh["rigid_position"]-task_grasp_p),
        "rigid_orientation_to_terminal_Task_grasp_deg": rotation_error_deg(fresh["rigid_rotation"], task_grasp_r)}
    q = fresh["base_pose"][:, 3:]
    q0 = np.asarray(task["base_pose"][3:])
    dots = np.abs((q@q0)/(norm(q)*norm(q0)))
    native["base_orientation_drift_deg"] = np.rad2deg(2*np.arccos(np.clip(dots, 0., 1.)))
    consumed = {"time_s": tt, "fresh_state_index": indices}
    for arm in ("continuum", "rigid"):
        p = trace["task_input_reference_"+arm+"_target_position"][:1350]
        r = trace["task_input_reference_"+arm+"_target_rotation"][:1350]
        consumed[arm+"_position_tracking_error_m"] = norm(fresh[arm+"_position"][indices]-p)
        consumed[arm+"_orientation_tracking_error_deg"] = rotation_error_deg(fresh[arm+"_rotation"][indices], r)
        for axis, label in enumerate("xyz"):
            consumed[arm+"_actual_"+label+"_m"] = fresh[arm+"_position"][indices, axis]
            consumed[arm+"_consumed_reference_"+label+"_m"] = p[:, axis]
    consumed["original_17D_intervention_rad_s"] = trace["task_avoidance_intervention"][:1350]
    vector_available = all(key in trace for key in ("task_qp_box_nominal_velocity", "task_qp_selected_velocity"))
    if vector_available:
        nominal, selected = trace["task_qp_box_nominal_velocity"][:1350], trace["task_qp_selected_velocity"][:1350]
        require(nominal.shape == selected.shape == (1350, 17) and np.isfinite(nominal).all() and np.isfinite(selected).all(), "invalid saved QP velocity vectors")
        consumed["box_nominal_17D_norm_rad_s"], consumed["selected_17D_norm_rad_s"] = norm(nominal), norm(selected)
        require(np.allclose(norm(selected-nominal), consumed["original_17D_intervention_rad_s"], atol=1e-9, rtol=0.), "saved vector intervention disagrees with original scalar")
    for key in ("task_pcc_clearance", "task_capsule_clearance", "task_interval_current_envelope_margin_m", "task_interval_ramp_minimum_envelope_margin_m"):
        if key in trace:
            consumed[key] = trace[key][:1350]
    torque = trace.pop("torque")
    require(torque.shape == (13500, 67) and np.isfinite(torque).all(), "saved applied torque invalid")
    control_native = {"applied_interval_end_time_s": trace["time"], "applied_interval_start_time_s": trace["time"]-.002,
        "actual_torque_L2_67channels_Nm": norm(torque), "actual_torque_max_abs_67channels_Nm": np.max(np.abs(torque), axis=1),
        "actual_torque_saturation_count": trace["torque_saturation_count"]}
    for arm in ("rigid", "continuum"):
        for axis, label in enumerate("xyz"):
            native[arm+"_actual_"+label+"_m"] = fresh[arm+"_position"][:, axis]
    for axis, label in enumerate("xyz"):
        native["continuum_Task_base_"+label+"_m"] = base[:, axis]
    route_native, route_consumed = mask(t, declaration["route_interval_s"]), mask(tt, declaration["route_interval_s"])
    summary = {"actual_steps": 13500, "fresh_states": 13501, "consumed_reference_samples": 1350,
        "actual_status": terminal["status"], "original_Task_and_safety_passed": True, "quality_eligible": True,
        "route_interval_s": declaration["route_interval_s"], "task_horizon_s": [0., 27.],
        "original_full_quality_vector": quality["full_metrics"],
        "original_independent_Task_requirement_results": evaluation["task_requirements"],
        "control_diagnostics": {"QP_vectors": "AVAILABLE_SAVED_17D_VECTORS" if vector_available else "NOT_MEASURED_OLD_EVIDENCE",
            "original_scalar_17D_intervention_rad_s": {"whole_task": stats(consumed["original_17D_intervention_rad_s"]), "route": stats(consumed["original_17D_intervention_rad_s"][route_consumed])},
            "native_applied_torque": {key: stats(value) for key, value in control_native.items() if key.startswith("actual_torque")},
            "applied_torque_scope": "All 13500 original applied 2ms intervals; 67 actuator channels; norms are diagnostics, not a new limit/gate",
            "existing_interval_PCC_fields": {key: {"saved_samples": int(len(value)), "finite_samples": int(np.isfinite(value).sum()), "nonfinite_CSV_cells": "empty, not zero"}
                for key, value in consumed.items() if key.startswith("task_")}},
        "consumed_reference_tracking": {key: {"whole_task": stats(value), "route": stats(value[route_consumed])}
            for key, value in consumed.items() if "tracking_error" in key},
        "native_diagnostics": {key: {"whole_task": stats(value), "route": stats(value[route_native])}
            for key, value in native.items() if key not in ("time_s", "fresh_state_index") and ("drift" in key or "minimum_m" in key or "minus_task_base" in key)},
        "independent_Task_metrics_are_original_evaluation_not_curve_RMSE": True}
    return {"slot": slot, "declaration": declaration, "task": task, "fresh": fresh, "trace": trace,
            "base": base, "task_grasp_p": task_grasp_p, "native": native, "consumed": consumed, "control_native": control_native,
            "summary": summary, "source_records": source_records(sources, paths)}


def label(record):
    d, s = record["declaration"], record["slot"]
    return f"{s['slot_id']} | d={1000*d['distance_m']:.0f} mm, c{'+' if d['side'] == 1 else '-'} | {s['mode']} | complete actual + original Task/safety passed"


def shade(ax, interval):
    ax.axvspan(*interval, color="#CC79A7", alpha=.12, lw=0)
    ax.set_xlim(0., 27.)
    ax.set_xlabel("Task time (s)")
    ax.grid(alpha=.18, lw=.5)


def footer(fig, text):
    fig.text(.012, .012, text, fontsize=8, ha="left", va="bottom")


def save(fig, directory, stem):
    png, pdf = directory/(stem+".png"), directory/(stem+".pdf")
    require(not png.exists() and not pdf.exists(), "append-only output already exists")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return "figures/"+png.name, "figures/"+pdf.name


def plot_trajectory(record, directory):
    f, tr, t = record["fresh"], record["trace"], record["fresh"]["time"]
    keep = np.unique(np.r_[np.arange(0, len(t), 10), len(t)-1])
    route = keep[mask(t[keep], record["declaration"]["route_interval_s"])]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.1))
    projections = ((0, 1), (1, 2), (0, 2))
    for row, arm in enumerate(("continuum", "rigid")):
        actual = f[arm+"_position"]
        reference = tr["task_input_reference_"+arm+"_target_position"][:1350]
        for col, (a, b) in enumerate(projections):
            ax = axes[row, col]
            ax.plot(actual[keep, a], actual[keep, b], color="#0072B2", lw=1.4)
            ax.plot(reference[:, a], reference[:, b], color="#D55E00", lw=1., ls="--")
            if arm == "continuum":
                base = record["base"]
                ax.plot(base[keep, a], base[keep, b], color="#555555", ls=":", lw=1.2)
                points = np.asarray(record["task"]["scenario"]["continuum_target"]["waypoint_points_m"])
                ax.scatter(points[:, a], points[:, b], marker="x", s=19, color="#222222", zorder=5)
            else:
                use = keep[t[keep] >= 25.5-1e-9]
                p = record["task_grasp_p"][use]
                ax.plot(p[:, a], p[:, b], color="#009E73", ls=":", lw=1.4)
            ax.plot(actual[route, a], actual[route, b], color="#CC79A7", lw=2.)
            ax.scatter(actual[0, a], actual[0, b], color="#0072B2", s=15, marker="o")
            ax.scatter(actual[-1, a], actual[-1, b], color="#0072B2", s=22, marker="s")
            ax.set_xlabel(f"World {'xyz'[a]} (m)"); ax.set_ylabel(f"{arm.capitalize()}: world {'xyz'[b]} (m)")
            ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.18, lw=.5)
    handles = [Line2D([], [], color="#0072B2", label="Actual fresh kinematics"),
        Line2D([], [], color="#D55E00", ls="--", label="Consumed QP reference (20 ms)"),
        Line2D([], [], color="#555555", ls=":", label="Continuum Task/base path"),
        Line2D([], [], color="#CC79A7", lw=2, label="Actual in frozen route interval"),
        Line2D([], [], color="#009E73", ls=":", label="Rigid Task grasp point, 25.5–27 s only")]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.5, .99), frameon=False)
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.13, wspace=.3, hspace=.35)
    footer(fig, label(record)+"\nIndependent Task anchors/grasp are separate from the consumed execution reference. Actual display: saved 20 ms subset; no interpolation.")
    return save(fig, directory, record["slot"]["slot_id"]+"_trajectory")


def plot_tracking(record, directory):
    c, n = record["consumed"], record["native"]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.1))
    for col, arm in enumerate(("continuum", "rigid")):
        axes[0, col].plot(c["time_s"], c[arm+"_position_tracking_error_m"]*1000., color="#0072B2", lw=1.1, label="Actual - consumed QP reference")
        axes[1, col].plot(c["time_s"], c[arm+"_orientation_tracking_error_deg"], color="#0072B2", lw=1.1, label="Actual vs consumed rotation")
        if arm == "continuum":
            axes[0, col].plot(n["time_s"][::10], n["continuum_actual_minus_task_base_m"][::10]*1000., color="#555555", ls=":", label="Actual - independent Task/base path")
            for req in record["task"]["requirements"]:
                if req["arm"] == "continuum":
                    axes[0, col].axvline(req["time_s"], color="#999999", lw=.55, alpha=.45)
        else:
            use = n["time_s"] >= 25.5-1e-9
            axes[0, col].plot(n["time_s"][use][::10], n["rigid_position_to_terminal_Task_grasp_m"][use][::10]*1000., color="#009E73", ls=":", label="Independent Task grasp (terminal only)")
            axes[1, col].plot(n["time_s"][use][::10], n["rigid_orientation_to_terminal_Task_grasp_deg"][use][::10], color="#009E73", ls=":", label="Independent Task grasp (terminal only)")
        axes[0, col].set_ylabel(arm.capitalize()+" position error (mm)")
        axes[1, col].set_ylabel(arm.capitalize()+" orientation error (deg)")
        for row in range(2):
            shade(axes[row, col], record["declaration"]["route_interval_s"])
            axes[row, col].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=.09, right=.985, top=.97, bottom=.16, wspace=.28, hspace=.3)
    footer(fig, label(record)+"\nShaded: complete frozen route interval. Tracking is measured at exact pre-step QP ticks (0–26.98 s).\nActual–base deviation is not a Task-failure metric; original Task windows/terminal checks remain in the supplied summary.")
    return save(fig, directory, record["slot"]["slot_id"]+"_tracking")


def plot_safety(record, directory):
    n = record["native"]; use = np.unique(np.r_[np.arange(0, len(n["time_s"]), 10), len(n["time_s"])-1])
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.1))
    definitions = (("base_translation_drift_m", 1000., "Base translation from declared pose (mm)"),
        ("base_orientation_drift_deg", 1., "Base orientation from declared pose (deg)"),
        ("robot_target_minimum_m", 1000., "Robot–target minimum clearance (mm)"),
        ("continuum_sphere_minimum_m", 1000., "Continuum–related sphere minimum (mm)"))
    for ax, (key, scale, ylabel) in zip(axes.flat, definitions):
        ax.plot(n["time_s"][use], n[key][use]*scale, color="#0072B2", lw=1.2)
        shade(ax, record["declaration"]["route_interval_s"]); ax.set_ylabel(ylabel)
    axes[1, 0].axhline(5., color="#D55E00", ls="--", lw=.9, label="Original robot–target 5 mm gate")
    axes[1, 0].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=.09, right=.985, top=.97, bottom=.15, wspace=.28, hspace=.3)
    footer(fig, label(record)+"\nSaved native 2 ms states; displayed 20 ms subset. Sphere curve is the sealed route-clearance series over the whole task.\nRobot–target and related-sphere pair sets differ. These discrete curves add no new geometry queries or continuous-time certificate.")
    return save(fig, directory, record["slot"]["slot_id"]+"_safety_base")


def plot_control(record, directory):
    c, n = record["consumed"], record["control_native"]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.1))
    axes[0, 0].plot(c["time_s"], c["original_17D_intervention_rad_s"], color="#CC79A7", lw=1.1, label="Original ||selected - box nominal||")
    if "selected_17D_norm_rad_s" in c:
        axes[0, 0].plot(c["time_s"], c["selected_17D_norm_rad_s"], color="#0072B2", lw=.9, label="Selected 17D speed norm")
        axes[0, 0].plot(c["time_s"], c["box_nominal_17D_norm_rad_s"], color="#D55E00", lw=.9, ls="--", label="Box nominal 17D speed norm")
    else:
        axes[0, 0].text(.03, .96, "QP vectors: NOT MEASURED in old evidence\nScalar intervention is original saved log.", transform=axes[0, 0].transAxes, va="top", fontsize=8)
    axes[0, 0].set_ylabel("QP velocity diagnostic (rad/s)")
    end = n["applied_interval_end_time_s"]
    axes[0, 1].plot(end, n["actual_torque_L2_67channels_Nm"], color="#0072B2", lw=.8, label="L2 norm, 67 applied channels")
    axes[0, 1].plot(end, n["actual_torque_max_abs_67channels_Nm"], color="#D55E00", lw=.8, label="Largest |applied channel torque|")
    axes[0, 1].set_ylabel("Applied torque diagnostic (N m)")
    for ax, fields in ((axes[1, 0], (("task_pcc_clearance", "Existing PCC clearance"), ("task_capsule_clearance", "Existing capsule clearance"))),
                       (axes[1, 1], (("task_interval_current_envelope_margin_m", "Current envelope margin"), ("task_interval_ramp_minimum_envelope_margin_m", "Ramp minimum envelope margin")))):
        for key, title in fields:
            if key in c:
                values = np.where(np.isfinite(c[key]), c[key]*1000., np.nan)
                ax.plot(c["time_s"], values, lw=.9, label=title)
        ax.set_ylabel("Saved original diagnostic (mm)")
    for ax in axes.flat:
        shade(ax, record["declaration"]["route_interval_s"])
        ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=.09, right=.985, top=.97, bottom=.16, wspace=.28, hspace=.3)
    footer(fig, label(record)+"\nQP/PCC/interval diagnostics: saved consumed 20 ms ticks. Torque: original applied 2 ms intervals, labeled by interval end.\nSix reused slots have no new QP vectors; no reconstruction. PCC, capsule and interval margins retain their distinct original scopes.")
    return save(fig, directory, record["slot"]["slot_id"]+"_control_diagnostics")


def write_csv(path, columns, indices=None):
    require(not path.exists(), "append-only CSV already exists")
    keys = list(columns)
    if indices is None:
        indices = range(len(columns[keys[0]]))
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(keys)
        for index in indices:
            writer.writerow([(format(float(columns[key][index]), ".17g") if np.isfinite(columns[key][index]) else "")
                if np.issubdtype(np.asarray(columns[key]).dtype, np.floating) else int(columns[key][index]) for key in keys])


def plot_task(records, directory):
    fig, axes = plt.subplots(3, 2, figsize=(12., 9.))
    fields = (("consumed", "continuum_position_tracking_error_m", 1000., "Continuum tracking error (mm)"),
        ("consumed", "rigid_position_tracking_error_m", 1000., "Rigid tracking error (mm)"),
        ("consumed", "continuum_orientation_tracking_error_deg", 1., "Continuum orientation error (deg)"),
        ("consumed", "rigid_orientation_tracking_error_deg", 1., "Rigid orientation error (deg)"),
        ("native", "base_translation_drift_m", 1000., "Base translation drift (mm)"),
        ("native", "continuum_sphere_minimum_m", 1000., "Related-sphere clearance (mm)"))
    for ax, (group, field, scale, ylabel) in zip(axes.flat, fields):
        for i, record in enumerate(records):
            values = record[group]; stride = 10 if group == "native" else 1
            ax.plot(values["time_s"][::stride], values[field][::stride]*scale, color=COLORS[i], ls=STYLES[i], lw=1.)
        shade(ax, records[0]["declaration"]["route_interval_s"]); ax.set_ylabel(ylabel)
    fig.legend(handles=[Line2D([], [], color=c, ls=s, label=l) for c, s, l in zip(COLORS, STYLES, LABELS)], ncol=4,
        loc="upper center", bbox_to_anchor=(.5, .995), frameon=False)
    fig.subplots_adjust(left=.09, right=.985, top=.92, bottom=.115, wspace=.28, hspace=.32)
    declaration = records[0]["declaration"]
    footer(fig, f"{declaration['task_id']} | d={declaration['distance_m']*1000.:.0f} mm | all seven fixed modes\nTracking: fresh actual vs actually consumed QP reference at the same 20 ms tick. Shaded: frozen route interval.\nTask/base deviations and independent Task acceptance are separate; no new trial or geometry query.")
    return save(fig, directory, "task_"+declaration["task_id"]+"_comparison")


def plot_matrix(task_groups, directory):
    panels = (("Route intervention (rad/s)", lambda r: r["summary"]["original_full_quality_vector"]["I_route_rad_s"]),
        ("Continuum tracking, route RMS (mm)", lambda r: r["summary"]["consumed_reference_tracking"]["continuum_position_tracking_error_m"]["route"]["rms"]*1000.),
        ("Rigid tracking, whole-task RMS (mm)", lambda r: r["summary"]["consumed_reference_tracking"]["rigid_position_tracking_error_m"]["whole_task"]["rms"]*1000.),
        ("Base translation, whole-task peak (mm)", lambda r: r["summary"]["native_diagnostics"]["base_translation_drift_m"]["whole_task"]["maximum"]*1000.),
        ("Related sphere, route minimum (mm)", lambda r: r["summary"]["native_diagnostics"]["continuum_sphere_minimum_m"]["route"]["minimum"]*1000.))
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 9.1))
    task_labels = [f"{group[0]['declaration']['distance_m']*1000.:.0f} mm c{'+' if group[0]['declaration']['side'] == 1 else '-'}" for group in task_groups]
    for ax, (title, getter) in zip(axes.flat, panels):
        values = np.asarray([[getter(record) for record in group] for group in task_groups])
        ax.imshow(values, cmap="cividis", aspect="auto")
        ax.set_xticks(range(7), ("z0", "v1 +12", "v1 -12", "v2 +12", "v2 -12", "v2 +20", "v2 -20"), rotation=25, ha="right")
        ax.set_yticks(range(4), task_labels); ax.set_xlabel(title)
        low, high = float(values.min()), float(values.max())
        for row in range(4):
            for col in range(7):
                ax.text(col, row, f"{values[row,col]:.4f}" if "rad/s" in title else f"{values[row,col]:.3f}",
                    ha="center", va="center", fontsize=7, color="white" if values[row,col] < (low+high)/2 else "black")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02, .9, "Four development Tasks × seven fixed candidates\nAll values from sealed saved evidence.\n\nEach panel has its own color scale.\nTracking RMS is reference tracking,\nnot independent Task failure.\n\nNo new rollout, geometry, solver, model,\ntraining, or continuous-time certification.", va="top", fontsize=11, linespacing=1.7)
    fig.subplots_adjust(left=.08, right=.985, top=.985, bottom=.12, wspace=.28, hspace=.6)
    footer(fig, "Complete route interval and whole-task scope are explicit per panel. Raw source digests, exact native/consumed clocks, and full-precision CSVs: plots.json.")
    return save(fig, directory, "all_tasks_metric_matrix")


def generate(source, output):
    started = time.perf_counter(); sources = Sources(source)
    plan, report = sources.read("plan.json"), sources.read("report.json")
    matrix = sources.read("quality_matrix.json")
    require(len(plan["slots"]) == 28 and len(plan["tasks"]) == 4 and report["complete_safe_slots"] == 28, "wrong sealed 4x7 complete study")
    require(tuple(plan["teacher"]["constant_policy_candidate_set"]) == MODES, "mode order differs")
    groups = []
    for declaration in plan["tasks"]:
        task = sources.read(declaration["task_path"], declaration["task_file_sha256"])
        members = [slot for slot in plan["slots"] if slot["task_id"] == declaration["task_id"]]
        require(tuple(slot["mode"] for slot in members) == MODES, "task fixed mode order differs")
        group = []
        for slot in members:
            record = load_slot(sources, slot, task, declaration)
            prior = next(row for row in matrix["rows"] if row["slot_id"] == slot["slot_id"])
            require(prior["eligible"] and prior["full_task_and_original_safety_passed"], "matrix eligibility disagrees")
            require(record["summary"]["original_full_quality_vector"] == prior["original_full_quality_vector"], "original quality payload differs from sealed matrix")
            group.append(record)
        groups.append(group)
    print(json.dumps({"source_contract_preflight": "PASS", "fixed_slots": 28, "sealed_sources": len(sources.files)}), flush=True)
    directory = output/"figures"
    require(not directory.exists(), "append-only figure directory already exists")
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"source").mkdir()
    script = Path(__file__).resolve(); script_bytes = script.read_bytes()
    with (directory/"source"/script.name).open("xb") as stream:
        stream.write(script_bytes)
    records, task_figures = [], []
    for group in groups:
        declaration = group[0]["declaration"]
        for record in group:
            slot = record["slot"]
            fig_paths, pdf_paths = {}, {}
            for key, plotter in (("trajectory", plot_trajectory), ("tracking", plot_tracking), ("safety_base", plot_safety), ("control_diagnostics", plot_control)):
                fig_paths[key], pdf_paths[key] = plotter(record, directory)
            native_csv = directory/(slot["slot_id"]+"_native_20ms.csv")
            consumed_csv = directory/(slot["slot_id"]+"_consumed_20ms.csv")
            control_csv = directory/(slot["slot_id"]+"_control_20ms.csv")
            native_indices = np.unique(np.r_[np.arange(0, 13501, 10), 13500])
            write_csv(native_csv, record["native"], native_indices)
            write_csv(consumed_csv, record["consumed"])
            write_csv(control_csv, record["control_native"], np.arange(9, 13500, 10))
            records.append({"slot_id": slot["slot_id"], "task_id": slot["task_id"], "mode": slot["mode"],
                "source_slot": slot["old_source_slot"] or slot["slot_id"], "source_role": slot["source_role"],
                "figures": fig_paths, "pdf": pdf_paths,
                "csv": {"native": "figures/"+native_csv.name, "consumed": "figures/"+consumed_csv.name, "control": "figures/"+control_csv.name},
                "summary": record["summary"], "source_records": record["source_records"]})
            print(json.dumps({"slot_id": slot["slot_id"], "figures_completed": 4, "csv_completed": 3}), flush=True)
        png, pdf = plot_task(group, directory)
        task_figures.append({"task_id": declaration["task_id"], "source_slots": [r["slot"]["slot_id"] for r in group],
                             "figures": {"comparison": png}, "pdf": {"comparison": pdf}})
    png, pdf = plot_matrix(groups, directory)
    sources.verify_unchanged()
    metrics = {"consumed_reference_tracking": "Euclidean Cartesian position norm and SO(3) principal angle at exact consumed pre-step QP ticks; original 0..26.98s, no resampling/interpolation",
        "native": "Saved independent fresh kinematics at all 13501 original 2ms states, 0..27s inclusive; statistics use all states",
        "native_csv": "Every tenth original fresh state plus terminal: 1351 exact 20ms samples with original fresh_state_index; full precision .17g; not a new simulation",
        "consumed_csv": "All 1350 actually consumed QP-input reference ticks with matching fresh_state_index; original time/position; .17g",
        "control_csv": "Every tenth applied 2ms torque interval (ends .020..27.0s), original interval start/end preserved; all-native torque statistics; .17g",
        "old_missing_QP_vectors": "Six original reused slots have original scalar intervention only; nominal/selected vector norms omitted and NOT_MEASURED_OLD_EVIDENCE; no vector inference",
        "existing_PCC_interval_diagnostics": "Saved existing PCC/capsule clearances and current/ramp envelope margins, each retains original pair/envelope scope; nonfinite unavailable CSV cells are empty, no zero imputation",
        "Task_base_path": "Original independent minimum-jerk Task position polynomial evaluated at saved times; no residual reference reconstructed; deviation is not Task failure",
        "rigid_Task": "Target-frame Task grasp transformed by fresh target pose, displayed only in terminal Task window [25.5,27]s",
        "Task_gate": "Original independent evaluator requirement results copied verbatim; no curve-wide tolerance or gate inferred",
        "base_drift": "Fresh base translation norm vs frozen Task.base_pose; rotation principal angle from normalized absolute quaternion dot product",
        "robot_target_clearance": "Saved native target_minimum_m over original robot-target pair set; censor flags retained in CSV",
        "related_sphere_clearance": "Sealed route_clearance.npz continuum_obstacle_minimum_m at all native states; distinct pair set; no new query",
        "route_window": "Per-Task frozen complete closed interval, no endpoint or horizon search",
        "display_thinning": "Native trajectory/safety display every tenth original state plus final; tracking plots all consumed ticks; statistics never thinned",
        "source_lifecycle": "All consumed inputs checked against sealed payload/external SHA256 and rehashed unchanged at completion; local source locators are provenance; generated CSV/PDF/PNG are self-contained review artifacts"}
    outputs = {p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(directory.rglob("*")) if p.is_file()}
    manifest = {"schema": "v64_b31_execution_aware_media_plots_v1", "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_report_sha256": sources.files[str(source/"report.json")]["sha256"],
        "source_manifest_sha256": sources.files[str(source/"manifest.json")]["sha256"], "source_producer_commit": report["source_producer_commit"],
        "plot_producer_source_sha256": hashlib.sha256(script_bytes).hexdigest(), "plot_producer_script": "figures/source/"+script.name,
        "records": sorted(records, key=lambda r: r["slot_id"]), "task_figures": task_figures,
        "all_tasks_figures": {"metric_matrix": png}, "all_tasks_pdf": {"metric_matrix": pdf},
        "metric_contracts": metrics, "source_records": list(sources.files.values()), "outputs": outputs,
        "source_files_unchanged": True, "new_physics_steps": 0, "new_geometry_queries": 0, "new_solver_calls": 0,
        "new_models_or_sampling": 0, "new_training_runs": 0, "existing_scientific_outputs_modified": False,
        "visual_QA": "PENDING_STATIC_IMAGE_INSPECTION", "elapsed_wall_s": time.perf_counter()-started}
    with (directory/"plots.json").open("x", encoding="utf8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    return {"status": "PASS", "slots": len(records), "png_figures": 117, "pdf_figures": 117, "csv_files": 84,
            "manifest": str(directory/"plots.json"), "elapsed_wall_s": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source.resolve(), args.output.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
