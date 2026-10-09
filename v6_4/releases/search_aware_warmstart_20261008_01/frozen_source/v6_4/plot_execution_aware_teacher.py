"""Exactly two PNGs from B3.1 terminal files and saved arrays only.

No repository/controller/model imports, simulation, geometry, inference or
browser. Old sources are read at their declared hashes, never reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SCHEMA = "v64_b31_execution_aware_teacher_figures_v1"
MODES = ("z0", "v1_plus12", "v1_minus12", "v2_plus12", "v2_minus12", "v2_plus20", "v2_minus20")
LABELS = ("z0", "v1 +12", "v1 -12", "v2 +12", "v2 -12", "v2 +20", "v2 -20")
TICK_LABELS = ("z0", "v1\n+12", "v1\n-12", "v2\n+12", "v2\n-12", "v2\n+20", "v2\n-20")
COLORS = ("#333333", "#D55E00", "#0072B2", "#CC79A7", "#009E73", "#E69F00", "#56B4E9")
OUTPUTS = ("fig_reference_actual.png", "fig_route_quality.png", "plot_manifest.json")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Inputs:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.bindings = {}
        self.hash_calls = 0
        self.bytes_hashed = 0

    def bind(self, path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = self.source/path
        path = path.resolve()
        if not path.is_relative_to(self.source) and expected is None:
            raise ValueError("external source needs an explicit frozen digest: "+str(path))
        digest = sha(path)
        self.hash_calls += 1
        self.bytes_hashed += path.stat().st_size
        if expected is not None and digest != expected:
            raise ValueError("source hash mismatch: "+str(path))
        key = str(path)
        if key in self.bindings and self.bindings[key]["sha256"] != digest:
            raise ValueError("source changed during plotting: "+key)
        self.bindings[key] = {"path": key, "sha256": digest, "bytes": path.stat().st_size,
                              "external_reused_source": not path.is_relative_to(self.source)}
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text(encoding="utf-8-sig"))

    def verify(self):
        for item in list(self.bindings.values()):
            self.bind(item["path"], item["sha256"])


def base_reference(task, times):
    """Declared original minimum-jerk position polynomial; no provider/model."""
    source = task["scenario"]["continuum_target"]
    if source.get("mode") != "irregular_waypoints" or source.get("reference_profile") != "minimum_jerk_c2":
        raise ValueError("unsupported base reference")
    times = np.asarray(times, dtype=float)
    points = np.asarray(source["waypoint_points_m"], dtype=float)
    durations = np.asarray(source["segment_durations_s"], dtype=float)
    initial = np.asarray(source["initial_position_w"], dtype=float)
    transition = float(source["transition_duration_s"])
    if (times.ndim != 1 or not np.isfinite(times).all() or np.any(times < -1e-9)
            or np.any(times > 27.+1e-9) or points.ndim != 2 or points.shape[1] != 3
            or durations.shape != (len(points)-1,) or not np.isfinite(points).all()
            or not np.isfinite(durations).all() or np.any(durations <= 0.)
            or abs(durations.sum()-source["path_duration_s"]) > 1e-9):
        raise ValueError("invalid declared minimum-jerk reference")
    cumulative = np.r_[0., np.cumsum(durations)]
    output = np.empty((len(times), 3))
    for index, t in enumerate(np.clip(times, 0., 27.)):
        if t < transition:
            first, last, u = initial, points[0], t/transition
        elif t >= transition+source["path_duration_s"]:
            output[index] = points[-1]
            continue
        else:
            elapsed = t-transition
            segment = min(max(int(np.searchsorted(cumulative, elapsed, side="right")-1), 0), len(durations)-1)
            first, last = points[segment], points[segment+1]
            u = (elapsed-cumulative[segment])/durations[segment]
        output[index] = first+(10*u**3-15*u**4+6*u**5)*(last-first)
    return output


def reference_offset(plan, times):
    definition = plan["definition"]
    version = definition["representation_version"]
    if version not in ("task_anchored_cartesian_residual_v1", "task_anchored_cartesian_residual_v2"):
        raise ValueError("unsupported residual version")
    if plan.get("representation_version") != version:
        raise ValueError("plan/definition version mismatch")
    z = np.asarray(plan["z_m"], dtype=float)
    times = np.asarray(times, dtype=float)
    if z.shape != (6, 2) or not np.isfinite(z).all() or times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("invalid residual coefficient or reference time")
    output = np.zeros((len(times), 3))
    for interval, basis, coefficient, enabled in zip(definition["intervals_s"], definition["transverse_bases"], z, definition["interval_mask"]):
        if not enabled:
            if np.any(coefficient):
                raise ValueError("nonzero inactive residual")
            continue
        lower, upper = interval
        inside = (times > lower) & (times < upper)
        u = np.clip((times-lower)/(upper-lower), 0., 1.)
        if version.endswith("_v1"):
            weight = 64.*u**3*(1.-u)**3
        else:
            basis_function = definition["basis_function"]
            if (basis_function["kind"] != "minimum_jerk_c2_plateau"
                    or basis_function["rise_ratio"] != .25 or basis_function["fall_start_ratio"] != .75):
                raise ValueError("unexpected frozen v2 plateau")
            v = np.where(u < .25, u/.25, np.where(u > .75, (1.-u)/.25, 1.))
            weight = 10.*v**3-15.*v**4+6.*v**5
        output += np.where(inside, weight, 0.)[:, None]*(np.asarray(basis)@coefficient)
    return output


def common_saved_ticks(first, second):
    """Align physical 2 ms saved ticks without interpolating either actual."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    ai, bi = np.rint(a/.002).astype(np.int64), np.rint(b/.002).astype(np.int64)
    if (np.max(np.abs(a-ai*.002), initial=0.) > 1e-8
            or np.max(np.abs(b-bi*.002), initial=0.) > 1e-8
            or len(np.unique(ai)) != len(ai) or len(np.unique(bi)) != len(bi)):
        raise ValueError("actual clocks are not unique saved native ticks")
    _, ia, ib = np.intersect1d(ai, bi, assume_unique=True, return_indices=True)
    if not np.allclose(a[ia], b[ib], atol=1e-9, rtol=0.):
        raise ValueError("actual physical clocks differ")
    return ia, ib


def load_fresh(inputs, result, quality, eligible):
    if not result.get("evaluation_path"):
        if eligible:
            raise ValueError("eligible candidate lacks evaluation")
        return None, "NO_INDEPENDENT_FRESH_STATES"
    evaluation_path = inputs.bind(result["evaluation_path"], result["evaluation_sha256"])
    evaluation = inputs.read(evaluation_path, result["evaluation_sha256"])
    manifest_path = evaluation_path.parent/"manifest.json"
    fresh = evaluation_path.parent/"fresh_replay.npz"
    # An external old manifest is bound by the already verified quality sources.
    expected_manifest = quality.get("sources", {}).get(str(manifest_path))
    if (not eligible and not manifest_path.exists() and expected_manifest is None and not fresh.exists()
            and evaluation.get("schema") == "v64_b2_unexecuted_evaluation_v1"
            and evaluation.get("status") == "NOT_RUN" and result.get("actual_steps", 0) == 0
            and not result.get("full_task_success") and not evaluation.get("evidence_valid")
            and not evaluation.get("full_task_success")):
        return None, "NO_INDEPENDENT_FRESH_STATES"
    if not manifest_path.is_relative_to(inputs.source) and expected_manifest is None:
        raise ValueError("old evaluation manifest lacks a frozen quality binding")
    manifest = inputs.read(manifest_path, expected_manifest)
    for relative, digest in manifest.items():
        inputs.bind(evaluation_path.parent/relative, digest)
    report_fresh_sha = evaluation.get("fresh_replay_sha256")
    if not fresh.exists():
        if ("fresh_replay.npz" in manifest or report_fresh_sha
                or eligible or evaluation.get("full_task_success")):
            raise ValueError("claimed independent fresh states missing")
        return None, "FAILED_EVALUATION_WITHOUT_FRESH_STATES"
    if "fresh_replay.npz" not in manifest:
        raise ValueError("present fresh states lack manifest digest")
    expected_fresh = report_fresh_sha or quality.get("sources", {}).get(str(fresh))
    if expected_fresh is None:
        raise ValueError("present fresh states lack an independently bound digest")
    if manifest["fresh_replay.npz"] != expected_fresh:
        raise ValueError("fresh replay digest differs between bound evaluation/quality and manifest")
    inputs.bind(fresh, expected_fresh)
    with np.load(fresh, allow_pickle=False) as saved:
        times = np.asarray(saved["time"], dtype=float).copy()
        position = np.asarray(saved["continuum_position"], dtype=float).copy()
    steps = int(result.get("actual_steps", 0))
    if (times.shape != (steps+1,) or position.shape != (steps+1, 3)
            or not np.isfinite(times).all() or not np.isfinite(position).all()
            or abs(times[0]) > 1e-9 or np.any(np.diff(times) <= 0)
            or abs(times[-1]-.002*steps) > 1e-8):
        raise ValueError("independent saved actual prefix invalid")
    common_saved_ticks(times, times)
    if eligible and (steps != 13500 or abs(times[-1]-27.) > 1e-9):
        raise ValueError("eligible candidate lacks complete saved horizon")
    return {"time": times, "position": position, "path": str(fresh),
            "sha256": manifest["fresh_replay.npz"]}, "INDEPENDENT_FRESH_FULL" if eligible else "INDEPENDENT_FRESH_PREFIX_OR_INELIGIBLE"


def load_study(source):
    inputs = Inputs(source)
    plan = inputs.read("plan.json")
    manifest = inputs.read("task_manifest.json", plan["task_manifest_sha256"])
    inputs.algorithm_producer_commit = inputs.read("source_identity.json")["algorithm_producer_commit"]
    complete = inputs.read("execution_complete.json")
    if complete.get("terminal_slots") != 28 or len(plan["slots"]) != 28 or manifest["slots"] != plan["slots"]:
        raise ValueError("plotting requires the original 28 fixed terminal slots")
    for name in ("quality_matrix.json", "teacher_records.json", "postprocessing_source_identity.json"):
        if (inputs.source/name).exists():
            inputs.bind(name)
    tasks = []
    for declared in plan["tasks"]:
        task = inputs.read(declared["task_path"], declared["task_file_sha256"])
        if object_sha(task) != declared["task_sha256"]:
            raise ValueError("Task content hash mismatch")
        members = [slot for slot in plan["slots"] if slot["task_id"] == declared["task_id"]]
        if tuple(slot["mode"] for slot in members) != MODES:
            raise ValueError("fixed candidate order differs")
        records = []
        for slot in members:
            terminal = inputs.read(Path("slots")/slot["slot_id"]/"slot_result.json")
            if (terminal["frozen_slot"] != slot or terminal["slot_id"] != slot["slot_id"]
                    or terminal["status"] == "STARTED"):
                raise ValueError("terminal slot differs from frozen declaration")
            for path, digest in terminal["evidence_bindings"].items():
                inputs.bind(path, digest)
            candidate = inputs.read(slot["plan_path"], slot["plan_file_sha256"])
            if object_sha(candidate) != slot["plan_sha256"]:
                raise ValueError("reference plan content hash mismatch")
            quality, result = terminal["quality"], terminal["source_result"]
            if quality["task_sha256"] != slot["task_sha256"] or result["task_sha256"] != slot["task_sha256"]:
                raise ValueError("quality/result Task identity mismatch")
            for path, digest in quality["sources"].items():
                inputs.bind(path, digest)
            eligible = bool(quality.get("quality_label_eligible") is True
                            and quality.get("safety", {}).get("full_task_and_original_safety_passed") is True)
            metrics = quality.get("full_metrics") if eligible else None
            if eligible and (not metrics or not result.get("full_task_success")):
                raise ValueError("qualified full metrics or result missing")
            if not eligible and quality.get("full_metrics"):
                raise ValueError("failed candidate has full trajectory metrics")
            actual, state_status = load_fresh(inputs, result, quality, eligible)
            records.append({"slot": slot, "plan": candidate, "quality": quality,
                "eligible": eligible, "metrics": metrics, "actual": actual,
                "status": terminal["status"], "state_status": state_status})
        basis = np.asarray(records[0]["plan"]["definition"]["transverse_bases"][declared["key_interval_slot"]])
        for record in records:
            definition = record["plan"]["definition"]
            if (definition["intervals_s"][declared["key_interval_slot"]] != declared["route_interval_s"]
                    or not np.array_equal(basis, definition["transverse_bases"][declared["key_interval_slot"]])):
                raise ValueError("candidate route window or fixed world basis differs")
        tasks.append({"declaration": declared, "task": task, "records": records, "axis": basis[:, 0]})
    if len(tasks) != 4:
        raise ValueError("plot requires all four declared development tasks")
    inputs.verify()
    return inputs, plan, tasks


def task_title(task):
    d = task["declaration"]
    return f"d={1000*d['distance_m']:.0f} mm, c{'+' if d['side'] == 1 else '-'}"


def window_mask(times, interval):
    return (times >= interval[0]-1e-9) & (times <= interval[1]+1e-9)


def failed_note(task):
    failed = [LABELS[i] for i, record in enumerate(task["records"]) if not record["eligible"]]
    note = "All 7 candidates complete + safe" if not failed else "FAILED / REFUSED / INELIGIBLE: "+", ".join(failed)
    return textwrap.fill(note, width=65)


def thin(x, y, count=900):
    indices = np.unique(np.linspace(0, len(x)-1, min(len(x), count)).round().astype(int))
    return x[indices], y[indices]


def response(task, record):
    zero, actual = task["records"][0]["actual"], record["actual"]
    if actual is None or zero is None:
        return None
    ia, iz = common_saved_ticks(actual["time"], zero["time"])
    times = actual["time"][ia]
    use = window_mask(times, task["declaration"]["route_interval_s"])
    times = times[use]
    lateral = ((actual["position"][ia[use]]-zero["position"][iz[use]]) @ task["axis"])*1000.
    full = bool(record["eligible"] and task["records"][0]["eligible"])
    return {"time": times, "lateral_mm": lateral, "full_comparable": full,
            "rms_mm": float(np.sqrt(np.mean(lateral*lateral))) if full and len(times) else None,
            "scope": "same-Task common native saved ticks; no actual interpolation"}


def draw_reference_actual(tasks, path):
    fig, axes = plt.subplots(4, 3, figsize=(18, 15), squeeze=False)
    fig.suptitle("B3.1 | Reference shape and measured lateral route response", fontsize=17, y=.993)
    titles = ("Reference - frozen base", "Actual - frozen base", "Actual - same-Task z0 actual")
    for row, task in enumerate(tasks):
        interval, axis = task["declaration"]["route_interval_s"], task["axis"]
        grid = np.linspace(*interval, 701)
        for column, ax in enumerate(axes[row]):
            ax.axhline(0., color="#888888", lw=.8)
            ax.set_xlim(interval)
            ax.set_xlabel("Physical time [s]")
            ax.set_ylabel("Lateral displacement [mm]")
            ax.grid(alpha=.2)
            ax.set_title(task_title(task)+" | "+titles[column], fontsize=11)
        for index, record in enumerate(task["records"]):
            color = COLORS[index]
            style = ":" if index == 0 else ("--" if record["slot"]["reference_version"] == "v1" else "-")
            reference = (reference_offset(record["plan"], grid) @ axis)*1000.
            axes[row, 0].plot(grid, reference, color=color, ls=style, lw=1.7)
            actual = record["actual"]
            if actual is not None:
                use = window_mask(actual["time"], interval)
                times = actual["time"][use]
                lateral = ((actual["position"][use]-base_reference(task["task"], times)) @ axis)*1000.
                if len(times):
                    axes[row, 1].plot(*thin(times, lateral), color=color, ls=style, lw=1.5,
                                      alpha=1. if record["eligible"] else .55)
                    if not record["eligible"]:
                        axes[row, 1].plot(times[-1], lateral[-1], marker="x", color=color, ms=7)
            delta = response(task, record)
            if delta is not None and len(delta["time"]):
                axes[row, 2].plot(*thin(delta["time"], delta["lateral_mm"]), color=color, ls=style,
                                  lw=1.5, alpha=1. if delta["full_comparable"] else .55)
                if not delta["full_comparable"]:
                    axes[row, 2].plot(delta["time"][-1], delta["lateral_mm"][-1], marker="x", color=color, ms=7)
        axes[row, 0].text(0., -.26, failed_note(task), transform=axes[row, 0].transAxes,
                          fontsize=7.5, color="#444444", wrap=True)
    handles = [Line2D([], [], color=COLORS[i], lw=2, label=LABELS[i]+" mm" if i else LABELS[i]) for i in range(7)]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(.5, .022), fontsize=10)
    fig.text(.5, .008, "Fixed world transverse axis; complete predeclared route window. Dashed=v1, solid=v2. Faint/x=failed or incomplete; missing actual stays missing.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.955, bottom=.09, hspace=.62, wspace=.27)
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=155, facecolor="white")
    plt.close(fig)


def full_metric(record, name):
    return float(record["metrics"][name]) if record["eligible"] else np.nan


def route_clearance(record):
    if not record["eligible"]:
        return np.nan
    return 1000.*float(record["metrics"]["continuum_route_obstacle_clearance"]["route_window_minimum_m"])


def missing_marks(ax, values):
    for index, value in enumerate(values):
        if not np.isfinite(value):
            ax.text(index, .03, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                    color="#777777", rotation=90)


def draw_route_quality(tasks, path):
    fig, axes = plt.subplots(4, 3, figsize=(18, 15), squeeze=False)
    fig.suptitle("B3.1 | Every fixed candidate: full-task quality and reference-to-actual response", fontsize=16, y=.993)
    x = np.arange(7)
    for row, task in enumerate(tasks):
        records, interval = task["records"], task["declaration"]["route_interval_s"]
        for ax in axes[row]:
            ax.set_xticks(x, TICK_LABELS, fontsize=8)
            ax.grid(axis="y", alpha=.2)
            ax.set_xlim(-.6, 6.6)
        intervention = [full_metric(record, "I_route_rad_s") for record in records]
        ax = axes[row, 0]
        ax.bar(x, intervention, color=COLORS, width=.68)
        ax.set_title(task_title(task)+" | Original route intervention", fontsize=11)
        ax.set_ylabel("17D I_route [rad/s]")
        missing_marks(ax, intervention)
        for index, value in enumerate(intervention):
            if np.isfinite(value):
                ax.annotate(f"{value:.4f}", (index, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        ax.set_ylim(bottom=0.)
        ax.margins(y=.18)
        clearance = [route_clearance(record) for record in records]
        path_length = [full_metric(record, "continuum_path_length_m") for record in records]
        ax = axes[row, 1]
        ax.bar(x, clearance, color=COLORS, width=.68, alpha=.75)
        ax.set_ylabel("Route min continuum-sphere clearance [mm]")
        ax.set_title(task_title(task)+" | Native geometry / full path", fontsize=11)
        missing_marks(ax, clearance)
        secondary = ax.twinx()
        secondary.scatter(x, path_length, marker="D", s=27, facecolor="white", edgecolor="#222222", zorder=5)
        secondary.set_ylabel("Full-task continuum path [m]")
        if not np.isfinite(path_length).any():
            secondary.set_ylim(0., 1.)
        ax = axes[row, 2]
        grid = np.arange(int(np.ceil(interval[0]/.002)), int(np.floor(interval[1]/.002))+1)*.002
        ref_rms = [float(np.sqrt(np.mean(((reference_offset(record["plan"], grid) @ task["axis"])*1000.)**2))) for record in records]
        actual_rms = []
        for record in records:
            delta = response(task, record)
            actual_rms.append(np.nan if delta is None or delta["rms_mm"] is None else delta["rms_mm"])
        ax.bar(x-.18, ref_rms, width=.34, color=COLORS, alpha=.35, label="Reference - base (native-time grid)")
        ax.bar(x+.18, actual_rms, width=.34, color=COLORS, label="Actual - z0 actual (native ticks)")
        missing_marks(ax, actual_rms)
        ax.set_title(task_title(task)+" | Complete-window RMS lateral response", fontsize=10.5)
        ax.set_ylabel("RMS lateral displacement [mm]")
        if row == 0:
            ax.legend(fontsize=7, loc="upper left")
        axes[row, 0].text(0., -.24, failed_note(task), transform=axes[row, 0].transAxes, fontsize=7.5, wrap=True)
    fig.text(.5, .02, "All 28 slots retained. Full quality shown only after original complete Task + safety gates. No failure cost imputation; N/A is not zero.", ha="center", fontsize=10)
    fig.text(.5, .006, "Clearance bars = measured local continuum-sphere pair policy; diamonds = full-task path. Response is a finite paired diagnostic, not a controller transfer gain.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.065, right=.96, top=.955, bottom=.09, hspace=.62, wspace=.41)
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=155, facecolor="white")
    plt.close(fig)


def compact_records(tasks):
    records = []
    for task in tasks:
        for record in task["records"]:
            delta = response(task, record)
            records.append({"slot_id": record["slot"]["slot_id"], "task_id": record["slot"]["task_id"],
                "mode": record["slot"]["mode"], "reference_version": record["plan"]["representation_version"],
                "status": record["status"], "full_quality_eligible": record["eligible"],
                "source_role": record["slot"]["source_role"], "fresh_state_status": record["state_status"],
                "state_source": None if record["actual"] is None else record["actual"]["path"],
                "source_state_count": 0 if record["actual"] is None else len(record["actual"]["time"]),
                "I_route_rad_s": None if not record["eligible"] else full_metric(record, "I_route_rad_s"),
                "route_clearance_mm": None if not record["eligible"] else route_clearance(record),
                "full_continuum_path_m": None if not record["eligible"] else full_metric(record, "continuum_path_length_m"),
                "complete_window_actual_minus_zero_rms_mm": None if delta is None else delta["rms_mm"],
                "response_common_saved_window_points": 0 if delta is None else len(delta["time"]),
                "new_component_vectors_reconstructed_from_old_scalar": False})
    return records


def build(source, output=None):
    started = time.perf_counter()
    source = Path(source).resolve()
    output = Path(output).resolve() if output else source/"figures"
    if any((output/name).exists() for name in OUTPUTS):
        raise FileExistsError("exclusive figure outputs already exist; do not overwrite")
    inputs, plan, tasks = load_study(source)
    records = compact_records(tasks)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "font.size": 10}):
        draw_reference_actual(tasks, output/OUTPUTS[0])
        draw_route_quality(tasks, output/OUTPUTS[1])
    inputs.verify()
    payload = {"schema": SCHEMA, "source": str(source), "run_id": plan["run_id"],
        "actual_algorithm_producer_commit": inputs.algorithm_producer_commit,
        "terminal_fixed_slots": 28, "tasks": 4, "png_count": 2, "PDF_count": 0, "video_count": 0,
        "task_panels": [{"task_id": task["declaration"]["task_id"],
            "distance_m": task["declaration"]["distance_m"], "side": task["declaration"]["side"],
            "route_interval_s": task["declaration"]["route_interval_s"],
            "fixed_world_lateral_axis": task["axis"].tolist(),
            "slot_ids": [record["slot"]["slot_id"] for record in task["records"]]} for task in tasks],
        "source_sha_bindings": list(inputs.bindings.values()), "records": records,
        "figures": [
            {"path": OUTPUTS[0], "sha256": sha(output/OUTPUTS[0]), "caption": "四任务完整预声明路线窗口：解析参考相对冻结基础参考、保存 actual 相对基础参考，以及同任务非零 actual 相对 z0 actual 的固定世界横向投影。失败前缀仅作透明诊断，缺失 actual 不重建。"},
            {"path": OUTPUTS[1], "sha256": sha(output/OUTPUTS[1]), "caption": "四任务全部七个固定候选：原 17D I_route、相关连续体—球原生净空与完整路径长度、参考和 actual−z0 的窗口 RMS 响应；完整失败/拒绝不填完整质量成本，N/A 不代表零。"}],
        "scope": {"time": "physical time; complete frozen route interval; native saved 2ms actual ticks",
            "actual": "independent hash-bound fresh_replay.npz only; no interpolation or model reconstruction",
            "reference": "declared analytic original minimum-jerk base and independent frozen v1/v2 basis",
            "displacements_distinguished": ["reference minus base", "actual minus base", "nonzero actual minus same-Task zero actual"],
            "response_not_claimed_as_transfer_gain_or_causal_minimum": True,
            "old_QP_component_vectors": "unavailable, never reverse-inferred from scalar",
            "full_failure_prefix_quality_separated": True, "development_data_not_independent_TEST": True,
            "deployment": "NOT_MET", "Diffusion_benefit_claim": False},
        "plot_cost": {"wall_s": time.perf_counter()-started, "source_files": len(inputs.bindings),
            "source_unique_bytes": sum(item["bytes"] for item in inputs.bindings.values()),
            "source_hash_calls": inputs.hash_calls, "source_bytes_hashed": inputs.bytes_hashed,
            "new_actual_slots": 0, "physics_steps": 0, "preview_steps": 0, "replay_steps": 0,
            "geometry_queries": 0, "QP_solves": 0, "model_calls": 0, "DDIM_samples": 0,
            "training_runs": 0, "optimizer_updates": 0, "PNGs": 2, "PDFs": 0, "videos": 0},
        "publication_generator_sha256": sha(Path(__file__))}
    with (output/OUTPUTS[2]).open("x", encoding="utf8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = build(args.source, args.output)
    print(json.dumps({"schema": payload["schema"], "png_count": 2, "slots": 28, "plot_cost": payload["plot_cost"]}))


if __name__ == "__main__":
    main()
