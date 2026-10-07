"""Complete nominal visualization bound to the latest accepted research evidence.

All quantities come from saved files. Fresh error curves describe the
post-integration nominal kinematics diagnostic; they do not rewrite the
original 25 functional checks or 11 execution checks.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.visualization.generate_visualizations import (
    _equal_3d_axes, _plot_coordinate_frame, _plot_obstacle_spheres, _scenario_obstacles,
)
from v6_lite.visualization.latest_saved_renderer import ROOT, SCOPE, relative, render_saved_scene

BASELINE = ROOT / "v6_lite/output/runs/research_acceptance_01"
SIMULATION = BASELINE / "simulation"
NOMINAL = ROOT / "v6_lite/output/runs/research_inertia_shadow_resumed_01"
BASELINE_MANIFEST_SHA = "379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0"
NOMINAL_MANIFEST_SHA = "ee98249e6ba936278e005943dc942724e57fb5ef2f59d2fe226b0e101d89158e"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {"path": relative(path), "sha256": sha(path), "bytes": path.stat().st_size}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _manifest(root: Path, expected: str) -> dict:
    path = root / "manifest.json"
    if sha(path) != expected:
        raise ValueError(f"frozen source manifest changed: {relative(path)}")
    return {key.replace("\\", "/"): value for key, value in _json(path).items()}


def _pinned(root: Path, manifest: dict, path: Path) -> dict:
    key = path.relative_to(root).as_posix()
    result = record(path)
    if manifest.get(key) != result["sha256"]:
        raise ValueError(f"source does not match the frozen manifest: {relative(path)}")
    return result


def angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    product = np.einsum("nij,nij->n", a, b)
    return np.rad2deg(np.arccos(np.clip((product - 1.) / 2., -1., 1.)))


def _statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "p95": float(np.percentile(values, 95)),
            "max": float(values.max()), "min": float(values.min()),
            "mean": float(values.mean()), "rmse": float(np.sqrt(np.mean(values ** 2)))}


def load_nominal_data(metrics: dict) -> tuple[list[dict], list[dict]]:
    """Load and SHA-bind all nominal display inputs without doing physics."""
    if (metrics.get("execution_mode") != "research_simulation" or not metrics.get("passed")
            or metrics.get("performance_is_acceptance_gate") is not False
            or len(metrics.get("scenarios", [])) != 5):
        raise ValueError("latest nominal visualization requires the complete accepted research cohort")
    baseline_manifest = _manifest(BASELINE, BASELINE_MANIFEST_SHA)
    nominal_manifest = _manifest(NOMINAL, NOMINAL_MANIFEST_SHA)
    source_refs = [record(BASELINE / "manifest.json"), record(NOMINAL / "manifest.json")]
    # Bind the reused pixel/geometry helpers as well as the new entry points.
    # These are current rendering inputs, distinct from the immutable old run.
    for name in ("v6_lite/visualization/latest_nominal.py",
                 "v6_lite/visualization/latest_saved_renderer.py",
                 "v6_lite/visualization/generate_visualizations.py",
                 "v6_lite/run_v6_lite.py", "v6_lite/hierarchical_qp.py",
                 "v6_lite/runtime_command.py", "model_test/robot_model_spec_v5.py",
                 "model_test/whole_body_verifier_v5.py"):
        source_refs.append({**record(ROOT/name), "role": "current_rendering_source"})
    spec = default_v6_lite_robot_spec()
    if spec.source_bundle_sha256() != metrics["runtime_identity"]["model_source_bundle_sha256"]:
        raise ValueError("rendering asset bundle differs from the nominal producer")
    for path in spec._source_assets():
        source_refs.append({**record(path), "role": "current_rendering_model_asset"})
    for path in (SIMULATION / "v6_lite_metrics.json", SIMULATION / "validation.json",
                 SIMULATION / "execution_validation.json"):
        source_refs.append(_pinned(BASELINE, baseline_manifest, path))
    source_refs.append(_pinned(NOMINAL, nominal_manifest, NOMINAL / "report.json"))
    original = _json(SIMULATION / "v6_lite_metrics.json")
    # Only trace path relocation is permitted in the supplied in-memory view.
    view = json.loads(json.dumps(metrics))
    for index, item in enumerate(view["scenarios"]):
        item["trace"]["path"] = original["scenarios"][index]["trace"]["path"]
    if view != original:
        raise ValueError("supplied metrics differ from the pinned nominal metrics")
    validation, execution = _json(SIMULATION / "validation.json"), _json(SIMULATION / "execution_validation.json")
    if (not validation["passed"] or (validation["passed_count"], validation["total_count"]) != (25, 25)
            or not execution["passed"] or (execution["passed_count"], execution["total_count"]) != (11, 11)):
        raise ValueError("original research acceptance records did not pass 25/25 and 11/11")
    report = _json(NOMINAL / "report.json")
    if not report["complete"] or not report["evidence_valid"]:
        raise ValueError("saved nominal observations are incomplete")
    bundles = []
    for index, scene in enumerate(metrics["scenarios"]):
        sid = f"v6_lite_scenario_{index:02d}"
        if scene["scenario"]["scenario_id"] != sid:
            raise ValueError("fixed scene order changed")
        key = f"scene_{index:02d}_alpha_1.00"
        trace_path = SIMULATION / "traces" / f"{sid}.npz"
        timeline_path = SIMULATION / "timing" / f"{sid}.jsonl"
        trajectory_path = NOMINAL / "replays" / key / "trajectory.npz"
        obs = NOMINAL / "observations" / key
        paths = [trace_path, timeline_path, trajectory_path, obs / "fresh_kinematics.npz",
                 obs / "robot_target_500hz.npz", obs / "whole_body_dense.npz", obs / "observation.json"]
        refs = [_pinned(BASELINE if path.is_relative_to(BASELINE) else NOMINAL,
                        baseline_manifest if path.is_relative_to(BASELINE) else nominal_manifest, path)
                for path in paths]
        if refs[0]["sha256"] != scene["trace"]["sha256"]:
            raise ValueError("scene trace binding differs from frozen input")
        trace, trajectory, fresh = _archive(trace_path), _archive(trajectory_path), _archive(obs / "fresh_kinematics.npz")
        target_distance, whole_distance = _archive(obs / "robot_target_500hz.npz"), _archive(obs / "whole_body_dense.npz")
        if (trajectory["qpos"].shape != (13501, 81) or trajectory["qvel"].shape != (13501, 79)
                or trajectory["torque_input"].shape != (13500, 67) or float(trajectory["alpha"]) != 1.
                or int(trajectory["scenario_index"]) != index
                or not np.array_equal(trajectory["torque_input"], trace["torque"])
                or not np.array_equal(trajectory["qpos"][0], trace["initial_qpos"])
                or not np.array_equal(trajectory["qvel"][0], trace["initial_qvel"])
                or not np.array_equal(trajectory["qpos"][::10], trace["task_qpos"])
                or not np.array_equal(trajectory["time"], fresh["time"])
                or not np.allclose(trajectory["time"], np.arange(13501)*.002, rtol=0., atol=1e-9)
                or not np.array_equal(trajectory["time"][1:], trace["time"])):
            raise ValueError(f"saved nominal initial state/grid/torque/parity mismatch: {sid}")
        if (not bool(target_distance["complete"]) or not bool(whole_distance["complete"])
                or target_distance["minimum_m"].shape != (13501,)
                or whole_distance["minimum_m"].shape != (5401,)):
            raise ValueError("saved geometry observations are incomplete")
        if any(not np.all(np.isfinite(array)) for archive in (trajectory, fresh, target_distance, whole_distance)
               for array in archive.values() if np.issubdtype(array.dtype, np.number)):
            raise ValueError("saved display arrays contain nonfinite values")
        contract = scene["scenario"]["continuum_target"]
        reference = {"rigid_target": fresh["rigid_target"], "rigid_target_rotation": fresh["rigid_target_rotation"],
                     "continuum_target": np.vstack([contract["initial_position_w"], trace["continuum_target"]]),
                     "continuum_target_rotation": np.concatenate([np.asarray(contract["target_rotation_world"])[None], trace["continuum_target_rotation"]])}
        base = trajectory["qpos"][:, :7]
        if not np.array_equal(base[1:], trace["base_qpos"]):
            raise ValueError("named free-base saved projection differs")
        quaternion = base[:, 3:] / np.linalg.norm(base[:, 3:], axis=1, keepdims=True)
        errors = {"rigid_error": np.linalg.norm(fresh["rigid_tip"] - reference["rigid_target"], axis=1),
                  "continuum_error": np.linalg.norm(fresh["continuum_tip"] - reference["continuum_target"], axis=1),
                  "rigid_orientation_error_deg": angles(fresh["rigid_rotation"], reference["rigid_target_rotation"]),
                  "continuum_orientation_error_deg": angles(fresh["continuum_rotation"], reference["continuum_target_rotation"]),
                  "base_translation_drift_m": np.linalg.norm(base[:, :3] - base[0, :3], axis=1),
                  "base_orientation_drift_deg": np.rad2deg(2*np.arccos(np.clip(np.abs(quaternion @ quaternion[0]), 0., 1.)))}
        rows = [json.loads(line) for line in timeline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        dispatch = np.asarray([(row["actual_dispatch_monotonic_ns"] - row["state_acquisition_monotonic_ns"])*1e-9 for row in rows])
        if len(rows) != len(trace["task_time"]) or not np.allclose(dispatch, [row["dispatch_latency_s"] for row in rows], rtol=0., atol=1e-12):
            raise ValueError("raw dispatch timing scope/formula differs")
        observation = next(item for item in report["runs"] if item["key"] == key)
        if (float(target_distance["minimum_m"].min()) != observation["robot_target_500hz"]["minimum_clearance_m"]
                or float(whole_distance["minimum_m"].min()) != observation["whole_body_dense_discrete"]["minimum_clearance_m"]):
            raise ValueError("geometry minimum differs from saved observation")
        bundle = {"scenario_id": sid, "scene": scene, "trace": trace, "trajectory": trajectory,
                  "kinematics": fresh, "references": reference, "errors": errors,
                  "target_distance": target_distance, "whole_distance": whole_distance,
                  "observation": observation, "dispatch_latency_s": dispatch, "source_refs": refs}
        bundles.append(bundle)
        source_refs.extend(refs)
    return bundles, source_refs


def _finish(figure: Any, path: Path, title: str) -> None:
    figure.suptitle(title + "\nNominal non-real-time research; original acceptance unchanged", fontsize=13)
    figure.tight_layout(rect=(0., 0., 1., .955))
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def plot_errors(metrics: dict, bundles: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13.5, 18.5), sharex=True)
    names = ["rigid_error", "continuum_error", "rigid_orientation_error_deg", "base_translation_drift_m", "base_orientation_drift_deg"]
    scales = [1000., 1000., 1., 1000., 1.]
    for index, bundle in enumerate(bundles):
        color, label = plt.get_cmap("tab10")(index), f"S{index:02d}"
        t = bundle["trajectory"]["time"]
        for axis, name, scale in zip(axes, names, scales):
            axis.plot(t, bundle["errors"][name]*scale, color=color, lw=1., label=label)
        axes[2].plot(t, bundle["errors"]["continuum_orientation_error_deg"], color=color, ls=":", lw=.9, label=label+" continuum")
        axes[5].plot(bundle["trace"]["task_time"], 1000*bundle["trace"]["task_minimum_queried_clearance"], color=color, label=label+" screened online")
        axes[5].plot(bundle["whole_distance"]["physics_grid_phase"]*.002, 1000*bundle["whole_distance"]["minimum_m"], color=color, ls=":", alpha=.8, label=label+" dense native")
    config = metrics["run_config"]
    for axis, threshold, label in ((axes[0], 1000*config["rigid_final_error_threshold_m"], "Original final 0.10 mm reference"),
                                   (axes[1], 1000*config["continuum_irregular_path_rmse_threshold_m"], "Original path-RMSE 0.18 mm reference"),
                                   (axes[2], config["orientation_error_threshold_deg"], "Original orientation reference"),
                                   (axes[5], 1000*config["whole_body_minimum_clearance_m"], "5 mm discrete geometry gate")):
        axis.axhline(threshold, color="#dc2626", ls="--", label=label)
    axes[0].axvspan(25.5, 27., color="#94a3b8", alpha=.12, label="original steady window")
    axes[0].axhline(.15, color="#9333ea", ls=":", label="Original steady-RMSE 0.15 mm reference")
    axes[1].axvspan(4.5, 25.5, color="#86efac", alpha=.12, label="original active path")
    for index, waypoint_time in enumerate(4.5+np.r_[0., np.cumsum(bundles[0]["scene"]["scenario"]["continuum_target"]["segment_durations_s"])], 1):
        axes[1].axvline(waypoint_time, color="#64748b", alpha=.2, lw=.8)
        axes[1].text(waypoint_time, .98, f"W{index}", transform=axes[1].get_xaxis_transform(), ha="center", va="top", fontsize=7)
    titles = ["Fresh rigid grasp-point error", "Fresh continuum-tip / frozen irregular reference error", "Fresh rigid and continuum orientation error", "Free-base translation drift", "Free-base attitude drift", "Screened online query and saved dense native minimum (distinct scopes)"]
    units = ["Error (mm)", "Error (mm)", "Error (deg)", "Drift (mm)", "Drift (deg)", "Clearance (mm)"]
    for axis, title, unit in zip(axes, titles, units):
        axis.set(title=title, ylabel=unit, xlim=(0., 27.)); axis.grid(alpha=.2); axis.legend(ncol=3, fontsize=7)
    axes[0].set_yscale("log"); axes[2].set_yscale("log"); axes[-1].set_xlabel("Saved simulation time (s)")
    _finish(fig, path, "Five-scene six-panel evidence | fresh state diagnostic")


def plot_safety(bundles: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    classes = list(bundles[0]["observation"]["whole_body_dense_discrete"]["minimum_by_class_m"])
    x = np.arange(len(classes)); width = .15
    for index, bundle in enumerate(bundles):
        obs = bundle["observation"]
        values = [1000*obs["whole_body_dense_discrete"]["minimum_by_class_m"][key] for key in classes]
        axes[0].bar(x + (index-2)*width, values, width, label=f"S{index:02d}")
        axes[1].plot(bundle["trajectory"]["time"], 1000*bundle["target_distance"]["minimum_m"], label=f"S{index:02d} native 75 target pairs")
        censored = bundle["target_distance"]["minimum_censored_lower_bound"]
        sampled = np.flatnonzero(censored)[::100]
        if len(sampled):
            axes[1].scatter(bundle["trajectory"]["time"][sampled], 1000*bundle["target_distance"]["minimum_m"][sampled], marker="^", facecolors="none", edgecolors=plt.get_cmap("tab10")(index), s=14)
        axes[1].plot(bundle["whole_distance"]["physics_grid_phase"]*.002, 1000*bundle["whole_distance"]["minimum_m"], ls=":", label=f"S{index:02d} dense 2927 pairs")
    axes[0].set(xticks=x, xticklabels=[name.replace("_", "\n") for name in classes], ylabel="Minimum saved native distance (mm)", title="Original whole-body classes: 5401 configurations / 2927 pairs per scene")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="Saved simulation time (s)", ylabel="Native distance / censored lower bound (mm)", title="Robot-target: all 13501 saved 2ms states; hollow triangles mark censored lower bounds (sampled for display)")
    for axis in axes:
        axis.axhline(5., color="#dc2626", ls="--", label="5 mm gate"); axis.grid(alpha=.2); axis.legend(ncol=3, fontsize=8)
    _finish(fig, path, "Discrete saved-state clearance | no continuous-time safety claim")


def plot_timing(bundles: list[dict], path: Path) -> dict:
    fig, axes = plt.subplots(4, 1, figsize=(13, 14))
    statistics = {}
    names = ("algorithm_full", "dispatch_release", "torque_compute")
    for index, bundle in enumerate(bundles):
        trace = bundle["trace"]
        samples = [trace["task_interval_full_control_latency_s"], bundle["dispatch_latency_s"], trace["torque_latency"]]
        clocks = [trace["task_time"], trace["task_time"], trace["time"]]
        for axis, name, clock, values in zip(axes, names, clocks, samples):
            axis.plot(clock, 1000*values, lw=.65, alpha=.8, label=f"S{index:02d}")
            statistics[f"S{index:02d}_{name}"] = {**_statistics(values), "above_budget_count": int(np.count_nonzero(values > (.002 if name == "torque_compute" else .02)))}
    for index, (axis, title) in enumerate(zip(axes, ["Full interval algorithm call (saved samples)", "State acquisition to actual torque publication (raw monotonic difference)", "Per-physics-step torque computation (saved samples)"])):
        axis.axhline(2. if index == 2 else 20., color="#dc2626", ls="--", label="2 ms performance reference" if index == 2 else "20 ms performance reference")
        axis.set(title=title, ylabel="Wall computation (ms)", xlabel="Simulation time (s)"); axis.legend(ncol=6, fontsize=8); axis.grid(alpha=.2)
    labels = list(statistics)
    x = np.arange(len(labels))
    axes[3].bar(x-.17, [1000*statistics[key]["p95"] for key in labels], .34, label="p95")
    axes[3].bar(x+.17, [1000*statistics[key]["max"] for key in labels], .34, label="maximum")
    axes[3].set(xticks=x, xticklabels=[name.replace("_", "\n", 1) for name in labels], ylabel="Wall computation (ms)", title="All samples and long tails retained; research acceptance does not gate wall-clock percentiles")
    axes[3].tick_params(axis="x", labelsize=7); axes[3].legend(); axes[3].grid(axis="y", alpha=.2)
    _finish(fig, path, "Compute/publication/torque timing | strict wall deployment remains NOT_MET")
    return statistics


def plot_execution(bundles: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 13))
    for index, bundle in enumerate(bundles):
        trace = bundle["trace"]; t = trace["task_time"]; label = f"S{index:02d}"
        difference = np.linalg.norm(trace["task_solver_candidate"]-trace["task_selected_command"], axis=1)
        if np.any(difference != 0.) or np.any(trace["task_execution_mode"] != "TRACK") or np.any(trace["task_failure_reason"] != "none"):
            raise ValueError("nominal command selection differs from certified saved execution")
        axes[0].plot(t, np.linalg.norm(trace["task_selected_command"], axis=1), label=label)
        axes[1].plot(t, difference, label=label)
        axes[2].plot(t, 1000*trace["task_ramp_clearance_min_slack_m_s"], label=label+" ten-step ramp")
        axes[2].plot(t, 1000*trace["task_lookahead_min_slack_m_s"], ls=":", label=label+" predicted next-start")
        axes[3].plot(t, 1000*trace["task_interval_realized_next_start_minimum_slack_m_s"], label=label+" realized next-start")
    axes[0].set(ylabel="Command norm (rad/s)", title="All 6750 selected commands")
    axes[1].set(ylabel="Difference norm (rad/s)", title="Solver candidate versus selected command: saved equality")
    axes[2].set(ylabel="Residual (mm/s)", title="Original ramp and lookahead constraint residuals")
    axes[3].set(ylabel="Residual (mm/s)", xlabel="Simulation time (s)", title="Realized next-start interval check, original tolerance retained")
    for axis in axes[2:]:
        axis.axhline(0., color="#64748b", ls=":", label="zero residual")
        axis.axhline(-.1, color="#dc2626", ls="--", label="original -1e-4 m/s tolerance")
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(ncol=3, fontsize=7)
    _finish(fig, path, "Saved execution contract | 25 functional / 11 execution checks + interval audit")


def plot_actions(bundles: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(15, 15))
    torque_limit = max(float(np.max(np.abs(bundle["trace"]["torque"]))) for bundle in bundles)
    velocity_limit = max(float(np.max(np.abs(bundle["trace"]["command_velocity"]))) for bundle in bundles)
    for index, bundle in enumerate(bundles):
        trace = bundle["trace"]
        for axis, values, limit, title in ((axes[index,0], trace["torque"], torque_limit, "67 actuator torques (N m)"),
                                           (axes[index,1], trace["command_velocity"], velocity_limit, "17 planner commanded velocities (rad/s)")):
            image = axis.imshow(values.T, aspect="auto", origin="lower", interpolation="nearest", cmap="coolwarm", vmin=-limit, vmax=limit, extent=(.002,27.,-.5,values.shape[1]-.5))
            axis.set(title=f"S{index:02d}: {title}", xlabel="Simulation time (s)", ylabel="Original actuator/planner index")
            fig.colorbar(image, ax=axis, shrink=.8)
    _finish(fig, path, "Actual saved actions | unchanged 67 torque channels and 17 planner commands")


def plot_paths(bundle: dict, path: Path) -> None:
    fig = plt.figure(figsize=(14., 6.8))
    obstacles = _scenario_obstacles(bundle["scene"]["scenario"])
    for index, branch in enumerate(("rigid", "continuum"), 1):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        tip = bundle["kinematics"][f"{branch}_tip"]; target = bundle["references"][f"{branch}_target"]
        axis.plot(*target.T, ls="--", lw=2, color="#f59e0b" if branch == "rigid" else "#16a34a", label="Frozen/current-pose target")
        axis.plot(*tip.T, lw=1.7, color="#2563eb" if branch == "rigid" else "#0891b2", label="Fresh nominal tip")
        axis.scatter(*tip[0], marker="o", s=35, label="Initial"); axis.scatter(*tip[-1], marker="*", s=80, label="Final")
        selected_obstacles = obstacles[index-1:index]
        _plot_obstacle_spheres(axis, selected_obstacles)
        _equal_3d_axes(axis, np.vstack([tip, target, [item.center for item in selected_obstacles]]))
        _plot_coordinate_frame(axis, tip[-1], bundle["kinematics"][f"{branch}_rotation"][-1], .075, 1.)
        _plot_coordinate_frame(axis, target[-1], bundle["references"][f"{branch}_target_rotation"][-1], .12, .55)
        if branch == "continuum":
            waypoints = np.asarray(bundle["scene"]["scenario"]["continuum_target"]["waypoint_points_m"])
            axis.scatter(*waypoints.T, marker="x", color="#dc2626", s=45, label="W1-W7")
            for number, point in enumerate(waypoints, 1):
                axis.text(*point, f" W{number}", fontsize=8)
                _plot_coordinate_frame(axis, point, bundle["references"]["continuum_target_rotation"][0], .05, .42)
        axis.set(title=branch.title()+" arm", xlabel="World x (m)", ylabel="World y (m)", zlabel="World z (m)"); axis.legend(fontsize=8)
    _finish(fig, path, bundle["scenario_id"]+" dual-arm 3D paths | RGB=XYZ")


def plot_base_gif(bundles: list[dict], path: Path) -> dict:
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
    artists = [[axis.plot([], [], lw=1.7, label=f"S{i:02d}")[0] for i in range(5)] for axis in axes]
    keys, scales = ("base_translation_drift_m", "base_orientation_drift_deg"), (1000., 1.)
    for axis, key, scale, ylabel in zip(axes, keys, scales, ("Translation drift (mm)", "Attitude drift (deg)")):
        maximum = max(float(np.max(bundle["errors"][key]))*scale for bundle in bundles)
        axis.set(xlim=(0.,27.), ylim=(0.,max(maximum*1.08,1e-5)), ylabel=ylabel); axis.grid(alpha=.2); axis.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("Saved simulation time (s)")
    title = fig.suptitle("Nominal research free-base drift | saved t=0.00s")
    fig.tight_layout(rect=(0.,0.,1.,.95))
    times = np.linspace(0., 27., 136)
    def update(number):
        for axis_index, (key, scale) in enumerate(zip(keys, scales)):
            for index, bundle in enumerate(bundles):
                stop = int(np.searchsorted(bundle["trajectory"]["time"], times[number], side="right"))
                artists[axis_index][index].set_data(bundle["trajectory"]["time"][:stop:10], scale*bundle["errors"][key][:stop:10])
        title.set_text(f"Nominal research free-base drift | saved t={times[number]:.2f}s")
        return [title, *artists[0], *artists[1]]
    animation.FuncAnimation(fig, update, frames=len(times), interval=200, blit=False).save(path, writer=animation.PillowWriter(fps=5), dpi=100)
    plt.close(fig)
    return {"path": relative(path), "fps": 5, "frame_count": 136, "source_duration_s": 27., "encoded_duration_s": 27.2,
            "quantities": ["base_translation_drift_mm", "base_attitude_drift_deg"], "scenario_count": 5}


def generate_nominal(metrics: dict, output: Path) -> dict:
    """Return Paths plus portable, source-bound metadata for the parent builder."""
    output = output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    bundles, source_refs = load_nominal_data(metrics)
    artifacts = []
    chart_names = ["error_curves.png", "safety_clearance_summary.png", "full_control_timing.png", "execution_contract.png", "actual_actions.png", "base_pose_drift.gif"]
    if any((output/name).exists() for name in chart_names):
        raise FileExistsError("nominal visualization artifacts already exist")
    for name, function in ((chart_names[0], lambda path: plot_errors(metrics, bundles, path)),
                           (chart_names[1], lambda path: plot_safety(bundles, path)),
                           (chart_names[3], lambda path: plot_execution(bundles, path)),
                           (chart_names[4], lambda path: plot_actions(bundles, path))):
        path = output/name; function(path); artifacts.append(path); print(f"generated {name}", flush=True)
    path = output/chart_names[2]; timing = plot_timing(bundles, path); artifacts.append(path)
    path = output/chart_names[5]; gif = plot_base_gif(bundles, path); artifacts.append(path)
    videos, scene_data = [], []
    baseline_manifest = _manifest(BASELINE, BASELINE_MANIFEST_SHA)
    for bundle in bundles:
        sid = bundle["scenario_id"]
        path = output/f"{sid}_tracking_paths_3d.png"
        if path.exists():
            raise FileExistsError(path)
        plot_paths(bundle, path); artifacts.append(path)
        files, metadata = render_saved_scene(metrics, bundle, output/"videos")
        artifacts.extend(files); videos.append(metadata)
        pcc_dest = output/"pcc_monitor"/sid/"plots"; pcc_dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in ("distance_comparison.png", "gradient_comparison.png", "minimum_clearance_comparison.png"):
            source = SIMULATION/"pcc_monitor"/sid/"plots"/name
            pin = _pinned(BASELINE, baseline_manifest, source)
            destination = pcc_dest/name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(source, destination)
            if sha(destination) != pin["sha256"]:
                raise ValueError("PCC plot byte copy differs")
            copied.append({"source": pin, "artifact": record(destination)})
            source_refs.append(pin); artifacts.append(destination)
        scene_data.append({"scenario_id": sid, "saved_state_count": len(bundle["trajectory"]["time"]),
                           "fresh_tracking_statistics_initial_excluded": {key: _statistics(value[1:]) for key,value in bundle["errors"].items()},
                           "original_phase_window_diagnostic_statistics": bundle["observation"]["fresh_kinematics"]["original_phase_window_diagnostic_statistics"],
                           "whole_body_dense_discrete": bundle["observation"]["whole_body_dense_discrete"],
                           "robot_target_500hz": bundle["observation"]["robot_target_500hz"], "copied_pcc_plots": copied})
        print(f"rendered {sid}: five views + grid + focus, 811 frames each", flush=True)
    return {"artifacts": artifacts, "videos": videos, "scenes": scene_data, "source_refs": source_refs,
            "scope": SCOPE, "acceptance": {"functional_checks": "25/25", "execution_checks": "11/11", "performance_is_acceptance_gate": False,
                                          "wall_deployment_certified": False, "fresh_diagnostic_relabels_acceptance": False},
            "physics_steps_executed": 0, "video_count": 35, "preview_count": 10,
            "timing_statistics_s": timing, "base_drift_gif": gif,
            "limitations": ["Fresh kinematic diagnostics have a current post-integration cache and retain frozen reference timing; original acceptance metrics are not replaced.",
                            "Screened online distances and full native saved-distance scopes are distinct.",
                            "Native distance records are discrete observations, not continuous-time safety or hardware evidence.",
                            "Legacy PCC monitor binding fields are placeholders; copied monitor plots do not establish interval-CBF activity.",
                            "Rendering is a presentation of saved states, not a new physical replay or acceptance run."]}
