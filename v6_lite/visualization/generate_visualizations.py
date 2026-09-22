"""Generate V6-lite figures, five-view replays, or one continuum-focus video."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model_test.whole_body_verifier_v5 import (
    WholeBodyCollisionVerifier,
    WholeBodyVerificationConfig,
    WorkspaceSphere,
)
from v6_lite.hierarchical_qp import CONTINUUM_EE_OFFSET_M, free_joint_slices
from v6_lite.run_v6_lite import default_v6_lite_robot_spec


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
DEFAULT_METRICS = ROOT / "output" / "v6_lite_metrics.json"
DEFAULT_OUTPUT = ROOT / "visualization" / "output"
VIEWS = ("overview", "front", "side", "top", "iso")
SOURCE_CONTRACT_VERSION = "v6_lite_6"
VISUALIZATION_CONTRACT_VERSION = "v6_lite_visualization_6"
CONTINUUM_FOCUS_AZIMUTH_DEG = -90.0
CONTINUUM_FOCUS_ELEVATION_DEG = -18.0
CONTINUUM_FOCUS_RIGID_ALPHA = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Use repository-relative paths in manifests whenever possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _scenario_obstacles(scenario: dict[str, Any]) -> tuple[WorkspaceSphere, ...]:
    return tuple(
        WorkspaceSphere(
            name=str(item["name"]),
            center=np.asarray(item["center_w"], dtype=np.float64),
            radius=float(item["radius_m"]),
        )
        for item in scenario["workspace_obstacles"]
    )


def _equal_3d_axes(axis: Any, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float(np.max(maximum - minimum)) * 0.58, 0.06)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _plot_obstacle_spheres(axis: Any, obstacles: tuple[WorkspaceSphere, ...]) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 15)
    for obstacle in obstacles:
        x = obstacle.center[0] + obstacle.radius * np.outer(np.cos(u), np.sin(v))
        y = obstacle.center[1] + obstacle.radius * np.outer(np.sin(u), np.sin(v))
        z = obstacle.center[2] + obstacle.radius * np.outer(np.ones_like(u), np.cos(v))
        axis.plot_surface(x, y, z, color="#f97316", alpha=0.28, linewidth=0)


def _clearance_summary_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract the two distinct safety quantities defined by V6-lite 6."""

    if metrics.get("contract_version") != SOURCE_CONTRACT_VERSION:
        raise ValueError(
            "clearance visualization requires source contract "
            f"{SOURCE_CONTRACT_VERSION!r}"
        )
    scenario_ids: list[str] = []
    whole_body_m: list[float] = []
    continuum_target_m: list[float] = []
    for scenario_result in metrics.get("scenarios", []):
        scenario_ids.append(str(scenario_result["scenario"]["scenario_id"]))
        report = scenario_result["metrics"]["whole_body_clearance"]
        whole_body_m.append(float(report["minimum_clearance"]))
        continuum_target_m.append(
            float(report["minimum_by_class"]["continuum_target"])
        )
    if not scenario_ids:
        raise ValueError("clearance visualization requires at least one scenario")
    values = np.asarray([whole_body_m, continuum_target_m], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("clearance summary values must be finite")
    return {
        "scenario_ids": scenario_ids,
        "whole_body_minimum_clearance_m": whole_body_m,
        "continuum_target_minimum_clearance_m": continuum_target_m,
        "verification_gate_m": float(
            metrics["run_config"]["whole_body_minimum_clearance_m"]
        ),
        "continuum_target_qp_nominal_margin_m": float(
            metrics["qp_config"]["clearance_safe_m"]
        ),
        "rigid_target_qp_nominal_margin_m": float(
            metrics["qp_config"]["rigid_target_clearance_safe_m"]
        ),
    }


def plot_clearance_summary(
    metrics: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Plot whole-body and continuum-target minima without conflating them."""

    summary = _clearance_summary_payload(metrics)
    scenario_ids = summary["scenario_ids"]
    x = np.arange(len(scenario_ids), dtype=np.float64)
    whole_body_mm = 1000.0 * np.asarray(
        summary["whole_body_minimum_clearance_m"], dtype=np.float64
    )
    continuum_target_mm = 1000.0 * np.asarray(
        summary["continuum_target_minimum_clearance_m"], dtype=np.float64
    )
    verification_gate_mm = 1000.0 * float(summary["verification_gate_m"])
    continuum_qp_margin_mm = 1000.0 * float(
        summary["continuum_target_qp_nominal_margin_m"]
    )

    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), sharex=True)
    panels = (
        (
            axes[0],
            whole_body_mm,
            "#2563eb",
            "Global whole-body minimum signed clearance",
        ),
        (
            axes[1],
            continuum_target_mm,
            "#059669",
            "Continuum-arm to moving-target minimum signed clearance",
        ),
    )
    for axis, values_mm, color, title in panels:
        bars = axis.bar(x, values_mm, width=0.62, color=color, alpha=0.86)
        axis.axhline(
            verification_gate_mm,
            color="#dc2626",
            ls="--",
            lw=1.5,
            label=f"dense-discrete verification gate ({verification_gate_mm:g} mm)",
        )
        axis.bar_label(bars, fmt="%.3f mm", padding=3, fontsize=8)
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_ylabel("Clearance (mm)")
        axis.grid(True, axis="y", alpha=0.24)
        axis.legend(frameon=False, loc="best")
        axis.set_ylim(
            0.0,
            max(float(np.max(values_mm)), verification_gate_mm) * 1.22 + 1.0,
        )
    axes[1].axhline(
        continuum_qp_margin_mm,
        color="#7c3aed",
        ls=":",
        lw=1.6,
        label=(
            "continuum/other-pair QP nominal margin "
            f"({continuum_qp_margin_mm:g} mm)"
        ),
    )
    axes[1].set_ylim(
        0.0,
        max(
            float(np.max(continuum_target_mm)),
            verification_gate_mm,
            continuum_qp_margin_mm,
        )
        * 1.22
        + 1.0,
    )
    axes[1].legend(frameon=False, loc="best")
    axes[1].set_xticks(x, scenario_ids, rotation=18, ha="right")
    axes[1].set_xlabel("Seeded scenario")
    figure.suptitle(
        "V6-lite safety clearance audit (dense-discrete replay)",
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.012,
        "The global minimum may be set by a 5 mm rigid-target class; "
        "it is not evidence that every pair maintained the 25 mm nominal margin.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return summary


def plot_error_curves(metrics: dict[str, Any], output_path: Path) -> None:
    run_config = metrics["run_config"]
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(6, 1, figsize=(13.5, 18.5), sharex=True)
    for index, scenario_result in enumerate(metrics["scenarios"]):
        trace = np.load(Path(scenario_result["trace"]["path"]), allow_pickle=False)
        label = f"S{index + 1}"
        color = colors(index)
        axes[0].plot(
            trace["time"], 1000.0 * trace["rigid_error"], color=color, lw=1.25, label=label
        )
        axes[1].plot(
            trace["time"],
            1000.0 * trace["continuum_error"],
            color=color,
            lw=1.25,
            label=label,
        )
        axes[2].plot(
            trace["time"],
            trace["rigid_orientation_error_deg"],
            color=color,
            lw=1.2,
            label=f"{label} rigid",
        )
        axes[2].plot(
            trace["time"],
            trace["continuum_orientation_error_deg"],
            color=color,
            lw=1.1,
            ls=":",
            label=f"{label} continuum",
        )
        axes[3].plot(
            trace["time"],
            1000.0 * trace["base_translation_drift_m"],
            color=color,
            lw=1.25,
            label=label,
        )
        axes[4].plot(
            trace["time"],
            trace["base_orientation_drift_deg"],
            color=color,
            lw=1.25,
            label=label,
        )
        axes[5].plot(
            trace["task_time"],
            1000.0 * trace["task_minimum_queried_clearance"],
            color=color,
            lw=1.25,
            label=label,
        )
    axes[0].axhline(
        1000.0 * run_config["rigid_final_error_threshold_m"],
        color="#dc2626",
        ls="--",
        lw=1.4,
        label="0.10 mm final-error gate",
    )
    axes[0].axhline(
        1000.0 * run_config["rigid_steady_rmse_threshold_m"],
        color="#9333ea",
        ls=":",
        lw=1.4,
        label="0.15 mm steady-RMSE gate",
    )
    axes[1].axhline(
        1000.0 * run_config["continuum_irregular_path_rmse_threshold_m"],
        color="#dc2626",
        ls="--",
        lw=1.4,
        label="0.18 mm active-path RMSE gate",
    )
    axes[2].axhline(
        run_config["orientation_error_threshold_deg"],
        color="#dc2626",
        ls="--",
        lw=1.4,
        label="0.25 deg full-run gate",
    )
    axes[5].axhline(
        1000.0 * run_config["whole_body_minimum_clearance_m"],
        color="#dc2626",
        ls="--",
        lw=1.4,
        label="5 mm verification gate",
    )
    titles = (
        "Rigid-arm grasp-point tracking error",
        "Continuum-tip irregular-waypoint tracking error",
        "Rigid and continuum orientation tracking error",
        "Free-base translation drift from initialized pose",
        "Free-base attitude drift from initialized pose",
        "Minimum online queried signed distance across all active pair classes",
    )
    ylabels = (
        "Position error (mm)",
        "Position error (mm)",
        "Orientation error (deg)",
        "Translation drift (mm)",
        "Attitude drift (deg)",
        "Clearance (mm)",
    )
    steady_start = float(run_config["duration_s"]) - float(
        run_config["steady_window_s"]
    )
    axes[0].axvspan(
        steady_start,
        float(run_config["duration_s"]),
        color="#94a3b8",
        alpha=0.10,
        label="rigid steady window",
    )
    target_contract = metrics["scenarios"][0]["scenario"]["continuum_target"]
    axes[1].axvspan(
        float(target_contract["path_start_s"]),
        float(target_contract["path_end_s"]),
        color="#86efac",
        alpha=0.10,
        label="active irregular path",
    )
    waypoint_times = float(target_contract["path_start_s"]) + np.concatenate(
        [np.zeros(1), np.cumsum(target_contract["segment_durations_s"])]
    )
    for waypoint_index, waypoint_time in enumerate(waypoint_times, start=1):
        axes[1].axvline(waypoint_time, color="#64748b", alpha=0.20, lw=0.8)
        axes[1].text(
            waypoint_time,
            0.98,
            f"W{waypoint_index}",
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="#475569",
        )
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.24)
        axis.legend(ncol=4, fontsize=8, frameon=False, loc="best")
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[-1].set_xlabel("Simulation time (s)")
    figure.suptitle(
        "V6-lite closed-loop evidence across five seeded scenarios",
        fontsize=15,
        fontweight="semibold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_base_pose_drift_gif(
    metrics: dict[str, Any],
    output_path: Path,
    *,
    fps: int = 5,
) -> dict[str, Any]:
    """Animate all five free-base translation and attitude drift traces."""

    colors = plt.get_cmap("tab10")
    series: list[dict[str, np.ndarray]] = []
    for scenario_result in metrics["scenarios"]:
        with np.load(Path(scenario_result["trace"]["path"]), allow_pickle=False) as trace:
            series.append(
                {
                    "time": np.asarray(trace["time"])[::10],
                    "translation_mm": 1000.0
                    * np.asarray(trace["base_translation_drift_m"])[::10],
                    "attitude_deg": np.asarray(
                        trace["base_orientation_drift_deg"]
                    )[::10],
                }
            )
    duration = float(metrics["run_config"]["duration_s"])
    frame_count = int(round(duration * fps)) + 1
    frame_times = np.linspace(0.0, duration, frame_count)
    figure, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
    translation_lines = []
    attitude_lines = []
    for index in range(len(series)):
        label = f"S{index + 1}"
        translation_line, = axes[0].plot([], [], color=colors(index), lw=1.8, label=label)
        attitude_line, = axes[1].plot([], [], color=colors(index), lw=1.8, label=label)
        translation_lines.append(translation_line)
        attitude_lines.append(attitude_line)
    cursor_lines = [
        axis.axvline(0.0, color="#334155", lw=1.0, ls="--", alpha=0.7)
        for axis in axes
    ]
    translation_max = max(float(np.max(item["translation_mm"])) for item in series)
    attitude_max = max(float(np.max(item["attitude_deg"])) for item in series)
    axes[0].set_ylim(0.0, max(translation_max * 1.08, 1e-4))
    axes[1].set_ylim(0.0, max(attitude_max * 1.08, 1e-4))
    axes[0].set_ylabel("Translation drift (mm)")
    axes[1].set_ylabel("Attitude drift (deg)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].set_title("Free-base translation drift", loc="left", fontweight="semibold")
    axes[1].set_title("Free-base attitude drift (SO(3) geodesic)", loc="left", fontweight="semibold")
    for axis in axes:
        axis.set_xlim(0.0, duration)
        axis.grid(True, alpha=0.24)
        axis.legend(ncol=5, fontsize=8, frameon=False, loc="upper left")
    title = figure.suptitle("V6-lite free-base pose drift | t=0.00 s", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def _update(frame_index: int) -> list[Any]:
        current_time = float(frame_times[frame_index])
        artists: list[Any] = []
        for index, item in enumerate(series):
            stop = int(np.searchsorted(item["time"], current_time, side="right"))
            translation_lines[index].set_data(
                item["time"][:stop], item["translation_mm"][:stop]
            )
            attitude_lines[index].set_data(
                item["time"][:stop], item["attitude_deg"][:stop]
            )
            artists.extend((translation_lines[index], attitude_lines[index]))
        for cursor in cursor_lines:
            cursor.set_xdata([current_time, current_time])
            artists.append(cursor)
        title.set_text(f"V6-lite free-base pose drift | t={current_time:5.2f} s")
        artists.append(title)
        return artists

    movie = animation.FuncAnimation(
        figure,
        _update,
        frames=frame_count,
        interval=1000.0 / fps,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return {
        "path": _portable_path(output_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": frame_count / float(fps),
        "quantities": ["base_translation_drift_mm", "base_attitude_drift_deg"],
        "scenario_count": len(series),
    }


def plot_tracking_paths(metrics: dict[str, Any], output_path: Path) -> None:
    scenario_result = metrics["scenarios"][0]
    trace = np.load(Path(scenario_result["trace"]["path"]), allow_pickle=False)
    obstacles = _scenario_obstacles(scenario_result["scenario"])
    figure = plt.figure(figsize=(14.0, 6.6))
    rigid_axis = figure.add_subplot(1, 2, 1, projection="3d")
    continuum_axis = figure.add_subplot(1, 2, 2, projection="3d")

    rigid_axis.plot(
        *trace["rigid_target"].T,
        color="#f59e0b",
        ls="--",
        lw=2.0,
        label="Satellite grasp point",
    )
    rigid_axis.plot(
        *trace["rigid_tip"].T,
        color="#2563eb",
        lw=2.0,
        label="Rigid tip",
    )
    rigid_axis.scatter(*trace["rigid_tip"][0], color="#2563eb", marker="o", s=40)
    rigid_axis.scatter(*trace["rigid_tip"][-1], color="#2563eb", marker="*", s=90)
    rigid_obstacles = obstacles[:1]
    _plot_obstacle_spheres(rigid_axis, rigid_obstacles)
    _equal_3d_axes(
        rigid_axis,
        np.vstack(
            [
                trace["rigid_tip"],
                trace["rigid_target"],
                np.asarray([item.center for item in rigid_obstacles]),
            ]
        ),
    )
    rigid_axis.set_title("Rigid arm: drifting satellite grasp point")

    continuum_axis.plot(
        *trace["continuum_target"].T,
        color="#16a34a",
        ls="--",
        lw=2.0,
        label="Irregular-waypoint reference",
    )
    continuum_axis.plot(
        *trace["continuum_tip"].T,
        color="#0891b2",
        lw=2.0,
        label="Continuum tip",
    )
    continuum_axis.scatter(
        *trace["continuum_tip"][0], color="#0891b2", marker="o", s=40
    )
    continuum_axis.scatter(
        *trace["continuum_tip"][-1], color="#0891b2", marker="*", s=90
    )
    continuum_obstacles = obstacles[1:2]
    waypoints = np.asarray(
        scenario_result["scenario"]["continuum_target"]["waypoint_points_m"],
        dtype=np.float64,
    )
    continuum_axis.scatter(
        *waypoints.T,
        color="#dc2626",
        marker="x",
        s=48,
        linewidths=1.6,
        label="W1-W7",
    )
    for waypoint_index, waypoint in enumerate(waypoints, start=1):
        continuum_axis.text(*waypoint, f" W{waypoint_index}", fontsize=8)
        _plot_coordinate_frame(
            continuum_axis,
            waypoint,
            trace["continuum_target_rotation"][0],
            0.050,
            0.42,
        )
    _plot_obstacle_spheres(continuum_axis, continuum_obstacles)
    _equal_3d_axes(
        continuum_axis,
        np.vstack(
            [
                trace["continuum_tip"],
                trace["continuum_target"],
                np.asarray([item.center for item in continuum_obstacles]),
            ]
        ),
    )
    continuum_axis.set_title("Continuum arm: original irregular waypoints")

    _plot_coordinate_frame(
        rigid_axis,
        trace["rigid_target"][-1],
        trace["rigid_target_rotation"][-1],
        0.120,
        0.55,
    )
    _plot_coordinate_frame(
        rigid_axis,
        trace["rigid_tip"][-1],
        trace["rigid_rotation"][-1],
        0.075,
        1.0,
    )
    _plot_coordinate_frame(
        continuum_axis,
        trace["continuum_target"][-1],
        trace["continuum_target_rotation"][-1],
        0.120,
        0.55,
    )
    _plot_coordinate_frame(
        continuum_axis,
        trace["continuum_tip"][-1],
        trace["continuum_rotation"][-1],
        0.075,
        1.0,
    )

    for axis in (rigid_axis, continuum_axis):
        axis.set_xlabel("World x (m)")
        axis.set_ylabel("World y (m)")
        axis.set_zlabel("World z (m)")
        axis.legend(frameon=False, fontsize=9)
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "V6-lite tracking paths | RGB = XYZ; W1-W7 frames shown on the irregular path",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _make_cameras(lookat: np.ndarray, distance: float) -> dict[str, mujoco.MjvCamera]:
    specifications = {
        "overview": (135.0, -18.0, 1.10),
        "front": (180.0, -12.0, 1.00),
        "side": (90.0, -18.0, 1.00),
        "top": (180.0, -82.0, 1.03),
        "iso": (45.0, -30.0, 1.08),
    }
    cameras: dict[str, mujoco.MjvCamera] = {}
    for name, (azimuth, elevation, multiplier) in specifications.items():
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.distance = distance * multiplier
        camera.azimuth = azimuth
        camera.elevation = elevation
        cameras[name] = camera
    return cameras


def _make_continuum_focus_camera(
    focus_points: np.ndarray,
) -> tuple[mujoco.MjvCamera, dict[str, Any]]:
    """Frame only the continuum branch and the moving target satellite.

    The continuum branch occupies the positive-y side of the V6-lite robot,
    while the rigid branch approaches from negative y.  MuJoCo azimuth -90
    observes the continuum side with the rigid branch predominantly behind
    the target/base instead of in front of the continuum backbone.
    """

    points = np.asarray(focus_points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 2 or np.any(~np.isfinite(points)):
        raise ValueError("continuum-focus framing requires finite 3D points")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    lookat = 0.5 * (minimum + maximum)
    radius = float(np.max(np.linalg.norm(points - lookat[None, :], axis=1)))
    # Keep a visible border around both ends of the continuum backbone and
    # the outer waypoint labels in the 4:3 video frame.
    distance = max(1.60, 1.80 * (radius + 0.08))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = CONTINUUM_FOCUS_AZIMUTH_DEG
    camera.elevation = CONTINUUM_FOCUS_ELEVATION_DEG
    return camera, {
        "lookat_m": lookat.tolist(),
        "distance_m": distance,
        "azimuth_deg": CONTINUUM_FOCUS_AZIMUTH_DEG,
        "elevation_deg": CONTINUUM_FOCUS_ELEVATION_DEG,
        "focus_bounds_min_m": minimum.tolist(),
        "focus_bounds_max_m": maximum.tolist(),
    }


def _continuum_focus_camera_from_replay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task_qpos: np.ndarray,
    continuum_body_ids: set[int],
    target_geom_id: int,
    extra_points: np.ndarray,
) -> tuple[mujoco.MjvCamera, dict[str, Any]]:
    samples = np.unique(
        np.linspace(0, task_qpos.shape[0] - 1, num=min(41, task_qpos.shape[0]))
        .round()
        .astype(np.int64)
    )
    signs = np.asarray(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    half_extents = np.asarray(model.geom_size[target_geom_id, :3], dtype=np.float64)
    ordered_bodies = np.asarray(sorted(continuum_body_ids), dtype=np.int64)
    points = [np.asarray(extra_points, dtype=np.float64).reshape(-1, 3)]
    for sample in samples:
        data.qpos[:] = task_qpos[int(sample)]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        points.append(np.asarray(data.xpos[ordered_bodies], dtype=np.float64).copy())
        rotation = np.asarray(data.geom_xmat[target_geom_id]).reshape(3, 3)
        center = np.asarray(data.geom_xpos[target_geom_id], dtype=np.float64)
        points.append(center[None, :] + (signs * half_extents[None, :]) @ rotation.T)
    return _make_continuum_focus_camera(np.vstack(points))


def _dim_rigid_arm_geometries(
    model: mujoco.MjModel, rigid_body_ids: set[int]
) -> int:
    """Make rigid-arm visual and collision geoms contextual but unobtrusive."""

    changed = 0
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in rigid_body_ids:
            continue
        original_alpha = float(model.geom_rgba[geom_id, 3])
        model.geom_rgba[geom_id, :3] = np.asarray([0.55, 0.58, 0.62])
        model.geom_rgba[geom_id, 3] = min(
            original_alpha, CONTINUUM_FOCUS_RIGID_ALPHA
        )
        changed += 1
    return changed


def _append_sphere(
    scene: mujoco.MjvScene, position: np.ndarray, radius: float, rgba: np.ndarray
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo scene geometry capacity exhausted")
    geometry = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geometry,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _append_label(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    label: str,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo scene geometry capacity exhausted")
    geometry = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geometry,
        mujoco.mjtGeom.mjGEOM_LABEL,
        np.zeros(3, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    geometry.label = str(label)
    scene.ngeom += 1


def _append_connector(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    stop: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo scene geometry capacity exhausted")
    geometry = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geometry,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geometry,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(start, dtype=np.float64),
        np.asarray(stop, dtype=np.float64),
    )
    scene.ngeom += 1


def _frame_axis_endpoints(
    position: np.ndarray, rotation: np.ndarray, length: float
) -> np.ndarray:
    origin = np.asarray(position, dtype=np.float64).reshape(3)
    frame = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.stack(
        [np.stack([origin, origin + float(length) * frame[:, axis]]) for axis in range(3)]
    )


def _quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    quaternion = quaternion / np.linalg.norm(quaternion)
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return rotation.reshape(3, 3)


def _plot_coordinate_frame(
    axis: Any,
    position: np.ndarray,
    rotation: np.ndarray,
    length: float,
    alpha: float,
) -> None:
    colors = ("#ef4444", "#22c55e", "#2563eb")
    for axis_index, endpoints in enumerate(
        _frame_axis_endpoints(position, rotation, length)
    ):
        delta = endpoints[1] - endpoints[0]
        axis.quiver(
            *endpoints[0],
            *delta,
            color=colors[axis_index],
            alpha=alpha,
            arrow_length_ratio=0.22,
            linewidth=1.5,
        )


def _append_coordinate_frame(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rotation: np.ndarray,
    length: float,
    alpha: float,
    radius: float,
) -> None:
    colors = (
        np.asarray([1.0, 0.12, 0.12, alpha]),
        np.asarray([0.12, 1.0, 0.20, alpha]),
        np.asarray([0.15, 0.42, 1.0, alpha]),
    )
    for axis_index, endpoints in enumerate(
        _frame_axis_endpoints(position, rotation, length)
    ):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo scene geometry capacity exhausted")
        geometry = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geometry,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            colors[axis_index].astype(np.float32),
        )
        mujoco.mjv_connector(
            geometry,
            mujoco.mjtGeom.mjGEOM_ARROW,
            float(radius),
            endpoints[0],
            endpoints[1],
        )
        scene.ngeom += 1


class _RawFfmpegWriter:
    def __init__(
        self, ffmpeg: str, path: Path, width: int, height: int, fps: float
    ) -> None:
        self.path = path
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.8f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = b"" if self.process.stderr is None else self.process.stderr.read()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed for {self.path}: {stderr.decode('utf-8', errors='replace')}"
            )


def _font(size: int) -> ImageFont.ImageFont:
    candidate = Path("C:/Windows/Fonts/arial.ttf")
    if candidate.is_file():
        return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _safety_overlay_text(
    online_whole_body_clearance_m: float,
    dense_continuum_target_minimum_clearance_m: float,
) -> str:
    return (
        "safety: online whole-body="
        f"{1000.0 * online_whole_body_clearance_m:6.2f} mm   "
        "dense run continuum-target min="
        f"{1000.0 * dense_continuum_target_minimum_clearance_m:6.2f} mm"
    )


def _annotate_frame(
    frame: np.ndarray,
    view: str,
    time_s: float,
    rigid_error_m: float,
    continuum_error_m: float,
    rigid_orientation_error_deg: float,
    continuum_orientation_error_deg: float,
    base_translation_drift_m: float,
    base_orientation_drift_deg: float,
    online_whole_body_clearance_m: float,
    dense_continuum_target_minimum_clearance_m: float,
    waypoint_status: str,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 138), fill=(10, 14, 24))
    title_font = _font(18)
    metric_font = _font(15)
    draw.text((12, 7), f"V6-lite | {view.upper()} | t={time_s:5.2f}s", font=title_font, fill=(255, 255, 255))
    draw.text(
        (12, 34),
        f"tracking: rigid={1000.0 * rigid_error_m:6.2f} mm   continuum={1000.0 * continuum_error_m:6.2f} mm",
        font=metric_font,
        fill=(203, 213, 225),
    )
    draw.text(
        (12, 54),
        _safety_overlay_text(
            online_whole_body_clearance_m,
            dense_continuum_target_minimum_clearance_m,
        ),
        font=metric_font,
        fill=(147, 197, 253),
    )
    draw.text(
        (12, 74),
        f"orientation: rigid={rigid_orientation_error_deg:6.3f} deg   continuum={continuum_orientation_error_deg:6.3f} deg",
        font=metric_font,
        fill=(167, 243, 208),
    )
    draw.text(
        (12, 94),
        f"base drift: position={1000.0 * base_translation_drift_m:7.3f} mm   attitude={base_orientation_drift_deg:7.4f} deg",
        font=metric_font,
        fill=(253, 230, 138),
    )
    draw.text(
        (12, 114),
        f"irregular path: {waypoint_status}   W1-W7 frames: faint RGB=XYZ",
        font=metric_font,
        fill=(191, 219, 254),
    )
    return np.asarray(image)


def _annotate_continuum_focus_frame(
    frame: np.ndarray,
    time_s: float,
    continuum_error_m: float,
    continuum_orientation_error_deg: float,
    online_whole_body_clearance_m: float,
    dense_continuum_target_minimum_clearance_m: float,
    waypoint_status: str,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 98), fill=(10, 14, 24))
    title_font = _font(18)
    metric_font = _font(15)
    draw.text(
        (12, 7),
        f"V6-lite | CONTINUUM FOCUS | t={time_s:5.2f}s",
        font=title_font,
        fill=(255, 255, 255),
    )
    draw.text(
        (12, 34),
        (
            f"continuum tracking={1000.0 * continuum_error_m:6.2f} mm   "
            f"orientation={continuum_orientation_error_deg:6.3f} deg"
        ),
        font=metric_font,
        fill=(167, 243, 208),
    )
    draw.text(
        (12, 54),
        _safety_overlay_text(
            online_whole_body_clearance_m,
            dense_continuum_target_minimum_clearance_m,
        ),
        font=metric_font,
        fill=(147, 197, 253),
    )
    draw.text(
        (12, 74),
        f"irregular path: {waypoint_status}   rigid arm dimmed to context",
        font=metric_font,
        fill=(253, 230, 138),
    )
    return np.asarray(image)


def _framing(model: mujoco.MjModel, data: mujoco.MjData, task_qpos: np.ndarray) -> tuple[np.ndarray, float]:
    samples = np.unique(
        np.linspace(0, task_qpos.shape[0] - 1, num=min(31, task_qpos.shape[0]))
        .round()
        .astype(np.int64)
    )
    points = []
    for index in samples:
        data.qpos[:] = task_qpos[int(index)]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        points.append(np.asarray(data.xpos[1:]).copy())
    values = np.vstack(points)
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    lookat = 0.5 * (minimum + maximum)
    radius = float(np.max(np.linalg.norm(values - lookat[None, :], axis=1)))
    return lookat, max(2.3, 1.75 * (radius + 0.15))


def _waypoint_status(target_contract: dict[str, Any], time_s: float) -> str:
    start = float(target_contract["path_start_s"])
    end = float(target_contract["path_end_s"])
    if time_s < start:
        return "transition to W1"
    if time_s >= end:
        return "W7 hold"
    cumulative = np.concatenate(
        [
            np.zeros(1),
            np.cumsum(target_contract["segment_durations_s"], dtype=np.float64),
        ]
    )
    segment = int(np.searchsorted(cumulative, time_s - start, side="right") - 1)
    segment = int(np.clip(segment, 0, len(cumulative) - 2))
    return f"W{segment + 1} to W{segment + 2}"


def render_five_views(
    metrics: dict[str, Any],
    output_dir: Path,
    *,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> tuple[list[Path], Path, Path, dict[str, Any]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode V6-lite videos")
    scenario_result = metrics["scenarios"][0]
    target_contract = scenario_result["scenario"]["continuum_target"]
    continuum_target_minimum_clearance_m = float(
        scenario_result["metrics"]["whole_body_clearance"]["minimum_by_class"][
            "continuum_target"
        ]
    )
    trace_path = Path(scenario_result["trace"]["path"])
    trace = np.load(trace_path, allow_pickle=False)
    spec = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        spec,
        _scenario_obstacles(scenario_result["scenario"]),
        WholeBodyVerificationConfig(
            minimum_clearance=float(
                metrics["run_config"]["whole_body_minimum_clearance_m"]
            ),
            adaptive_subdivisions=int(
                metrics["run_config"]["verification_subdivisions"]
            ),
        ),
    )
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    for obstacle_index, _obstacle in enumerate(
        scenario_result["scenario"]["workspace_obstacles"]
    ):
        prefix = f"v5_workspace_sphere_{obstacle_index:03d}_"
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(prefix):
                model.geom_rgba[geom_id] = np.asarray([1.0, 0.32, 0.05, 0.78])
    data = mujoco.MjData(model)
    base_qpos_slice, _base_dof_slice = free_joint_slices(
        model, spec.base_joint_name
    )
    data.qpos[:] = trace["initial_qpos"]
    data.qvel[:] = trace["initial_qvel"]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    lookat, distance = _framing(model, data, trace["task_qpos"])
    data.qpos[:] = trace["initial_qpos"]
    data.qvel[:] = trace["initial_qvel"]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    cameras = _make_cameras(lookat, distance)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"v6_lite_scenario_00_{view}.mp4" for view in VIEWS]
    writers = {
        view: _RawFfmpegWriter(ffmpeg, path, width, height, fps)
        for view, path in zip(VIEWS, paths)
    }
    renderer = mujoco.Renderer(model, height=height, width=width)
    duration = float(trace["time"][-1])
    frame_count = int(round(duration * fps)) + 1
    requested_times = np.arange(frame_count, dtype=np.float64) / fps
    frame_indices = np.searchsorted(trace["time"], requested_times, side="left")
    frame_indices = np.clip(frame_indices, 0, trace["time"].shape[0] - 1)
    requested_by_step: dict[int, list[int]] = {}
    for output_index, step_index in enumerate(frame_indices):
        requested_by_step.setdefault(int(step_index), []).append(output_index)
    rendered = 0
    waypoints = np.asarray(target_contract["waypoint_points_m"], dtype=np.float64)
    waypoint_centroid = np.mean(waypoints, axis=0)
    waypoint_radial = waypoints - waypoint_centroid[None, :]
    waypoint_radial /= np.maximum(
        np.linalg.norm(waypoint_radial, axis=1, keepdims=True), 1e-12
    )
    waypoint_label_positions = waypoints + 0.050 * waypoint_radial
    waypoint_label_positions[0] = waypoints[0] + np.asarray([0.0, -0.110, 0.055])
    waypoint_label_positions[6] = waypoints[6] + np.asarray([0.0, -0.110, -0.055])
    waypoint_rotation = np.asarray(
        target_contract["target_rotation_world"], dtype=np.float64
    )
    initial_base_pose = np.asarray(
        trace["initial_qpos"][base_qpos_slice], dtype=np.float64
    )
    initial_base_rotation = _quaternion_wxyz_to_rotation(initial_base_pose[3:7])
    try:
        for step_index, torque in enumerate(trace["torque"]):
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            if step_index not in requested_by_step:
                continue
            task_index = min(step_index // 10, trace["task_time"].shape[0] - 1)
            for _output_index in requested_by_step[step_index]:
                for view in VIEWS:
                    renderer.update_scene(data, camera=cameras[view])
                    for waypoint_index, waypoint in enumerate(waypoints):
                        _append_sphere(
                            renderer.scene,
                            waypoint,
                            0.0065,
                            np.asarray([1.0, 0.16, 0.16, 0.72]),
                        )
                        if view == "front":
                            _append_label(
                                renderer.scene,
                                waypoint_label_positions[waypoint_index],
                                f"W{waypoint_index + 1}",
                                np.asarray([1.0, 0.86, 0.86, 1.0]),
                            )
                        _append_coordinate_frame(
                            renderer.scene,
                            waypoint,
                            waypoint_rotation,
                            0.055,
                            0.48,
                            0.0022,
                        )
                        if waypoint_index > 0:
                            _append_connector(
                                renderer.scene,
                                waypoints[waypoint_index - 1],
                                waypoint,
                                0.0018,
                                np.asarray([1.0, 0.24, 0.24, 0.50]),
                            )
                    _append_coordinate_frame(
                        renderer.scene,
                        initial_base_pose[:3],
                        initial_base_rotation,
                        0.145,
                        0.34,
                        0.0040,
                    )
                    _append_coordinate_frame(
                        renderer.scene,
                        trace["base_qpos"][step_index, :3],
                        _quaternion_wxyz_to_rotation(
                            trace["base_qpos"][step_index, 3:7]
                        ),
                        0.115,
                        0.95,
                        0.0035,
                    )
                    _append_sphere(
                        renderer.scene,
                        trace["rigid_target"][step_index],
                        0.018,
                        np.asarray([1.0, 0.82, 0.05, 1.0]),
                    )
                    _append_sphere(
                        renderer.scene,
                        trace["rigid_tip"][step_index],
                        0.013,
                        np.asarray([0.15, 0.48, 1.0, 1.0]),
                    )
                    _append_connector(
                        renderer.scene,
                        trace["rigid_tip"][step_index],
                        trace["rigid_target"][step_index],
                        0.0025,
                        np.asarray([0.98, 0.72, 0.08, 0.85]),
                    )
                    _append_sphere(
                        renderer.scene,
                        trace["continuum_target"][step_index],
                        0.018,
                        np.asarray([0.20, 1.0, 0.32, 1.0]),
                    )
                    _append_sphere(
                        renderer.scene,
                        trace["continuum_tip"][step_index],
                        0.013,
                        np.asarray([0.05, 0.82, 0.95, 1.0]),
                    )
                    _append_connector(
                        renderer.scene,
                        trace["continuum_tip"][step_index],
                        trace["continuum_target"][step_index],
                        0.0025,
                        np.asarray([0.15, 0.95, 0.55, 0.85]),
                    )
                    _append_coordinate_frame(
                        renderer.scene,
                        trace["rigid_target"][step_index],
                        trace["rigid_target_rotation"][step_index],
                        0.160,
                        0.62,
                        0.0060,
                    )
                    _append_coordinate_frame(
                        renderer.scene,
                        trace["rigid_tip"][step_index],
                        trace["rigid_rotation"][step_index],
                        0.100,
                        1.0,
                        0.0040,
                    )
                    _append_coordinate_frame(
                        renderer.scene,
                        trace["continuum_target"][step_index],
                        trace["continuum_target_rotation"][step_index],
                        0.160,
                        0.62,
                        0.0060,
                    )
                    _append_coordinate_frame(
                        renderer.scene,
                        trace["continuum_tip"][step_index],
                        trace["continuum_rotation"][step_index],
                        0.100,
                        1.0,
                        0.0040,
                    )
                    frame = renderer.render()
                    annotated = _annotate_frame(
                        frame,
                        view,
                        float(trace["time"][step_index]),
                        float(trace["rigid_error"][step_index]),
                        float(trace["continuum_error"][step_index]),
                        float(trace["rigid_orientation_error_deg"][step_index]),
                        float(
                            trace["continuum_orientation_error_deg"][step_index]
                        ),
                        float(trace["base_translation_drift_m"][step_index]),
                        float(trace["base_orientation_drift_deg"][step_index]),
                        float(trace["task_minimum_queried_clearance"][task_index]),
                        continuum_target_minimum_clearance_m,
                        _waypoint_status(
                            target_contract, float(trace["time"][step_index])
                        ),
                    )
                    writers[view].append(annotated)
                rendered += 1
    finally:
        renderer.close()
        for writer in writers.values():
            writer.close()
    if rendered != frame_count:
        raise RuntimeError(f"rendered {rendered} frames, expected {frame_count}")

    composite = output_dir / "v6_lite_scenario_00_five_view_grid.mp4"
    blank_duration = frame_count / fps
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for path in paths:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x0a0e18:s={width}x{height}:r={fps}:d={blank_duration}",
            "-filter_complex",
            "[0:v][1:v][2:v]hstack=inputs=3[top];"
            "[3:v][4:v][5:v]hstack=inputs=3[bottom];"
            "[top][bottom]vstack=inputs=2[out]",
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(composite),
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"five-view composition failed: {completed.stderr}")
    preview = output_dir / "v6_lite_scenario_00_five_view_preview.png"
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{0.5 * duration:.3f}",
            "-i",
            str(composite),
            "-frames:v",
            "1",
            str(preview),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"five-view preview extraction failed: {completed.stderr}")
    metadata = {
        "scenario_id": scenario_result["scenario"]["scenario_id"],
        "source_trace": _portable_path(trace_path),
        "source_trace_sha256": _sha256(trace_path),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_s": duration,
        "camera_lookat_m": lookat.tolist(),
        "camera_distance_m": distance,
        "views": list(VIEWS),
        "coordinate_frames": {
            "axis_color_order": ["x_red", "y_green", "z_blue"],
            "target_axis_length_m": 0.160,
            "end_effector_axis_length_m": 0.100,
            "waypoint_axis_length_m": 0.055,
            "waypoint_labels_rendered": [f"W{index}" for index in range(1, 8)],
            "waypoint_label_view": "front",
            "continuum_end_effector_body_name": spec.continuum_tip_body_name,
            "continuum_end_effector_local_offset_body_m": CONTINUUM_EE_OFFSET_M.tolist(),
            "continuum_end_effector_position_source": "trace_body_origin_plus_rotated_local_offset",
            "base_initial_axis_length_m": 0.145,
            "base_current_axis_length_m": 0.115,
            "target_style": "long_translucent_arrows",
            "end_effector_style": "short_opaque_arrows",
            "frames_rendered": [
                "rigid_grasp_target",
                "rigid_end_effector",
                "continuum_irregular_target",
                "continuum_end_effector",
                "free_base_initial_pose",
                "free_base_current_pose",
                "continuum_waypoint_frames_W1_to_W7",
            ],
        },
        "overlay_metrics": [
            "rigid_position_error_mm",
            "continuum_position_error_mm",
            "rigid_orientation_error_deg",
            "continuum_orientation_error_deg",
            "base_translation_drift_mm",
            "base_attitude_drift_deg",
            "online_whole_body_clearance_mm",
            "dense_discrete_continuum_target_minimum_clearance_mm",
        ],
        "marker_legend": {
            "rigid_target": "yellow sphere",
            "rigid_tip": "blue sphere",
            "continuum_target": "green sphere",
            "continuum_tip": "cyan sphere",
            "workspace_obstacles": "orange translucent spheres",
        },
    }
    return paths, composite, preview, metadata


def render_continuum_focus_view(
    metrics: dict[str, Any],
    output_path: Path,
    *,
    width: int = 960,
    height: int = 720,
    fps: float = 30.0,
    time_limit_s: float | None = None,
) -> dict[str, Any]:
    """Render one continuum-side video without invoking the five-view path."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the continuum-focus video")
    scenario_result = metrics["scenarios"][0]
    target_contract = scenario_result["scenario"]["continuum_target"]
    trace_path = Path(scenario_result["trace"]["path"])
    trace = np.load(trace_path, allow_pickle=False)
    spec = default_v6_lite_robot_spec()
    verifier = WholeBodyCollisionVerifier(
        spec,
        _scenario_obstacles(scenario_result["scenario"]),
        WholeBodyVerificationConfig(
            minimum_clearance=float(
                metrics["run_config"]["whole_body_minimum_clearance_m"]
            ),
            adaptive_subdivisions=int(
                metrics["run_config"]["verification_subdivisions"]
            ),
        ),
    )
    model = verifier.model
    # The audited URDF defaults to a 640x480 offscreen framebuffer.  This
    # dedicated view is rendered at 960x720, so size MuJoCo's framebuffer
    # explicitly before constructing the renderer.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    continuum_body_ids = verifier._descendant_body_ids(
        verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP
    )
    rigid_body_ids = verifier._descendant_body_ids(
        verifier._RIGID_ROOT, verifier._RIGID_TIP
    )
    dimmed_geom_count = _dim_rigid_arm_geometries(model, rigid_body_ids)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    for obstacle_index, _obstacle in enumerate(
        scenario_result["scenario"]["workspace_obstacles"]
    ):
        prefix = f"v5_workspace_sphere_{obstacle_index:03d}_"
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(prefix):
                model.geom_rgba[geom_id] = np.asarray([1.0, 0.32, 0.05, 0.70])
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_satellite"
    )
    target_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
    )
    if target_body_id < 0 or target_geom_id < 0:
        raise RuntimeError("moving target satellite body/geom is missing")
    data = mujoco.MjData(model)
    waypoints = np.asarray(target_contract["waypoint_points_m"], dtype=np.float64)
    camera, camera_metadata = _continuum_focus_camera_from_replay(
        model,
        data,
        trace["task_qpos"],
        continuum_body_ids,
        int(target_geom_id),
        waypoints,
    )
    data.qpos[:] = trace["initial_qpos"]
    data.qvel[:] = trace["initial_qvel"]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    source_duration = float(trace["time"][-1])
    if time_limit_s is not None and time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be positive when provided")
    duration = (
        source_duration
        if time_limit_s is None
        else min(source_duration, float(time_limit_s))
    )
    frame_count = int(round(duration * fps)) + 1
    requested_times = np.arange(frame_count, dtype=np.float64) / fps
    frame_indices = np.searchsorted(trace["time"], requested_times, side="left")
    frame_indices = np.clip(frame_indices, 0, trace["time"].shape[0] - 1)
    requested_by_step: dict[int, list[int]] = {}
    for output_index, step_index in enumerate(frame_indices):
        requested_by_step.setdefault(int(step_index), []).append(output_index)

    continuum_target_minimum_clearance_m = float(
        scenario_result["metrics"]["whole_body_clearance"]["minimum_by_class"][
            "continuum_target"
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _RawFfmpegWriter(ffmpeg, output_path, width, height, fps)
    renderer = mujoco.Renderer(model, height=height, width=width)
    rendered = 0
    waypoint_centroid = np.mean(waypoints, axis=0)
    waypoint_radial = waypoints - waypoint_centroid[None, :]
    waypoint_radial /= np.maximum(
        np.linalg.norm(waypoint_radial, axis=1, keepdims=True), 1e-12
    )
    waypoint_label_positions = waypoints + 0.045 * waypoint_radial
    last_required_step = int(np.max(frame_indices))
    try:
        for step_index, torque in enumerate(trace["torque"]):
            if step_index > last_required_step:
                break
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            if step_index not in requested_by_step:
                continue
            task_index = min(step_index // 10, trace["task_time"].shape[0] - 1)
            for _output_index in requested_by_step[step_index]:
                renderer.update_scene(data, camera=camera)
                for waypoint_index, waypoint in enumerate(waypoints):
                    _append_sphere(
                        renderer.scene,
                        waypoint,
                        0.0065,
                        np.asarray([1.0, 0.16, 0.16, 0.72]),
                    )
                    _append_label(
                        renderer.scene,
                        waypoint_label_positions[waypoint_index],
                        f"W{waypoint_index + 1}",
                        np.asarray([1.0, 0.86, 0.86, 1.0]),
                    )
                    if waypoint_index > 0:
                        _append_connector(
                            renderer.scene,
                            waypoints[waypoint_index - 1],
                            waypoint,
                            0.0018,
                            np.asarray([1.0, 0.24, 0.24, 0.50]),
                        )
                _append_sphere(
                    renderer.scene,
                    trace["continuum_target"][step_index],
                    0.018,
                    np.asarray([0.20, 1.0, 0.32, 1.0]),
                )
                _append_sphere(
                    renderer.scene,
                    trace["continuum_tip"][step_index],
                    0.013,
                    np.asarray([0.05, 0.82, 0.95, 1.0]),
                )
                _append_connector(
                    renderer.scene,
                    trace["continuum_tip"][step_index],
                    trace["continuum_target"][step_index],
                    0.0025,
                    np.asarray([0.15, 0.95, 0.55, 0.85]),
                )
                _append_coordinate_frame(
                    renderer.scene,
                    trace["continuum_target"][step_index],
                    trace["continuum_target_rotation"][step_index],
                    0.135,
                    0.62,
                    0.0050,
                )
                _append_coordinate_frame(
                    renderer.scene,
                    trace["continuum_tip"][step_index],
                    trace["continuum_rotation"][step_index],
                    0.085,
                    1.0,
                    0.0035,
                )
                _append_label(
                    renderer.scene,
                    np.asarray(data.xpos[target_body_id])
                    + np.asarray([0.0, 0.0, 0.24]),
                    "moving target satellite",
                    np.asarray([1.0, 0.72, 0.55, 1.0]),
                )
                frame = renderer.render()
                annotated = _annotate_continuum_focus_frame(
                    frame,
                    float(trace["time"][step_index]),
                    float(trace["continuum_error"][step_index]),
                    float(trace["continuum_orientation_error_deg"][step_index]),
                    float(trace["task_minimum_queried_clearance"][task_index]),
                    continuum_target_minimum_clearance_m,
                    _waypoint_status(
                        target_contract, float(trace["time"][step_index])
                    ),
                )
                writer.append(annotated)
                rendered += 1
    finally:
        renderer.close()
        writer.close()
    if rendered != frame_count:
        raise RuntimeError(f"rendered {rendered} focus frames, expected {frame_count}")
    return {
        "view": "continuum_focus",
        "scenario_id": scenario_result["scenario"]["scenario_id"],
        "source_trace": _portable_path(trace_path),
        "source_trace_sha256": _sha256(trace_path),
        "path": _portable_path(output_path),
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration,
        "source_duration_s": source_duration,
        "rigid_arm_rendering": {
            "mode": "dimmed_context",
            "alpha": CONTINUUM_FOCUS_RIGID_ALPHA,
            "geom_count": dimmed_geom_count,
        },
        "camera": camera_metadata,
        "overlay_metrics": [
            "continuum_position_error_mm",
            "continuum_orientation_error_deg",
            "online_whole_body_clearance_mm",
            "dense_discrete_continuum_target_minimum_clearance_mm",
        ],
    }


def generate_continuum_focus(
    metrics_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Generate only the dedicated continuum-side video and its sidecar."""

    metrics = _load_json(metrics_path)
    if metrics.get("contract_version") != SOURCE_CONTRACT_VERSION:
        raise ValueError(
            f"expected {SOURCE_CONTRACT_VERSION!r} metrics, got "
            f"{metrics.get('contract_version')!r}"
        )
    if not metrics.get("passed"):
        raise ValueError("refusing to visualize a V6-lite run that did not pass")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "videos" / "v6_lite_scenario_00_continuum_focus.mp4"
    video = render_continuum_focus_view(metrics, video_path)
    manifest = {
        "contract_version": "v6_lite_continuum_focus_1",
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_metrics": _portable_path(metrics_path),
        "source_metrics_sha256": _sha256(metrics_path),
        "video": video,
    }
    manifest_path = output_dir / "continuum_focus_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest


def generate(metrics_path: Path, output_dir: Path) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    if metrics.get("contract_version") != SOURCE_CONTRACT_VERSION:
        raise ValueError(
            f"expected {SOURCE_CONTRACT_VERSION!r} metrics, got "
            f"{metrics.get('contract_version')!r}"
        )
    if not metrics.get("passed"):
        raise ValueError("refusing to visualize a V6-lite run that did not pass")
    error_path = output_dir / "error_curves.png"
    tracking_path = output_dir / "tracking_paths_3d.png"
    clearance_path = output_dir / "safety_clearance_summary.png"
    base_drift_gif_path = output_dir / "base_pose_drift.gif"
    plot_error_curves(metrics, error_path)
    plot_tracking_paths(metrics, tracking_path)
    clearance_metadata = plot_clearance_summary(metrics, clearance_path)
    gif_metadata = plot_base_pose_drift_gif(metrics, base_drift_gif_path)
    video_paths, composite_path, preview_path, video_metadata = render_five_views(
        metrics, output_dir / "videos"
    )
    artifacts = [
        error_path,
        tracking_path,
        clearance_path,
        base_drift_gif_path,
        preview_path,
        *video_paths,
        composite_path,
    ]
    manifest = {
        "contract_version": VISUALIZATION_CONTRACT_VERSION,
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_metrics": _portable_path(metrics_path),
        "source_metrics_sha256": _sha256(metrics_path),
        "video": video_metadata,
        "safety_clearance_plot": {
            "path": _portable_path(clearance_path),
            "quantities": [
                "whole_body_minimum_clearance_m",
                "continuum_target_minimum_clearance_m",
            ],
            "time_scope": "dense_discrete_replay_over_each_complete_run",
            **clearance_metadata,
        },
        "base_pose_drift_gif": gif_metadata,
        "artifacts": [
            {
                "path": _portable_path(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
    }
    manifest_path = output_dir / "visualization_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--continuum-focus-only",
        action="store_true",
        help=(
            "render only the continuum-side single-view video; do not render "
            "or replace the five-view visualization bundle"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.continuum_focus_only:
        manifest = generate_continuum_focus(
            args.metrics.resolve(), args.output_dir.resolve()
        )
        print(
            json.dumps(
                {
                    "artifact_count": 1,
                    "output_dir": args.output_dir.resolve().as_posix(),
                    "view": manifest["video"]["view"],
                    "camera": manifest["video"]["camera"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    manifest = generate(args.metrics.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "artifact_count": len(manifest["artifacts"]),
                "output_dir": args.output_dir.resolve().as_posix(),
                "views": manifest["video"]["views"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
