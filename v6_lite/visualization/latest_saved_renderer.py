"""Render saved nominal research states; never advance a physics integrator.

The displayed clock selects the first saved 2 ms state at or after each
30 Hz presentation time.  Both initial and final states are included.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_lite.hierarchical_qp import CONTINUUM_EE_OFFSET_M, free_joint_slices
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.runtime_command import model_id
from v6_lite.visualization.generate_visualizations import (
    VIEWS, CONTINUUM_FOCUS_RIGID_ALPHA, _RawFfmpegWriter,
    _append_connector, _append_coordinate_frame, _append_label, _append_sphere,
    _continuum_focus_camera_from_replay, _dim_rigid_arm_geometries, _font,
    _framing, _make_cameras, _quaternion_wxyz_to_rotation, _scenario_obstacles,
    _waypoint_status,
)

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "Nominal non-real-time research | saved states + mj_forward"


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def frame_schedule(times: np.ndarray, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or times[0] != 0.0 or np.any(np.diff(times) <= 0)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("finite, strictly increasing saved clock starting at zero required")
    duration = float(times[-1])
    count = int(round(duration * fps)) + 1
    requested = np.arange(count, dtype=float) / fps
    if abs(requested[-1] - duration) > 1e-8:
        raise ValueError("saved duration must end on the requested presentation grid")
    indices = np.searchsorted(times, requested, side="left")
    indices = np.clip(indices, 0, len(times) - 1)
    indices[0], indices[-1] = 0, len(times) - 1
    return requested, indices


def make_saved_model(metrics: dict, scene: dict) -> tuple[Any, Any, Any]:
    """Compile and bind the V6 model before making cosmetic rendering changes."""
    spec = default_v6_lite_robot_spec()
    expected_contract = metrics["runtime_identity"]["model_runtime_contract_sha256"]
    if spec.runtime_contract_sha256() != expected_contract:
        raise ValueError("current V6 model contract differs from the saved nominal source")
    if spec.source_bundle_sha256() != metrics["runtime_identity"]["model_source_bundle_sha256"]:
        raise ValueError("current model assets differ from the saved nominal source")
    verifier = WholeBodyCollisionVerifier(
        spec, _scenario_obstacles(scene["scenario"]),
        WholeBodyVerificationConfig(
            minimum_clearance=float(metrics["run_config"]["whole_body_minimum_clearance_m"]),
            query_distance_max=2.5,
            adaptive_subdivisions=int(metrics["run_config"]["verification_subdivisions"]),
            self_collision_ancestor_exclusion_depth=3,
            include_target_satellite_pairs=True,
        ),
    )
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    if model_id(model, expected_contract) != scene["execution_contract"]["source_compiled_model_sha256"]:
        raise ValueError("compiled model differs from saved nominal model identity")
    if verifier._pair_policy_sha256 != scene["metrics"]["whole_body_clearance"]["pair_policy_sha256"]:
        raise ValueError("saved nominal pair policy differs")
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("v5_workspace_sphere_"):
            model.geom_rgba[index] = [1.0, 0.32, 0.05, 0.78]
    return spec, verifier, mujoco.MjData(model)


def _annotations(frame: np.ndarray, bundle: dict, state: int, view: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 151), fill=(10, 14, 24))
    font, small = _font(17), _font(13 if image.width == 640 else 16)
    time = float(bundle["trajectory"]["time"][state])
    errors = bundle["errors"]
    native_prefix = ">=" if bundle["target_distance"]["minimum_censored_lower_bound"][state] else "="
    task = max(0, min(int(np.searchsorted(bundle["trace"]["task_time"], time,
                                         side="right") - 1), len(bundle["trace"]["task_time"]) - 1))
    lines = [
        f"{bundle['scenario_id']} | {view.upper()} | saved t={time:5.3f}s",
        "Nominal non-real-time research; saved qpos/qvel + forward",
        f"fresh tracking: rigid={1000*errors['rigid_error'][state]:7.3f} mm  continuum={1000*errors['continuum_error'][state]:7.3f} mm",
        f"orientation: rigid={errors['rigid_orientation_error_deg'][state]:6.4f} deg  continuum={errors['continuum_orientation_error_deg'][state]:6.4f} deg",
        f"base drift: {1000*errors['base_translation_drift_m'][state]:7.3f} mm / {errors['base_orientation_drift_deg'][state]:7.4f} deg",
        f"saved screened task={1000*bundle['trace']['task_minimum_queried_clearance'][task]:7.2f} mm; native robot-target{native_prefix}{1000*bundle['target_distance']['minimum_m'][state]:7.2f} mm",
        f"dense whole-body run min={1000*bundle['observation']['whole_body_dense_discrete']['minimum_clearance_m']:7.2f} mm; " + _waypoint_status(bundle["scene"]["scenario"]["continuum_target"], time),
    ]
    for index, line in enumerate(lines):
        draw.text((10, 5 + 21 * index), line, font=font if index == 0 else small,
                  fill=(255, 255, 255) if index == 0 else (203, 213, 225))
    return np.asarray(image)


def _markers(scene: Any, bundle: dict, state: int, base_slice: slice, *, focus: bool, labels: bool = True) -> None:
    kinematic = bundle["kinematics"]
    contract = bundle["scene"]["scenario"]["continuum_target"]
    waypoints = np.asarray(contract["waypoint_points_m"], dtype=float)
    direction = waypoints - waypoints.mean(axis=0)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    label_positions = waypoints + .05 * direction
    label_positions[0] = waypoints[0] + [0., -.11, .055]
    label_positions[6] = waypoints[6] + [0., -.11, -.055]
    rotation = np.asarray(contract["target_rotation_world"])
    for index, point in enumerate(waypoints):
        _append_sphere(scene, point, .0065, np.array([1., .16, .16, .72]))
        if labels:
            _append_label(scene, label_positions[index], f"W{index+1}", np.array([1., .86, .86, 1.]))
        _append_coordinate_frame(scene, point, rotation, .055, .48, .0022)
        if index:
            _append_connector(scene, waypoints[index-1], point, .0018, np.array([1., .24, .24, .50]))
    branches = ("continuum",) if focus else ("rigid", "continuum")
    for branch in branches:
        target = bundle["references"][f"{branch}_target"][state]
        target_rotation = bundle["references"][f"{branch}_target_rotation"][state]
        tip, tip_rotation = kinematic[f"{branch}_tip"][state], kinematic[f"{branch}_rotation"][state]
        target_color = np.array([1., .82, .05, 1.]) if branch == "rigid" else np.array([.20, 1., .32, 1.])
        tip_color = np.array([.15, .48, 1., 1.]) if branch == "rigid" else np.array([.05, .82, .95, 1.])
        _append_sphere(scene, target, .018, target_color)
        _append_sphere(scene, tip, .013, tip_color)
        _append_connector(scene, tip, target, .0025, target_color * [1, 1, 1, .85])
        _append_coordinate_frame(scene, target, target_rotation, .160, .62, .006)
        _append_coordinate_frame(scene, tip, tip_rotation, .100, 1., .004)
    if not focus:
        initial = bundle["trajectory"]["qpos"][0, base_slice]
        current = bundle["trajectory"]["qpos"][state, base_slice]
        _append_coordinate_frame(scene, initial[:3], _quaternion_wxyz_to_rotation(initial[3:]), .145, .34, .004)
        _append_coordinate_frame(scene, current[:3], _quaternion_wxyz_to_rotation(current[3:]), .115, .95, .0035)


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def _preview(ffmpeg: str, video: Path, path: Path, time: float) -> None:
    _run_ffmpeg([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{time:.8f}",
                 "-i", str(video), "-frames:v", "1", str(path)])


def render_saved_scene(metrics: dict, bundle: dict, output: Path, *, fps: float = 30.,
                       width: int = 640, height: int = 480,
                       time_limit_s: float | None = None) -> tuple[list[Path], dict]:
    """Produce five full views, their grid, and a dedicated continuum view.

    A time limit is solely for a explicitly labelled rendering smoke test;
    production callers leave it unset. No control or collision checks run.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    times = bundle["trajectory"]["time"]
    requested, indices = frame_schedule(times, fps)
    if time_limit_s is not None:
        if not np.isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError("smoke time limit must be positive and finite")
        keep = requested <= time_limit_s + 1e-12
        requested, indices = requested[keep], indices[keep]
    duration = float(requested[-1])
    output.mkdir(parents=True, exist_ok=True)
    sid = bundle["scenario_id"]
    paths = [output / f"{sid}_{view}.mp4" for view in VIEWS]
    grid, preview = output / f"{sid}_five_view_grid.mp4", output / f"{sid}_five_view_preview.png"
    focus_path, focus_preview = output / f"{sid}_continuum_focus.mp4", output / f"{sid}_continuum_focus_preview.png"
    expected_paths = [*paths, grid, preview, focus_path, focus_preview]
    if any(path.exists() for path in expected_paths):
        raise FileExistsError("saved-state rendering output already exists; use an exclusive directory")
    spec, verifier, data = make_saved_model(metrics, bundle["scene"])
    model = verifier.model
    base_slice, _ = free_joint_slices(model, spec.base_joint_name)
    lookat, distance = _framing(model, data, bundle["trajectory"]["qpos"][::10])
    # The information banner occupies the top 151 pixels. Keep the original
    # robot/path framing inside the remaining visible canvas in all views.
    distance *= 1.30
    cameras = _make_cameras(lookat, distance)
    renderer = None
    writers = {}
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        for view, path in zip(VIEWS, paths, strict=True):
            writers[view] = _RawFfmpegWriter(ffmpeg, path, width, height, fps)
        for state in indices:
            state = int(state)
            data.qpos[:], data.qvel[:], data.time = bundle["trajectory"]["qpos"][state], bundle["trajectory"]["qvel"][state], float(times[state])
            data.ctrl[:] = 0.
            mujoco.mj_forward(model, data)
            for view in VIEWS:
                renderer.update_scene(data, camera=cameras[view])
                _markers(renderer.scene, bundle, state, base_slice, focus=False, labels=view == "front")
                writers[view].append(_annotations(renderer.render(), bundle, state, view))
    finally:
        if renderer is not None:
            renderer.close()
        errors = []
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for path in paths:
        command.extend(["-i", str(path)])
    command.extend(["-f", "lavfi", "-i", f"color=c=0x0a0e18:s={width}x{height}:r={fps}:d={len(indices)/fps}",
                    "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[out]",
                    "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-frames:v", str(len(indices)), str(grid)])
    _run_ffmpeg(command)
    _preview(ffmpeg, grid, preview, duration / 2)
    focus_width, focus_height = 960, 720
    continuum = verifier._descendant_body_ids(verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP)
    rigid = verifier._descendant_body_ids(verifier._RIGID_ROOT, verifier._RIGID_TIP)
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision")
    if target_geom < 0:
        raise ValueError("saved target geometry missing")
    camera, camera_metadata = _continuum_focus_camera_from_replay(
        model, data, bundle["trajectory"]["qpos"][::10], continuum, target_geom,
        np.asarray(bundle["scene"]["scenario"]["continuum_target"]["waypoint_points_m"]))
    camera.distance *= 1.20
    camera_metadata["distance_m"] = float(camera.distance)
    dimmed = _dim_rigid_arm_geometries(model, rigid)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, focus_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, focus_height)
    renderer, writer = None, None
    try:
        renderer = mujoco.Renderer(model, height=focus_height, width=focus_width)
        writer = _RawFfmpegWriter(ffmpeg, focus_path, focus_width, focus_height, fps)
        for state in indices:
            state = int(state)
            data.qpos[:], data.qvel[:], data.time = bundle["trajectory"]["qpos"][state], bundle["trajectory"]["qvel"][state], float(times[state])
            data.ctrl[:] = 0.
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            _markers(renderer.scene, bundle, state, base_slice, focus=True)
            writer.append(_annotations(renderer.render(), bundle, state, "continuum focus"))
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.close()
    _preview(ffmpeg, focus_path, focus_preview, duration / 2)
    metadata = {
        "scenario_id": sid, "views": list(VIEWS), "fps": fps, "frame_count": len(indices),
        "source_trace": bundle["source_refs"][0]["path"],
        "source_trace_sha256": bundle["source_refs"][0]["sha256"],
        "source_saved_states": bundle["source_refs"][2],
        "source_fresh_kinematics": bundle["source_refs"][3],
        "duration_s": duration, "source_duration_s": float(times[-1]), "encoded_duration_s": len(indices)/fps,
        "width": width, "height": height, "scope": SCOPE, "physics_steps_executed": 0,
        "state_feed": "saved qpos/qvel/time + mj_forward; no integration or interpolation",
        "frame_state_selection": "first saved 2ms state at/after requested 30Hz time; exact endpoints",
        "presentation_time_first_s": float(requested[0]), "presentation_time_last_s": float(requested[-1]),
        "saved_state_first_index": int(indices[0]), "saved_state_last_index": int(indices[-1]),
        "saved_state_count": len(times), "smoke_truncated": time_limit_s is not None,
        "camera_lookat_m": lookat.tolist(), "camera_distance_m": float(distance),
        "files": [relative(path) for path in expected_paths],
        "media": [{"path": relative(path), "kind": "video", "view": view,
                    "width": width, "height": height, "fps": fps, "frame_count": len(indices)}
                   for path, view in zip(paths, VIEWS, strict=True)] + [
            {"path": relative(grid), "kind": "video", "view": "five_view_grid", "width": 3*width, "height": 2*height, "fps": fps, "frame_count": len(indices)},
            {"path": relative(focus_path), "kind": "video", "view": "continuum_focus", "width": focus_width, "height": focus_height, "fps": fps, "frame_count": len(indices)},
            {"path": relative(preview), "kind": "preview", "width": 3*width, "height": 2*height},
            {"path": relative(focus_preview), "kind": "preview", "width": focus_width, "height": focus_height}],
        "focus": {"camera": camera_metadata, "rigid_alpha": CONTINUUM_FOCUS_RIGID_ALPHA, "dimmed_geometries": dimmed},
        "coordinate_frames": {"axis_color_order": ["x_red", "y_green", "z_blue"],
                              "waypoint_labels_rendered": [f"W{i}" for i in range(1, 8)],
                              "waypoint_label_views": ["front", "continuum_focus"],
                              "continuum_end_effector_local_offset_body_m": CONTINUUM_EE_OFFSET_M.tolist(),
                              "frames_rendered": ["rigid_grasp_target", "rigid_end_effector", "continuum_irregular_target", "continuum_end_effector", "free_base_initial_pose", "free_base_current_pose", "continuum_waypoints_W1_to_W7"]},
        "overlay_scope": {"tracking": "fresh current-state kinematics diagnostic; original 25-check acceptance unchanged",
                          "screened_distance": "last saved task query; screened pair scope, not dense native minimum",
                          "native_robot_target": "75 original pairs at displayed saved 2ms state",
                          "dense_whole_body": "global run minimum over 5401 saved configurations and 2927 original pairs"},
        "source_refs": bundle["source_refs"],
    }
    return expected_paths, metadata
