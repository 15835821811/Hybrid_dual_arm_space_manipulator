"""Render the fourteen fixed B.2 saved replays without advancing physics.

The source pilot is read-only.  Presentation selects saved states, including
each exact final state; failed attempts end at their saved prefix boundary.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier, WholeBodyVerificationConfig
from v6_4.task_protocol import TaskSpec
from v6_4.reference_adapter import scenario_from_task
from v6_4.task_anchored_reference import TaskAnchoredResidualPlan, TaskAnchoredResidualReferenceProvider
from v6_lite.hierarchical_qp import free_joint_slices
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.runtime_command import model_id
from v6_lite.visualization.latest_saved_renderer import (
    VIEWS, _append_connector, _append_coordinate_frame, _append_label,
    _append_sphere, _continuum_focus_camera_from_replay, _dim_rigid_arm_geometries,
    _font, _framing, _make_cameras, _preview, _quaternion_wxyz_to_rotation,
    _run_ffmpeg,
)

ROOT = Path(__file__).resolve().parents[2]
SLOTS = tuple(f"TEST_{index:02d}_{method}" for index in range(4)
              for method in ("E0", "E1", "E2")) + ("teacher_00", "teacher_17")
SCHEMA = "v64_b2_saved_state_replay_v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def path_record(path):
    path = Path(path).resolve()
    try:
        portable = path.relative_to(ROOT).as_posix()
    except ValueError:
        portable = path.as_posix()
    return {"path": portable, "sha256": sha(path), "bytes": path.stat().st_size}


def schedule(times, fps, smoke_seconds=None):
    """A CFR presentation grid plus the exact saved endpoint, never past it.

    A non-grid endpoint occupies the final frame.  The encoded duration is
    at most two presentation frame periods longer than the physical prefix;
    no simulated state is extrapolated or interpolated.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0.
            or np.any(np.diff(times) <= 0.) or not np.all(np.isfinite(times))
            or not np.isfinite(fps) or fps <= 0.):
        raise ValueError("finite saved clock and positive FPS required")
    end = float(times[-1])
    if smoke_seconds is not None:
        end = min(end, float(smoke_seconds))
    requested = np.arange(int(np.floor(end * fps + 1e-9)) + 1) / fps
    if abs(requested[-1] - end) < 1e-8:
        requested[-1] = end
    else:
        requested = np.r_[requested, end]
    indices = np.searchsorted(times, requested, side="left")
    indices = np.minimum(indices, len(times) - 1)
    indices[0] = 0
    if smoke_seconds is None:
        indices[-1] = len(times) - 1
    if np.any(times[indices] > float(times[-1]) + 1e-12):
        raise ValueError("presentation crossed saved prefix")
    return requested, indices


class Writer:
    """Keep concurrent six-view H.264 encoding within a bounded CPU budget."""
    def __init__(self, ffmpeg, path, width, height, fps):
        self.path = path
        self.process = subprocess.Popen([
            ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def append(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        error = self.process.stderr.read()
        if self.process.wait() != 0:
            raise RuntimeError(f"FFmpeg {self.path}: {error.decode(errors='replace')}")


def load_slot(source, slot):
    folder = source / "attempts" / slot
    result_path = folder / "attempt_result.json"
    task_path, plan_path = folder / "task.json", folder / "plan.json"
    replay_path = folder / "actual/evaluation/fresh_replay.npz"
    result = read(result_path)
    task = TaskSpec.from_dict(read(task_path))
    plan = TaskAnchoredResidualPlan.from_dict(read(plan_path))
    evaluation_path = folder / "actual/evaluation/report.json"
    evaluation = read(evaluation_path)
    traces = list((folder / "actual").rglob(Path(result["trace_path"]).name))
    trace_path = traces[0] if len(traces) == 1 else None
    if trace_path is None:
        raise ValueError(f"missing actual trace for {slot}")
    if (result["slot_id"] != slot or result["task_sha256"] != task.sha256()
            or result["trace_sha256"] != sha(trace_path)
            or result["evaluation_sha256"] != sha(evaluation_path)
            or evaluation["fresh_replay_sha256"] != sha(replay_path)
            or result["plan_file_sha256"] != sha(plan_path)):
        raise ValueError(f"frozen evidence binding failed: {slot}")
    with np.load(replay_path, allow_pickle=False) as archive:
        replay = {key: archive[key] for key in archive.files}
    n = int(result["actual_steps"])
    if (replay["qpos"].shape != (n + 1, 81) or replay["qvel"].shape != (n + 1, 79)
            or not np.array_equal(replay["qpos"][0], task.initial_qpos)
            or not np.array_equal(replay["qvel"][0], task.initial_qvel)
            or not np.allclose(replay["time"], np.arange(n + 1) * .002, rtol=0., atol=1e-9)
            or any(not np.all(np.isfinite(v)) for v in replay.values()
                   if np.issubdtype(v.dtype, np.number))):
        raise ValueError(f"saved state shape/clock/finite/initial parity failed: {slot}")
    sources = {name: path_record(path) for name, path in (
        ("attempt_result", result_path), ("task", task_path), ("plan", plan_path),
        ("trace", trace_path), ("fresh_replay", replay_path), ("evaluation", evaluation_path))}
    return task, plan, result, evaluation, replay, sources


def make_model(task, evaluation, source, slot):
    spec = default_v6_lite_robot_spec()
    native = evaluation["native_geometry"]
    if (spec.runtime_contract_sha256() != task.model_contract_sha256
            or spec.runtime_contract_sha256() != native["model_contract_sha256"]
            or spec.source_bundle_sha256() != native["model_source_bundle_sha256"]):
        raise ValueError("render model differs from frozen native geometry model")
    scenario = scenario_from_task(task)
    verifier = WholeBodyCollisionVerifier(spec, scenario.obstacles, WholeBodyVerificationConfig(
        minimum_clearance=.005, query_distance_max=2.5, adaptive_subdivisions=4,
        self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
    if verifier._pair_policy_sha256 != native["whole_body"]["pair_policy_sha256"]:
        raise ValueError("render model pair policy differs from frozen independent evaluation")
    model = verifier.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    compiled = model_id(model, spec.runtime_contract_sha256())
    historic = source / "attempts" / slot / "actual/historical_metric_observations.json"
    if historic.exists() and read(historic)["execution_contract"]["source_compiled_model_sha256"] != compiled:
        raise ValueError("compiled render model differs from frozen complete actual")
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("v5_workspace_sphere_"):
            model.geom_rgba[index] = [1., .32, .05, .78]
    return spec, verifier, mujoco.MjData(model), compiled


def polyline(scene, points, color, radius=.0016):
    for a, b in zip(points[:-1], points[1:]):
        if np.linalg.norm(b - a) > 1e-10:
            _append_connector(scene, a, b, radius, np.asarray(color))


def markers(scene, task, replay, state, ref, base, *, focus, labels):
    current = int(state)
    stride = max(1, len(replay["time"]) // 95)
    path_indices = np.unique(np.r_[np.arange(0, current + 1, stride), current])
    polyline(scene, base[::stride], [.95, .30, .22, .34], .0013)
    polyline(scene, ref[::stride], [.28, .95, .28, .58], .0017)
    polyline(scene, replay["continuum_position"][path_indices], [.05, .90, 1., .95], .0022)
    source = task.scenario
    rotation = np.asarray(source["continuum_target_rotation_world"])
    points = np.asarray(source["continuum_target"]["waypoint_points_m"])
    for index, point in enumerate(points):
        _append_sphere(scene, point, .0058, np.array([1., .16, .16, .8]))
        _append_coordinate_frame(scene, point, rotation, .035, .34, .0013)
        if labels:
            _append_label(scene, point + [0., -.03, .035], f"W{index + 1}", np.array([1., .86, .86, 1.]))
    rigid_target = (replay["target_position"][current]
                    + replay["target_rotation"][current] @ np.asarray(source["grasp_point_target_frame_m"]))
    rigid_rotation = replay["target_rotation"][current] @ np.asarray(source["grasp_rotation_target_frame"])
    branches = [(ref[current], rotation, replay["continuum_position"][current],
                 replay["continuum_rotation"][current], [.20, 1., .32, 1.], [.05, .82, .95, 1.])]
    if not focus:
        branches.append((rigid_target, rigid_rotation, replay["rigid_position"][current],
                         replay["rigid_rotation"][current], [1., .82, .05, 1.], [.15, .48, 1., 1.]))
    for target, tr, tip, ar, tc, ac in branches:
        _append_sphere(scene, target, .012, np.asarray(tc))
        _append_sphere(scene, tip, .009, np.asarray(ac))
        _append_connector(scene, target, tip, .002, np.asarray(tc))
        _append_coordinate_frame(scene, target, tr, .10, .55, .004)
        _append_coordinate_frame(scene, tip, ar, .07, .95, .003)
    if not focus:
        for index, alpha in ((0, .33), (current, .90)):
            pose = replay["base_pose"][index]
            _append_coordinate_frame(scene, pose[:3], _quaternion_wxyz_to_rotation(pose[3:]), .12, alpha, .003)


def annotate(frame, task, result, replay, state, view, ref, base, *, smoke):
    # Reserve a separate information strip; overlays must not obscure the
    # base or articulated geometry at any replay time or camera angle.
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 102), (10, 14, 24))
    image.paste(Image.fromarray(frame), (0, 102))
    draw = ImageDraw.Draw(image)
    width = image.width
    draw.rectangle((0, 0, width, 101), fill=(10, 14, 24))
    font = _font(12 if width < 640 else 15)
    time = float(replay["time"][state])
    end = float(replay["time"][-1])
    failed = not result["full_task_success"]
    status = "REFUSED PREFIX" if failed else "FULL TASK PASS"
    method = result["slot_id"].split("_")[-1] if result["slot_id"].startswith("TEST") else "fixed teacher"
    error = 1000 * np.linalg.norm(replay["continuum_position"][state] - ref[state])
    offset = 1000 * np.linalg.norm(ref[state] - base[state])
    lines = [f"{result['slot_id']} | {method} | {view.upper()}",
             f"Task: {task.task_id}",
             f"{status} | saved t={time:6.3f}s / {end:.3f}s" + (" | SMOKE" if smoke else ""),
             f"continuum ref error={error:.3f}mm  residual={offset:.3f}mm",
             "green=residual ref  red=base  cyan=actual; RGB=XYZ",
             "saved states + mj_forward; 0 physics; NOT_MET"]
    for i, line in enumerate(lines):
        color = (255, 184, 110) if i == 2 and failed else (225, 234, 245)
        draw.text((7, 3 + i * 16), line, font=font, fill=color)
    return np.asarray(image)


def probe(path, expected_frames, width, height, fps):
    command = [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,duration", "-of", "json", str(path)]
    stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    if (int(stream["nb_frames"]) != expected_frames or int(stream["width"]) != width
            or int(stream["height"]) != height or abs(float(stream["duration"]) - expected_frames / fps) > .002):
        raise ValueError(f"encoded media differs from saved schedule: {path}")
    return stream


def render_slot(source, output, slot, fps=15., width=480, height=360, smoke_seconds=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    folder = output / "replays" / slot
    if folder.exists():
        raise FileExistsError(f"render directory exists: {folder}")
    task, plan, result, evaluation, replay, sources = load_slot(source, slot)
    requested, indices = schedule(replay["time"], fps, smoke_seconds)
    provider = TaskAnchoredResidualReferenceProvider(task, plan)
    reference, _, _ = provider.continuum_kinematics(replay["time"])
    offsets, _, _ = plan.offset_kinematics(replay["time"])
    base = reference - offsets
    spec, verifier, data, compiled = make_model(task, evaluation, source, slot)
    model = verifier.model
    lookat, distance = _framing(model, data, replay["qpos"][::10])
    distance *= 1.22
    cameras = _make_cameras(lookat, distance)
    continuum = verifier._descendant_body_ids(verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP)
    rigid = verifier._descendant_body_ids(verifier._RIGID_ROOT, verifier._RIGID_TIP)
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision")
    focus, focus_info = _continuum_focus_camera_from_replay(
        model, data, replay["qpos"][::10], continuum, target_geom,
        np.asarray(task.scenario["continuum_target"]["waypoint_points_m"]))
    focus.distance *= 1.17
    cameras["continuum_focus"] = focus
    normal_rgba = model.geom_rgba.copy()
    _dim_rigid_arm_geometries(model, rigid)
    focus_rgba = model.geom_rgba.copy()
    model.geom_rgba[:] = normal_rgba
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    folder.mkdir(parents=True)
    ffmpeg = shutil.which("ffmpeg")
    views = (*VIEWS, "continuum_focus")
    paths = {view: folder / f"{slot}_{view}.mp4" for view in views}
    writers, renderer = {}, None
    last_frames = {}
    try:
        renderer = mujoco.Renderer(model, height=height - 102, width=width)
        for view, path in paths.items():
            writers[view] = Writer(ffmpeg, path, width, height, fps)
        for number, state in enumerate(indices):
            state = int(state)
            data.qpos[:], data.qvel[:], data.time = replay["qpos"][state], replay["qvel"][state], float(replay["time"][state])
            data.ctrl[:] = 0.
            mujoco.mj_forward(model, data)
            for view in views:
                is_focus = view == "continuum_focus"
                model.geom_rgba[:] = focus_rgba if is_focus else normal_rgba
                renderer.update_scene(data, camera=cameras[view])
                markers(renderer.scene, task, replay, state, reference, base,
                        focus=is_focus, labels=view in ("front", "continuum_focus"))
                frame = annotate(renderer.render(), task, result, replay, state, view, reference, base,
                                 smoke=smoke_seconds is not None)
                writers[view].append(frame)
                if number == len(indices) - 1:
                    last_frames[view] = frame
    finally:
        if renderer is not None:
            renderer.close()
        errors = []
        for writer in writers.values():
            try:
                writer.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
    grid = folder / f"{slot}_five_view_grid.mp4"
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for view in VIEWS:
        command += ["-i", str(paths[view])]
    command += ["-f", "lavfi", "-i", f"color=c=0x0a0e18:s={width}x{height}:r={fps}:d={len(indices)/fps}",
                "-filter_complex_threads", "1", "-filter_complex",
                "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[out]",
                "-map", "[out]", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-frames:v", str(len(indices)), str(grid)]
    _run_ffmpeg(command)
    paths["five_view_grid"] = grid
    preview = folder / f"{slot}_five_view_preview.png"
    focus_preview = folder / f"{slot}_continuum_focus_preview.png"
    _preview(ffmpeg, grid, preview, len(indices) / fps / 2)
    _preview(ffmpeg, paths["continuum_focus"], focus_preview, len(indices) / fps / 2)
    last = folder / f"{slot}_last_saved_state.png"
    Image.fromarray(last_frames["continuum_focus"]).save(last)
    records = {}
    for view, path in paths.items():
        dims = (width * 3, height * 2) if view == "five_view_grid" else (width, height)
        records[view] = {**path_record(path), "relative_path": path.relative_to(output).as_posix(),
                         "probe": probe(path, len(indices), *dims, fps)}
    gifs = []
    if slot in ("teacher_00", "TEST_00_E2") and smoke_seconds is None:
        gif = folder / f"{slot}_short_preview.gif"
        _run_ffmpeg([ffmpeg, "-y", "-loglevel", "error", "-ss", "3", "-t", "6", "-i", str(paths["continuum_focus"]),
                     "-filter_complex_threads", "1", "-vf", "fps=6,scale=400:-1:flags=lanczos", "-loop", "0", str(gif)])
        gifs.append({**path_record(gif), "relative_path": gif.relative_to(output).as_posix(),
                     "source_start_s": 3., "source_end_s": 9., "scope": "six-second excerpt at original speed"})
    for expected in sources.values():
        source_path = Path(expected["path"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if sha(source_path) != expected["sha256"]:
            raise ValueError("frozen source changed while rendering")
    metadata = {"schema": SCHEMA, "slot_id": slot, "task_id": task.task_id,
        "task_sha256": task.sha256(), "method": slot.split("_")[-1] if slot.startswith("TEST") else "TEACHER",
        "status": result["status"], "full_task_success": result["full_task_success"],
        "execution_failure": result.get("execution_failure"), "source_start_s": 0.,
        "source_end_s": float(replay["time"][-1]), "frame_count": len(indices), "fps": fps,
        "encoded_duration_s": len(indices) / fps, "source_state_count": len(replay["time"]),
        "selected_state_indices": indices.tolist(), "presentation_times_s": requested.tolist(),
        "selected_saved_times_s": replay["time"][indices].tolist(),
        "last_selected_state_index": int(indices[-1]), "last_selected_time_s": float(replay["time"][indices[-1]]),
        "exact_source_endpoint_included": int(indices[-1]) == len(replay["time"]) - 1,
        "no_state_after_saved_endpoint": True, "smoke_truncated": smoke_seconds is not None,
        "state_selection": "first saved 2ms state at/after CFR presentation time; exact final endpoint; no interpolation",
        "width": width, "height": height, "videos": {view: item["relative_path"] for view, item in records.items()},
        "video_records": records, "preview": preview.relative_to(output).as_posix(),
        "preview_record": path_record(preview), "focus_preview": focus_preview.relative_to(output).as_posix(),
        "focus_preview_record": path_record(focus_preview), "last_saved_state": last.relative_to(output).as_posix(),
        "last_saved_state_record": path_record(last), "gifs": gifs, "sourcepaths": sources,
        "source_compiled_model_sha256": compiled, "model_contract_sha256": spec.runtime_contract_sha256(),
        "model_source_bundle_sha256": spec.source_bundle_sha256(),
        "focus_camera": focus_info, "physics_steps_executed": 0, "optimizer_updates": 0, "new_samples": 0,
        "reference_scope": "declared physical-time Cartesian residual; rigid target transported using saved fresh target pose",
        "path_scope": "reference paths stop at saved endpoint; actual trail stops at displayed saved state",
        "coordinate_frame_axis_order": ["X_red", "Y_green", "Z_blue"],
        "deployment": "NOT_MET", "visualization_is_new_acceptance": False}
    write(folder / "replay_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--slot", choices=SLOTS, help="smoke only: render one fixed slot")
    parser.add_argument("--smoke-seconds", type=float)
    args = parser.parse_args()
    if args.width < 480 or args.height < 360 or args.width % 2 or args.height % 2:
        parser.error("even dimensions of at least 480x360 are required")
    if (args.slot is None) != (args.smoke_seconds is None):
        parser.error("--slot and --smoke-seconds must be used together for a labelled smoke run")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required")
    slots = (args.slot,) if args.slot else SLOTS
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "videos.json").exists():
        raise FileExistsError("existing videos.json cannot be overwritten")
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(render_slot, args.source, args.output, slot, args.fps, args.width,
                               args.height, args.smoke_seconds): slot for slot in slots}
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            print(json.dumps({"slot": record["slot_id"], "frames": record["frame_count"],
                              "source_end_s": record["source_end_s"], "videos": len(record["videos"])}), flush=True)
    ordered = sorted(records, key=lambda record: slots.index(record["slot_id"]))
    write(args.output / "videos.json", {"schema": "v64_b2_replay_collection_v1",
          "created_utc": datetime.now(timezone.utc).isoformat(), "generator": path_record(__file__),
          "source_manifest": path_record(args.source / "manifest.json"),
          "slot_order": list(slots), "records": ordered, "record_count": len(ordered),
          "video_count": sum(len(record["videos"]) for record in ordered),
          "physics_steps_executed": 0, "optimizer_updates": 0, "new_samples": 0,
          "historical_outputs_overwritten": False, "smoke_truncated": args.smoke_seconds is not None})


if __name__ == "__main__":
    main()
