"""Render B.3.1 actual saved states; never advance physics or query clearance.

Five 640x480 cameras, their 1920x960 grid and a 960x720 continuum focus
produce seven videos per frozen EA slot. Existing fresh kinematics are used
only after exact full-state parity with the original actual trace is checked.
The sealed experiment, prior media and publication modules remain read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from v6_4.task_protocol import TaskSpec
from v6_4.task_anchored_reference import TaskAnchoredResidualPlan, TaskAnchoredResidualReferenceProvider
from v6_4.visualization import render_residual_replays as legacy

ROOT = Path(__file__).resolve().parents[2]
SLOTS = tuple(f"EA_{index:02d}" for index in range(28))
SCHEMA = "v64_b31_saved_actual_state_media_v1"
COLLECTION_SCHEMA = "v64_b31_saved_actual_media_collection_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


class Sources:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.manifest_path = self.source/"manifest.json"
        self.manifest = read(self.manifest_path)
        require(self.manifest.get("status") == "FINAL", "render only a sealed terminal source")
        verification = read(self.source/"verification.json")
        require(verification.get("status") == "PASS" and verification.get("manifest_sha256") == sha(self.manifest_path),
                "source verification does not bind its final manifest")
        self.paths = {"source_manifest": record(self.manifest_path), "source_verification": record(self.source/"verification.json")}

    def bind(self, name, path, expected=None):
        path = Path(path).resolve()
        if path.is_relative_to(self.source):
            row = self.manifest["payload"].get(path.relative_to(self.source).as_posix())
        else:
            row = self.manifest["external"].get(str(path))
        require(row is not None, "render input is not part of the sealed source: "+str(path))
        item = record(path)
        require(item["sha256"] == row["sha256"] and item["bytes"] == row["bytes"], "sealed source changed: "+str(path))
        require(expected is None or item["sha256"] == expected, "independent source digest differs: "+str(path))
        self.paths[name] = item
        return path

    def unchanged(self):
        for item in self.paths.values():
            require(record(item["path"]) == item, "saved rendering input changed")


class RenderOnly(AbstractContextManager):
    """Forbid integrators, native distance queries and any accidental QP solve."""
    def __init__(self):
        self.forward_calls = 0
        self.forbidden_attempts = []
        self.restore = []

    def __enter__(self):
        def patch(owner, name, replacement):
            old = getattr(owner, name)
            self.restore.append((owner, name, old))
            setattr(owner, name, replacement)
            return old
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_geomDistance"):
            def reject(*args, _name=name, **kwargs):
                self.forbidden_attempts.append(_name)
                raise RuntimeError("saved renderer forbids "+_name)
            patch(mujoco, name, reject)
        old_forward = mujoco.mj_forward
        def forward(*args, **kwargs):
            self.forward_calls += 1
            return old_forward(*args, **kwargs)
        patch(mujoco, "mj_forward", forward)
        osqp = sys.modules.get("osqp")
        if osqp is not None and hasattr(osqp, "OSQP"):
            def reject_qp(*args, **kwargs):
                self.forbidden_attempts.append("OSQP.solve")
                raise RuntimeError("saved renderer forbids QP solve")
            patch(osqp.OSQP, "solve", reject_qp)
        return self

    def __exit__(self, *args):
        for owner, name, old in reversed(self.restore):
            setattr(owner, name, old)
        return False


def load_slot(source, slot_id):
    """Read actual states and verify independently saved replay state parity."""
    sources = Sources(source)
    plan_path = sources.bind("frozen_study_plan", sources.source/"plan.json")
    study = read(plan_path)
    terminal = read(sources.bind("execution_complete", sources.source/"execution_complete.json"))
    require(terminal["terminal_slots"] == 28 and tuple(s["slot_id"] for s in study["slots"]) == SLOTS,
            "fixed 28-slot terminal protocol required")
    frozen = next(s for s in study["slots"] if s["slot_id"] == slot_id)
    fixed = read(sources.bind("slot_result", sources.source/"slots"/slot_id/"slot_result.json"))
    require(fixed["frozen_slot"] == frozen and fixed["slot_id"] == slot_id, "fixed slot differs")
    result, quality = fixed["source_result"], fixed["quality"]
    require(result.get("actual_steps") == 13500 and result.get("full_task_success") is True,
            "this complete-media collection requires the saved full 27s actual")
    evaluation_path = sources.bind("evaluation", result["evaluation_path"], result["evaluation_sha256"])
    attempt = evaluation_path.parent.parent.parent
    original = read(sources.bind("attempt_result", attempt/"attempt_result.json"))
    require(original == result, "embedded actual result differs from original")
    expected_sources = quality["sources"]
    task_path, reference_path = attempt/"task.json", attempt/"plan.json"
    sources.bind("task", task_path, expected_sources[str(task_path)])
    sources.bind("plan", reference_path, result["plan_file_sha256"])
    task = TaskSpec.from_dict(read(task_path))
    plan = TaskAnchoredResidualPlan.from_dict(read(reference_path))
    require(task.sha256() == frozen["task_sha256"] == result["task_sha256"]
            and plan.sha256() == frozen["plan_sha256"] == result["plan_content_sha256"], "Task/reference content differs")
    trace_path = sources.bind("actual_trace", result["trace_path"], result["trace_sha256"])
    evaluation = read(evaluation_path)
    fresh_path, evaluation_manifest_path = evaluation_path.parent/"fresh_replay.npz", evaluation_path.parent/"manifest.json"
    evaluation_manifest = read(sources.bind("evaluation_manifest", evaluation_manifest_path, expected_sources[str(evaluation_manifest_path)]))
    fresh_sha = evaluation["fresh_replay_sha256"]
    require(evaluation_manifest["fresh_replay.npz"] == fresh_sha == expected_sources[str(fresh_path)], "fresh independent digest differs")
    sources.bind("fresh_replay", fresh_path, fresh_sha)
    sources.bind("historical_model_observation", attempt/"actual/historical_metric_observations.json")
    with np.load(trace_path, allow_pickle=False) as archive:
        actual = {key: archive[key].copy() for key in ("initial_qpos", "initial_qvel", "actual_full_qpos",
                  "actual_full_qvel", "time", "generated_continuum_position", "generated_rigid_position")}
    needed = ("time", "qpos", "qvel", "base_pose", "target_position", "target_rotation", "rigid_position",
              "rigid_rotation", "continuum_position", "continuum_rotation")
    with np.load(fresh_path, allow_pickle=False) as archive:
        replay = {key: archive[key].copy() for key in needed}
    qpos = np.vstack((actual["initial_qpos"], actual["actual_full_qpos"]))
    qvel = np.vstack((actual["initial_qvel"], actual["actual_full_qvel"]))
    times = np.r_[0., actual["time"]]
    require(qpos.shape == (13501, 81) and qvel.shape == (13501, 79) and times.shape == (13501,), "actual state shape differs")
    require(np.array_equal(qpos[0], task.initial_qpos) and np.array_equal(qvel[0], task.initial_qvel), "actual initial state differs from Task")
    require(all(np.isfinite(value).all() for value in (*actual.values(), *replay.values())), "nonfinite saved input")
    qpos_exact, qvel_exact = np.array_equal(qpos, replay["qpos"]), np.array_equal(qvel, replay["qvel"])
    time_delta = float(np.max(np.abs(times-replay["time"])))
    require(qpos_exact and qvel_exact and time_delta <= 1e-9 and np.all(np.diff(times) > 0.)
            and np.allclose(times, np.arange(13501)*.002, rtol=0., atol=1e-9), "actual/fresh full-state or saved-clock parity failed")
    parity = {"state_count": 13501, "qpos_bit_exact": bool(qpos_exact), "qvel_bit_exact": bool(qvel_exact),
              "qpos_max_abs_difference": float(np.max(np.abs(qpos-replay["qpos"]))),
              "qvel_max_abs_difference": float(np.max(np.abs(qvel-replay["qvel"]))),
              "time_max_abs_difference_s": time_delta, "time_tolerance_s": 1e-9,
              "initial_state_matches_Task": True, "fresh_replay_is_actual_state_source": False}
    # Geometry uses actual, not replay, state arrays. Kinematics retain their
    # independently saved provenance after exact state parity established above.
    replay["qpos"], replay["qvel"], replay["time"] = qpos, qvel, times
    provider = TaskAnchoredResidualReferenceProvider(task, plan)
    reference, _, _ = provider.continuum_kinematics(times)
    offset, _, _ = plan.offset_kinematics(times)
    base = reference-offset
    rigid_reference = replay["target_position"] + np.einsum("nij,j->ni", replay["target_rotation"],
                                                       np.asarray(task.scenario["grasp_point_target_frame_m"]))
    c_error = float(np.max(np.abs(reference[1:]-actual["generated_continuum_position"])))
    # Runtime's post-step provider sample sees the cached pre-step rigid target
    # pose. Do not confuse this logged reference with a fresh current Task pose
    # or with the held QP input at the separate 20ms task clock.
    r_error = float(np.max(np.abs(rigid_reference[:-1]-actual["generated_rigid_position"])))
    require(c_error <= 1e-12 and r_error <= 1e-12, "post-step generated reference/cached target phase differs from saved records")
    binding = {"binding_scope": "post-step generated references only; not held/consumed QP task inputs",
               "continuum_poststep_generated_reference_max_abs_m": c_error,
               "rigid_poststep_generated_cached_prestep_reference_max_abs_m": r_error,
               "rigid_display_target_scope": "fresh current independent Task grasp pose; not generated rigid reference",
               "poststep_saved_reference_samples": 13500, "reference_tolerance_m": 1e-12}
    return sources, frozen, task, plan, result, quality, evaluation, replay, reference, base, parity, binding


def annotate(frame, frozen, task, plan, result, replay, state, view, reference, base, *, smoke, strip):
    image = Image.new("RGB", (frame.shape[1], frame.shape[0]+strip), (10, 14, 24))
    image.paste(Image.fromarray(frame), (0, strip))
    draw = ImageDraw.Draw(image)
    font_size = 12 if image.width < 900 else 16
    actual, target = replay["continuum_position"][state], reference[state]
    rigid_target = replay["target_position"][state] + replay["target_rotation"][state] @ np.asarray(task.scenario["grasp_point_target_frame_m"])
    c_error = 1000.*float(np.linalg.norm(actual-target))
    base_error = 1000.*float(np.linalg.norm(actual-base[state]))
    r_error = 1000.*float(np.linalg.norm(replay["rigid_position"][state]-rigid_target))
    status = "ACTUAL TASK/SAFETY PASS" if result["full_task_success"] else "ACTUAL FAILED PREFIX"
    source_slot = result["slot_id"]
    lines = [f"{frozen['slot_id']} | {frozen['mode']} | source {source_slot} | {view.upper()}",
             f"Task {task.task_id} | {plan.representation_version.rsplit('_', 1)[-1]} | z={frozen['amplitude_m']*1000:+g}mm",
             f"{status} | saved t={replay['time'][state]:6.3f}/{replay['time'][-1]:.3f}s" + (" | SMOKE" if smoke else ""),
             f"Generated C error={c_error:.2f}mm | independent Task C/R={base_error:.2f}/{r_error:.2f}mm",
             "green=generated C ref; red=Task C; cyan=actual C; yellow/blue=Task/actual R",
             "ACTUAL saved states + forward | 0 physics | deployment NOT_MET / NOT_ESTABLISHED"]
    spacing = (strip-6)//len(lines)
    for index, line in enumerate(lines):
        font = legacy._font(font_size)
        fit_size = font_size
        while draw.textlength(line, font=font) > image.width-14 and fit_size > 9:
            fit_size -= 1
            font = legacy._font(fit_size)
        draw.text((7, 3+index*spacing), line, font=font, fill=(225, 234, 245))
    return np.asarray(image)


def markers(scene, task, replay, state, reference, base, *, focus, labels):
    """Retain B.2 marker geometry; separate coincident W1/W7 labels locally."""
    legacy.markers(scene, task, replay, state, reference, base, focus=focus, labels=False)
    if labels:
        points = np.asarray(task.scenario["continuum_target"]["waypoint_points_m"])
        for index, point in enumerate(points):
            offset = np.array([0., -.03, .035])
            if index == 0:
                offset[2] = .07
            elif index == 6:
                offset[2] = -.04
            legacy._append_label(scene, point+offset, f"W{index+1}", np.array([1., .86, .86, 1.]))


def render_slot(source, output, slot_id, fps=15., width=640, height=480, focus_width=960, focus_height=720, smoke_seconds=None):
    started = time.perf_counter()
    producer_bindings = generator_records()
    output = Path(output).resolve()
    folder = output/"replays"/slot_id
    require(not folder.exists(), "exclusive slot render directory already exists")
    loaded = load_slot(source, slot_id)
    sources, frozen, task, plan, result, quality, evaluation, replay, reference, base, parity, ref_binding = loaded
    requested, indices = legacy.schedule(replay["time"], fps, smoke_seconds)
    strip = 120
    views = (*legacy.VIEWS, "continuum_focus")
    paths = {view: folder/f"{slot_id}_{view}.mp4" for view in views}
    writers, renderers, last_frames = {}, {}, {}
    timings = {}
    with RenderOnly() as guard:
        model_started = time.perf_counter()
        attempt = Path(sources.paths["attempt_result"]["path"]).parent
        spec, verifier, data, compiled = legacy.make_model(task, evaluation, attempt.parent.parent, attempt.name)
        model = verifier.model
        lookat, distance = legacy._framing(model, data, replay["qpos"][::10])
        cameras = legacy._make_cameras(lookat, distance*1.22)
        continuum = verifier._descendant_body_ids(verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP)
        rigid = verifier._descendant_body_ids(verifier._RIGID_ROOT, verifier._RIGID_TIP)
        target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision")
        focus, focus_info = legacy._continuum_focus_camera_from_replay(model, data, replay["qpos"][::10], continuum,
                     target_geom, np.asarray(task.scenario["continuum_target"]["waypoint_points_m"]))
        focus.distance *= 1.17
        cameras["continuum_focus"] = focus
        normal_rgba = model.geom_rgba.copy()
        legacy._dim_rigid_arm_geometries(model, rigid)
        focus_rgba = model.geom_rgba.copy()
        model.geom_rgba[:] = normal_rgba
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width, focus_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height, focus_height)
        timings["model_and_camera_setup_s"] = time.perf_counter()-model_started
        folder.mkdir(parents=True)
        ffmpeg = shutil.which("ffmpeg")
        render_started = time.perf_counter()
        try:
            renderers["normal"] = mujoco.Renderer(model, height=height-strip, width=width)
            renderers["focus"] = mujoco.Renderer(model, height=focus_height-strip, width=focus_width)
            for view, path in paths.items():
                w, h = (focus_width, focus_height) if view == "continuum_focus" else (width, height)
                writers[view] = legacy.Writer(ffmpeg, path, w, h, fps)
            for number, state in enumerate(indices):
                state = int(state)
                data.qpos[:], data.qvel[:], data.time = replay["qpos"][state], replay["qvel"][state], float(replay["time"][state])
                data.ctrl[:] = 0.
                mujoco.mj_forward(model, data)
                for view in views:
                    is_focus = view == "continuum_focus"
                    renderer = renderers["focus" if is_focus else "normal"]
                    model.geom_rgba[:] = focus_rgba if is_focus else normal_rgba
                    renderer.update_scene(data, camera=cameras[view])
                    markers(renderer.scene, task, replay, state, reference, base,
                            focus=is_focus, labels=view in ("front", "continuum_focus"))
                    frame = annotate(renderer.render(), frozen, task, plan, result, replay, state, view, reference, base,
                                     smoke=smoke_seconds is not None, strip=strip)
                    writers[view].append(frame)
                    if number == len(indices)-1:
                        last_frames[view] = frame
        finally:
            for renderer in renderers.values():
                renderer.close()
            errors = []
            for writer in writers.values():
                try:
                    writer.close()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
        timings["native_render_and_encode_s"] = time.perf_counter()-render_started
    compose_started = time.perf_counter()
    grid = folder/f"{slot_id}_five_view_grid.mp4"
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for view in legacy.VIEWS:
        command += ["-i", str(paths[view])]
    command += ["-f", "lavfi", "-i", f"color=c=0x0a0e18:s={width}x{height}:r={fps}:d={len(indices)/fps}",
        "-filter_complex_threads", "1", "-filter_complex",
        "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[out]",
        "-map", "[out]", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-frames:v", str(len(indices)), str(grid)]
    legacy._run_ffmpeg(command)
    paths["five_view_grid"] = grid
    preview, focus_preview = folder/f"{slot_id}_five_view_preview.png", folder/f"{slot_id}_continuum_focus_preview.png"
    legacy._preview(ffmpeg, grid, preview, len(indices)/fps/2.)
    legacy._preview(ffmpeg, paths["continuum_focus"], focus_preview, len(indices)/fps/2.)
    last = folder/f"{slot_id}_last_saved_state.png"
    Image.fromarray(last_frames["continuum_focus"]).save(last)
    video_records = {}
    for view, path in paths.items():
        w, h = ((width*3, height*2) if view == "five_view_grid" else
                (focus_width, focus_height) if view == "continuum_focus" else (width, height))
        video_records[view] = {**record(path), "relative_path": path.relative_to(output).as_posix(), "width": w, "height": h,
                              "probe": legacy.probe(path, len(indices), w, h, fps)}
    timings["grid_previews_probe_s"] = time.perf_counter()-compose_started
    sources.unchanged()
    require(generator_records() == producer_bindings, "render producer source changed during slot")
    timings["total_wall_s"] = time.perf_counter()-started
    schedule = {"presentation_times_s": requested.tolist(), "selected_saved_indices": indices.tolist(),
                "selected_saved_times_s": replay["time"][indices].tolist(), "fps": fps,
                "frame_count": len(indices), "source_state_count": len(replay["time"]),
                "includes_exact_saved_endpoint": smoke_seconds is None, "smoke_truncated": smoke_seconds is not None,
                "selection_rule": "first actual saved state at/after CFR presentation time; exact complete saved endpoint; no interpolation"}
    metadata = {"schema": SCHEMA, "renderer_producer_bindings": producer_bindings,
        "slot_id": slot_id, "source_slot": result["slot_id"], "source_role": frozen["source_role"],
        "task_id": task.task_id, "task_sha256": task.sha256(), "mode": frozen["mode"], "status": result["status"],
        "reference_version": plan.representation_version, "reference_family": frozen["reference_version"],
        "amplitude_m": frozen["amplitude_m"], "full_task_success": result["full_task_success"],
        "quality_label_eligible": quality["quality_label_eligible"], "reference_plan_sha256": plan.sha256(),
        "reference_execution_binding": ref_binding, "render_state_kind": "ACTUAL_TRACE_SAVED_STATES",
        "actual_saved_state_source": {**sources.paths["actual_trace"], "qpos_fields": ["initial_qpos", "actual_full_qpos"],
             "qvel_fields": ["initial_qvel", "actual_full_qvel"], "clock_fields": ["literal_initial_zero", "time"]},
        "independent_replay_state_source": {**sources.paths["fresh_replay"],
             "role": "saved kinematics only after exact actual/replay full-state parity; not render-state source"},
        "actual_fresh_parity": parity, "schedule": schedule,
        "fps": fps, "frame_count": len(indices), "source_start_s": float(replay["time"][0]),
        "source_end_s": float(replay["time"][-1]), "displayed_saved_end_s": float(replay["time"][indices[-1]]),
        "source_duration_s": float(replay["time"][-1]), "encoded_duration_s": len(indices)/fps,
        "includes_exact_saved_endpoint": smoke_seconds is None, "smoke_truncated": smoke_seconds is not None,
        "width": width, "height": height, "focus_width": focus_width, "focus_height": focus_height,
        "videos": {view: row["relative_path"] for view, row in video_records.items()}, "video_records": video_records,
        "preview": preview.relative_to(output).as_posix(), "preview_record": record(preview),
        "focus_preview": focus_preview.relative_to(output).as_posix(), "focus_preview_record": record(focus_preview),
        "last_saved_state": last.relative_to(output).as_posix(), "last_saved_state_record": record(last),
        "sourcepaths": sources.paths, "source_compiled_model_sha256": compiled,
        "model_contract_sha256": spec.runtime_contract_sha256(), "model_source_bundle_sha256": spec.source_bundle_sha256(),
        "focus_camera": focus_info, "render_cost": {"wall_s": timings, "mj_forward_calls": guard.forward_calls,
             "native_view_frames": len(indices)*len(views), "encoded_video_frames": len(indices)*len(paths),
             "physics_steps": 0, "geometry_queries": 0, "QP_solves": 0, "model_samples": 0,
             "forbidden_call_attempts": guard.forbidden_attempts},
        "physics_steps_executed": 0, "optimizer_updates": 0, "new_samples": 0,
        "reference_scope": "green=frozen analytic continuum reference on actual post-step clocks, matching saved generated_continuum_position; not held/consumed QP task inputs; red=independent Task/base; cyan=actual tip; yellow/blue=current Task/actual rigid pose",
        "kinematics_scope": "fresh independent same-torque replay kinematics reused after bit-exact actual qpos/qvel parity",
        "path_scope": "complete reference paths through saved endpoint; actual trail only through displayed actual saved state",
        "coordinate_frame_axis_order": ["X_red", "Y_green", "Z_blue"],
        "waypoint_label_placement": "W1 +0.07m world Z; W7 -0.04m world Z; remaining labels retain legacy offset",
        "deployment": result.get("deployment", "NOT_MET"), "deployment_visualization_label": "NOT_ESTABLISHED",
        "visualization_is_new_acceptance": False, "new_safety_evaluation": False}
    write(folder/"replay_metadata.json", metadata)
    return metadata


def generator_records():
    return [record(path) for path in (Path(__file__), Path(legacy.__file__),
        ROOT/"v6_lite/visualization/latest_saved_renderer.py", ROOT/"v6_lite/visualization/generate_visualizations.py")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=15.)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--focus-width", type=int, default=960)
    parser.add_argument("--focus-height", type=int, default=720)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all-slots", action="store_true", help="explicit full 28-slot / 196-video production")
    mode.add_argument("--slot", choices=SLOTS, help="labelled smoke only; pair with --smoke-seconds")
    mode.add_argument("--check-inputs", action="store_true", help="read/parity-check all 28 slots; allocate no render model")
    parser.add_argument("--smoke-seconds", type=float)
    args = parser.parse_args()
    if (args.slot is not None) != (args.smoke_seconds is not None):
        parser.error("--slot and --smoke-seconds are required together")
    if not np.isfinite(args.fps) or args.fps <= 0. or args.fps > 60.:
        parser.error("FPS must be finite and in (0,60]")
    if args.smoke_seconds is not None and (not np.isfinite(args.smoke_seconds) or not 0. < args.smoke_seconds < 27.):
        parser.error("smoke duration must be finite, positive and less than 27s")
    if (min(args.width, args.focus_width) < 640 or min(args.height, args.focus_height) < 480
            or any(value % 2 for value in (args.width, args.height, args.focus_width, args.focus_height))):
        parser.error("even video dimensions of at least 640x480 required")
    source, output = args.source.resolve(), args.output.resolve()
    producer_bindings = generator_records()
    shared_figures_only = (args.all_slots and output.is_dir()
                          and {path.name for path in output.iterdir()} <= {"figures"})
    if ((output.exists() and not shared_figures_only)
            or output.is_relative_to(source) or source.is_relative_to(output)):
        parser.error("output must be exclusive, or a production media directory containing only the separate figures subdirectory")
    if args.check_inputs:
        records = []
        started = time.perf_counter()
        for slot in SLOTS:
            loaded = load_slot(source, slot)
            sources, frozen, task, plan, result, _, _, replay, _, _, parity, binding = loaded
            sources.unchanged()
            records.append({"slot_id": slot, "source_slot": result["slot_id"], "source_role": frozen["source_role"],
                "actual_fresh_parity": parity, "reference_execution_binding": binding, "sourcepaths": sources.paths,
                "source_end_s": float(replay["time"][-1])})
        write(output/"input_checks.json", {"schema": "v64_b31_saved_actual_media_input_checks_v1", "status": "PASS",
            "records": records, "record_count": len(records), "wall_s": time.perf_counter()-started,
            "physics_steps": 0, "geometry_queries": 0, "QP_solves": 0, "mj_forward_calls": 0, "videos_generated": 0,
            "generator_records": producer_bindings})
        print(json.dumps({"status": "PASS", "input_slots": 28, "videos_generated": 0, "output": str(output)}))
        return
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required")
    # Root's separately generated end-effector figures may already occupy only
    # figures/. This renderer owns exclusive replays/ and videos.json paths.
    output.mkdir(parents=True, exist_ok=shared_figures_only)
    slots = (args.slot,) if args.slot else SLOTS
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(render_slot, source, output, slot, args.fps, args.width, args.height,
                   args.focus_width, args.focus_height, args.smoke_seconds): slot for slot in slots}
        for future in as_completed(pending):
            row = future.result()
            records.append(row)
            print(json.dumps({"slot": row["slot_id"], "source_slot": row["source_slot"], "frames": row["frame_count"],
                "videos": len(row["videos"]), "wall_s": row["render_cost"]["wall_s"]}, ensure_ascii=False), flush=True)
    ordered = sorted(records, key=lambda row: slots.index(row["slot_id"]))
    require(generator_records() == producer_bindings, "render producer source changed during collection")
    write(output/"videos.json", {"schema": COLLECTION_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": record(source/"manifest.json"), "source_verification": record(source/"verification.json"),
        "generator": producer_bindings[0], "generator_records": producer_bindings, "slot_order": list(slots), "records": ordered,
        "record_count": len(ordered), "video_count": sum(len(row["videos"]) for row in ordered),
        "wall_s": time.perf_counter()-started, "smoke_truncated": args.smoke_seconds is not None,
        "physics_steps_executed": 0, "geometry_queries": 0, "optimizer_updates": 0, "new_samples": 0,
        "actual_saved_state_source_verified": True, "independent_replay_states_used_for_render": False,
        "historical_outputs_overwritten": False, "visualization_is_new_acceptance": False})


if __name__ == "__main__":
    main()
