"""C.2 dashboard, figures and seven-view media from sealed saved actual states.

Rendering never advances physics, queries clearance, solves QP, or proposes a
new reference. Alias methods share their unique actual's media. Failed actual
prefixes end at their exact saved endpoint; NO_PLAN receives no invented video.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import render_residual_replays as legacy
from .render_execution_aware_replays import RenderOnly, markers
from ..route_candidate_evaluator import verify_seal
from ..route_optimizer_protocol import digest, read, sha
from ..task_anchored_reference import TaskAnchoredResidualPlan, TaskAnchoredResidualReferenceProvider
from ..task_protocol import TaskSpec


ENDPOINTS = ("R8", "R12", "N8", "D8")
PREFERENCES = ("A", "B")
SCHEMA = "v64_c2_saved_actual_media_v1"
COLORS = {"R": "#586b89", "N": "#d08a26", "D": "#0f947b"}


class C2RenderOnly(RenderOnly):
    """Also forbid the actual controller's custom ADMM entry points."""

    def __enter__(self):
        super().__enter__()
        from v6_lite.hierarchical_qp import HierarchicalVelocityQP
        for name in ("solve", "_solve_qp_admm"):
            old = getattr(HierarchicalVelocityQP, name)
            self.restore.append((HierarchicalVelocityQP, name, old))
            def reject(*args, _name=name, **kwargs):
                self.forbidden_attempts.append("HierarchicalVelocityQP." + _name)
                raise RuntimeError("saved C.2 renderer forbids QP " + _name)
            setattr(HierarchicalVelocityQP, name, reject)
        return self


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def record(path, root=None):
    path = Path(path).resolve()
    return {"path": path.relative_to(root).as_posix() if root and path.is_relative_to(root) else str(path),
        "sha256": sha(path), "bytes": path.stat().st_size}


def terminal_slots(run):
    run = Path(run).resolve()
    if not (run / "actual_complete.json").is_file():
        raise ValueError("all actual endpoints must be terminal before media generation")
    complete = read(run / "actual_complete.json")
    sealed = read(run / "sealed_selections" / "all_selections.json")
    for relative, expected in sealed["files"].items():
        if sha(run / relative) != expected:
            raise ValueError("sealed search/selection evidence changed")
    if sha(run / "model_freeze.json") != sealed["model_freeze_sha256"]:
        raise ValueError("model freeze differs from pre-actual selection seal")
    plan = read(run / "plan.json")
    tasks = [r for r in plan["tasks"] if r.get("learning_split", r.get("split")) == "test"]
    if len(tasks) != 4:
        raise ValueError("the frozen four TEST Tasks are required")
    result = []
    for task in tasks:
        for endpoint in ENDPOINTS:
            selection_path = run / "sealed_selections" / task["task_id"] / f"{endpoint}.json"
            selection = read(selection_path)
            for preference in PREFERENCES:
                path = run / "actual" / task["task_id"] / f"{endpoint}_{preference}" / "slot.json"
                verify_seal(path.parent)
                slot = read(path)
                if complete["slot_hashes"].get(path.relative_to(run).as_posix()) != sha(path):
                    raise ValueError("actual terminal receipt does not bind logical slot")
                if (slot["task_sha256"] != task["task_sha256"] or slot["selection_sha256"] != sha(selection_path)
                        or slot["endpoint"] != endpoint or slot["preference"] != preference):
                    raise ValueError("actual logical slot differs from its sealed endpoint")
                selected = selection["preferences"][preference]["selected_plan"]
                expected = TaskAnchoredResidualPlan.from_dict(selected).sha256() if selected else None
                if slot["plan_sha256"] != expected:
                    raise ValueError("actual reference differs from frozen selection")
                result.append(slot)
    return result


def resolve_unique(slots, task_id, method):
    by_key = {(row["task_id"], row["method"]): row for row in slots}
    current = by_key[(task_id, method)]; seen = set()
    while current.get("alias_of_method"):
        key = (task_id, current["method"])
        if key in seen:
            raise ValueError("actual alias cycle")
        seen.add(key)
        source = by_key[(task_id, current["alias_of_method"])]
        if (source["alias_identity"] != current["alias_identity"] or source["plan_sha256"] != current["plan_sha256"]
                or source["task_sha256"] != current["task_sha256"]):
            raise ValueError("actual media alias lacks complete frozen identity")
        current = source
    return current


def _local_path(run, attempt, value, name):
    path = Path(value) if value else attempt / name
    if path.exists():
        return path.resolve()
    # Evidence remains usable when its complete run is copied to a release.
    matches = list(attempt.rglob(path.name))
    if len(matches) != 1:
        raise ValueError("saved evidence path cannot be rebound unambiguously: " + str(path))
    return matches[0].resolve()


def execution_model_binding(run, attempt, sources):
    """Bind the renderer to a saved actual model, including failed prefixes."""
    hashes = []
    observation = attempt / "actual" / "historical_metric_observations.json"
    if observation.exists():
        compiled = (read(observation).get("execution_contract") or {}).get("source_compiled_model_sha256")
        if compiled:
            hashes.append(compiled)
            sources["execution_model_observation"] = record(observation, run)
    for path in sorted((attempt / "actual").rglob("timing/*.jsonl")):
        with path.open(encoding="utf8") as stream:
            for line in stream:
                certificate = json.loads(line).get("certificate")
                if certificate and certificate.get("source_model_hash"):
                    hashes.append(certificate["source_model_hash"])
                    sources["execution_model_certificate"] = record(path, run)
                    break
        if "execution_model_certificate" in sources:
            break
    if not hashes or len(set(hashes)) != 1:
        raise ValueError("saved actual execution compiled-model identity missing or inconsistent")
    return hashes[0]


def verify_render_model(compiled, parity):
    if compiled != parity["saved_execution_compiled_model_sha256"]:
        raise ValueError("compiled render model differs from saved actual execution model")


def load_actual(run, slot):
    run = Path(run).resolve()
    directory = run / "actual" / slot["task_id"] / slot["method"]
    attempt = directory / "attempt"
    task_path = run / "frozen_tasks" / slot["task_id"] / "task.json"
    plan_path = directory / "selected_plan.json"
    result_path = attempt / "attempt_result.json"
    evaluation_path = attempt / "actual" / "evaluation" / "report.json"
    replay_path = evaluation_path.parent / "fresh_replay.npz"
    task = TaskSpec.from_dict(read(task_path)); plan = TaskAnchoredResidualPlan.from_dict(read(plan_path))
    executed_plan = TaskAnchoredResidualPlan.from_dict(read(attempt / "plan.json"))
    result, evaluation = read(result_path), read(evaluation_path)
    trace_path = _local_path(run, attempt, result.get("trace_path"), "trace.npz")
    if (task.sha256() != slot["task_sha256"] or plan.sha256() != slot["plan_sha256"]
            or executed_plan.sha256() != plan.sha256() or result["plan_content_sha256"] != plan.sha256()
            or result["task_sha256"] != task.sha256() or result["plan_file_sha256"] != sha(attempt / "plan.json")
            or result["trace_sha256"] != sha(trace_path) or evaluation["fresh_replay_sha256"] != sha(replay_path)
            or result["evaluation_sha256"] != sha(evaluation_path)):
        raise ValueError("actual rendering evidence bindings differ")
    with np.load(trace_path, allow_pickle=False) as archive:
        actual = {k: archive[k].copy() for k in
            ("initial_qpos", "initial_qvel", "actual_full_qpos", "actual_full_qvel", "time", "generated_continuum_position")}
    needed = ("time", "qpos", "qvel", "base_pose", "target_position", "target_rotation",
        "rigid_position", "rigid_rotation", "continuum_position", "continuum_rotation")
    with np.load(replay_path, allow_pickle=False) as archive:
        replay = {k: archive[k].copy() for k in needed}
    n = int(evaluation.get("actual_physics_steps", slot["actual_steps"]))
    qpos = np.vstack((actual["initial_qpos"], actual["actual_full_qpos"][:n]))
    qvel = np.vstack((actual["initial_qvel"], actual["actual_full_qvel"][:n]))
    times = np.r_[0., actual["time"][:n]]
    shapes = {"time": (n + 1,), "base_pose": (n + 1, 7),
        **{k: (n + 1, 3) for k in ("target_position", "rigid_position", "continuum_position")},
        **{k: (n + 1, 3, 3) for k in ("target_rotation", "rigid_rotation", "continuum_rotation")}}
    if (n <= 0 or n != slot["actual_steps"] or (slot.get("full_task_success") and n != 13500)
            or qpos.shape != (n + 1, 81) or qvel.shape != (n + 1, 79)
            or replay["qpos"].shape != qpos.shape or replay["qvel"].shape != qvel.shape
            or any(replay[k].shape != shape for k, shape in shapes.items())
            or not np.array_equal(qpos[0], task.initial_qpos) or not np.array_equal(qvel[0], task.initial_qvel)
            or not all(np.isfinite(v).all() for v in (*actual.values(), *replay.values()))
            or not np.allclose(times, np.arange(n + 1) * .002, atol=1e-9, rtol=0.)
            or not np.allclose(times, replay["time"], atol=1e-9, rtol=0.)):
        raise ValueError("actual saved state dimension/clock/initial/finite check failed")
    qpos_error = float(np.max(np.abs(qpos - replay["qpos"])))
    qvel_error = float(np.max(np.abs(qvel - replay["qvel"])))
    if qpos_error > 1e-9 or qvel_error > 1e-8:
        raise ValueError("actual state and independent replay parity failed")
    parity = {"qpos_max_abs_difference": qpos_error, "qvel_max_abs_difference": qvel_error,
        "qpos_bit_exact": bool(np.array_equal(qpos, replay["qpos"])),
        "qvel_bit_exact": bool(np.array_equal(qvel, replay["qvel"])), "initial_state_matches_Task": True}
    replay["qpos"], replay["qvel"], replay["time"] = qpos, qvel, times
    reference, _, _ = TaskAnchoredResidualReferenceProvider(task, plan).continuum_kinematics(times)
    generated = actual["generated_continuum_position"][:n]
    if generated.shape != (n, 3):
        raise ValueError("saved generated continuum reference shape differs")
    reference_error = float(np.max(np.abs(reference[1:] - generated)))
    if reference_error > 1e-12:
        raise ValueError("frozen continuum reference differs from saved actual generated reference")
    parity["generated_continuum_reference_max_abs_difference_m"] = reference_error
    offset, _, _ = plan.offset_kinematics(times)
    base = reference - offset
    sources = {name: record(path, run) for name, path in (
        ("slot", directory / "slot.json"), ("task", task_path), ("plan", plan_path), ("executed_plan", attempt / "plan.json"),
        ("attempt_result", result_path), ("evaluation", evaluation_path), ("actual_trace", trace_path),
        ("fresh_replay", replay_path))}
    parity["saved_execution_compiled_model_sha256"] = execution_model_binding(run, attempt, sources)
    return task, plan, result, evaluation, replay, reference, base, parity, sources


def unchanged(run, sources):
    for row in sources.values():
        path = Path(row["path"])
        if not path.is_absolute(): path = Path(run) / path
        if sha(path) != row["sha256"]:
            raise ValueError("source actual evidence changed during media generation")


def _angle(actual, target):
    cosine = (np.einsum("nij,nij->n", actual, target) - 1.) * .5
    return np.arccos(np.clip(cosine, -1., 1.))


def trajectory_figures(run, output, slot):
    task, plan, _, _, replay, reference, base, parity, sources = load_actual(run, slot)
    folder = Path(output) / "figures"; folder.mkdir(parents=True, exist_ok=True)
    sid = task.task_id + "_" + slot["method"]
    rigid_target = replay["target_position"] + np.einsum("nij,j->ni", replay["target_rotation"],
        np.asarray(task.scenario["grasp_point_target_frame_m"]))
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    for i, (actual, target, title) in enumerate(((replay["continuum_position"], reference, "Continuum tip"),
            (replay["rigid_position"], rigid_target, "Rigid tip")), 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot(*target.T, color="#35975b", label="Frozen reference" if i == 1 else "Current Task grasp")
        if i == 1: ax.plot(*base.T, color="#bb655c", alpha=.7, label="Task/base path")
        ax.plot(*actual.T, color="#1573af", linewidth=1.5, label="Actual saved tip")
        ax.scatter(*actual[-1], color="#d47614", s=24, label="Exact saved endpoint")
        ax.set(xlabel="World X [m]", ylabel="World Y [m]", zlabel="World Z [m]", title=title)
        ax.legend(fontsize=8)
    fig.suptitle(f"{sid} | {'full actual' if slot['full_task_success'] else 'failed actual prefix'} | saved end {replay['time'][-1]:.3f}s")
    trajectory = folder / f"{sid}_trajectory.png"; fig.savefig(trajectory, dpi=155); plt.close(fig)
    c_error = np.linalg.norm(replay["continuum_position"] - reference, axis=1)
    c_task = np.linalg.norm(replay["continuum_position"] - base, axis=1)
    r_error = np.linalg.norm(replay["rigid_position"] - rigid_target, axis=1)
    c_rotation = np.broadcast_to(np.asarray(task.scenario["continuum_target_rotation_world"]), replay["continuum_rotation"].shape)
    r_rotation = np.einsum("nij,jk->nik", replay["target_rotation"], np.asarray(task.scenario["grasp_rotation_target_frame"]))
    c_angle, r_angle = _angle(replay["continuum_rotation"], c_rotation), _angle(replay["rigid_rotation"], r_rotation)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(replay["time"], c_error * 1000, label="Continuum vs frozen generated reference")
    axes[0].plot(replay["time"], c_task * 1000, label="Continuum vs Task/base path", alpha=.7)
    axes[0].set_ylabel("Position error [mm]"); axes[0].legend(fontsize=8)
    axes[1].plot(replay["time"], r_error * 1000, color="#b17b23", label="Rigid vs current Task grasp")
    axes[1].set_ylabel("Position error [mm]"); axes[1].legend(fontsize=8)
    axes[2].plot(replay["time"], c_angle * 180 / np.pi, label="Continuum")
    axes[2].plot(replay["time"], r_angle * 180 / np.pi, label="Rigid")
    axes[2].set(xlabel="Saved physical time [s]", ylabel="Orientation error [deg]")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2); ax.set_xlim(0., max(.02, float(replay["time"][-1])))
        for point in task.requirements:
            lo, hi = point.time_window_s
            if lo <= replay["time"][-1]: ax.axvspan(lo, min(hi, replay["time"][-1]), color="#688ca4", alpha=.08)
    fig.suptitle(f"{sid} | descriptive tracking, original independent gates determine acceptance")
    tracking = folder / f"{sid}_tracking.png"; fig.savefig(tracking, dpi=155); plt.close(fig)
    csv = folder / f"{sid}_tracking_native.csv"
    np.savetxt(csv, np.column_stack((replay["time"], c_error, c_task, r_error, c_angle, r_angle)), delimiter=",",
        header="physical_time_s,continuum_generated_position_error_m,continuum_Task_base_position_error_m,rigid_current_Task_position_error_m,continuum_orientation_error_rad,rigid_orientation_error_rad", comments="")
    unchanged(run, sources)
    return {"trajectory": trajectory.relative_to(output).as_posix(), "tracking": tracking.relative_to(output).as_posix(),
        "tracking_csv": csv.relative_to(output).as_posix(), "source_end_s": float(replay["time"][-1]),
        "actual_fresh_parity": parity, "sources": sources, "physics_steps": 0, "geometry_queries": 0}


def _annotation(frame, slot, task, replay, state, view, reference, strip=120):
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + strip), (10, 14, 24))
    image.paste(Image.fromarray(frame), (0, strip)); draw = ImageDraw.Draw(image)
    complete = slot["full_task_success"] and slot["original_independent_gates_passed"]
    lines = [f"C.2 {slot['method']} | {view.upper()}", f"Task {task.task_id}",
        f"{'FULL ACTUAL + 5 GATES PASS' if complete else 'FAILED ACTUAL PREFIX'} | saved {replay['time'][state]:.3f}/{replay['time'][-1]:.3f}s",
        f"Continuum frozen reference error {1000*np.linalg.norm(replay['continuum_position'][state]-reference[state]):.2f}mm",
        "green=frozen reference; red=Task/base; cyan=actual; RGB=XYZ",
        "Saved actual states + forward only | deployment NOT_MET"]
    for i, line in enumerate(lines):
        draw.text((7, 4 + i * 19), line, fill=(245, 193, 130) if i == 2 and not complete else (225, 234, 245),
            font=legacy._font(12 if image.width < 900 else 16))
    return np.asarray(image)


def render_actual(run, output, slot, fps=15., width=640, height=480, focus_width=960, focus_height=720):
    started = time.perf_counter(); output = Path(output).resolve(); run = Path(run).resolve()
    sid = slot["task_id"] + "_" + slot["method"]; folder = output / "replays" / sid
    metadata_path = folder / "replay_metadata.json"
    if metadata_path.exists():
        saved = read(metadata_path)
        unchanged(run, saved["sourcepaths"])
        for row in saved["video_records"].values():
            if sha(output / row["relative_path"]) != row["sha256"]: raise ValueError("retained C.2 video changed")
        return saved
    if folder.exists(): raise FileExistsError("unfinished render directory retained: " + str(folder))
    task, plan, result, evaluation, replay, reference, base, parity, sources = load_actual(run, slot)
    requested, indices = legacy.schedule(replay["time"], fps)
    views = (*legacy.VIEWS, "continuum_focus"); paths = {v: folder / f"{sid}_{v}.mp4" for v in views}
    ffmpeg = shutil.which("ffmpeg"); writers, renderers, last_frames = {}, {}, {}
    with C2RenderOnly() as guard:
        # Same model contract and native pair geometry as the existing renderer;
        # no pair is queried under the render-only guard.
        spec, verifier, data, compiled = legacy.make_model(task, evaluation, run, sid)
        verify_render_model(compiled, parity)
        model = verifier.model
        lookat, distance = legacy._framing(model, data, replay["qpos"][::10])
        cameras = legacy._make_cameras(lookat, distance * 1.22)
        continuum = verifier._descendant_body_ids(verifier._CONTINUUM_ROOT, verifier._CONTINUUM_TIP)
        rigid = verifier._descendant_body_ids(verifier._RIGID_ROOT, verifier._RIGID_TIP)
        target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision")
        focus, focus_info = legacy._continuum_focus_camera_from_replay(model, data, replay["qpos"][::10], continuum,
            target_geom, np.asarray(task.scenario["continuum_target"]["waypoint_points_m"]))
        focus.distance *= 1.17; cameras["continuum_focus"] = focus
        normal_rgba = model.geom_rgba.copy(); legacy._dim_rigid_arm_geometries(model, rigid)
        focus_rgba = model.geom_rgba.copy(); model.geom_rgba[:] = normal_rgba
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width, focus_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height, focus_height)
        folder.mkdir(parents=True)
        try:
            renderers["normal"] = mujoco.Renderer(model, height=height - 120, width=width)
            renderers["focus"] = mujoco.Renderer(model, height=focus_height - 120, width=focus_width)
            for view, path in paths.items():
                w, h = (focus_width, focus_height) if view == "continuum_focus" else (width, height)
                writers[view] = legacy.Writer(ffmpeg, path, w, h, fps)
            for number, state in enumerate(indices):
                state = int(state)
                data.qpos[:], data.qvel[:], data.time = replay["qpos"][state], replay["qvel"][state], float(replay["time"][state])
                data.ctrl[:] = 0.; mujoco.mj_forward(model, data)
                for view in views:
                    is_focus = view == "continuum_focus"; renderer = renderers["focus" if is_focus else "normal"]
                    model.geom_rgba[:] = focus_rgba if is_focus else normal_rgba
                    renderer.update_scene(data, camera=cameras[view])
                    markers(renderer.scene, task, replay, state, reference, base, focus=is_focus, labels=view in ("front", "continuum_focus"))
                    frame = _annotation(renderer.render(), slot, task, replay, state, view, reference)
                    writers[view].append(frame)
                    if number == len(indices) - 1: last_frames[view] = frame
        finally:
            for renderer in renderers.values(): renderer.close()
            errors = []
            for writer in writers.values():
                try: writer.close()
                except Exception as error: errors.append(error)
            if errors: raise errors[0]
    grid = folder / f"{sid}_five_view_grid.mp4"
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for view in legacy.VIEWS: command += ["-i", str(paths[view])]
    command += ["-f", "lavfi", "-i", f"color=c=0x0a0e18:s={width}x{height}:r={fps}:d={len(indices)/fps}",
        "-filter_complex_threads", "1", "-filter_complex",
        "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[out]",
        "-map", "[out]", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-frames:v", str(len(indices)), str(grid)]
    legacy._run_ffmpeg(command); paths["five_view_grid"] = grid
    preview = folder / f"{sid}_five_view_preview.png"; focus_preview = folder / f"{sid}_continuum_focus_preview.png"
    legacy._preview(ffmpeg, grid, preview, len(indices) / fps / 2.)
    legacy._preview(ffmpeg, paths["continuum_focus"], focus_preview, len(indices) / fps / 2.)
    last = folder / f"{sid}_last_saved_state.png"; Image.fromarray(last_frames["continuum_focus"]).save(last)
    video_records = {}
    for view, path in paths.items():
        w, h = (width * 3, height * 2) if view == "five_view_grid" else (focus_width, focus_height) if view == "continuum_focus" else (width, height)
        video_records[view] = {**record(path, output), "relative_path": path.relative_to(output).as_posix(),
            "probe": legacy.probe(path, len(indices), w, h, fps)}
    unchanged(run, sources)
    metadata = {"schema": SCHEMA, "slot_id": sid, "task_id": task.task_id, "method": slot["method"],
        "status": slot["status"], "full_task_success": slot["full_task_success"], "plan_sha256": plan.sha256(),
        "actual_fresh_parity": parity, "sourcepaths": sources, "render_state_kind": "ACTUAL_TRACE_SAVED_STATES",
        "independent_replay_use": "kinematics after saved actual state parity; no new replay",
        "source_end_s": float(replay["time"][-1]), "displayed_saved_end_s": float(replay["time"][indices[-1]]),
        "frame_count": len(indices), "fps": fps, "encoded_duration_s": len(indices) / fps,
        "selected_saved_indices": indices.tolist(), "selected_saved_times_s": replay["time"][indices].tolist(),
        "presentation_times_s": requested.tolist(), "includes_exact_saved_endpoint": int(indices[-1]) == len(replay["time"]) - 1,
        "videos": {v: p.relative_to(output).as_posix() for v, p in paths.items()}, "video_records": video_records,
        "preview": preview.relative_to(output).as_posix(), "focus_preview": focus_preview.relative_to(output).as_posix(),
        "last_saved_state": last.relative_to(output).as_posix(), "source_compiled_model_sha256": compiled,
        "focus_camera": focus_info, "wall_s": time.perf_counter() - started,
        "render_cost": {"mj_forward_calls": guard.forward_calls, "physics_steps": 0, "geometry_queries": 0,
            "QP_solves": 0, "forbidden_call_attempts": guard.forbidden_attempts},
        "deployment": "NOT_MET", "visualization_is_new_acceptance": False}
    write(metadata_path, metadata)
    return metadata


def cold_cost_bracket(run, task_id, endpoint, jobs, workers):
    """Measured inner timer and enclosing worker interval; never a fake exact cold time."""
    run = Path(run)
    stream = run / "benchmark_search" / task_id / endpoint[0]
    planning = stream / "planning" / task_id
    cost_path = stream / "planning_cost.json"; final_path = planning / "selection.json"
    cost, final = read(cost_path), read(final_path)
    job = jobs[(task_id, endpoint[0])]
    stream_inner, final_elapsed, worker = [float(v) for v in
        (cost["end_to_end_cold_planning_s"], final["elapsed_wall_s"], job["elapsed_wall_s"])]
    if not all(np.isfinite(v) for v in (stream_inner, final_elapsed, worker)) or not 0. <= final_elapsed <= stream_inner <= worker:
        raise ValueError("cold timing bracket requires finite final <= inner <= worker intervals")
    prefix_elapsed = None
    sources = {"search_phase": record(run / "search_phase.json", run),
        "planning_cost": record(cost_path, run), "final_selection": record(final_path, run)}
    if endpoint == "R8":
        prefix_path = planning / "prefix_08.json"
        prefix_elapsed = float(read(prefix_path)["elapsed_wall_s"])
        if not np.isfinite(prefix_elapsed) or not 0. <= prefix_elapsed <= final_elapsed:
            raise ValueError("R8 prefix elapsed must lie within final search elapsed")
        lower = stream_inner - final_elapsed + prefix_elapsed
        upper = worker - final_elapsed + prefix_elapsed
        formula = "outer_worker_elapsed_s - final_selection_elapsed_s + prefix08_elapsed_s"
        sources["prefix08_selection"] = record(prefix_path, run)
    else:
        lower, upper, formula = stream_inner, worker, "outer_worker_elapsed_s"
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError("cold timing bracket upper bound is below lower bound")
    value = {"task_id": task_id, "endpoint": endpoint, "planner_internal_cold_s": lower,
        "cold_lower_bound_s": lower, "cold_upper_bound_s": upper, "outer_worker_elapsed_s": worker,
        "stream_internal_cold_s": stream_inner, "final_selection_elapsed_s": final_elapsed,
        "prefix08_elapsed_s": prefix_elapsed, "startup_and_tail_residual_s": worker - stream_inner,
        "bound_width_s": upper - lower, "upper_bound_formula": formula, "source_receipts": sources,
        "worker_count": workers,
        "timing_scope": "inner endpoint timer through enclosing worker dispatch-to-return; startup and post-selection tail are bracketed, not separately measured",
        "execution_context": "shared worker-resource load; no isolated repeated-task latency experiment",
        "warm_path_is_decomposition_estimate": True,
        "warm_resident_model_path_estimate_s": cost.get("warm_resident_model_path_estimate_s")}
    table = run / "tables" / "report_interpretation" / "endpoint_cold_cost_brackets.json"
    if table.exists():
        rows = read(table)
        match = next((r for r in rows if r["task_id"] == task_id and r["endpoint"] == endpoint), None)
        if match is None or any(not np.isclose(value[k], match[k], atol=1e-9, rtol=0.) for k in
                ("cold_lower_bound_s", "cold_upper_bound_s", "outer_worker_elapsed_s", "bound_width_s")):
            raise ValueError("media cold bracket differs from final report interpretation")
        value["report_interpretation_receipt"] = record(table, run)
    return value


def core_charts(run, output, slots):
    tasks = list(dict.fromkeys(row["task_id"] for row in slots)); prefixes = []
    phase = read(Path(run) / "search_phase.json")
    jobs = {(row["task_id"], row["method"]): row for row in phase["jobs"]}
    if (phase.get("passed") is not True or len(jobs) != len(tasks) * 3 or len(phase["jobs"]) != len(jobs)
            or any(row["exit_code"] != 0 for row in phase["jobs"])):
        raise ValueError("terminal search worker receipts are required for cold timing brackets")
    costs = {}
    for task_id in tasks:
        for method in ("R", "N", "D"):
            stream = Path(run) / "benchmark_search" / task_id / method
            planning = stream / "planning" / task_id
            for budget in ((4, 8, 12) if method == "R" else (4, 8)):
                prefix = read(planning / f"prefix_{budget:02d}.json")
                prefixes.append({"task_id": task_id, "method": method, "budget": budget, "snapshot": prefix})
                endpoint = f"{method}{budget}"
                if endpoint in ENDPOINTS:
                    costs[(task_id, endpoint)] = cold_cost_bracket(run, task_id, endpoint, jobs, phase["workers"])
    folder = Path(output) / "figures"; folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, pref in enumerate(PREFERENCES):
        metric, unit = ("I_support", "rad/s") if pref == "A" else ("L_full", "m")
        for method in ("R", "N", "D"):
            budgets = (4, 8, 12) if method == "R" else (4, 8)
            counts, qualities = [], []
            for budget in budgets:
                items = [r["snapshot"]["preferences"][pref] for r in prefixes if r["method"] == method and r["budget"] == budget]
                counts.append(sum(item["selected_plan"] is not None for item in items))
                values = [(item.get("prediction_metrics") or {}).get(metric) for item in items if item["selected_plan"] is not None]
                qualities.append(float(np.mean(values)) if values else np.nan)
            axes[0, column].plot(budgets, counts, "o-", color=COLORS[method], label=method)
            axes[1, column].plot(budgets, qualities, "o-", color=COLORS[method], label=method)
        axes[0, column].set(title=f"Preference {pref}: predicted plan coverage", ylabel="Tasks with a plan / 4", ylim=(-.1, 4.2), yticks=range(5))
        axes[1, column].set(title=f"Preference {pref}: eligible predicted {metric}", ylabel=f"Mean {metric} [{unit}]", xlabel="Consumed evaluation-slot budget")
        for ax in axes[:, column]: ax.set_xticks([4, 8, 12]); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("C.2 sealed search prefixes | 4-slot points are prediction only; quality means exclude NO_PLAN")
    budget_chart = folder / "budget_quality_coverage.png"; fig.savefig(budget_chart, dpi=155); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(ENDPOINTS))
    means = np.asarray([np.mean([costs[(tid, ep)]["cold_lower_bound_s"] for tid in tasks]) for ep in ENDPOINTS])
    upper_means = np.asarray([np.mean([costs[(tid, ep)]["cold_upper_bound_s"] for tid in tasks]) for ep in ENDPOINTS])
    axes[0].bar(x, means, color=[COLORS[ep[0]] for ep in ENDPOINTS], alpha=.65, label="Mean inner timer (lower)")
    axes[0].errorbar(x, means, yerr=[np.zeros(len(x)), upper_means - means], fmt="none", color="#253249", capsize=5,
        label="Mean enclosing worker bound (upper)")
    for j, tid in enumerate(tasks):
        lower = np.asarray([costs[(tid, ep)]["cold_lower_bound_s"] for ep in ENDPOINTS])
        upper = np.asarray([costs[(tid, ep)]["cold_upper_bound_s"] for ep in ENDPOINTS])
        axes[0].errorbar(x + (j - 1.5) * .055, lower, yerr=[np.zeros(len(x)), upper - lower], fmt="o", label=tid, markersize=3, linewidth=.8, alpha=.65)
    axes[0].set(xticks=x, xticklabels=ENDPOINTS, ylabel="Observed cold planning bracket [s]", title="Inner timer lower / worker enclosing upper")
    axes[0].legend(fontsize=6); axes[0].grid(axis="y", alpha=.2)
    for j, pref in enumerate(PREFERENCES):
        counts = [sum(r["full_task_success"] and r["original_independent_gates_passed"] and
            (pref != "B" or r["clearance_30mm_met"] is True) for r in slots if r["endpoint"] == ep and r["preference"] == pref) for ep in ENDPOINTS]
        axes[1].bar(x + (j - .5) * .35, counts, width=.35, label=f"{pref}: actual Task + 5 gates" + (" + 30mm" if pref == "B" else ""))
    axes[1].set(xticks=x, xticklabels=ENDPOINTS, ylabel="Qualifying actual Tasks / 4", ylim=(0, 4.5), yticks=range(5), title="All failures and NO_PLAN retained in denominator")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.2)
    fig.suptitle(f"Worker startup/tail bracket under shared {phase['workers']}-worker load; no isolated repeat | warm path is an estimate")
    cost_chart = folder / "planning_cost_actual_results.png"; fig.savefig(cost_chart, dpi=155); plt.close(fig)
    return {"budget_quality_coverage": budget_chart.relative_to(output).as_posix(),
        "planning_cost_actual_results": cost_chart.relative_to(output).as_posix()}, costs


def dashboard(output, payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C.2 preference-conditioned initialization</title><style>
body{font:15px system-ui,sans-serif;background:#f4f6fa;color:#253249;max-width:1400px;margin:auto;padding:24px}h1{font-size:28px;margin-bottom:8px}p{line-height:1.6}section{background:white;border:1px solid #dce2eb;border-radius:12px;padding:20px;margin:20px 0}.charts,.figures{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{width:100%;height:auto}video{width:100%;max-height:760px;background:#111827}select{padding:8px;margin:5px;max-width:40vw;border:1px solid #aabbcc;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e5eaf1}.pass{color:#148067}.fail{color:#b25c25}.badge{background:#e7edf5;padding:3px 8px;border-radius:6px}.scroll{overflow:auto}#status{font-size:16px;padding:12px 0}@media(max-width:800px){.charts,.figures{grid-template-columns:1fr}body{padding:12px}}
</style><h1>V6.4-C.2 · Preference-conditioned initialization</h1>
<p>R8 / R12: frozen rule search. N8: TRAIN-only retrieval. D8: trained Diffusion initialization + C.1 search + original control. Four TEST Tasks; A/B share each search pool. Prefix predictions, final actual execution and independent replay remain separate evidence.</p>
<p id="conclusion"></p>
<p><span class="badge">deployment NOT_MET</span> <span class="badge">continuous-time safety NOT_ESTABLISHED</span> <span class="badge">hardware safety NOT_ESTABLISHED</span></p>
<section><div class="charts"><img id="budget"><img id="cost"></div><p>Cold planning is an observed bracket: the inner planner timer is the lower bound; the enclosing worker dispatch-to-return interval supplies the upper bound, including startup and post-selection tail. R8 carries the measured R-worker residual around its sealed eighth-slot prefix. Runs share four workers and hardware resources; these are not isolated repeated-task latency measurements. Warm paths are decomposition estimates. Protocol budget reductions alone do not establish speedup.</p></section>
<section><label>Task <select id="task"></select></label><label>Endpoint <select id="endpoint"></select></label><label>Preference <select id="preference"></select></label><label>View <select id="view"></select></label>
<div id="status"></div><p id="costNote"></p><video id="video" controls preload="metadata"></video><p id="videoNote"></p><div id="downloads"></div><div class="figures"><img id="trajectory"><img id="tracking"></div></section>
<section><h2>All 32 logical actual slots</h2><p>Aliases link the same unique actual media and contribute no second physical run. Failed prefixes end at their exact saved state; NO_PLAN has zero execution steps and no video. Tracking is descriptive and does not replace the original five independent gates.</p><div class="scroll"><table><thead><tr><th>Task</th><th>Endpoint / pref</th><th>Status</th><th>Steps</th><th>Task + five gates</th><th>B ≥30mm</th><th>I [rad/s]</th><th>L [m]</th><th>d [mm]</th><th>Cold bracket [s]</th><th>Media source</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<script>const D=__DATA__;const $=id=>document.getElementById(id);function esc(v){const p=document.createElement('span');p.textContent=String(v);return p.innerHTML}function fmt(x,d=4){return x==null?'—':Number(x).toFixed(d)}function options(id,values){$(id).innerHTML=values.map(v=>'<option>'+esc(v)+'</option>').join('')}
$('conclusion').textContent='Research delivery complete: '+D.summary?.research_delivery_complete+'; trained initializer operational: '+D.summary?.learned_initializer_operational+'; learning benefit: '+D.summary?.learning_benefit_established_in_pilot+'; default: '+D.summary?.default_initializer_decision+'.';
$('budget').src=D.charts.budget_quality_coverage;$('cost').src=D.charts.planning_cost_actual_results;options('task',D.tasks);options('endpoint',['R8','R12','N8','D8']);options('preference',['A','B']);options('view',['five_view_grid','continuum_focus','overview','front','side','top','iso']);
function show(){const r=D.slots.find(r=>r.task_id===$('task').value&&r.endpoint===$('endpoint').value&&r.preference===$('preference').value);if(!r)return;const v=r.media?.videos?.[$('view').value];$('video').style.display=v?'block':'none';if(v){if(!$('video').src.endsWith(v)){$('video').src=v;$('video').poster=($('view').value==='continuum_focus'?r.media.focus_preview:r.media.preview)||''}}else{$('video').removeAttribute('src');$('video').load()}
$('status').textContent=r.status+' | '+r.actual_steps+' saved steps | '+(r.alias_of_method?'alias of '+r.alias_of_method:r.unique_run?'unique actual':'no physical run');$('status').className=r.full_task_success?'pass':'fail';$('videoNote').textContent=r.media?'Saved endpoint '+fmt(r.media.source_end_s,3)+' s; exact endpoint included. All frames use saved actual states.':r.actual_steps?'Video not rendered for this actual.':'No saved execution states: '+r.status;for(const name of ['trajectory','tracking']){$(name).style.display=r.figures?'block':'none';if(r.figures)$(name).src=r.figures[name]}
$('costNote').textContent=r.planning_cold_bracket?'Observed cold planning bracket: '+fmt(r.planning_cold_bracket.cold_lower_bound_s,2)+'–'+fmt(r.planning_cold_bracket.cold_upper_bound_s,2)+' s (inner timer to enclosing worker interval; startup/tail bracketed). Shared '+r.planning_cold_bracket.worker_count+'-worker load; warm path is an estimate.':'Cold timing bracket unavailable.';
$('downloads').innerHTML=Object.entries(r.media?.videos||{}).map(([name,path])=>'<a href="'+esc(path)+'">'+esc(name)+'</a>').join(' · ')+(r.figures?' · <a href="'+esc(r.figures.tracking_csv)+'">native tracking CSV</a>':'')}
$('rows').innerHTML=D.slots.map(r=>{const q=r.full_task_success?r.quality||{}:{};const c=r.planning_cold_bracket;return '<tr><td>'+esc(r.task_id)+'</td><td>'+esc(r.endpoint+' / '+r.preference)+'</td><td class="'+(r.full_task_success?'pass':'fail')+'">'+esc(r.status)+'</td><td>'+r.actual_steps+'</td><td>'+((r.full_task_success&&r.original_independent_gates_passed)?'PASS':'FAIL')+'</td><td>'+(r.clearance_30mm_met===true?'YES':r.clearance_30mm_met===false?'NO':'—')+'</td><td>'+fmt(q.I_support)+'</td><td>'+fmt(q.L_full)+'</td><td>'+fmt(q.d_support==null?null:q.d_support*1000,2)+'</td><td>'+(c?fmt(c.cold_lower_bound_s,1)+'–'+fmt(c.cold_upper_bound_s,1):'—')+'</td><td>'+esc(r.alias_of_method||r.method)+'</td></tr>'}).join('');for(const id of ['task','endpoint','preference','view'])$(id).addEventListener('change',show);show();</script></html>'''
    (Path(output) / "index.html").write_text(page.replace("__DATA__", encoded), encoding="utf8")


def build(run, output, *, render=False, check_inputs=False, workers=2, fps=15., width=640, height=480, focus_width=960, focus_height=720):
    started = time.perf_counter(); run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("media output must be separate from frozen scientific run")
    slots = terminal_slots(run); output.mkdir(parents=True, exist_ok=True)
    request = {"schema": SCHEMA, "plan_sha256": sha(run / "plan.json"), "actual_complete_sha256": sha(run / "actual_complete.json"),
        "search_phase_sha256": sha(run / "search_phase.json"),
        "fps": fps, "width": width, "height": height, "focus_width": focus_width, "focus_height": focus_height}
    request_path = output / "build_request.json"
    if request_path.exists() and read(request_path) != request:
        raise ValueError("media directory belongs to a different frozen run or rendering configuration")
    if not request_path.exists():
        if any(output.iterdir()): raise FileExistsError("exclusive new C.2 media output required")
        write(request_path, request)
    unique = [s for s in slots if not s.get("alias_of_method") and s["actual_steps"] > 0]
    checks = []
    for slot in unique:
        loaded = load_actual(run, slot)
        checks.append({"task_id": slot["task_id"], "method": slot["method"], "actual_fresh_parity": loaded[-2],
            "source_end_s": float(loaded[4]["time"][-1]), "sources": loaded[-1]})
    write(output / "input_checks.json", {"schema": SCHEMA, "status": "PASS", "unique_saved_actuals": len(unique),
        "logical_slots": len(slots), "checks": checks, "physics_steps": 0, "geometry_queries": 0, "QP_solves": 0})
    if check_inputs: return {"status": "PASS", "unique_saved_actuals": len(unique), "logical_slots": len(slots)}
    figures = {(s["task_id"], s["method"]): trajectory_figures(run, output, s) for s in unique}
    charts, costs = core_charts(run, output, slots); videos = {}
    if render:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ValueError("ffmpeg and ffprobe must be available for requested video refresh")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(render_actual, run, output, s, fps, width, height, focus_width, focus_height): s for s in unique}
            for future in as_completed(pending):
                row = future.result(); videos[(row["task_id"], row["method"])] = row
                print(json.dumps({"event": "C2_MEDIA", "task": row["task_id"], "method": row["method"],
                    "videos": len(row["videos"]), "saved_endpoint_s": row["source_end_s"]}), flush=True)
    else:
        for s in unique:
            path = output / "replays" / (s["task_id"] + "_" + s["method"]) / "replay_metadata.json"
            if path.exists(): videos[(s["task_id"], s["method"])] = read(path)
    display_slots = []
    for s in slots:
        original = resolve_unique(slots, s["task_id"], s["method"]); key = (original["task_id"], original["method"])
        display_slots.append({**s, "figures": figures.get(key), "media": videos.get(key),
            "planning_cold_bracket": costs[(s["task_id"], s["endpoint"])], "media_source_method": original["method"]})
    payload = {"schema": SCHEMA, "tasks": list(dict.fromkeys(s["task_id"] for s in slots)), "charts": charts,
        "slots": display_slots, "summary": read(run / "summary.json") if (run / "summary.json").exists() else None}
    write(output / "dashboard_data.json", payload); dashboard(output, payload)
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "run_identity": request,
        "generator": record(__file__), "logical_actual_slots": len(slots), "unique_actual_media": len(videos),
        "video_count": sum(len(row["videos"]) for row in videos.values()), "alias_slots": sum(bool(s.get("alias_of_method")) for s in slots),
        "no_plan_slots": sum(s["status"] == "NO_PLAN" for s in slots), "wall_s": time.perf_counter() - started,
        "files": {p.relative_to(output).as_posix(): record(p, output) for p in output.rglob("*") if p.is_file() and p.name != "visualization_manifest.json"},
        "physics_steps_executed": 0, "geometry_queries": 0, "QP_solves": 0, "optimizer_updates": 0,
        "new_model_samples": 0, "historical_outputs_overwritten": False, "visualization_is_new_acceptance": False}
    write(output / "visualization_manifest.json", manifest)
    return {"status": "PASS", "output": str(output), "logical_slots": len(slots), "unique_actual_media": len(videos),
        "video_count": manifest["video_count"], "physics_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true"); parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2); parser.add_argument("--fps", type=float, default=15.)
    parser.add_argument("--width", type=int, default=640); parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--focus-width", type=int, default=960); parser.add_argument("--focus-height", type=int, default=720)
    args = parser.parse_args()
    if args.check_inputs and args.render: parser.error("check-inputs cannot render")
    if not np.isfinite(args.fps) or not 0. < args.fps <= 60.: parser.error("FPS must be in (0,60]")
    if min(args.width, args.focus_width) < 640 or min(args.height, args.focus_height) < 480 or any(v % 2 for v in (args.width, args.height, args.focus_width, args.focus_height)):
        parser.error("even dimensions of at least 640x480 required")
    print(json.dumps(build(args.run, args.output, render=args.render, check_inputs=args.check_inputs, workers=args.workers,
        fps=args.fps, width=args.width, height=args.height, focus_width=args.focus_width, focus_height=args.focus_height)), flush=True)


if __name__ == "__main__": main()
