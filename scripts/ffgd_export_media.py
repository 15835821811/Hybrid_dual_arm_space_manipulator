"""Read-only views of an execution log; never planning, stepping, or replay.

Run with the project Python. Optional video needs ffmpeg on PATH or --ffmpeg.
The output is a new directory. Source hashes are checked after export.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ffgd.problem_io import load_problem
from ffgd.execution import refine_physics
from ffgd.trajectory import basis
from ffgd.temporal import target_body, world_goal, interpolate_qpos


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def export(execution, output, fps=15, ffmpeg=None, videos=False, geometry_pairs=False):
    execution, output = Path(execution).resolve(), Path(output).resolve()
    if output == execution or execution in output.parents:
        raise ValueError('OUTPUT_MUST_BE_OUTSIDE_SOURCE_EXECUTION')
    if not 1 <= fps <= 60:
        raise ValueError('FPS_OUT_OF_RANGE')
    provenance = read_json(execution / 'provenance.json')
    summary = read_json(execution / 'execution.json')
    planning = Path(provenance['planning_directory']).resolve()
    sources = [execution / p for p in ('execution.npz', 'execution.json', 'provenance.json', 'acceptance.json')]
    sources += [planning / 'problem.json', planning / 'manifest.json']
    original = {str(p): digest(p) for p in sources}
    robot, initial, task, public = load_problem(planning / 'problem.json')
    r, _ = refine_physics(robot, initial, provenance['physics_substeps'])
    if r.model_hash != provenance['execution_model_hash'] or summary['model_hash'] != r.model_hash:
        raise ValueError('SAVED_MODEL_IDENTITY_MISMATCH')
    if summary['candidate_hash'] != provenance['candidate_hash']:
        raise ValueError('SAVED_CANDIDATE_IDENTITY_MISMATCH')
    with np.load(execution / 'execution.npz', allow_pickle=False) as archive:
        saved = {k: archive[k] for k in archive.files}
    states = saved['states']
    dt, duration = float(saved['physics_dt']), float(saved['duration'])
    count = len(states)
    if count != len(saved['ctrl']) + 1 or dt != provenance['physics_dt']:
        raise ValueError('INCONSISTENT_NATIVE_SAMPLE_COUNTS_OR_TIMESTEP')
    output.mkdir(parents=True, exist_ok=False)
    # These are source records, not new acceptance or replay conclusions.
    (output / 'source_task.json').write_text(json.dumps(public, indent=2), encoding='utf-8')
    for filename in ('execution.json', 'acceptance.json'):
        shutil.copyfile(execution / filename, output / ('source_' + filename))
    data = mujoco.MjData(r.model)
    mask = int(saved['mask'])
    if mask != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError('INCOMPLETE_SAVED_STATE_MASK')
    if states.shape[1] != mujoco.mj_stateSize(r.model, mask):
        raise ValueError('SAVED_STATE_SIZE_MISMATCH')
    C = saved['control_points']
    times = np.arange(count) * dt
    # A display loop may restore saved states. This is explicitly NOT replay.
    def restore(index):
        mujoco.mj_setState(r.model, data, states[index], mask)
        mujoco.mj_kinematics(r.model, data)
        return data
    def tips():
        positions = data.xpos[r.tips].copy()
        rotations = data.xmat[r.tips].reshape(2, 3, 3).copy()
        positions[0] += rotations[0] @ np.array([.0475, 0, 0])
        return positions, rotations
    def names(prefix, n):
        return [f'{prefix}_{i}' for i in range(n)]
    headers = ['native_index', 'local_time_s', 'absolute_time_s', 'clearance_m']
    headers += names('base_xyz_wxyz', 7) + names('actual_q17_rad', 17) + names('actual_q67_rad', 67)
    headers += names('reference_q17_rad', 17) + names('reference_v17_rad_s', 17) + names('reference_a17_rad_s2', 17)
    headers += names('raw_torque_Nm', 67) + names('applied_control_Nm', 67)
    headers += names('momentum_residual', 6) + names('manifold_residual_rad', 67)
    headers += names('surface_centers_xyz_m', 6) + names('surface_rotations_row_major', 18)
    with (output / 'native_samples.csv').open('w', newline='', encoding='utf-8') as stream, (output / 'interval_errors.csv').open('w', newline='', encoding='utf-8') as intervals:
        writer, iw = csv.writer(stream), csv.writer(intervals)
        writer.writerow(headers)
        iw.writerow(['native_index', 'local_time_s', 'interval_index', 'tip', 'frame', 'position_error_m', 'rotation_error_rad', 'position_enabled', 'rotation_enabled', 'position_tolerance_m', 'rotation_tolerance_rad', 'goal_x_m', 'goal_y_m', 'goal_z_m'])
        for i, t in enumerate(times):
            restore(i)
            positions, rotations = tips()
            low = data.qpos[r.qpos_ids]
            refs = [basis(len(C), float(t), duration, order) @ C for order in range(3)]
            row = [i, t, data.time, saved['clearance'][i]]
            row += data.qpos[r.base_qpos].tolist() + (r.Bplus @ low).tolist() + low.tolist()
            for ref in refs:
                row += ref.tolist()
            # No command is applied after the final saved state.
            row += saved['raw_torque'][i].tolist() if i < count - 1 else [''] * 67
            row += saved['ctrl'][i][r.actuator_ids].tolist() if i < count - 1 else [''] * 67
            row += saved['momentum_residual'][i].tolist() + saved['manifold_residual'][i].tolist()
            row += positions.ravel().tolist() + rotations.ravel().tolist()
            writer.writerow(row)
            for j, interval in enumerate(task.intervals):
                if interval.start <= t <= interval.end:
                    P, R = world_goal(interval, data.xpos, data.xmat.reshape(-1, 3, 3), target_body(r))
                    iw.writerow([i, t, j, interval.tip, interval.frame, np.linalg.norm(positions[interval.tip] - P),
                                 Rotation.from_matrix(R.T @ rotations[interval.tip]).magnitude(), interval.position_enabled, interval.rotation_enabled,
                                 interval.position_tolerance, interval.rotation_tolerance, *P])
    with (output / 'pose_errors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['local_time_s', 'pose_index', 'tip', 'position_error_m', 'rotation_error_rad', 'position_enabled', 'rotation_enabled', 'position_tolerance_m', 'rotation_tolerance_rad', 'interpolation_bracket_s', 'covered'])
        for j, pose in enumerate(task.poses):
            if pose.time > times[-1]:
                writer.writerow([pose.time, j, pose.tip, '', '', pose.position_enabled, pose.rotation_enabled, pose.position_tolerance, pose.rotation_tolerance, '', False])
                continue
            right = min(int(np.searchsorted(times, pose.time)), count - 1)
            left = max(0, right - 1)
            qpos = np.array([restore(i).qpos.copy() for i in (left, right)])
            data.qpos[:], bracket = interpolate_qpos(r, times[[left, right]], qpos, pose.time)
            mujoco.mj_kinematics(r.model, data)
            positions, rotations = tips()
            writer.writerow([pose.time, j, pose.tip, np.linalg.norm(positions[pose.tip] - pose.position),
                             Rotation.from_matrix(pose.rotation.T @ rotations[pose.tip]).magnitude(), pose.position_enabled, pose.rotation_enabled,
                             pose.position_tolerance, pose.rotation_tolerance, bracket, True])
    pair_report = dict(status='NOT_RUN')
    if geometry_pairs:
        from ffgd.geometry import Geometry
        geometry = Geometry(r)
        error = 0.
        with (output / 'geometry_pairs.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['native_index', 'local_time_s', 'clearance_m', 'geom_a', 'geom_b', 'name_a', 'name_b', 'native_queries', 'capped'])
            for i, t in enumerate(times):
                scan = geometry.native_scan(restore(i))
                error = max(error, abs(scan['minimum'] - float(saved['clearance'][i])))
                a, b = scan['pair'] if scan['pair'] is not None else (-1, -1)
                names_pair = [mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_GEOM, k) if k >= 0 else '' for k in (a, b)]
                writer.writerow([i, t, scan['minimum'], a, b, *names_pair, scan['queries'], scan['capped']])
        if error > 1e-10:
            raise ValueError('DISPLAY_GEOMETRY_DIFFERS_FROM_SAVED_CLEARANCE')
        pair_report = dict(status='SAVED_STATE_REQUERY_COMPLETE', maximum_saved_clearance_difference=error, policy_hash=geometry.policy_hash)
    video_records = []
    if videos:
        executable = ffmpeg or shutil.which('ffmpeg')
        if not executable:
            raise FileNotFoundError('FFMPEG_REQUIRED_FOR_VIDEO')
        indices = np.rint(np.linspace(0, count - 1, max(2, round(times[-1] * fps) + 1))).astype(int)
        np.savetxt(output / 'video_frame_indices.csv', np.c_[np.arange(len(indices)), indices, times[indices]],
                   delimiter=',', header='video_frame,native_index,local_time_s', comments='')
        width, height = 640, 480
        camera = mujoco.MjvCamera()
        camera.lookat[:] = r.model.stat.center
        camera.distance = max(2., 2.5 * r.model.stat.extent)
        renderer = mujoco.Renderer(r.model, height=height, width=width)
        # Windows ffmpeg often ships without fontconfig; use the installed
        # font explicitly, without copying it into the delivery bundle.
        font = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/arial.ttf'
        font_option = "fontfile='" + font.as_posix().replace(':', r'\:') + "':" if os.name == 'nt' and font.is_file() else ''
        try:
            for label, azimuth, elevation in [('overview', 135, -30), ('opposite', -45, -20)]:
                camera.azimuth, camera.elevation = azimuth, elevation
                target = output / f'actual_{label}.mp4'
                command = [str(executable), '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                           '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '1',
                           '-pix_fmt', 'yuv420p', '-vf', 'drawtext=' + font_option + 'text=SAVED ACTUAL STATES - NOT REPLAY:fontcolor=white:box=1:boxcolor=black:x=8:y=8', str(target)]
                with (output / f'{label}_ffmpeg.log').open('wb') as log:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL)
                    try:
                        for i in indices:
                            restore(int(i))
                            renderer.update_scene(data, camera=camera)
                            process.stdin.write(renderer.render().tobytes())
                        process.stdin.close()
                        if process.wait(timeout=60) != 0:
                            raise RuntimeError('FFMPEG_FAILED_SEE_SAVED_LOG')
                    finally:
                        if process.stdin and not process.stdin.closed:
                            try:
                                process.stdin.close()
                            except BrokenPipeError:
                                pass
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                video_records.append(dict(file=target.name, frames=len(indices), fps=fps, actual_state_frames_only=True))
        finally:
            renderer.close()
    after = {str(p): digest(p) for p in sources}
    if original != after:
        raise RuntimeError('SOURCE_EVIDENCE_CHANGED_DURING_EXPORT')
    report = dict(kind='READ_ONLY_MEDIA_EXPORT_NOT_EXECUTION_OR_REPLAY', execution=str(execution), planning=str(planning),
                  source_hashes=original, source_files_unchanged=True, native_steps_executed=0, saved_states=count,
                  full_state_csv=True, csv_units='SI; angles radians; quaternion wxyz; world surface centers',
                  terminal_control_fields='blank; no command after final state', interval_error_scope='SAVED_NATIVE_GRID_ONLY_NOT_NEW_ACCEPTANCE',
                  task_poses='source_task.json; no interpolated hidden path', source_execution_status=summary['status'],
                  source_acceptance_status=read_json(execution / 'acceptance.json')['status'],
                  source_method=provenance['method'], source_candidate_hash=summary['candidate_hash'],
                  source_planning_manifest=read_json(planning / 'manifest.json'), videos=video_records, geometry_pair_export=pair_report,
                  generator_sha256=digest(Path(__file__)))
    (output / 'media_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return dict(output=str(output), states=count, videos=len(video_records), source_unchanged=True, new_physics_steps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--videos', action='store_true')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--geometry-pairs', action='store_true', help='Re-query geometry on each saved state; no physics stepping')
    args = parser.parse_args()
    print(json.dumps(export(args.execution, args.output, args.fps, args.ffmpeg, args.videos, args.geometry_pairs), indent=2))
