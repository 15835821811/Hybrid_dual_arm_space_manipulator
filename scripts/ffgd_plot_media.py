"""Plot exported actual records with matplotlib; never import the simulator.

Use the optional isolated media environment from requirements-media-lock.txt.
All CSV samples are plotted. No success judgment is produced by this script.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plot(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('PLOT_OUTPUT_MUST_BE_NEW_AND_OUTSIDE_SOURCE')
    inputs = [source / p for p in ('native_samples.csv', 'interval_errors.csv', 'pose_errors.csv', 'source_task.json', 'source_execution.json', 'media_manifest.json')]
    before = {str(p): digest(p) for p in inputs}
    manifest = json.loads((source / 'media_manifest.json').read_text())
    task = json.loads((source / 'source_task.json').read_text())['task']
    execution = json.loads((source / 'source_execution.json').read_text())
    data = np.atleast_1d(np.genfromtxt(source / 'native_samples.csv', delimiter=',', names=True, filling_values=np.nan))
    intervals, poses = rows(source / 'interval_errors.csv'), rows(source / 'pose_errors.csv')
    t = data['local_time_s']
    def matrix(prefix, count):
        return np.stack([data[f'{prefix}_{i}'] for i in range(count)], axis=1)
    positions = matrix('surface_centers_xyz_m', 6).reshape(-1, 2, 3)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .2, 'svg.hashsalt': 'ffgd-actual-records'})
    saved = []
    def save(fig, name):
        fig.suptitle(f"Saved actual execution: {Path(manifest['execution']).name}\n{execution['status']} | source acceptance: {manifest['source_acceptance_status']} | display only", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .94))
        for extension in ('png', 'svg'):
            path = output / f'{name}.{extension}'
            fig.savefig(path, dpi=160)
            saved.append(path.name)
        plt.close(fig)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(221, projection='3d')
    for tip, color, label in [(0, '#157ea8', 'Continuum surface center'), (1, '#c24c29', 'Rigid surface center')]:
        P = positions[:, tip]
        ax.plot(*P.T, color=color, label=label)
        goals = np.array([p['position'] for p in task.get('poses', []) if p['tip'] == tip and p.get('position_enabled', True)])
        if len(goals):
            ax.scatter(*goals.T, marker='x', color=color, s=35)
    ax.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)', title='Actual paths; crosses = public pose requests')
    ax.legend(fontsize=7)
    ap, ar = fig.add_subplot(222), fig.add_subplot(223)
    for group in sorted({r['interval_index'] for r in intervals}, key=int):
        selected = [r for r in intervals if r['interval_index'] == group]
        x = [float(r['local_time_s']) for r in selected]
        label = f"Interval {group}, tip {selected[0]['tip']}"
        if selected[0]['position_enabled'] == 'True':
            line, = ap.plot(x, [1000 * float(r['position_error_m']) for r in selected], label=label)
            ap.plot(x, [1000 * float(r['position_tolerance_m']) for r in selected], '--', color=line.get_color(), alpha=.6)
        if selected[0]['rotation_enabled'] == 'True':
            line, = ar.plot(x, [np.degrees(float(r['rotation_error_rad'])) for r in selected], label=label)
            ar.plot(x, [np.degrees(float(r['rotation_tolerance_rad'])) for r in selected], '--', color=line.get_color(), alpha=.6)
    for p in poses:
        if p['covered'] != 'True':
            continue
        if p['position_enabled'] == 'True':
            ap.scatter(float(p['local_time_s']), 1000 * float(p['position_error_m']), color='black', marker='x', s=25)
            ap.scatter(float(p['local_time_s']), 1000 * float(p['position_tolerance_m']), color='#157ea8', marker='_', s=80)
        if p['rotation_enabled'] == 'True':
            ar.scatter(float(p['local_time_s']), np.degrees(float(p['rotation_error_rad'])), color='black', marker='x', s=25)
            ar.scatter(float(p['local_time_s']), np.degrees(float(p['rotation_tolerance_rad'])), color='#157ea8', marker='_', s=80)
    ap.set(xlabel='Local time (s)', ylabel='Position error (mm)', title='Task errors; dashed = declared interval limits')
    ar.set(xlabel='Local time (s)', ylabel='Rotation error (deg)', title='Pose marks: black errors, blue limits; brackets recorded in CSV')
    for a in (ap, ar):
        a.set_xlim(0, max(t[-1], 1e-6) * 1.02)
        a.set_ylim(bottom=0)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=6)
    ac = fig.add_subplot(224)
    ac.plot(t, 1000 * data['clearance_m'], color='#157ea8')
    ac.axhline(0, color='red', linestyle='--')
    ac.set(xlabel='Local time (s)', ylabel='Minimum clearance (mm)', title=f"Saved whole-body clearance; minimum {1000 * min(data['clearance_m']):.4g} mm")
    save(fig, 'task_and_geometry')

    q = matrix('actual_q17_rad', 17)
    ref = matrix('reference_q17_rad', 17)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for a, indices, title in [(axes[0], range(10), 'Continuum q17 coordinates'), (axes[1], range(10, 17), 'Rigid q17 coordinates')]:
        for i in indices:
            color = plt.get_cmap('tab20')(i)
            a.plot(t, q[:, i], color=color, label=str(i))
            a.plot(t, ref[:, i], '--', color=color, linewidth=.7)
        a.set(ylabel='Radians', title=title + '; solid actual, dashed reference')
        a.legend(ncol=10, fontsize=7)
    axes[2].plot(t, matrix('actual_q67_rad', 67), linewidth=.5)
    axes[2].set(ylabel='Radians', title='All 67 actual joint coordinates (no projection)')
    axes[3].plot(t, np.max(abs(q - ref), axis=1), label='max |q17 actual-reference|')
    axes[3].plot(t, np.max(abs(matrix('manifold_residual_rad', 67)), axis=1), label='max |q67-B q17|')
    axes[3].set(xlabel='Local time (s)', ylabel='Radians', title='Tracking and manifold residual')
    axes[3].legend()
    save(fig, 'joint_tracking')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(t, matrix('raw_torque_Nm', 67), linewidth=.5)
    axes[0, 0].set(ylabel='Raw torque (Nm)', title=f"All 67 actuators; clipped steps: {execution['clipped_control_steps']}")
    axes[0, 1].plot(t, matrix('applied_control_Nm', 67), linewidth=.5)
    axes[0, 1].set(ylabel='Applied torque (Nm)', title=f"Recorded controls; safety abort: {execution['safety_abort']}")
    base = matrix('base_xyz_wxyz', 7)
    axes[1, 0].plot(t, base[:, :3] - base[0, :3], label=['X', 'Y', 'Z'])
    axes[1, 0].set(ylabel='Base translation from first state (m)', title='Actual floating base (no resetting)')
    axes[1, 0].legend()
    H = matrix('momentum_residual', 6)
    a = axes[1, 1]
    a.plot(t, np.linalg.norm(H[:, :3], axis=1), color='#157ea8')
    a.set(ylabel='|P-P0| (kg m/s)', title='Saved physical momentum change\nExternal forcing may contribute')
    b = a.twinx()
    b.plot(t, np.linalg.norm(H[:, 3:], axis=1), color='#c24c29')
    b.set_ylabel('|L-L0| (kg m^2/s)', color='#c24c29')
    for a in axes.ravel():
        a.set_xlabel('Local time (s)')
    save(fig, 'actuators_base_momentum')
    if before != {str(p): digest(p) for p in inputs}:
        raise RuntimeError('SOURCE_MEDIA_CHANGED_DURING_PLOTTING')
    report = dict(kind='READ_ONLY_PLOTS_NOT_NEW_ACCEPTANCE', source_hashes=before, source_unchanged=True,
                  source_native_samples=len(data), plotted_native_samples=len(data), subsampled=False,
                  files=saved, matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                  generator_sha256=digest(Path(__file__)))
    (output / 'plot_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return dict(output=str(output), files=saved, plotted_samples=len(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.media, args.output), indent=2))
