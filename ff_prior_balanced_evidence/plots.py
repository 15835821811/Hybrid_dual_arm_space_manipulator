"""Observed body-distance and continuum-shape figures; no causal comparison."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ff_prior.storage import read_json


def time_cell_edges(times):
    """Midpoint cell boundaries preserve nonuniform actual sample times."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('INVALID_BODY_PLOT_TIMES')
    if len(times) == 1:
        # A zero-duration recorded prefix has only an isolated observation.
        return np.array([times[0]-1e-9, times[0]+1e-9])
    if np.any(np.diff(times) <= 0):
        raise ValueError('NONINCREASING_BODY_PLOT_TIMES')
    return np.r_[times[0], times[:-1]+.5*np.diff(times), times[-1]]


class BodySeries:
    def __init__(self, directories):
        self.parts = []
        self.meta = None
        self.complete = True
        self.source_statuses = []
        for directory in directories:
            directory = Path(directory)
            mp = directory/'body_obstacle_metadata.json'
            if not mp.exists():
                self.complete = False
                self.source_statuses.append(dict(directory=str(directory), status='MISSING'))
                continue
            meta = read_json(mp)
            self.source_statuses.append(dict(directory=str(directory), status=meta['status'],
                samples=meta.get('samples'), expected_samples=meta.get('expected_samples')))
            self.complete = self.complete and meta['status'] == 'COMPLETE'
            if not meta.get('samples'):
                continue
            if self.meta is not None and (meta['links'] != self.meta['links'] or meta['obstacles'] != self.meta['obstacles']):
                raise ValueError('BODY_MEDIA_SCENE_OR_TOPOLOGY_MISMATCH')
            self.meta = meta
            with np.load(directory/'body_obstacle_trace.npz') as z:
                self.parts.append({k: z[k] for k in z.files})
        self.a = {}
        if self.parts:
            for key in self.parts[0]:
                self.a[key] = np.concatenate([part[key] for part in self.parts])
            # The continuous handoff can be observed by both segment recorders.
            order = np.argsort(self.a['time'], kind='stable')
            order = order[np.r_[True, np.diff(self.a['time'][order]) > 1e-9]]
            self.a = {k: v[order] for k, v in self.a.items()}

    def index(self, t, tolerance=1e-7):
        if not self.a:
            return None
        times = self.a['time']
        j = min(len(times)-1, int(np.searchsorted(times, t)))
        if j and abs(times[j-1]-t) < abs(times[j]-t):
            j -= 1
        return j if abs(times[j]-t) <= tolerance else None

    def frame(self, t):
        j = self.index(t)
        if j is None:
            return None
        links, obstacles = self.meta['links'], self.meta['obstacles']
        nearest = []
        for oi, obs in enumerate(obstacles):
            li = int(np.argmin(self.a['distance_m'][j, oi]))
            geom = int(self.a['nearest_geom_id'][j, oi, li])
            link = links[li]
            nearest.append(dict(obstacle=obs['object_id'], body_id=link['body_id'],
                body_name=link['body_name'], arm=link['arm'], geom_id=geom,
                geom_name=self.meta['geom_names'][geom], distance_m=float(self.a['distance_m'][j, oi, li]),
                connector=self.a['connector'][j, oi, li].tolist()))
        return dict(time=float(self.a['time'][j]), nearest=nearest)


def _save(fig, out, name, files):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        path = out/f'{name}.{ext}'
        fig.savefig(path, dpi=155, bbox_inches='tight')
        files.append(path.name)
    plt.close(fig)


def generate_body_figures(out, series):
    out = Path(out)
    files = []
    if not series.a:
        return dict(status='BLOCKED_NO_BODY_DIAGNOSTICS', files=[])
    a, meta = series.a, series.meta
    times = a['time']
    links, objects = meta['links'], meta['obstacles']
    labels = [link['body_name'] for link in links]
    distance = a['distance_m']
    minimum = distance.min(axis=1)
    time_edges = time_cell_edges(times)
    fig, ax = plt.subplots(figsize=(15, 8))
    image = ax.pcolormesh(time_edges, np.arange(len(links)+1)-.5, (1000*minimum).T,
        shading='flat', rasterized=True, cmap='viridis', vmin=0, vmax=100)
    ax.set(title='Exact obstacle distance at every saved control tick, by physical arm body',
        xlabel='absolute simulation time (s)', ylabel='collision body')
    stride = max(1, len(links)//30)
    ax.set_yticks(np.arange(0, len(links), stride), labels[::stride], fontsize=7)
    fig.colorbar(image, ax=ax, label='minimum over 7 obstacles (mm); color saturates at 100 mm only')
    _save(fig, out, '09_body_distance_heatmap', files)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for oi, obj in enumerate(objects):
        axes[0].plot(times, distance[:, oi].min(axis=1)*1000, label=obj['object_id'])
        axes[1].step(times, distance[:, oi].argmin(axis=1), where='post', label=obj['object_id'])
    axes[0].axhline(0, color='red', label='physical gate 0 mm')
    axes[0].axhline(10, color='grey', ls='--', label='display band only 10 mm')
    axes[0].set(ylabel='nearest body distance (mm)', title='Each obstacle against the entire two-arm shape')
    axes[1].set(yticks=np.arange(0, len(links), stride), yticklabels=labels[::stride],
        ylabel='nearest collision body', xlabel='absolute simulation time (s)')
    axes[1].tick_params(axis='y', labelsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(ncol=5, fontsize=8)
    _save(fig, out, '10_obstacle_nearest_body', files)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    im = axes[0].pcolormesh(time_edges, np.arange(29)+1.5, a['continuum_curvature'].T,
        shading='flat', rasterized=True, cmap='magma')
    fig.colorbar(im, ax=axes[0], label='turn angle / mean chord (1/m)')
    axes[0].set(title='Observed continuum centerline discrete curvature (not material strain)', ylabel='middle link number')
    axes[1].plot(times, a['continuum_curvature'])
    axes[1].set(ylabel='1/m', xlabel='absolute simulation time (s)', title='All 28 curvature observations')
    _save(fig, out, '11_continuum_curvature', files)
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection='3d')
    bx = fig.add_subplot(122)
    ids = np.unique(np.linspace(0, len(times)-1, min(8, len(times))).round().astype(int))
    for j in ids:
        skeleton = a['body_positions'][j, meta['continuum_body_ids']]
        ax.plot(*skeleton.T, label=f'{times[j]:.2f}s')
        bx.plot(np.arange(2, 30), a['continuum_curvature'][j], label=f'{times[j]:.2f}s')
    ax.set(xlabel='world X (m)', ylabel='world Y (m)', zlabel='world Z (m)', title='Actual continuum body-origin shape snapshots')
    bx.set(xlabel='middle link number', ylabel='1/m', title='Shape curvature at the same saved times')
    ax.legend(fontsize=7)
    bx.legend(fontsize=7, ncol=2)
    _save(fig, out, '12_continuum_shape', files)
    before, after = a['command_before'], a['command_after']
    available = np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
    if available.any():
        fig, axes = plt.subplots(2, 1, figsize=(14, 7))
        axes[0].plot(times[available], before[available], alpha=.65)
        axes[0].set(title='All 17 desired controller velocities before filtering', ylabel='rad/s')
        axes[1].plot(times[available], after[available]-before[available])
        axes[1].set(title='Recorded filtered minus desired velocity; not a causal advantage test', ylabel='rad/s', xlabel='absolute time (s)')
        _save(fig, out, '13_controller_filter_effect', files)
    with (out/'obstacle_nearest_body.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['absolute_time_s','obstacle','nearest_body','arm','nearest_geom_id','nearest_geom_name','distance_m','inside_display_10mm'])
        for ti, t in enumerate(times):
            for oi, obj in enumerate(objects):
                li = int(distance[ti, oi].argmin())
                geom = int(a['nearest_geom_id'][ti, oi, li])
                writer.writerow([t, obj['object_id'], labels[li], links[li]['arm_name'], geom,
                    meta['geom_names'][geom], distance[ti, oi, li], bool(distance[ti, oi, li] < .010)])
    files.append('obstacle_nearest_body.csv')
    summary = []
    with (out/'obstacle_body_minimum_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['obstacle','minimum_distance_m','time_s','body','geom','display_band_observations','total_observations'])
        for oi, obj in enumerate(objects):
            ti, li = np.unravel_index(np.argmin(distance[:, oi]), distance[:, oi].shape)
            geom = int(a['nearest_geom_id'][ti, oi, li])
            item = [obj['object_id'], float(distance[ti, oi, li]), float(times[ti]), labels[li],
                meta['geom_names'][geom], int(np.sum(distance[:, oi].min(axis=1)<.010)), len(times)]
            writer.writerow(item)
            summary.append(item)
    files.append('obstacle_body_minimum_summary.csv')
    return dict(status='COMPLETE' if series.complete else 'PARTIAL_BODY_DIAGNOSTICS',
        source_statuses=series.source_statuses, samples=len(times), files=files, summary=summary,
        causal_effect_claim=False, display_band_m=.010, physical_gate_m=0.)
