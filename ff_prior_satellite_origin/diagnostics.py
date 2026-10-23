"""Read-only geometry observations, never simulation integration or acceptance.

Each obstacle is queried against every collidable geometry on each arm branch.
Distances are reduced only within the SAME named body; no tip-only surrogate,
bounding-sphere pruning, hidden query cap, or contact-list approximation is used.
"""
from pathlib import Path
import time
import numpy as np
import mujoco
from ff_prior.storage import file_digest
from ff_prior_pilot_train.storage import atomic_npz
from .compact_storage import atomic_json


DISPLAY_BAND_M = .010


def discrete_curvature(points):
    """Turning angle / mean adjacent chord length; a shape diagnostic, not strain."""
    points = np.asarray(points, dtype=float)
    segments = np.diff(points, axis=-2)
    lengths = np.linalg.norm(segments, axis=-1)
    a, b = segments[..., :-1, :], segments[..., 1:, :]
    numerator = np.linalg.norm(np.cross(a, b), axis=-1)
    denominator = np.sum(a * b, axis=-1)
    angle = np.arctan2(numerator, denominator)
    scale = .5 * (lengths[..., :-1] + lengths[..., 1:])
    return np.divide(angle, scale, out=np.zeros_like(angle), where=scale > 1e-12)


def _ancestors(model, body):
    result = []
    while body:
        result.append(int(body))
        body = int(model.body_parentid[body])
    return list(reversed(result))


class BodyObstacleRecorder:
    def __init__(self, robot, scene_dict, source_mode='actual_control_tick', deadline=None):
        self.robot, self.model = robot, robot.model
        self.scene = scene_dict.to_dict() if hasattr(scene_dict, 'to_dict') else scene_dict
        self.source_mode, self.deadline = source_mode, deadline
        m = self.model
        paths = [_ancestors(m, robot.tip_c), _ancestors(m, robot.tip_r)]
        shared = set(paths[0]) & set(paths[1])
        collidable = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        self.links = []
        for arm, path in enumerate(paths):
            for body in path:
                if body in shared:
                    continue
                geoms = [g for g in collidable if int(m.geom_bodyid[g]) == body]
                if geoms:
                    self.links.append(dict(body_id=body,
                        body_name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body),
                        arm=arm, arm_name='continuum' if arm == 0 else 'rigid', geom_ids=geoms))
        self.obstacles = []
        for obj in self.scene['objects']:
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'v7_' + obj['object_id'])
            if geom < 0:
                raise ValueError('UNRESOLVED_OBSTACLE_' + obj['object_id'])
            self.obstacles.append(dict(object_id=obj['object_id'], geom_id=int(geom)))
        if not self.links or len(self.obstacles) != 7:
            raise ValueError('BALANCED_DIAGNOSTIC_TOPOLOGY_MISMATCH')
        self.continuum_body_ids = [robot.id(mujoco.mjtObj.mjOBJ_BODY, f'link_{i}') for i in range(1, 31)]
        self.geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        self.rows, self.events = [], []
        self.query_count = 0
        self.started = time.perf_counter()
        self.failure = None

    def sample(self, data, command_before=None, command_after=None, active_constraints=None,
               filter_reason='', native_index=None):
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError('BODY_DIAGNOSTIC_WALL_BUDGET')
        if self.rows and float(data.time) <= self.rows[-1]['time'] + 1e-10:
            raise ValueError('BODY_DIAGNOSTIC_NONINCREASING_TIME')
        m = self.model
        shape = (len(self.obstacles), len(self.links))
        distances = np.full(shape, np.inf)
        nearest = np.full(shape, -1, dtype=np.int32)
        lines = np.full(shape + (2, 3), np.nan)
        scratch = np.zeros(6)
        for oi, obs in enumerate(self.obstacles):
            if self.deadline is not None and time.time() >= self.deadline:
                raise TimeoutError('BODY_DIAGNOSTIC_WALL_BUDGET')
            og = obs['geom_id']
            for li, link in enumerate(self.links):
                for geom in link['geom_ids']:
                    # A geometry-center distance plus both rbound radii is an
                    # upper bound on the nearest distance. The extra metre
                    # prevents a capped result without guessing a scene size.
                    limit = float(np.linalg.norm(data.geom_xpos[og] - data.geom_xpos[geom])
                                  + m.geom_rbound[og] + m.geom_rbound[geom] + 1.)
                    distance = float(mujoco.mj_geomDistance(m, data, og, geom, limit, scratch))
                    self.query_count += 1
                    if not np.isfinite(distance) or distance >= limit:
                        raise RuntimeError('BODY_DIAGNOSTIC_NONEXACT_DISTANCE')
                    if distance < distances[oi, li]:
                        distances[oi, li] = distance
                        nearest[oi, li] = geom
                        lines[oi, li] = scratch.reshape(2, 3)
        skeleton = data.xpos[self.continuum_body_ids].copy()
        before = np.full(17, np.nan) if command_before is None else np.asarray(command_before, dtype=float).copy()
        after = np.full(17, np.nan) if command_after is None else np.asarray(command_after, dtype=float).copy()
        if before.shape != (17,) or after.shape != (17,):
            raise ValueError('BODY_DIAGNOSTIC_COMMAND_DIMENSION')
        self.rows.append(dict(time=float(data.time), native_index=-1 if native_index is None else int(native_index),
            distance_m=distances, nearest_geom_id=nearest, connector=lines,
            body_positions=data.xpos.copy(), continuum_curvature=discrete_curvature(skeleton),
            command_before=before, command_after=after))
        self.events.append(dict(time=float(data.time), native_index=native_index,
            filter_reason=filter_reason, active_constraints=[] if active_constraints is None else active_constraints,
            command_available=command_before is not None, filtered_command_available=command_after is not None))

    def save(self, out_dir, source_native_path=None, expected_samples=None, failure=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            arrays = dict(time=np.empty(0), native_index=np.empty(0, dtype=int))
        else:
            arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
        atomic_npz(out / 'body_obstacle_trace.npz', **arrays)
        atomic_json(out / 'body_filter_events.json', self.events)
        complete = failure is None and (expected_samples is None or len(self.rows) == expected_samples)
        meta = dict(schema='balanced_whole_arm_observation_v1', status='COMPLETE' if complete else 'PARTIAL',
            source_mode=self.source_mode, samples=len(self.rows), expected_samples=expected_samples,
            failure=failure, query_count=self.query_count, elapsed_s=time.perf_counter() - self.started,
            links=self.links, obstacles=self.obstacles, geom_names=self.geom_names, body_names=self.body_names,
            parents=self.model.body_parentid.tolist(), continuum_body_ids=self.continuum_body_ids,
            control_commands='NaN means no recorded filter command; never replaced with fabricated zeros',
            distance_policy='Exact mj_geomDistance for every obstacle x collidable arm geom; minimum within each body',
            query_cap_policy='per-pair center distance + both bounding radii + 1 metre; capped result raises',
            display_band_m=DISPLAY_BAND_M, physical_nonpenetration_gate_m=0.,
            curvature_definition='turning angle divided by mean adjacent body-origin chord length; not material strain',
            coverage='Arm branches only for body influence diagnosis; original full-machine geometry is separate acceptance',
            no_counterfactual_comparison=True, new_native_steps=0, new_coupled_calls=0,
            source_native_sha256=file_digest(source_native_path) if source_native_path else None,
            files={name: dict(bytes=(out/name).stat().st_size, sha256=file_digest(out/name))
                   for name in ('body_obstacle_trace.npz', 'body_filter_events.json')})
        atomic_json(out / 'body_obstacle_metadata.json', meta)
        return meta


def record_saved_trace(robot, scene, native_path, out_dir, control_dt=.002, deadline=None, notify=None):
    """Observe saved actual states in private data; no simulation advance/replay."""
    native_path = Path(native_path)
    source = np.load(native_path)
    times = np.asarray(source['time'])
    if not len(times):
        raise ValueError('NO_NATIVE_STATES_FOR_BODY_DIAGNOSTICS')
    # Select by recorded time; always include the actual final state.
    targets = times[0] + np.arange(int(round((times[-1]-times[0])/control_dt))+1) * control_dt
    ids = np.clip(np.searchsorted(times, targets-1e-9), 0, len(times)-1)
    ids = np.unique(np.r_[ids, len(times)-1])
    recorder = BodyObstacleRecorder(robot, scene, 'saved_actual_static_control_ticks', deadline)
    scratch = mujoco.MjData(robot.model)
    failure = None
    try:
        for j, i in enumerate(ids):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError('BODY_DIAGNOSTIC_WALL_BUDGET')
            scratch.qpos[:] = source['qpos'][i]
            scratch.qvel[:] = source['qvel'][i]
            scratch.time = float(times[i])
            mujoco.mj_kinematics(robot.model, scratch)
            recorder.sample(scratch, native_index=int(i),
                filter_reason='Static qualified feedforward reference; no online geometry velocity filter')
            if notify and j % 250 == 0:
                notify('saved_body_geometry', sample=j+1, total=len(ids), exact_queries=recorder.query_count)
    except Exception as exc:
        failure = repr(exc)
    finally:
        source.close()
    return recorder.save(out_dir, native_path, expected_samples=len(ids), failure=failure)
