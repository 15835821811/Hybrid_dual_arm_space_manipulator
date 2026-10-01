"""Independent 172-case shape diagnostic audit; no producer/query wrappers.

Rebuild URDF FK and all mesh/box capsules, use matrix exponentials for PCC,
derive OBB point/segment distances, enumerate every capsule, and recompute
the original 17 PCC x 5 axis finite-sampling sufficient containment bound.
Only mj_forward/mj_geomDistance are used; no steps, budgets or timing tests.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from v6_lite.continuum_model_spec import default_continuum_model_spec
from v6_lite.run_v6_lite import default_v6_lite_robot_spec
from v6_lite.runtime_command import model_id

TRIAL = ROOT / 'v6_lite/output/runs/research_shape_decomposition_01'
OLD = ROOT / 'v6_lite/output/runs/research_conservatism_01'
PREVIOUS = ROOT / 'v6_lite/output/runs/research_shape_decomposition_independent_audit_01'
COMMIT = '32c3b04764669402f06d4a843b4ef36c81138908'
INPUT_SHA = 'ec620f3396bc06f98597e76270501a1bd7a88a941e5740239bb10d775d3efe8c'
INDICES_SHA = '5d528aa0b01942484d9f9d0375ffe4c79c1f3c6c016b8c2cc9303af15e50c243'
CONTRACT_SHA = 'd3228423e4352d29025631c6d34663d95b6b20ee2c70a7e32db58d2076701db0'
GATE, PAD, MARGIN = .005, 1e-9, 1e-4
CHECKS = []


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'),
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def jsonrows(p):
    return [json.loads(x, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            for x in Path(p).read_text(encoding='utf-8').splitlines()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def write(p, v):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(v, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.write('\n')


def finite(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def near(a, b, tol=1e-12):
    return finite(a) and finite(b) and abs(a-b) <= tol


def vecnear(a, b, tol=1e-12):
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return aa.shape == bb.shape and np.all(np.isfinite(aa)) and np.all(np.isfinite(bb)) and bool(np.max(np.abs(aa-bb), initial=0) <= tol)


def check(name, passed, detail=None):
    CHECKS.append({'name': name, 'passed': bool(passed), 'detail': detail})


def verify_manifest(directory, manifest_name='artifact_manifest.json'):
    manifest = read(directory/manifest_name)
    entries = {x['path'].replace('\\', '/'): x for x in manifest['artifacts']}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
              if p.is_file() and p.name != manifest_name}
    return len(entries) == len(manifest['artifacts']) and set(entries) == actual and all(
        sha(directory/name) == x['sha256'] and (directory/name).stat().st_size == x['bytes']
        for name, x in entries.items())


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    a = skew(axis)
    return np.eye(3)+math.sin(angle)*a+(1-math.cos(angle))*(a@a)


def origin(joint):
    x = joint.find('origin')
    t = np.eye(4)
    if x is not None:
        t[:3, 3] = np.fromstring(x.get('xyz', '0 0 0'), sep=' ')
        r, p, y = np.fromstring(x.get('rpy', '0 0 0'), sep=' ')
        t[:3, :3] = rot([0, 0, 1], y) @ rot([0, 1, 0], p) @ rot([1, 0, 0], r)
    return t


def chain_from_urdf(contract):
    xml = ET.parse(ROOT/contract['source_urdf']).getroot()
    bychild = {j.find('child').get('link'): j for j in xml.findall('joint')}
    chain = []
    name = contract['end_effector_body_name']
    while name != contract['base_body_name']:
        j = bychild[name]
        axis = j.find('axis')
        chain.append({'name': j.get('name'), 'kind': j.get('type'), 'child': name,
                      'origin': origin(j), 'axis': np.fromstring('1 0 0' if axis is None else axis.get('xyz'), sep=' ')})
        name = j.find('parent').get('link')
    chain.reverse()
    return chain


def fk(chain, q60, base_name):
    t = np.eye(4)
    result = {base_name: t.copy()}
    k = 0
    for j in chain:
        t = t @ j['origin']
        if j['kind'] == 'revolute':
            a = np.eye(4)
            a[:3, :3] = rot(j['axis'], float(q60[k]))
            t = t @ a
            k += 1
        elif j['kind'] != 'fixed':
            raise ValueError('unsupported chain joint')
        result[j['child']] = t.copy()
    if k != 60:
        raise ValueError('incomplete chain')
    return result


def pcc_prefix_and_generator(q, c):
    prefixes = [np.asarray(c['base_to_shape_start'], dtype=float)]
    generators = []
    for i, length in enumerate(c['segment_lengths_m']):
        g = np.zeros((4, 4))
        g[:3, :3] = skew(np.asarray(c['pcc_bending_map']) @ q[2*i:2*i+2]/length)
        g[0, 3] = 1.
        generators.append(g)
        prefixes.append(prefixes[-1] @ expm(g*length))
    return prefixes, generators


def pcc_at(prefixes, generators, segment, local_s):
    return (prefixes[segment] @ expm(generators[segment]*local_s))[:3, 3]


def point_sd(point, center, rotation, extent):
    local = rotation.T @ (point-center)
    outside = np.maximum(np.abs(local)-extent, 0.)
    distance = float(np.linalg.norm(outside)+min(float(np.max(np.abs(local)-extent)), 0.))
    boxpoint = np.clip(local, -extent, extent)
    if distance > 0:
        normal = rotation @ ((local-boxpoint)/distance)
    else:
        axis = int(np.argmax(np.abs(local)-extent))
        boxpoint[axis] = extent[axis]*(1 if local[axis] >= 0 else -1)
        direction = np.zeros(3)
        direction[axis] = 1 if local[axis] >= 0 else -1
        normal = rotation @ direction
    return distance, center+rotation@boxpoint, normal


def segment_sd(start, end, center, rotation, extent):
    """All active-face regions of the squared exterior distance quadratic.

    Interior candidates provide only an unsafe indicator, not exact depth.
    """
    a, v = rotation.T@(start-center), rotation.T@(end-start)
    events = {0., 1.}
    for i in range(3):
        if abs(v[i]) > 1e-15:
            events.update(float(t) for t in ((-extent[i]-a[i])/v[i], (extent[i]-a[i])/v[i]) if 0 < t < 1)
    boundaries = sorted(events)
    candidates = set(boundaries)
    for lo, hi in zip(boundaries, boundaries[1:]):
        mid = (lo+hi)/2
        candidates.add(mid)
        x = a+mid*v
        mask = np.abs(x) > extent
        target = np.sign(x)*extent
        vv = float(v[mask]@v[mask])
        if vv > 1e-24:
            candidates.add(float(np.clip(-float(v[mask]@(a-target)[mask])/vv, lo, hi)))
    evaluated = []
    for t in sorted(candidates):
        p = start+t*(end-start)
        sd, bp, normal = point_sd(p, center, rotation, extent)
        evaluated.append((sd, t, p, bp, normal))
    return min(evaluated, key=lambda v: v[0])


def vertices_body(model, gid):
    typ = int(model.geom_type[gid])
    if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        offset, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        vertices = np.asarray(model.mesh_vert[offset:offset+count], dtype=float)
    elif typ == int(mujoco.mjtGeom.mjGEOM_BOX):
        vertices = np.asarray(list(itertools.product([-1., 1.], repeat=3)))*model.geom_size[gid]
    else:
        raise ValueError('unsupported capsule source geometry')
    flat = np.empty(9)
    mujoco.mju_quat2Mat(flat, model.geom_quat[gid])
    return model.geom_pos[gid]+vertices@flat.reshape(3, 3).T


def derive_capsules(model, geoms):
    result = []
    for gid in geoms:
        vertices = vertices_body(model, gid)
        mean = vertices.mean(axis=0)
        values, axes = np.linalg.eigh((vertices-mean).T@(vertices-mean))
        axis = axes[:, int(np.argmax(values))]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        projected = (vertices-mean)@axis
        start, end = mean+projected.min()*axis, mean+projected.max()*axis
        direction = end-start
        t = np.clip((vertices-start)@direction/(direction@direction), 0., 1.)
        distances = np.linalg.norm(vertices-(start+t[:, None]*direction), axis=1)
        radius = float(distances.max()+PAD)
        body = int(model.geom_bodyid[gid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        section = 4 if name == 'end_effector' else (int(name.split('_')[-1])-1)//6
        result.append({'geom_id': gid, 'geom_name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                       'body_id': body, 'body_name': name, 'segment_id': section,
                       'local_start': start, 'local_end': end, 'radius_m': radius,
                       'source_vertex_excess_m': float((distances-radius).max()),
                       'vertex_count': len(vertices), 'axis_length_m': float(np.linalg.norm(end-start))})
    return result


def distribution(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('invalid statistic')
    return {'count': len(x), 'min': float(x.min()), 'p50': float(np.percentile(x, 50)),
            'p95': float(np.percentile(x, 95)), 'max': float(x.max())}


def run():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    plan, producer, provenance = [read(TRIAL/x) for x in ('plan.json', 'report.json', 'source_provenance.json')]
    oldplan, oldselection = read(OLD/'plan.json'), read(OLD/'selections.json')
    oldrows, produced = jsonrows(OLD/'baseline.jsonl'), jsonrows(TRIAL/'cases.jsonl')
    ids = oldselection['proven_proxy_below_actual_safe_indices']
    expected_ids = [r['index'] for r in oldrows if r['actual_mujoco_distance_m'] >= GATE and r['proxy_status'] == 'PROXY_CLEARANCE_BELOW_GATE']
    indices_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()
    check('exact_172_indices_original_selection', len(ids) == len(set(ids)) == 172 and ids == expected_ids
          == plan['indices'] == [r['index'] for r in produced] and indices_sha == INDICES_SHA == plan['indices_sha256'])
    check('producer_and_original_manifests', verify_manifest(TRIAL) and verify_manifest(OLD)
          and verify_manifest(ROOT/'v6_lite/output/runs/research_conservatism_independent_audit_01'))
    check('producer_complete_checks_and_scope', producer['complete'] is True and producer['evidence_valid'] is True
          and all(x is True for x in producer['checks'].values()) and producer['safety_contract_changed'] is False
          and producer['closed_loop_claim'] is False and producer['hardware_claim'] is False and producer['wall_deployment_status'] == 'NOT_MET')
    previous = read(PREVIOUS/'report.json')
    previous_audit = {'path': PREVIOUS.relative_to(ROOT).as_posix(),
        'report_sha256': sha(PREVIOUS/'report.json'), 'manifest_sha256': sha(PREVIOUS/'artifact_manifest.json'),
        'evidence_valid': previous['evidence_valid'], 'checks_passed': previous['checks_passed'],
        'checks_count': previous['checks_count'],
        'tool_error': 'audit01 selected last eigenvector for tied maximal PCA eigenvalues; original capsule protocol uses first maximal eigenvalue via argmax. The terminal cube gives multiple equally valid axes, but matching the frozen capsule requires the declared tie rule. Original producer artifacts are unchanged.'}
    check('previous_audit_tool_failure_preserved', verify_manifest(PREVIOUS) and previous['evidence_valid'] is False
          and previous['checks_passed'] == 10 and previous['checks_count'] == 12)
    check('producer_references_unchanged', all(sha(ROOT/x['path']) == x['sha256']
          and (ROOT/x['path']).stat().st_size == x['bytes'] for x in plan['references']))
    source = plan['source']
    current_hashes = {name: sha(ROOT/name) for name in source['files']}
    check('source_commit_snapshot_and_raw_hashes', commit == COMMIT == producer['source_commit'] == source['git_commit']
          == provenance['before']['git_commit'] == provenance['after']['git_commit'] and not tracked
          and source['tracked_worktree_dirty'] is False and not source['capture_errors']
          and provenance['capture_complete'] is True and provenance['source_unchanged'] is True
          and provenance['git_commit_unchanged'] is True and not provenance['source_changes']
          and source['files'] == provenance['before']['files'] == provenance['after']['files']
          and all(current_hashes[name] == x['sha256_raw'] and (ROOT/name).stat().st_size == x['size_bytes'] for name, x in source['files'].items())
          and sha(TRIAL/'producer.py') == sha(ROOT/'v6_lite/audit_research_shape_decomposition.py'))
    input_path = ROOT/'v6_lite/output/v6_2_b1/formal_audit_r02/independent_heldout_inputs.npz'
    check('frozen_input_identity_and_old_manifest', sha(input_path) == INPUT_SHA == plan['frozen_input_sha256']
          == oldplan['frozen_input']['sha256'] and verify_manifest(input_path.parent, 'audit_manifest.json'))
    with np.load(input_path, allow_pickle=False) as z:
        inputs = {k: z[k].copy() for k in z.files}
    c = plan['model_contract']
    spec = default_continuum_model_spec()
    radii = np.asarray(plan['tube_radii_m'])
    check('fixed_original_model_contract_radii_and_protocol', canonical(c) == CONTRACT_SHA == spec.contract_sha256()
          == plan['model_contract_sha256'] == oldplan['model_contract_sha256']
          and c == oldplan['model_contract'] == spec.to_dict() and plan['tube_radii_m'] == oldplan['tube_radii_m']
          and plan['gate_m'] == GATE and plan['PCC_samples_per_section'] == 17 and plan['capsule_axis_samples'] == 5
          and plan['original_envelope_numerical_margin_m'] == MARGIN and plan['capsule_count'] == 61
          and plan['fallback_geom_names'] == ['collision_0003'] and plan['fallback_included_in_envelope_radius_or_distances'] is False
          and plan['simulation_periods_unchanged'] == {'planning_s': .02, 'physics_s': .002}
          and plan['online_radius_reduction_authorized'] is False and plan['compute_performance_or_wall_test'] is False
          and plan['negative_clearance_rule'] == 'intersecting centerline and capsule negative values are unsafe indicators, not exact penetration depths'
          and plan['negative_envelope_margin_rule'] == 'sufficient upper bound exceeds fixed radius; inconclusive, not proof of actual noncontainment')
    selected_arrays = {k: {'dtype': str(inputs[k].dtype), 'shape': list(inputs[k][ids].shape),
        'sha256_c_order_rawbytes': hashlib.sha256(np.ascontiguousarray(inputs[k][ids]).tobytes(order='C')).hexdigest()}
        for k in ('configurations', 'centers', 'rotations', 'half_extents')}
    write(OUT/'audit_plan.json', {'schema': 'independent_shape_decomposition_plan_v1', 'declared_before_numeric_checks': True,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'auditor_source_commit': commit, 'source_trial_commit': COMMIT,
        'auditor_script_sha256': sha(__file__), 'input_sha256': sha(input_path), 'indices': ids, 'indices_sha256': indices_sha,
        'indices_encoding': "json.dumps(indices,separators=(',',':')).encode('ascii'); no newline",
        'selected_arrays': selected_arrays, 'trial_manifest_sha256': sha(TRIAL/'artifact_manifest.json'),
        'previous_retained_audit': previous_audit,
        'scope': '172 existing cases only; no new B1/budget queries, mj_step, closed-loop or wall testing',
        'methods': ['independent analytic point/segment OBB formulas', 'independent XML URDF FK',
                    'independent PCC matrix exponential', 'all 61 independently reconstructed capsules; no minimum wrappers',
                    'direct native distance for all 61 distal collision geoms', '17x5 envelope sufficient bound per capsule and section']})
    write(OUT/'source_inventory.json', {'source_trial_commit': COMMIT, 'auditor_source_commit': commit, 'raw_sha256': current_hashes})
    robot = default_v6_lite_robot_spec()
    model = robot.compile_dynamic_model()
    data = mujoco.MjData(model)
    check('nominal_compiled_model_hash', model_id(model, CONTRACT_SHA) == plan['nominal_compiled_model_sha256'])
    target_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target_satellite_collision'))
    target_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, robot.target_free_joint_name))
    target_addr = int(model.jnt_qposadr[target_joint])
    qpos_ids = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in robot.low_level_joint_names]
    rootbody = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c['mount_body_name']))
    current = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c['end_effector_body_name']))
    bodyids = set()
    while current != 0:
        bodyids.add(current)
        if current == rootbody:
            break
        current = int(model.body_parentid[current])
    geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodyids and int(model.geom_bodyid[i]) != rootbody
             and (int(model.geom_contype[i]) or int(model.geom_conaffinity[i]) or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('collision_'))]
    capsules = derive_capsules(model, geoms)
    chain = chain_from_urdf(c)
    check('independent_geometry_inventory_and_urdf_identity', rootbody in bodyids and len(capsules) == 61
          and [cap['geom_name'] for cap in capsules] == [x['geom_name'] for x in produced[0]['capsule_envelope_samples']]
          and [x['name'] for x in chain if x['kind'] == 'revolute'] == c['low_level_joint_names']
          and sha(ROOT/c['source_urdf']) == c['source_urdf_sha256'])
    write(OUT/'capsule_definitions.json', [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cap.items()} for cap in capsules])
    lengths, modules = np.asarray(c['segment_lengths_m']), np.asarray(c['module_lengths_m'])
    boundaries, module_ends = np.r_[0., np.cumsum(lengths)], np.cumsum(modules)
    mapping = np.asarray(c['planner_to_actuated'])
    all_case_checks = []
    all_numeric = []
    stream = (OUT/'case_numeric_checks.jsonl').open('x', encoding='utf-8', newline='\n')
    try:
        for ordinal, (index, row) in enumerate(zip(ids, produced)):
            old = oldrows[index]
            q = inputs['configurations'][index]
            center, rotation, extent = (inputs[k][index] for k in ('centers', 'rotations', 'half_extents'))
            local_checks = {}
            def ck(name, passed):
                local_checks[name] = bool(passed)
            ck('exact_frozen_input_arrays', np.array_equal(np.asarray(row['q_continuum_rad']), q)
               and np.array_equal(np.asarray(row['target_center_m']), center) and np.array_equal(np.asarray(row['target_rotation']), rotation)
               and np.array_equal(np.asarray(row['target_half_extents_m']), extent) and np.all(np.isfinite(q))
               and np.all(np.isfinite(rotation)) and np.all(np.isfinite(center)) and np.all(extent > 0)
               and vecnear(rotation.T@rotation, np.eye(3)) and near(float(np.linalg.det(rotation)), 1.)
               and np.all(q >= np.asarray(c['work_domain_lower_rad'])) and np.all(q <= np.asarray(c['work_domain_upper_rad'])))
            model.geom_size[target_id] = extent
            data.qpos[:] = model.qpos0
            planner = robot.planner_zero.copy()
            planner[:10] = q
            data.qpos[qpos_ids] = robot.encode_position(planner)
            flat = np.empty(9)
            mujoco.mju_quat2Mat(flat, model.geom_quat[target_id])
            bodyrot = rotation@flat.reshape(3, 3).T
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, bodyrot.reshape(-1))
            data.qpos[target_addr:target_addr+3] = center-bodyrot@model.geom_pos[target_id]
            data.qpos[target_addr+3:target_addr+7] = quat
            mujoco.mj_forward(model, data)
            q60 = mapping@q
            ck('matched_actual_configuration_target_pose_and_model', vecnear(data.qpos[np.asarray(qpos_ids)[:60]], q60)
               and vecnear(data.geom_xpos[target_id], center) and vecnear(data.geom_xmat[target_id].reshape(3, 3), rotation)
               and row['compiled_case_model_sha256'] == old['compiled_case_model_sha256'] == model_id(model, CONTRACT_SHA))
            transforms = fk(chain, q60, c['base_body_name'])
            position_error, rotation_error = 0., 0.
            for name, t in transforms.items():
                bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
                if bid < 0:
                    raise ValueError('missing independent FK body')
                position_error = max(position_error, float(np.linalg.norm(data.xpos[bid]-t[:3, 3])))
                rotation_error = max(rotation_error, float(np.max(np.abs(data.xmat[bid].reshape(3, 3)-t[:3, :3]))))
            ck('all_independent_urdf_fk_matches_mujoco', position_error <= 1e-12 and rotation_error <= 1e-12
               and near(row['discrete_fk_vs_mujoco_position_error_max_m'], position_error)
               and near(row['discrete_fk_vs_mujoco_rotation_entry_error_max'], rotation_error))
            direct = []
            for gid in geoms:
                pair = np.empty(6)
                distance = float(mujoco.mj_geomDistance(model, data, gid, target_id, .5, pair))
                direct.append({'geom_name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid), 'distance_m': distance,
                               'witness_arm_world': pair[:3].tolist(), 'witness_box_world': pair[3:].tolist()})
            actual = min(direct, key=lambda x: x['distance_m'])
            ck('direct_all_geoms_actual_matches_original', near(actual['distance_m'], old['actual_mujoco_distance_m'])
               and near(actual['distance_m'], row['actual_mujoco_distance_m']) and actual['geom_name'] == old['actual_source_geom'] == row['actual_source_geom']
               and row['actual_query_truncated'] is False and actual['distance_m'] >= GATE
               and old['upper_m'] < GATE and near(row['original_upper_m'], old['upper_m']) and near(row['original_lower_m'], old['lower_m'])
               and row['matches_original_distance'] is True and row['matches_original_model'] is True)
            prefixes, generators = pcc_prefix_and_generator(q, c)
            w = row['refusal_witness']
            originalw = old['query_result']
            section, local_s = originalw['best_segment_id'], originalw['best_local_arclength_m']
            s = float(boundaries[section]+local_s)
            module = 29 if s >= module_ends[-1] else int(np.searchsorted(module_ends, s, side='left'))
            module_s = s-(0. if module == 0 else float(module_ends[module-1]))
            pcc = pcc_at(prefixes, generators, section, local_s)
            body = c['module_body_names'][module]
            bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body))
            actual_s = data.xpos[bid]+data.xmat[bid].reshape(3, 3)@np.array([module_s, 0., 0.])
            pcc_sd, _, _ = point_sd(pcc, center, rotation, extent)
            actual_sd, _, _ = point_sd(actual_s, center, rotation, extent)
            radius, norm, shift = float(radii[section]), float(np.linalg.norm(actual_s-pcc)), actual_sd-pcc_sd
            raw_proxy = pcc_sd-radius
            total, residual = actual['distance_m']-raw_proxy, actual['distance_m']-actual_sd
            ck('independent_same_material_witness_and_pcc_exponential', section in range(5) and 0 <= local_s <= lengths[section]
               and module in range(30) and -1e-12 <= module_s <= modules[module]+1e-12
               and near(w['material_arclength_m'], s) and w['segment_id'] == section and w['discrete_module_index'] == module
               and near(w['discrete_module_arclength_m'], module_s) and vecnear(w['pcc_world'], pcc)
               and vecnear(originalw['best_point_world'], pcc) and vecnear(w['discrete_world'], actual_s)
               and near(float(np.nextafter(raw_proxy+PAD, math.inf)), old['upper_m'])
               and near(w['pcc_point_sd_m'], pcc_sd) and near(w['discrete_point_sd_m'], actual_sd)
               and near(w['pcc_same_radius_sample_clearance_m'], raw_proxy) and near(w['radius_m'], radius)
               and near(w['discrete_same_radius_sample_clearance_m'], actual_sd-radius)
               and near(w['position_discrepancy_norm_m'], norm) and near(w['directional_distance_effect_m'], shift)
               and abs(shift) <= norm+1e-12 and w['matches_saved_proxy_upper'] is True and w['matches_saved_pcc_point'] is True)
            arithmetic_error = total-(radius+shift+residual)
            ck('independent_residual_identity_and_interpretation', near(w['actual_minus_raw_pcc_witness_m'], total)
               and near(w['remaining_global_surface_correspondence_term_m'], residual) and abs(arithmetic_error) <= 1e-12
               and near(w['decomposition_arithmetic_residual_m'], arithmetic_error) and w['remaining_term_is_pure_physical_radius'] is False)
            chain_values = []
            chain_ok = len(row['chain_modules']) == 30
            for i, (name, length, saved) in enumerate(zip(c['module_body_names'], modules, row['chain_modules'])):
                bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
                start, rr = data.xpos[bid], data.xmat[bid].reshape(3, 3)
                end = start+rr@np.array([length, 0., 0.])
                d, param, _, _, _ = segment_sd(start, end, center, rotation, extent)
                net = d-radii[i//6]
                chain_values.append({'module_index': i, 'centerline_distance_or_unsafe_indicator_m': d,
                                     'net_indicator_m': float(net), 'witness_parameter': param})
                chain_ok &= (saved['module_index'] == i and saved['body_name'] == name and saved['segment_id'] == i//6
                    and vecnear(saved['start_world'], start) and vecnear(saved['end_world'], end) and near(saved['tube_radius_m'], float(radii[i//6]))
                    and near(saved['centerline_distance_or_unsafe_indicator_m'], d) and near(saved['same_radius_net_clearance_indicator_m'], float(net))
                    and near(saved['witness_parameter'], param) and saved['centerline_exterior_exact'] is (d > 0))
            chain_min = min(chain_values, key=lambda x: x['net_indicator_m'])
            chain_status = 'SAFE' if chain_min['net_indicator_m'] >= GATE else 'BELOW'
            ck('all_30_modules_independent_piecewise_distance', chain_ok and chain_status == row['same_radius_discrete_chain_gate_status']
               and near(chain_min['net_indicator_m'], row['same_radius_discrete_chain_clearance_indicator_m'])
               and chain_min['module_index'] == row['chain_minimum_module_index']
               and row['all_chain_centerlines_exterior'] is all(x['centerline_distance_or_unsafe_indicator_m'] > 0 for x in chain_values))
            lastbody = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c['module_body_names'][-1]))
            lastend = data.xpos[lastbody]+data.xmat[lastbody].reshape(3, 3)@np.array([modules[-1], 0., 0.])
            terminal = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c['end_effector_body_name']))
            ck('terminal_offset_counted_once', vecnear(lastend, data.xpos[terminal]) and row['last_module_endpoint_matches_terminal_frame'] is True)
            pcc_samples = [np.asarray([pcc_at(prefixes, generators, i, float(u)) for u in np.linspace(0, lengths[i], 17)]) for i in range(5)]
            cap_values, envelope_caps = [], []
            capsule_ok = len(row['capsule_envelope_samples']) == 61
            for cap, saved in zip(capsules, row['capsule_envelope_samples']):
                bid = cap['body_id']
                rr, pos = data.xmat[bid].reshape(3, 3), data.xpos[bid]
                start, end = pos+rr@cap['local_start'], pos+rr@cap['local_end']
                d, param, cp, bp, normal = segment_sd(start, end, center, rotation, extent)
                cap_values.append({'geom_name': cap['geom_name'], 'axis_signed_distance_or_unsafe_indicator_m': d,
                    'capsule_indicator_m': d-cap['radius_m'], 'axis_witness_parameter': param,
                    'witness_arm_world': (cp-cap['radius_m']*normal).tolist(), 'witness_box_world': bp.tolist()})
                axes = np.asarray([pos+rr@((1-t)*cap['local_start']+t*cap['local_end']) for t in np.linspace(0, 1, 5)])
                sampled = float(np.min(np.linalg.norm(axes[:, None, :]-pcc_samples[cap['segment_id']][None, :, :], axis=2), axis=1).max())
                allow = cap['axis_length_m']/8
                item = {'geom_name': cap['geom_name'], 'segment_id': cap['segment_id'], 'sampled_axis_to_pcc_max_m': sampled,
                        'capsule_radius_m': cap['radius_m'], 'sampled_required_radius_m': sampled+cap['radius_m'],
                        'axis_sampling_allowance_m': allow, 'source_vertex_excess_m': cap['source_vertex_excess_m']}
                envelope_caps.append(item)
                capsule_ok &= (saved['geom_name'] == cap['geom_name'] and saved['body_name'] == cap['body_name'] and saved['segment_id'] == cap['segment_id']
                    and all(near(saved[k], v) for k, v in item.items() if isinstance(v, float)) and cap['source_vertex_excess_m'] <= 0)
            cap_min = min(cap_values, key=lambda x: x['capsule_indicator_m'])
            cap_status = 'SAFE' if cap_min['capsule_indicator_m'] >= GATE else 'BELOW'
            ck('all_61_capsule_candidates_and_minimum', capsule_ok and near(cap_min['capsule_indicator_m'], row['capsule_clearance_indicator_m'])
               and cap_status == row['capsule_gate_status'] and cap_min['geom_name'] == row['capsule_minimum_source_geom']
               and vecnear(cap_min['witness_arm_world'], row['capsule_witness_arm_world']) and vecnear(cap_min['witness_box_world'], row['capsule_witness_box_world']))
            envelope_rows = []
            env_ok = len(row['original_protocol_segment_envelope']) == 5
            for i, saved in enumerate(row['original_protocol_segment_envelope']):
                group = [cap for cap in envelope_caps if cap['segment_id'] == i]
                required, allowance = max(x['sampled_required_radius_m'] for x in group), max(x['axis_sampling_allowance_m'] for x in group)
                pcc_allowance = float(lengths[i]/32)
                upper = required+allowance+pcc_allowance+MARGIN
                item = {'segment_id': i, 'sampled_required_radius_m': required, 'axis_sampling_allowance_m': allowance,
                        'pcc_sampling_allowance_m': pcc_allowance, 'numerical_margin_m': MARGIN,
                        'sufficient_radius_upper_original_protocol_m': upper, 'declared_radius_m': float(radii[i]),
                        'sufficient_radius_margin_m': float(radii[i]-upper)}
                envelope_rows.append(item)
                env_ok &= saved['segment_id'] == i and all(near(saved[k], v) for k, v in item.items() if isinstance(v, float))
            margin = min(x['sufficient_radius_margin_m'] for x in envelope_rows)
            supported = all(x['sufficient_radius_margin_m'] >= 0 for x in envelope_rows)
            ck('all_capsules_original_17x5_sufficient_bound', env_ok and near(margin, row['minimum_sufficient_radius_margin_m'])
               and supported == row['original_protocol_sufficient_coverage_at_this_state'])
            numeric = {'index': index, 'checks': local_checks, 'direct_actual_all_geoms': direct,
                'same_material_witness': {'s_m': s, 'module_index': module, 'module_s_m': module_s, 'pcc_world': pcc.tolist(),
                    'actual_mujoco_world': actual_s.tolist(), 'pcc_sd_m': pcc_sd, 'actual_sd_m': actual_sd,
                    'radius_m': radius, 'position_error_norm_m': norm, 'directional_distance_effect_m': shift,
                    'raw_proxy_m': raw_proxy, 'actual_minus_raw_proxy_m': total, 'remaining_global_surface_correspondence_m': residual,
                    'identity_residual_m': arithmetic_error},
                'chain_modules': chain_values, 'all_capsule_candidates': cap_values,
                'all_capsule_envelope_samples': envelope_caps, 'segment_envelope': envelope_rows,
                'same_radius_chain_status': chain_status, 'capsule_status': cap_status,
                'minimum_sufficient_radius_margin_m': margin, 'sufficient_coverage': supported,
                'actual_minus_capsule_indicator_m': actual['distance_m']-cap_min['capsule_indicator_m'],
                'fk_position_error_max_m': position_error, 'fk_rotation_entry_error_max': rotation_error}
            stream.write(json.dumps(numeric, ensure_ascii=True, allow_nan=False)+'\n')
            stream.flush()
            all_numeric.append(numeric)
            all_case_checks.append({'index': index, 'passed': all(local_checks.values()), 'checks': local_checks})
            if (ordinal+1) % 32 == 0 or ordinal+1 == 172:
                print(f'[independent-shape] {ordinal+1}/172; failed_cases={sum(not x["passed"] for x in all_case_checks)}', flush=True)
    finally:
        stream.close()
    check('all_172_case_numeric_checks', len(all_case_checks) == 172 and all(x['passed'] for x in all_case_checks))
    chain_counts = dict(Counter(x['same_radius_chain_status'] for x in all_numeric))
    cap_counts = dict(Counter(x['capsule_status'] for x in all_numeric))
    stats = {'witness_position_discrepancy_norm_m': distribution([x['same_material_witness']['position_error_norm_m'] for x in all_numeric]),
        'witness_directional_distance_effect_m': distribution([x['same_material_witness']['directional_distance_effect_m'] for x in all_numeric]),
        'minimum_sufficient_radius_margin_m': distribution([x['minimum_sufficient_radius_margin_m'] for x in all_numeric]),
        'actual_minus_capsule_indicator_m': distribution([x['actual_minus_capsule_indicator_m'] for x in all_numeric])}
    check('independent_top_counts_and_all_report_statistics', producer['case_count'] == 172 and producer['indices_sha256'] == INDICES_SHA
          and chain_counts == producer['same_radius_chain_status_counts'] and cap_counts == producer['capsule_gate_status_counts']
          and producer['centerline_replacement_alone_flips_case_count'] == chain_counts.get('SAFE', 0)
          and producer['sufficient_envelope_support_at_this_state_count'] == sum(x['sufficient_coverage'] for x in all_numeric)
          and all(set(producer[k]) == set(v) and all(near(producer[k][kk], vv) for kk, vv in v.items()) for k, v in stats.items()))
    check('sources_and_producer_original_artifacts_unchanged_after_audit', all(sha(ROOT/name) == value for name, value in current_hashes.items())
          and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == COMMIT
          and verify_manifest(TRIAL) and verify_manifest(OLD) and sha(input_path) == INPUT_SHA)
    write(OUT/'by_case_checks.json', all_case_checks)
    write(OUT/'by_item_checks.json', CHECKS)
    report = {'schema': 'independent_shape_decomposition_audit_v1', 'complete': True,
        'evidence_valid': all(x['passed'] for x in CHECKS), 'source_trial_commit': COMMIT, 'auditor_source_commit': commit,
        'auditor_script_sha256': sha(__file__), 'case_count': len(all_case_checks), 'indices_sha256': INDICES_SHA,
        'checks_passed': sum(x['passed'] for x in CHECKS), 'checks_count': len(CHECKS),
        'per_case_check_count': sum(len(x['checks']) for x in all_case_checks),
        'per_case_checks_passed': sum(sum(x['checks'].values()) for x in all_case_checks),
        'same_radius_chain_status_counts': chain_counts, 'capsule_status_counts': cap_counts,
        'sufficient_envelope_support_count': sum(x['sufficient_coverage'] for x in all_numeric),
        'independent_actual_native_query_count': 172*len(geoms), 'independent_chain_segment_count': 172*30,
        'independent_capsule_candidate_count': 172*61, 'independent_per_capsule_envelope_count': 172*61,
        'original_PCC_samples_per_section': 17, 'original_axis_samples_per_capsule': 5,
        'statistics': stats, 'previous_retained_audit': previous_audit,
        'failed_checks': [x for x in CHECKS if not x['passed']],
        'failed_cases': [x for x in all_case_checks if not x['passed']],
        'limits': ['172 selected offline states; no full-1024 geometry/budget rerun',
            'all sampled centerline minima and capsule minima are exterior here; negative values elsewhere only unsafe indicators',
            'negative sufficient-envelope margin would be inconclusive; no actual noncontainment inference',
            'same-radius chain is a classification counterfactual, not an independently authorized online replacement',
            'PCC/capsule sampling bound uses unit-speed Lipschitz distances and fixed allowances plus original empirical numerical margin',
            'global actual minimum need not occur at the PCC material witness; residual is not pure physical radius',
            'shape/radius/correspondence effects coupled; no independent contribution percentages or online radius reduction'],
        'wall_deployment_status': 'NOT_MET', 'closed_loop_claim': False, 'hardware_claim': False}
    write(OUT/'report.json', report)
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'artifact_manifest.json')
    write(OUT/'artifact_manifest.json', {'schema': 'independent_shape_decomposition_manifest_v1',
        'artifacts': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]})
    print(json.dumps({'evidence_valid': report['evidence_valid'], 'checks': f'{report["checks_passed"]}/{report["checks_count"]}',
        'per_case_checks': f'{report["per_case_checks_passed"]}/{report["per_case_check_count"]}',
        'chain_counts': chain_counts, 'capsule_counts': cap_counts, 'output': str(OUT)}, ensure_ascii=True), flush=True)
    return 0 if report['evidence_valid'] else 1


if __name__ == '__main__':
    try:
        sys.exit(run())
    except Exception:
        failure = traceback.format_exc()
        write(OUT/'audit_failure.json', {'complete': False, 'evidence_valid': False, 'error': failure})
        print(failure, file=sys.stderr)
        sys.exit(1)
