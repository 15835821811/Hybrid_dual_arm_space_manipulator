"""Independent bounded geometry audit of completed resumed inertia-shadow evidence.

Plan stage uses only the standard library and never reads in-flight replay data.
Geometry stage requires the completed resumed observer manifest and inherited audit pins.
The producer and resumer are never imported or called. This script contains no
physical stepping path; the previous 100-step nominal audit is inherited by SHA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
sys.dont_write_bytecode = True

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / '.git').exists() and (p / 'model_test').is_dir())
OUTPUT = Path(__file__).resolve().parent
BASELINE = ROOT / 'v6_lite/output/runs/research_acceptance_01'
TRIAL = ROOT / 'v6_lite/output/runs/research_inertia_shadow_resumed_01'
ORIGINAL = ROOT / 'v6_lite/output/runs/research_inertia_shadow_01'
INHERITED = ROOT / 'v6_lite/output/runs/research_inertia_shadow_independent_audit_02'
FINAL_MANIFEST_SHA = 'ee98249e6ba936278e005943dc942724e57fb5ef2f59d2fe226b0e101d89158e'
REPLAY_MANIFEST_SHA = 'a8543ecf3a62c7adef87e7a698a0536860ab4a8a1800ed95905c870eef256a0a'
INHERITED_SCRIPT_SHA = '71af899c04f43824c6839828915ca26de8fc5fb2b922604785bf7792cdf4066a'
INHERITED_REPORT_SHA = 'c652b09b07b8ed34cf557d662bd1bca1e0a5ed35aa7528704a2571e7ba0cf90c'
INHERITED_MANIFEST_SHA = 'e3010334c4204582b6fd15f1f8708ba5f8d7e9eaa8a86d0f7857d0000f91a055'
SOURCE_COMMIT = '19ad85c7db81a2e254fd176077b6aae39333a439'
PRODUCER_SHA = '1d18141729d95195d94c1b46dfe80baf3035f949ad754a73175e0d94db88e2d6'
BASELINE_COMMIT = '9cd1831e7c77a340464a669eac3b5aa7cd9c3c3c'
BASELINE_MANIFEST_SHA = '379b9977a441c540a20cfc50a9e90614a1e0985fa2a5a73bbb7c9690f823eec0'
CONTRACT = 'ec36bd1e6c1f47e5dcfb94be8da4c147729f4ccd2ed62aaf1ce8cf505584ac3f'
BUNDLE = '638cee34a2dca5c79d62f1898c99caa810510791faa0e8caca18dab5c7693536'
DT, STEPS, ALPHAS = .002, 13500, (1., .95, 1.05)
TORQUE_SHA = (
    '0eaa9e47f2398ec91e7d57ba21ecb3f52ff3f6f0d24c2d6cf4e27304a12ff092',
    '9ae518f8a3e1d128be7a5f1372c567e4e3392b3ed2d8d08c29c704786bcd62f4',
    '487f93f956e7c4c55e1a7355b3f5a482a737701b0c1edb1c9b582424993d3b86',
    '0a32c80ee382b8973500182e211d87d62804e2d6ba2c7ff51f359a17c038eadc',
    'bad41848faa2de5c3d37f1d763a9a95d441a3a2590fd7f2d667739237148507d')
DIRECT = (
    'qpos0','body_parentid','body_pos','body_quat','body_ipos','body_iquat',
    'body_mass','body_inertia','body_gravcomp','body_mocapid','jnt_type',
    'jnt_bodyid','jnt_qposadr','jnt_dofadr','jnt_pos','jnt_axis','jnt_range',
    'jnt_limited','jnt_stiffness','jnt_margin','dof_armature','dof_damping',
    'dof_frictionloss','dof_solref','dof_solimp','geom_type','geom_bodyid',
    'geom_dataid','geom_pos','geom_quat','geom_size','geom_contype',
    'geom_conaffinity','geom_friction','geom_solref','geom_solimp','geom_margin',
    'geom_gap','geom_rgba','mesh_vert','mesh_face','site_pos','site_quat',
    'site_size','site_type','site_bodyid','actuator_trntype','actuator_trnid',
    'actuator_gear','actuator_gaintype','actuator_biastype','actuator_dyntype',
    'actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrllimited',
    'actuator_forcelimited','actuator_ctrlrange','actuator_forcerange',
    'actuator_actrange')
MODEL_FIELDS = (
    'body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat',
    'dof_damping','dof_armature','dof_frictionloss','jnt_stiffness','jnt_axis',
    'jnt_pos','jnt_range','geom_pos','geom_quat','geom_size','geom_type',
    'geom_contype','geom_conaffinity','actuator_gear','actuator_gainprm',
    'actuator_biasprm','actuator_dynprm','actuator_ctrlrange')
LIMITS = {
    'new_producer_imported_or_called': False,
    'physical_replay_scope': 'no new physical stepping; inherited audit_02 nominal 20 steps per scene, 100 total by SHA',
    'new_physics_steps': 0,
    'independent_full_15_run_physics_replay': False,
    'geometry_scope': 'all saved minima/count records; direct native queries at fixed samples and saved worst state',
    'independent_full_geometry_recalculation': False,
    'all_pair_truncation_count_independently_reconstructed': False,
    'all_pair_class_minima_independently_reconstructed': False,
    'source_state_id_fully_reconstructed_from_old_qpos_only_trace': False,
    'closed_loop_robustness_established': False,
    'runtime_certificate_applies_to_shadow': False,
    'continuous_time_collision_certified': False,
    'wall_deployment_certified': False,
    'hardware_established': False,
    'new_acceptance_claim': False}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def source_files():
    result = {}
    excluded = {'output','__pycache__','cache','caches','.cache','.pytest_cache',
                '.mypy_cache','.ruff_cache','.git'}
    for package in ('v6_lite', 'model_test'):
        for directory, children, names in os.walk(ROOT / package, followlinks=False):
            children[:] = sorted(n for n in children if n not in excluded
                                 and not (Path(directory) / n).is_symlink()
                                 and not (Path(directory) / n).is_junction())
            for name in sorted(names):
                path = Path(directory) / name
                if name.endswith('.py') and not path.is_symlink():
                    result[path.relative_to(ROOT).as_posix()] = file_record(path)
    return result


def assets():
    urdf = ROOT / 'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf'
    root = ET.parse(urdf).getroot()
    compiler = root.find('mujoco/compiler')
    meshdir = '' if compiler is None else compiler.attrib.get('meshdir', '')
    paths = {urdf.resolve()}
    for mesh in root.findall('.//mesh'):
        if mesh.attrib.get('filename'):
            paths.add((urdf.parent / meshdir / mesh.attrib['filename']).resolve())
    return {p.relative_to(ROOT).as_posix(): file_record(p)
            for p in sorted(paths, key=lambda p: str(p).lower())}


def git_state():
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip()
    status = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],
                                     cwd=ROOT).decode().splitlines()
    return {'head': head, 'tracked_status': status}


def baseline_records():
    require(sha(BASELINE / 'manifest.json') == BASELINE_MANIFEST_SHA,
            'baseline manifest pin mismatch')
    manifest = read(BASELINE / 'manifest.json')
    require(len(manifest) == 46, 'baseline must retain all 46 artifacts')
    result = {}
    for name, digest in manifest.items():
        path = (BASELINE / name.replace('\\','/')).resolve()
        require(path.is_relative_to(BASELINE.resolve()), 'baseline manifest escaped')
        record = file_record(path)
        require(record['sha256'] == digest, 'baseline artifact changed: ' + name)
        result[name.replace('\\','/')] = record
    require(read(BASELINE / 'plan.json')['source']['git_commit'] == BASELINE_COMMIT,
            'baseline source commit differs')
    return result


def artifact_manifest(directory):
    return {p.relative_to(directory).as_posix(): file_record(p)
            for p in sorted(directory.rglob('*')) if p.is_file()
            and p != directory / 'artifact_manifest.json'}


def inherited_records():
    require(sha(INHERITED / 'audit_saved_inertia_shadow.py') == INHERITED_SCRIPT_SHA,
            'inherited auditor script pin mismatch')
    require(sha(INHERITED / 'replay/report.json') == INHERITED_REPORT_SHA,
            'inherited audit report pin mismatch')
    require(sha(INHERITED / 'replay/artifact_manifest.json') == INHERITED_MANIFEST_SHA,
            'inherited audit manifest pin mismatch')
    manifest = read(INHERITED / 'replay/artifact_manifest.json')
    result = {}
    for name, record in manifest.items():
        path = (INHERITED / 'replay' / name).resolve()
        require(path.is_relative_to((INHERITED / 'replay').resolve()), 'inherited path escaped')
        require(file_record(path) == record, 'inherited audit artifact changed: ' + name)
        result['replay/' + name] = record
    for name in ('audit_saved_inertia_shadow.py','plan.json','source_manifest.json',
                 'prepare_manifest.json','migration_receipt.json','replay/artifact_manifest.json'):
        result[name] = file_record(INHERITED / name)
    report = read(INHERITED / 'replay/report.json')
    require(report['evidence_valid'] and report['passed_checks'] == 2395 and report['total_checks'] == 2395
            and report['result']['physics_steps_independently_replayed'] == 100
            and report['result']['saved_physics_steps_verified'] == 202500,
            'inherited evidence was not the qualified bounded replay audit')
    return result


def completed_producer_records():
    require(sha(TRIAL / 'manifest.json') == FINAL_MANIFEST_SHA, 'completed observer manifest pin differs')
    manifest = read(TRIAL / 'manifest.json')
    require(len(manifest) == 146, 'completed observer coverage differs')
    result = {}
    for name, digest in manifest.items():
        path = (TRIAL / name).resolve()
        require(path.is_relative_to(TRIAL.resolve()), 'completed observer path escaped')
        record = file_record(path)
        require(record['sha256'] == digest, 'completed observer file changed: ' + name)
        result[name] = record
    return result


def prepare():
    require(not (OUTPUT / 'plan.json').exists(), 'audit plan already exists')
    source = source_files(); original = baseline_records(); state = git_state()
    require(state == {'head': SOURCE_COMMIT, 'tracked_status': []}, 'core is not frozen source')
    producer_plan = read(TRIAL / 'plan.json')
    require(producer_plan['schema'] == 'inertia_shadow_plan_v1'
            and producer_plan['current_source']['git_commit'] == SOURCE_COMMIT
            and producer_plan['smoke_steps'] is None, 'producer predeclaration differs')
    require(sha(ROOT / 'v6_lite/run_research_inertia_shadow.py') == PRODUCER_SHA
            and producer_plan['tool_sha256'] == PRODUCER_SHA, 'producer source pin differs')
    asset_inventory = assets(); inheritance = inherited_records(); produced = completed_producer_records()
    write(OUTPUT / 'source_manifest.json', {'git': state, 'source_files': source,
          'model_assets': asset_inventory, 'baseline_artifacts': original,
          'producer_predeclared_plan': file_record(TRIAL / 'plan.json'),
          'completed_producer_artifacts': produced, 'inherited_audit02_artifacts': inheritance})
    plan = {'schema': 'independent_inertia_shadow_geometry_audit_plan_v1',
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PREPARED_NOT_RUN',
            'source_commit': SOURCE_COMMIT, 'script_sha256': sha(__file__),
            'source_manifest_sha256': sha(OUTPUT / 'source_manifest.json'),
            'producer_path': str(TRIAL), 'baseline_path': str(BASELINE),
            'producer_sha256': PRODUCER_SHA, 'baseline_manifest_sha256': BASELINE_MANIFEST_SHA,
            'completed_observer_manifest_sha256': FINAL_MANIFEST_SHA,
            'inherited_audit02_report_sha256': INHERITED_REPORT_SHA,
            'inherited_audit02_manifest_sha256': INHERITED_MANIFEST_SHA,
            'alpha_order': list(ALPHAS), 'scene_count': 5, 'steps_per_run': STEPS,
            'state_count_per_run': STEPS + 1, 'physics_period_s': DT,
            'new_physics_steps': 0, 'inherited_nominal_physical_total_steps': 100,
            'target_fixed_state_indices': [0, 4500, 9000, 13500],
            'whole_body_fixed_state_indices': [0, 1350, 2700, 4050, 5400],
            'frame_and_reference_indices': [0,1,2249,2250,2251,4500,9000,12749,12750,12751,13499,13500],
            'include_reported_worst_state': True,
            'unfavorable_case_explicitly_included': 'scene_02_alpha_0.95, saved minimum and full stored phase statistics',
            'saved_record_statistics_scope': 'all saved minima and inexpensive saved-frame position-error statistics',
            'geometry_status': 'NOT_RUN', 'limitations': LIMITS}
    write(OUTPUT / 'plan.json', plan)
    write(OUTPUT / 'prepare_manifest.json', {name: file_record(OUTPUT / name) for name in
          ('audit_saved_inertia_geometry.py','source_manifest.json','plan.json')})
    return {'status': 'PREPARED_NOT_RUN', 'source_count': len(source),
            'asset_count': len(asset_inventory), 'baseline_artifact_count': len(original),
            'completed_producer_artifact_count': len(produced), 'script_sha256': plan['script_sha256']}


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, key, passed, details=None):
        result = bool(passed)
        self.rows.append({'check': key, 'passed': result, 'details': details})
        return result

    def need(self, key, passed, details=None):
        require(self.add(key, passed, details), key)


def stable(checks, prefix):
    plan = read(OUTPUT / 'plan.json')
    checks.need(prefix + ':script_pin', sha(__file__) == plan['script_sha256'])
    checks.need(prefix + ':source_manifest_pin', sha(OUTPUT / 'source_manifest.json')
                == plan['source_manifest_sha256'])
    manifest = read(OUTPUT / 'source_manifest.json')
    checks.need(prefix + ':source_frozen', git_state() == manifest['git']
                and source_files() == manifest['source_files'])
    checks.need(prefix + ':assets_frozen', assets() == manifest['model_assets'])
    checks.need(prefix + ':baseline_46_unchanged', baseline_records() == manifest['baseline_artifacts'])
    checks.need(prefix + ':producer_predeclared_plan_unchanged',
                file_record(TRIAL / 'plan.json') == manifest['producer_predeclared_plan'])
    checks.need(prefix + ':inherited_audit02_unchanged', inherited_records() == manifest['inherited_audit02_artifacts'])
    checks.need(prefix + ':all_146_completed_producer_files_unchanged', completed_producer_records() == manifest['completed_producer_artifacts'])
    return plan


def verify_manifest(path, pin, checks, prefix):
    checks.need(prefix + ':external_manifest_pin', bool(pin) and sha(path) == pin)
    manifest = read(path)
    require(isinstance(manifest, dict) and manifest, 'empty producer manifest')
    inventory = {}
    for name, digest in manifest.items():
        target = (TRIAL / name.replace('\\','/')).resolve()
        checks.need(prefix + ':owned:' + name, target.is_relative_to(TRIAL.resolve()))
        record = file_record(target)
        checks.need(prefix + ':sha:' + name, record['sha256'] == digest)
        inventory[name.replace('\\','/')] = record
    return inventory


def numeric_modules():
    # MuJoCo 3.3.2 Windows path parser needs the original ASCII relative URDF.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import mujoco
    from model_test.robot_model_spec_v5 import RobotModelSpecV5
    from model_test.whole_body_verifier_v5 import (
        WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere)
    return np, mujoco, RobotModelSpecV5, WholeBodyCollisionVerifier, WholeBodyVerificationConfig, WorkspaceSphere


def model_hash(model, np):
    digest = hashlib.sha256(CONTRACT.encode())
    for name in MODEL_FIELDS:
        digest.update(getattr(model, name).tobytes())
    digest.update(np.asarray([model.opt.timestep, int(model.opt.integrator), int(model.opt.solver),
                  int(model.opt.disableflags), int(model.opt.enableflags), model.opt.iterations,
                  model.opt.tolerance, model.opt.density, model.opt.viscosity]).tobytes())
    digest.update(model.opt.gravity.tobytes())
    return digest.hexdigest()


def array_record(value, np):
    value = np.ascontiguousarray(value)
    return {'sha256_raw_c_order': hashlib.sha256(value.tobytes()).hexdigest(),
            'dtype': value.dtype.str, 'shape': list(value.shape)}


def load_npz(path, np):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def options(model, np):
    result = {}
    for name in dir(model.opt):
        if name.startswith('_'):
            continue
        value = getattr(model.opt, name)
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, (int, float, np.number)):
            result[name] = value.item() if isinstance(value, np.number) else value
        elif hasattr(value, '__int__'):
            result[name] = int(value)
    return result


def construct(index, scenario, np, mj, cls, verifier_cls, config_cls, obstacle_cls, checks):
    urdf = Path('dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf')
    spec = cls.from_urdf(urdf, continuum_kp=100., continuum_kd=5., continuum_torque_limit=4.,
                         rigid_kp=200., rigid_kd=20., rigid_torque_limit=80.)
    checks.need(f'scene{index}:model_contract', spec.runtime_contract_sha256() == CONTRACT
                and spec.source_bundle_sha256() == BUNDLE)
    # Independently assemble MuJoCo model; do not invoke new producer model helpers.
    description = mj.MjSpec.from_file(str(urdf))
    description.option.gravity[:] = 0.
    description.option.timestep = DT
    description.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    for name, armature, damping, limit in zip(spec.low_level_joint_names,
            spec.joint_armature, spec.joint_damping, spec.torque_limits):
        joint = description.joint(name)
        joint.armature, joint.damping = float(armature), float(damping)
        motor = description.add_actuator(name='v5_torque_' + name, target=name)
        motor.trntype, motor.gaintype = mj.mjtTrn.mjTRN_JOINT, mj.mjtGain.mjGAIN_FIXED
        motor.gainprm[0], motor.gear[0] = 1., 1.
        motor.biastype = mj.mjtBias.mjBIAS_NONE
        motor.ctrllimited = motor.forcelimited = True
        motor.ctrlrange[:], motor.forcerange[:] = (-float(limit), float(limit)), (-float(limit), float(limit))
    obstacles = []
    for k, obstacle in enumerate(scenario['workspace_obstacles']):
        obstacles.append(obstacle_cls(str(obstacle['name']), np.array(obstacle['center_w']),
                                       float(obstacle['radius_m'])))
        description.worldbody.add_geom(name=f'v5_workspace_sphere_{k:03d}_' + obstacle['name'],
            type=mj.mjtGeom.mjGEOM_SPHERE, size=[float(obstacle['radius_m']),0.,0.],
            pos=obstacle['center_w'], contype=1, conaffinity=1)
    model = description.compile()
    # The established original verifier is used only to bind its named pair policy.
    verifier = verifier_cls(spec, obstacles, config_cls(minimum_clearance=.005,
        query_distance_max=2.5, adaptive_subdivisions=4,
        self_collision_ancestor_exclusion_depth=3, include_target_satellite_pairs=True))
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    verifier.model.geom_contype[:] = verifier.model.geom_conaffinity[:] = 0
    checks.need(f'scene{index}:manual_compile_matches_original_verifier',
                (model.nq, model.nv, model.nu) == (81,79,67)
                and all(np.array_equal(getattr(model,n),getattr(verifier.model,n)) for n in DIRECT)
                and options(model,np) == options(verifier.model,np))
    pairs = verifier.pairs_for_model(model)
    return spec, model, pairs, verifier._pair_policy_sha256


def dense_configuration(model, raw, index, np, mj):
    if index==5400:
        return raw[-1].copy()
    segment,quarter=divmod(index,4)
    q0,q1=raw[segment*10],raw[(segment+1)*10]
    velocity=np.empty(model.nv)
    mj.mj_differentiatePos(model,velocity,1.,q0,q1)
    current=q0.copy(); mj.mj_integratePos(model,current,velocity,quarter/4.)
    return current


def distance_sample(model,pairs,qpos,qvel,phase,maximum,winner_index,np,mj):
    observer=mj.MjData(model)
    observer.qpos[:],observer.qvel[:],observer.time=qpos,qvel,phase*DT
    mj.mj_forward(model,observer)
    values=[]; flags=[]; fromto=np.empty(6); classes={}; winner_witness=None
    for index,pair in enumerate(pairs):
        raw=float(mj.mj_geomDistance(model,observer,int(pair.geom_a),int(pair.geom_b),maximum,fromto))
        require(np.isfinite(raw),'native distance nonfinite')
        censor=raw>=maximum-1e-12; value=maximum if censor else raw
        values.append(value); flags.append(censor)
        if index==winner_index and not censor:
            winner_witness=fromto.tolist()
        classes[pair.pair_class]=min(classes.get(pair.pair_class,float('inf')),value)
    k=min(range(len(values)),key=lambda n:(values[n],flags[n]))
    return {'minimum_m':values[k],'minimum_censored_lower_bound':flags[k],
            'minimum_pair_index':k,'truncated_query_count':sum(flags),
            'query_count':len(pairs),'class_minimum_m':classes,
            'stored_winner_witness_m':winner_witness,
            'censoring_lower_bound_for_raw_near_max':maximum-1e-12,
            'raw_values':values,'censored_flags':flags}


def position_statistics(value,np):
    require(value.size and np.all(np.isfinite(value)), 'nonfinite saved-frame statistic')
    return {'count':int(value.size),'mean':float(np.mean(value)),
            'rmse':float(np.sqrt(np.mean(value**2))), 'p95':float(np.percentile(value,95.)),
            'max':float(np.max(value)), 'min':float(np.min(value))}


def compare_statistics(actual,expected):
    return set(actual)==set(expected) and actual['count']==expected['count'] and all(
        abs(actual[k]-expected[k])<=1e-12 for k in actual if k!='count')


def independent_reference(definition,time_s,np):
    # Reproduce the declared minimum-jerk interpolation from frozen points;
    # no new scenario RNG calls and no call to controller/reference helpers.
    t=max(float(time_s),0.); transition=float(definition['transition_duration_s'])
    points=np.asarray(definition['waypoint_points_m']); durations=np.asarray(definition['segment_durations_s'])
    if t<transition:
        duration=max(transition,1e-12); tau=t/duration
        start=np.asarray(definition['initial_position_w']); delta=points[0]-start
    else:
        elapsed=t-transition
        if elapsed>=float(definition['path_duration_s']):
            return points[-1].copy(),np.zeros(3)
        cumulative=np.concatenate([np.zeros(1),np.cumsum(durations)])
        segment=int(np.clip(np.searchsorted(cumulative,elapsed,side='right')-1,0,len(durations)-1))
        duration=max(float(durations[segment]),1e-12); tau=(elapsed-cumulative[segment])/duration
        start=points[segment]; delta=points[segment+1]-start
    tau=float(np.clip(tau,0.,1.))
    progress=tau**3*(10.-15.*tau+6.*tau**2)
    rate=30.*tau**2*(1.-tau)**2/duration
    return start+progress*delta,rate*delta


def phase_reference_audit(key,model,spec,trajectory,frames,old,scenario,declared,plan,np,mj,checks):
    times=trajectory['time'][1:]
    checks.need(key+':original_postintegration_reference_clock',np.array_equal(times,old['time']))
    definition=scenario['continuum_target']
    checks.need(key+':original_phase_contract',float(definition['path_start_s'])==4.5
        and float(definition['path_end_s'])==25.5 and float(definition['transition_duration_s'])==4.5
        and float(definition['path_duration_s'])==21.)
    path=(times>=float(definition['path_start_s']))&(times<=float(definition['path_end_s']))
    steady=times>=25.5
    rigid_errors=np.linalg.norm(frames['rigid_tip'][1:]-frames['rigid_target'][1:],axis=1)
    continuum_errors=np.linalg.norm(frames['continuum_tip'][1:]-old['continuum_target'],axis=1)
    computed={'rigid_position_error_m':position_statistics(rigid_errors,np),
              'continuum_position_error_m':position_statistics(continuum_errors,np),
              'continuum_original_path_window_error_m':position_statistics(continuum_errors[path],np),
              'rigid_original_last_1_5s_position_error_m':position_statistics(rigid_errors[steady],np),
              'rigid_final_position_error_m':float(rigid_errors[-1])}
    phase=declared['original_phase_window_diagnostic_statistics']
    checks.need(key+':all_saved_position_statistics_recomputed',
        compare_statistics(computed['rigid_position_error_m'],declared['tracking_statistics']['rigid_position_error_m'])
        and compare_statistics(computed['continuum_position_error_m'],declared['tracking_statistics']['continuum_position_error_m'])
        and compare_statistics(computed['continuum_original_path_window_error_m'],phase['continuum_original_path_window_error_m'])
        and compare_statistics(computed['rigid_original_last_1_5s_position_error_m'],phase['rigid_original_last_1_5s_position_error_m'])
        and abs(computed['rigid_final_position_error_m']-phase['rigid_final_position_error_m'])<=1e-12,computed)
    checks.need(key+':fresh_phase_claim_scope',declared['original_acceptance_relabelled'] is False)
    rid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,spec.rigid_tip_body_name)
    cid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,spec.continuum_tip_body_name)
    tid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'target_satellite')
    grasp_p=np.asarray(scenario['grasp_point_target_frame_m']); grasp_R=np.asarray(scenario['grasp_rotation_target_frame'])
    observer=mj.MjData(model); samples=[]
    selected=set(plan['frame_and_reference_indices'])
    selected.add(int(declared['_saved_target_worst_state']))
    for state in sorted(selected):
        observer.qpos[:],observer.qvel[:],observer.time=trajectory['qpos'][state],trajectory['qvel'][state],float(trajectory['time'][state])
        mj.mj_forward(model,observer)
        rr=observer.xmat[rid].reshape(3,3); cr=observer.xmat[cid].reshape(3,3); tr=observer.xmat[tid].reshape(3,3)
        fresh={'rigid_tip':observer.xpos[rid].copy(),
               'continuum_tip':observer.xpos[cid]+cr@np.array([.0475,0.,0.]),
               'rigid_target':observer.xpos[tid]+tr@grasp_p,
               'rigid_rotation':rr,'continuum_rotation':cr,'rigid_target_rotation':tr@grasp_R}
        frame_residual={n:float(np.max(np.abs(value-frames[n][state]))) for n,value in fresh.items()}
        checks.need(key+f':phase_fresh_frame{state}',all(v<=1e-11 for v in frame_residual.values()),frame_residual)
        reference_residual=None
        if state>0:
            p,v=independent_reference(definition,float(times[state-1]),np)
            reference_residual={'position_m':float(np.max(np.abs(p-old['continuum_target'][state-1]))),
                                'velocity_m_s':float(np.max(np.abs(v-old['continuum_target_velocity'][state-1])))}
            checks.need(key+f':poststep_reference{state}',max(reference_residual.values())<=1e-12,reference_residual)
        samples.append({'state':state,'time_s':float(trajectory['time'][state]),
                        'frame_residual':frame_residual,'reference_residual':reference_residual})
    return {'statistics_from_all_saved_frames':computed,'path_saved_sample_count':int(np.sum(path)),
            'steady_saved_sample_count':int(np.sum(steady)),'independent_frame_reference_samples':samples,
            'metric_scope':'inexpensive arithmetic on saved fresh frames plus limited independently refreshed frame/reference samples; no new 25/11 acceptance'}


def geometry_audit(pin,stage,checks):
    plan=stable(checks,'before')
    replay=read(INHERITED/'replay/report.json')
    checks.need('prior_independent_replay_evidence_valid',replay['evidence_valid'])
    inventory=verify_manifest(TRIAL/'manifest.json',pin,checks,'observer_manifest')
    report=read(TRIAL/'report.json')
    checks.need('producer_full_observation_complete',report['complete'] and report['evidence_valid']
        and report['status']=='DIAGNOSTIC_COMPLETE' and report['physics_steps_completed']==202500)
    expected=[(i,a,f'scene_{i:02d}_alpha_{a:.2f}') for a in ALPHAS for i in range(5)]
    checks.need('all_15_observer_runs_unique',[(r['scenario_index'],r['alpha'],r['key']) for r in report['runs']]==expected)
    checks.need('no_runtime_or_robustness_certificate_claim',report['limitations']['closed_loop_robustness_established'] is False
        and report['limitations']['continuous_time_collision_certified'] is False
        and report['limitations']['runtime_certificate_reused'] is False)
    required={'plan.json','producer.py','input_identities.json','replay_report.json','replay_manifest.json','report.json'}
    required.update('replays/'+key+'/'+n for _,_,key in expected for n in ('model_parameters.npz','model.json','trajectory.npz','replay.json'))
    required.update('observations/'+key+'/'+n for _,_,key in expected for n in
        ('observer_plan.json','robot_target_500hz.npz','whole_body_dense.npz','fresh_kinematics.npz','observation.json'))
    required.update({'interrupted_source_inventory.json','original_goal_pause_receipt.json',
                     'resume_observations.py','resumption_plan.json','test_qualification.json'})
    checks.need('exact_complete_artifact_coverage',set(inventory)==required and not (TRIAL/'failure.json').exists())
    checks.need('completed_observer_pinned_source_replay',sha(TRIAL/'replay_manifest.json')==REPLAY_MANIFEST_SHA
        and report['caller_pinned_replay_manifest_sha256']==REPLAY_MANIFEST_SHA)
    resumption=read(TRIAL/'resumption_plan.json')
    checks.need('resumption_14_reused_1_new_0_physics',resumption['inherited_complete_keys']==[v[2] for v in expected[:-1]]
        and resumption['new_observation_keys']==[expected[-1][2]]
        and resumption['physics_steps_replayed_by_resumption']==0
        and report['resumption']['complete_observations_reused']==14
        and report['resumption']['new_observations_computed']==1
        and report['resumption']['physics_steps_recomputed']==0)
    original_inventory=read(TRIAL/'interrupted_source_inventory.json')
    checks.need('resumption_interrupted_inventory_pin',sha(TRIAL/'interrupted_source_inventory.json')
        ==resumption['interrupted_source_inventory_sha256'])
    for filename,record in original_inventory.items():
        path=(ORIGINAL/filename).resolve()
        checks.need('original_interrupted_source_unchanged:'+filename,path.is_relative_to(ORIGINAL.resolve()) and file_record(path)==record)
    replay_manifest=read(TRIAL/'replay_manifest.json')
    for filename,digest in replay_manifest.items():
        checks.need('all_64_replay_files_reused:'+filename,sha(ORIGINAL/filename)==digest and sha(TRIAL/filename)==digest)
    for key in resumption['inherited_complete_keys']:
        for filename in ('observer_plan.json','robot_target_500hz.npz','whole_body_dense.npz','fresh_kinematics.npz','observation.json'):
            relative='observations/'+key+'/'+filename
            checks.need('all_14_complete_observations_reused:'+relative,file_record(TRIAL/relative)==original_inventory[relative])
    np,mj,cls,vcls,ccls,ocls=numeric_modules()
    metrics=read(BASELINE/'simulation/v6_lite_metrics.json')
    all_samples=[]; query_count=0; summaries=[]
    rowmap={r['key']:r for r in report['runs']}
    for index,item in enumerate(metrics['scenarios']):
        with np.load(BASELINE/'simulation/traces'/f'v6_lite_scenario_{index:02d}.npz',allow_pickle=False) as archive:
            old={n:archive[n].copy() for n in ('time','continuum_target','continuum_target_velocity')}
        spec,model,pairs,policy=construct(index,item['scenario'],np,mj,cls,vcls,ccls,ocls,checks)
        target=tuple(p for p in pairs if p.pair_class.endswith('_target'))
        checks.need(f'scene{index}:original_compiled_pair_counts',len(pairs)==2927 and len(target)==75)
        named=[{'pair_class':p.pair_class,'geom_a':p.geom_a_name,'geom_b':p.geom_b_name,
                'geom_a_id':int(p.geom_a),'geom_b_id':int(p.geom_b)} for p in pairs]
        for alpha in ALPHAS:
            key=f'scene_{index:02d}_alpha_{alpha:.2f}'; obs=TRIAL/'observations'/key
            observer_plan=read(obs/'observer_plan.json'); observation=read(obs/'observation.json')
            checks.need(key+':observer_plan_and_source',observer_plan['pairs']==named
                and observer_plan['pair_policy_sha256']==policy and observer_plan['observer_model_id']==model_hash(model,np)
                and observer_plan['whole_body_pair_count']==2927 and observer_plan['target_pair_count']==75
                and observation==rowmap[key] and observation['status']=='OBSERVATION_COMPLETED')
            trajectory=load_npz(TRIAL/'replays'/key/'trajectory.npz',np)
            kinematic=load_npz(obs/'fresh_kinematics.npz',np)
            checks.need(key+':observer_trajectory_binding',observer_plan['input_trajectory_sha256']==sha(TRIAL/'replays'/key/'trajectory.npz'))
            checks.need(key+':fresh_frame_saved_layout',np.array_equal(kinematic['time'],trajectory['time'])
                and all(kinematic[n].shape==(13501,3) and np.all(np.isfinite(kinematic[n]))
                        for n in ('rigid_tip','continuum_tip','rigid_target'))
                and all(kinematic[n].shape==(13501,3,3) and np.all(np.isfinite(kinematic[n]))
                        for n in ('rigid_rotation','continuum_rotation','rigid_target_rotation')))
            phase_declaration=dict(observation['fresh_kinematics'])
            phase_declaration['_saved_target_worst_state']=observation['robot_target_500hz']['minimum_pair']['state_index']
            phase_result=phase_reference_audit(key,model,spec,trajectory,kinematic,old,item['scenario'],
                                              phase_declaration,plan,np,mj,checks)
            write(stage/(key+'_phase_reference.json'),phase_result)
            scopes=[('robot_target_500hz','robot_target_500hz.npz',target,13501,.25,plan['target_fixed_state_indices']),
                    ('whole_body_dense_discrete','whole_body_dense.npz',pairs,5401,2.5,plan['whole_body_fixed_state_indices'])]
            for label,filename,current_pairs,count,maximum,fixed in scopes:
                arrays=load_npz(obs/filename,np); declared=observation[label]
                values=arrays['minimum_m']; flags=arrays['minimum_censored_lower_bound']; winners=arrays['minimum_pair_index']
                phase=np.arange(count,dtype=float)*(1. if count==13501 else 2.5)
                checks.need(key+':'+label+':all_saved_record_shapes',values.shape==(count,) and flags.shape==(count,)
                    and winners.shape==(count,) and np.array_equal(arrays['physics_grid_phase'],phase)
                    and bool(arrays['complete']) and int(arrays['planned_state_count'])==count
                    and int(arrays['completed_state_count'])==count and int(arrays['completed_query_count'])==count*len(current_pairs)
                    and np.all(np.isfinite(values)) and np.all((winners>=0)&(winners<len(current_pairs))))
                best=min(range(count),key=lambda n:(values[n],flags[n])); worst=declared['minimum_pair']
                checks.need(key+':'+label+':saved_summary_arithmetic',declared['checked_state_count']==count
                    and declared['pair_count']==len(current_pairs) and declared['query_count']==count*len(current_pairs)
                    and declared['distance_max_m']==maximum and declared['minimum_clearance_m']==float(values[best])
                    and declared['minimum_is_censored_lower_bound']==bool(flags[best])
                    and declared['below_required_clearance_state_count']==int(np.sum(values<.005))
                    and declared['negative_native_signed_state_count']==int(np.sum(values<0.))
                    and declared['measured_discrete_clearance_at_least_gate']==bool(np.min(values)>=.005)
                    and declared['continuous_time_certified'] is False and declared['nativeccd_claimed'] is False)
                checks.need(key+':'+label+':censor_bound_scope',np.all(values[flags]==maximum)
                    and np.all(values[~flags]<maximum-1e-12)
                    and int(np.sum(flags))*len(current_pairs)<=declared['truncated_query_count']<=count*len(current_pairs))
                wi=int(worst['state_index'])
                checks.need(key+':'+label+':worst_state_identity',0<=wi<count and float(worst['physics_grid_phase'])==phase[wi]
                    and float(worst['time_s'])==phase[wi]*DT and worst['native_signed_distance_or_lower_bound_m']==float(values[wi])
                    and worst['censored_lower_bound']==bool(flags[wi])
                    and current_pairs[int(winners[wi])].geom_a_name==worst['geom_a']
                    and current_pairs[int(winners[wi])].geom_b_name==worst['geom_b'])
                selected=sorted(set(fixed+[wi])); scoped_samples=[]
                for state in selected:
                    if count==13501:
                        q,v=trajectory['qpos'][state],trajectory['qvel'][state]
                    else:
                        q=dense_configuration(model,trajectory['qpos'],state,np,mj); v=np.zeros(model.nv)
                    fresh=distance_sample(model,current_pairs,q,v,float(phase[state]),maximum,int(winners[state]),np,mj)
                    query_count+=fresh['query_count']
                    checks.need(key+':'+label+f':fresh_native_state{state}',abs(fresh['minimum_m']-float(values[state]))<=1e-10
                        and fresh['minimum_censored_lower_bound']==bool(flags[state]),
                        {'saved':float(values[state]),'fresh':fresh['minimum_m'],'phase':float(phase[state])})
                    # Stored winner may tie another pair; independently query that exact pair.
                    checks.need(key+':'+label+f':stored_winner{state}',
                        abs(fresh['raw_values'][int(winners[state])]-float(values[state]))<=1e-10
                        and fresh['censored_flags'][int(winners[state])]==bool(flags[state]))
                    if state==wi:
                        checks.need(key+':'+label+':worst_witness_censor_scope',
                            (worst['witness_fromto_m'] is None)==bool(flags[state])
                            and (bool(flags[state]) or np.max(np.abs(np.array(worst['witness_fromto_m'])
                                -np.array(fresh['stored_winner_witness_m'])))<=1e-9))
                    if count==13501:
                        frame=mj.MjData(model); frame.qpos[:],frame.qvel[:],frame.time=q,v,phase[state]*DT
                        mj.mj_forward(model,frame)
                        rid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,spec.rigid_tip_body_name)
                        cid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,spec.continuum_tip_body_name)
                        rr=frame.xmat[rid].reshape(3,3); cr=frame.xmat[cid].reshape(3,3)
                        checks.need(key+f':fresh_frame_state{state}',
                            np.max(np.abs(frame.xpos[rid]-kinematic['rigid_tip'][state]))<=1e-11
                            and np.max(np.abs(frame.xpos[cid]+cr@np.array([.0475,0.,0.])-kinematic['continuum_tip'][state]))<=1e-11
                            and np.max(np.abs(rr-kinematic['rigid_rotation'][state]))<=1e-11
                            and np.max(np.abs(cr-kinematic['continuum_rotation'][state]))<=1e-11)
                    scoped_samples.append({'state':state,'phase':float(phase[state]),'fresh':fresh})
                all_samples.append({'key':key,'scope':label,'samples':scoped_samples})
                summaries.append({'key':key,'scope':label,'saved_state_count':count,'saved_queries':count*len(current_pairs),
                                  'independent_state_count':len(selected),'minimum_m':float(np.min(values))})
            write(stage/(key+'_samples.json'),all_samples[-2:])
    verify_manifest(TRIAL/'manifest.json',pin,checks,'observer_manifest_after')
    stable(checks,'after')
    return {'producer_manifest_sha256':pin,'saved_geometric_minimum_records_verified':15*(13501+5401),
            'producer_total_native_queries_declared':252319530,'independent_native_query_count':query_count,
            'physics_steps_independently_replayed':0,'summaries':summaries,'limitations':LIMITS}


def execute(stage_name,pin):
    stage=OUTPUT/stage_name
    require(not stage.exists(),'independent stage output already exists')
    stage.mkdir()
    checks=Checks(); started=time.perf_counter(); result={}; error=None
    try:
        result=geometry_audit(pin,stage,checks)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    valid=error is None and all(r['passed'] for r in checks.rows)
    report={'schema':'independent_inertia_shadow_'+stage_name+'_audit_v1','stage':stage_name,
            'evidence_valid':valid,'passed_checks':sum(r['passed'] for r in checks.rows),
            'total_checks':len(checks.rows),'failed_checks':[r for r in checks.rows if not r['passed']],
            'scope_complete':valid,'full_independent_experiment_replayed':False,
            'shadow_is_runtime_certified':False,'research_acceptance_passed':None,
            'error':error,'elapsed_s':time.perf_counter()-started,'result':result,
            'limitations':LIMITS,'script_sha256':sha(__file__)}
    write(stage/'checks.json',checks.rows); write(stage/'report.json',report)
    write(stage/'terminal_execution.json',{'intended_exit_code':0 if valid else 1,
          'command':sys.argv,'python':sys.version,'stage':stage_name,
          'producer_manifest_pin':pin,'finished_utc':datetime.now(timezone.utc).isoformat()})
    write(stage/'artifact_manifest.json',artifact_manifest(stage))
    if stage_name=='geometry':
        prior=read(INHERITED/'replay/report.json')
        final_valid=valid and prior['evidence_valid']
        write(OUTPUT/'report.json',{'schema':'independent_inertia_shadow_final_audit_v1',
            'evidence_valid':final_valid,'complete':final_valid,
            'passed_checks':report['passed_checks']+prior['passed_checks'],
            'total_checks':report['total_checks']+prior['total_checks'],
            'inherited_replay_report':file_record(INHERITED/'replay/report.json'),
            'geometry_report':file_record(stage/'report.json'),
            'script_sha256':sha(__file__),'source_commit':SOURCE_COMMIT,
            'new_physics_steps':0,'inherited_independent_physics_steps':100,'producer_saved_physics_steps':202500,
            'producer_saved_geometry_queries':252319530,
            'limitations':LIMITS,'research_acceptance_passed':None})
        write(OUTPUT/'artifact_manifest.json',artifact_manifest(OUTPUT))
    return report,0 if valid else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('plan','geometry'))
    parser.add_argument('--producer-manifest-sha')
    args=parser.parse_args()
    if args.stage=='plan':
        result=prepare(); print(json.dumps(result,ensure_ascii=False)); return 0
    require(args.producer_manifest_sha,'caller must pin a completed producer manifest')
    report,code=execute(args.stage,args.producer_manifest_sha)
    print(json.dumps({k:report[k] for k in ('stage','evidence_valid','passed_checks','total_checks','error')},ensure_ascii=False))
    return code


if __name__=='__main__':
    raise SystemExit(main())
