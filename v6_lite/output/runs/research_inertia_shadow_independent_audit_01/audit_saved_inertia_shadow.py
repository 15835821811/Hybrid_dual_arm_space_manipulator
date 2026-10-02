"""Independent bounded audit of saved inertia-shadow evidence.

Plan stage uses only the standard library and never reads in-flight replay data.
Replay/geometry stages require caller-pinned completed producer manifests.
The new producer is never imported or called. Exactly 100 nominal mj_step calls
are allowed in replay audit; geometry audit never advances physics.
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

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / '.git').exists() and (p / 'model_test').is_dir())
OUTPUT = Path(__file__).resolve().parent
BASELINE = ROOT / 'v6_lite/output/runs/research_acceptance_01'
TRIAL = ROOT / 'v6_lite/output/runs/research_inertia_shadow_01'
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
    'physical_replay_scope': 'only nominal alpha=1 first 20 steps per scene; total 100 steps',
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


def prepare():
    require(not (OUTPUT / 'plan.json').exists(), 'audit plan already exists')
    source = source_files()
    original = baseline_records()
    state = git_state()
    require(state == {'head': SOURCE_COMMIT, 'tracked_status': []}, 'core is not frozen source')
    producer_plan = read(TRIAL / 'plan.json')  # predeclared plan only, never in-flight results
    require(producer_plan['schema'] == 'inertia_shadow_plan_v1'
            and producer_plan['current_source']['git_commit'] == SOURCE_COMMIT
            and producer_plan['smoke_steps'] is None, 'producer predeclaration differs')
    require(sha(ROOT / 'v6_lite/run_research_inertia_shadow.py') == PRODUCER_SHA
            and producer_plan['tool_sha256'] == PRODUCER_SHA, 'producer source pin differs')
    asset_inventory = assets()
    write(OUTPUT / 'source_manifest.json', {'git': state, 'source_files': source,
          'model_assets': asset_inventory, 'baseline_artifacts': original,
          'producer_predeclared_plan': file_record(TRIAL / 'plan.json')})
    plan = {'schema': 'independent_inertia_shadow_audit_plan_v1',
            'prepared_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'PREPARED_NOT_RUN', 'source_commit': SOURCE_COMMIT,
            'script_sha256': sha(__file__), 'source_manifest_sha256': sha(OUTPUT / 'source_manifest.json'),
            'producer_path': str(TRIAL), 'baseline_path': str(BASELINE),
            'producer_sha256': PRODUCER_SHA, 'baseline_manifest_sha256': BASELINE_MANIFEST_SHA,
            'alpha_order': list(ALPHAS), 'scene_count': 5, 'steps_per_run': STEPS,
            'state_count_per_run': STEPS + 1, 'physics_period_s': DT,
            'nominal_physical_steps_per_scene': 20, 'nominal_physical_total_steps': 100,
            'target_fixed_state_indices': [0, 4500, 9000, 13500],
            'whole_body_fixed_state_indices': [0, 1350, 2700, 4050, 5400],
            'include_reported_worst_state': True,
            'manifest_pins_required': ['completed replay_manifest SHA', 'completed observer manifest SHA'],
            'replay_status': 'NOT_RUN', 'geometry_status': 'NOT_RUN', 'limitations': LIMITS}
    write(OUTPUT / 'plan.json', plan)
    write(OUTPUT / 'prepare_manifest.json', {name: file_record(OUTPUT / name) for name in
          ('audit_saved_inertia_shadow.py','source_manifest.json','plan.json')})
    return {'status': 'PREPARED_NOT_RUN', 'source_count': len(source),
            'asset_count': len(asset_inventory), 'baseline_artifact_count': len(original),
            'script_sha256': plan['script_sha256']}


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
    urdf = ROOT / 'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf'
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


def robot_body_ids(model, spec, np, mj, checks, prefix):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'world_joint')
    tid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'target_world_joint')
    checks.need(prefix + ':both_named_free_joints', jid >= 0 and tid >= 0
        and int(model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE)
        and int(model.jnt_type[tid]) == int(mj.mjtJoint.mjJNT_FREE))
    root, target = int(model.jnt_bodyid[jid]), int(model.jnt_bodyid[tid])
    selected = []
    for body in range(1, model.nbody):
        ancestors = []
        current = body
        while current:
            ancestors.append(current)
            current = int(model.body_parentid[current])
        if root in ancestors:
            selected.append(body)
    checks.need(prefix + ':robot_root_name', mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,root) == 'base_of_satelltte')
    checks.need(prefix + ':world_and_target_excluded', 0 not in selected and target not in selected)
    joint_ids = [mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n) for n in spec.low_level_joint_names]
    checks.need(prefix + ':all_67_motor_joint_bodies_in_robot_subtree', len(joint_ids) == 67
                and all(j >= 0 and int(model.jnt_bodyid[j]) in selected for j in joint_ids))
    return np.array(selected,dtype=int), jid, tid, joint_ids


def mapping(np, spec, checks):
    result = np.zeros((67,17))
    pattern = (1,0,0,1,1,0,0,1,1,0,0,1)
    for segment in range(5):
        for row, column in enumerate(pattern):
            result[12*segment+row,2*segment+column] = 1./6.
    result[60:,10:] = np.eye(7)
    decoder = np.linalg.pinv(result)
    checks.need('manual_named_17_mapping', np.array_equal(result,spec.planner_to_low_level)
                and np.array_equal(decoder,spec.low_level_to_planner))
    return decoder


def certificate_model_checks(index, model, hashes, old, np, mj, checks):
    from v6_lite.runtime_command import CommandCertificate, DispatchGate
    path = BASELINE / 'simulation/timing' / f'v6_lite_scenario_{index:02d}.jsonl'
    with path.open(encoding='utf-8') as handle:
        row = json.loads(next(handle))
    c = CommandCertificate(**row['certificate'])
    command = np.ascontiguousarray(old['command_velocity'][0])
    checks.need(f'scene{index}:original_certificate_payload',
        hashlib.sha256(command.tobytes()).hexdigest() == c.command_sha256
        and c.source_model_hash == hashes[1.])
    results = {}
    for alpha in ALPHAS:
        gate = DispatchGate()
        before = (gate.last_command_id, gate.next_substep)
        result = gate.check(c, now=max(c.validation_finished_time, c.solve_finished_time),
            model_hash=hashes[alpha], partition_hash=c.source_partition_id, command=command,
            observed_state_id=c.source_state_id, observed_simulation_s=c.planned_execution_start_simulation_s,
            substep=0, microstate_matches=True, policy='research_simulation')
        if alpha == 1.:
            checks.need(f'scene{index}:nominal_certificate_model_binding_control', result['accepted'])
        else:
            checks.need(f'scene{index}:alpha{alpha}:old_model_certificate_rejected_without_step',
                        not result['accepted'] and result['reason'] == 'MODEL_ID_MISMATCH'
                        and before == (gate.last_command_id, gate.next_substep))
        results[str(alpha)] = result
    return {'scope':'pure saved-certificate model-binding check; original source_state_id copied, not reconstructed',
            'physics_steps':0, 'checks':results}


def replay_audit(pin, stage, checks):
    plan = stable(checks, 'before')
    inventory = verify_manifest(TRIAL/'replay_manifest.json',pin,checks,'replay_manifest')
    report = read(TRIAL/'replay_report.json')
    expected = [(i,a,f'scene_{i:02d}_alpha_{a:.2f}') for a in ALPHAS for i in range(5)]
    checks.need('full_saved_replay_report', report['complete'] and report['physics_steps_completed'] == 202500
                and [(r.get('scenario_index'),r.get('alpha'),r.get('key')) for r in report['runs']] == expected)
    required = {'plan.json','producer.py','input_identities.json','replay_report.json'}
    required.update('replays/'+key+'/'+n for _,_,key in expected
                    for n in ('model_parameters.npz','model.json','trajectory.npz','replay.json'))
    checks.need('exact_replay_manifest_coverage', set(inventory) == required)
    checks.need('saved_producer_source_sha', sha(TRIAL/'producer.py') == PRODUCER_SHA)
    np,mj,cls,vcls,ccls,ocls = numeric_modules()
    metrics = read(BASELINE/'simulation/v6_lite_metrics.json')
    producer_inputs = read(TRIAL/'input_identities.json')
    checks.need('five_unique_input_records', len(producer_inputs) == 5
                and [i['scenario']['scenario_id'] for i in producer_inputs] ==
                    [f'v6_lite_scenario_{i:02d}' for i in range(5)])
    by_scene, physical_steps = [], 0
    rows_by_key = {r['key']:r for r in report['runs']}
    for index,item in enumerate(metrics['scenarios']):
        prefix=f'scene{index}'
        old=load_npz(BASELINE/'simulation/traces'/f'v6_lite_scenario_{index:02d}.npz',np)
        scenario=item['scenario']
        checks.need(prefix+':frozen_input_scenario', producer_inputs[index]['scenario'] == scenario
            and producer_inputs[index]['scenario_sha256'] == hashlib.sha256(canonical(scenario)).hexdigest())
        checks.need(prefix+':original_67_torque_bytes', old['torque'].shape == (STEPS,67)
            and old['torque'].dtype == np.dtype('float64')
            and hashlib.sha256(old['torque'].tobytes()).hexdigest() == TORQUE_SHA[index])
        spec,model,pairs,policy=construct(index,scenario,np,mj,cls,vcls,ccls,ocls,checks)
        selected,base_jid,target_jid,joints=robot_body_ids(model,spec,np,mj,checks,prefix)
        qids=np.array([int(model.jnt_qposadr[j]) for j in joints]); dids=np.array([int(model.jnt_dofadr[j]) for j in joints])
        checks.need(prefix+':67_motor_named_order', [mj.mj_id2name(model,mj.mjtObj.mjOBJ_ACTUATOR,k)
            for k in range(model.nu)] == ['v5_torque_'+n for n in spec.low_level_joint_names])
        decoder=mapping(np,spec,checks)
        original={n:getattr(model,n).copy() for n in DIRECT}
        nominal_id=model_hash(model,np)
        checks.need(prefix+':compiled_model_and_pair_policy', nominal_id==item['execution_contract']['source_compiled_model_sha256']
            and policy==item['metrics']['whole_body_clearance']['pair_policy_sha256'] and len(pairs)==2927)
        saved_nominal=None; hashes={}; run_summaries=[]
        for alpha in ALPHAS:
            key=f'scene_{index:02d}_alpha_{alpha:.2f}'; rp=TRIAL/'replays'/key
            metadata=read(rp/'model.json'); replay=read(rp/'replay.json')
            saved=load_npz(rp/'trajectory.npz',np); parameters=load_npz(rp/'model_parameters.npz',np)
            checks.need(key+':complete_saved_run', replay['status']=='REPLAY_COMPLETED' and replay['full_27s_complete']
                and replay['steps_completed']==STEPS and replay['planned_steps']==STEPS
                and replay==rows_by_key[key] and replay['trajectory_sha256']==sha(rp/'trajectory.npz'))
            checks.need(key+':raw_torque_identical', array_record(saved['torque_input'],np)==array_record(old['torque'],np)
                        and np.array_equal(saved['torque_input'],old['torque']))
            checks.need(key+':raw_initial_state_and_grid', saved['qpos'].shape==(STEPS+1,81)
                and saved['qvel'].shape==(STEPS+1,79) and saved['time'].shape==(STEPS+1,)
                and np.array_equal(saved['qpos'][0],old['initial_qpos'])
                and np.array_equal(saved['qvel'][0],old['initial_qvel'])
                and np.all(np.isfinite(saved['qpos'])) and np.all(np.isfinite(saved['qvel']))
                and np.allclose(saved['time'],np.arange(STEPS+1)*DT,rtol=0.,atol=1e-9))
            checks.need(key+':saved_factor_and_timestep',float(saved['alpha'])==alpha
                and int(saved['scenario_index'])==index and float(saved['physics_period_s'])==DT)
            expected_I=original['body_inertia'].copy(); expected_I[selected]*=alpha
            checks.need(key+':direct_parameter_file_coverage', set(parameters)=={p+n for p in ('before_','after_') for n in DIRECT})
            for name in DIRECT:
                expected_after=expected_I if name=='body_inertia' else original[name]
                checks.need(key+':direct:'+name, np.array_equal(parameters['before_'+name],original[name])
                            and np.array_equal(parameters['after_'+name],expected_after))
                checks.need(key+':metadata_array_identity:'+name,
                    metadata['before'][name]==array_record(parameters['before_'+name],np)
                    and metadata['after'][name]==array_record(parameters['after_'+name],np))
            checks.need(key+':full_robot_selection', [r['body_id'] for r in metadata['selection']['bodies']]==selected.tolist()
                        and metadata['selection']['root_joint']=='world_joint'
                        and metadata['selection']['target_root_body_id_excluded']==int(model.jnt_bodyid[target_jid]))
            checks.need(key+':options_and_geometry_fixed', metadata['options_before_after']==options(model,np)
                        and metadata['same_geometry_and_actuator_contract'] and metadata['runtime_certificates_applicable'] is False)
            model.body_inertia[:]=expected_I
            hashes[alpha]=model_hash(model,np)
            checks.need(key+':mutable_compiled_model_hash', metadata['nominal_model_id']==nominal_id
                and metadata['effective_model_id']==hashes[alpha] and ((hashes[alpha]==nominal_id)==(alpha==1.)))
            # Reconstruct the saved *new* initial integration state exactly.
            data=mj.MjData(model); mj.mj_setConst(model,data)
            data.qpos[:],data.qvel[:],data.ctrl[:],data.time=old['initial_qpos'],old['initial_qvel'],0.,0.
            mj.mj_forward(model,data)
            initial=np.empty(mj.mj_stateSize(model,mj.mjtState.mjSTATE_INTEGRATION))
            mj.mj_getState(model,data,initial,mj.mjtState.mjSTATE_INTEGRATION)
            checks.need(key+':new_initial_integration_state', np.array_equal(initial,saved['initial_integration_state']))
            if alpha==1.:
                saved_nominal=saved
                planner_q=saved['qpos'][1:,qids] @ decoder.T
                planner_dq=saved['qvel'][1:,dids] @ decoder.T
                badr=int(model.jnt_qposadr[base_jid])
                residuals={'planner_q':float(np.max(np.abs(planner_q-old['planner_q']))),
                    'planner_dq':float(np.max(np.abs(planner_dq-old['planner_dq']))),
                    'base_qpos':float(np.max(np.abs(saved['qpos'][1:,badr:badr+7]-old['base_qpos']))),
                    'task_qpos':float(np.max(np.abs(saved['qpos'][::10]-old['task_qpos'])))}
                checks.need(key+':full_saved_projection_matches_old_trace', all(v<=1e-11 for v in residuals.values()),residuals)
                checks.need(key+':producer_nominal_parity_bounded', replay['nominal_parity']['passed']
                            and set(replay['nominal_parity']['maximum_absolute_residuals'])==
                                {'planner_q','planner_dq','base_qpos','task_qpos','rigid_tip','rigid_rotation',
                                 'continuum_tip','continuum_tip_body_origin','continuum_rotation'}
                            and max(replay['nominal_parity']['maximum_absolute_residuals'].values())<=1e-11)
            run_summaries.append({'key':key,'effective_model_hash':hashes[alpha],
                'body_count':len(selected),'saved_states':len(saved['time']),'saved_steps':STEPS})
            model.body_inertia[:]=original['body_inertia']
        # Exactly 20 steps per nominal scene. No perturbed physics replay.
        data=mj.MjData(model); mj.mj_setConst(model,data)
        data.qpos[:],data.qvel[:],data.ctrl[:],data.time=old['initial_qpos'],old['initial_qvel'],0.,0.
        mj.mj_forward(model,data)
        rigid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,spec.rigid_tip_body_name)
        continuum=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,spec.continuum_tip_body_name)
        sample=[]
        for step in range(20):
            data.ctrl[:]=old['torque'][step]; mj.mj_step(model,data); physical_steps+=1
            cr=data.xmat[continuum].reshape(3,3)
            residual={'qpos':float(np.max(np.abs(data.qpos-saved_nominal['qpos'][step+1]))),
                'qvel':float(np.max(np.abs(data.qvel-saved_nominal['qvel'][step+1]))),
                'rigid_tip':float(np.max(np.abs(data.xpos[rigid]-old['rigid_tip'][step]))),
                'continuum_tip':float(np.max(np.abs(data.xpos[continuum]+cr@np.array([.0475,0.,0.])-old['continuum_tip'][step]))),
                'rigid_rotation':float(np.max(np.abs(data.xmat[rigid].reshape(3,3)-old['rigid_rotation'][step]))),
                'continuum_rotation':float(np.max(np.abs(cr-old['continuum_rotation'][step])))}
            checks.need(prefix+f':nominal_physics_step{step}:raw_frame_parity', all(v<=1e-11 for v in residual.values())
                        and abs(data.time-(step+1)*DT)<=1e-9,residual)
            sample.append({'step':step,'time_s':float(data.time),'residuals':residual})
        cert=certificate_model_checks(index,model,hashes,old,np,mj,checks)
        by_scene.append({'index':index,'runs':run_summaries,'nominal_physical_samples':sample,
                         'certificate_model_binding':cert})
        write(stage/f'scene_{index:02d}.json',by_scene[-1])
    checks.need('exact_limited_independent_physics_scope',physical_steps==100)
    verify_manifest(TRIAL/'replay_manifest.json',pin,checks,'replay_manifest_after')
    stable(checks,'after')
    return {'producer_manifest_sha256':pin,'physics_steps_independently_replayed':physical_steps,
            'saved_runs_verified':15,'saved_physics_steps_verified':202500,'saved_state_vectors_verified':202515,
            'source_and_baseline_unchanged':True,'by_scene':by_scene,'limitations':LIMITS}


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


def geometry_audit(pin,stage,checks):
    plan=stable(checks,'before')
    require((OUTPUT/'replay/report.json').is_file(),'independent replay audit missing')
    replay=read(OUTPUT/'replay/report.json')
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
    checks.need('exact_complete_artifact_coverage',set(inventory)==required and not (TRIAL/'failure.json').exists())
    np,mj,cls,vcls,ccls,ocls=numeric_modules()
    metrics=read(BASELINE/'simulation/v6_lite_metrics.json')
    all_samples=[]; query_count=0; summaries=[]
    rowmap={r['key']:r for r in report['runs']}
    for index,item in enumerate(metrics['scenarios']):
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
        result=(replay_audit if stage_name=='replay' else geometry_audit)(pin,stage,checks)
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
        prior=read(OUTPUT/'replay/report.json')
        final_valid=valid and prior['evidence_valid']
        write(OUTPUT/'report.json',{'schema':'independent_inertia_shadow_final_audit_v1',
            'evidence_valid':final_valid,'complete':final_valid,
            'passed_checks':report['passed_checks']+prior['passed_checks'],
            'total_checks':report['total_checks']+prior['total_checks'],
            'replay_report':file_record(OUTPUT/'replay/report.json'),
            'geometry_report':file_record(stage/'report.json'),
            'script_sha256':sha(__file__),'source_commit':SOURCE_COMMIT,
            'independent_physics_steps':100,'producer_saved_physics_steps':202500,
            'producer_saved_geometry_queries':252319530,
            'limitations':LIMITS,'research_acceptance_passed':None})
        write(OUTPUT/'artifact_manifest.json',artifact_manifest(OUTPUT))
    return report,0 if valid else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('plan','replay','geometry'))
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
