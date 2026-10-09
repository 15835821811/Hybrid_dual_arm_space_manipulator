"""Finite A.1 development regression of four already frozen proposals.

No training, sampling, historical-output mutation, or force playback is an
execution mode of this command.  The old successful force replay is used
only for explicitly offline reference diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from v6_4.contracts import TaskSpec, TrajectoryProposal
from v6_4.reference_adapter import CartesianPassThroughReferenceProvider, scenario_from_task
from v6_4.run_planning import prepare_provider, run_attempt
from v6_4.trajectory_codec import CubicBSplineCodec
from v6_lite.run_v6_lite import default_v6_lite_robot_spec, _body_pose_and_twist
from v6_lite.hierarchical_qp import HierarchicalQPConfig

ROOT = Path(__file__).resolve().parents[1]


def write(path, payload):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_sources():
    paths = [*ROOT.joinpath('v6_4').glob('*.py'),
             *ROOT.joinpath('v6_lite').glob('*.py'),
             *ROOT.joinpath('model_test').glob('*.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def vector_error(actual, reference):
    e = np.asarray(actual)-np.asarray(reference)
    return {'RMS_L2': float(np.sqrt(np.mean(np.sum(e*e, axis=1)))),
            'maximum_L2': float(np.max(np.linalg.norm(e, axis=1))),
            'coordinate_RMSE': np.sqrt(np.mean(e*e, axis=0)).tolist()}


def diagnose(mapping, output):
    """R1 exact feedback snapshots and R2 derived-reference codec errors."""
    output.mkdir(parents=True, exist_ok=False)
    previous = Path(mapping['positive_control']['root'])
    task = TaskSpec.from_dict(json.loads((previous/'task.json').read_text(encoding='utf8')))
    prior = json.loads((previous/'result.json').read_text(encoding='utf8'))
    if prior.get('task_success') is not True or prior.get('complete') is not True:
        raise ValueError('R0 source is not a complete successful task')
    source = previous/'evaluation/fresh_replay.npz'
    with np.load(source, allow_pickle=False) as f:
        state = {k:f[k].copy() for k in f.files}
    if len(state['time']) != 13501:
        raise ValueError('R0 full physical-state denominator differs')
    spec = default_v6_lite_robot_spec()
    scene = scenario_from_task(task)
    codec = CubicBSplineCodec(task.initial_planner_q, task.initial_planner_dq)
    # This is derived from actual feedback for diagnostics; it is never
    # represented as the missing original Cartesian joint-reference input.
    controls = codec.fit(state['time'], state['q'])
    provider, verifier = prepare_provider(task, controls, spec)
    model = verifier.model
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = task.initial_qpos, task.initial_qvel
    mujoco.mj_forward(model, data)
    passthrough = CartesianPassThroughReferenceProvider(task).prepare(spec, model, data, scene)
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target_satellite')
    exact = {}
    for i in range(0, len(state['time']), 10):
        data.qpos[:], data.qvel[:], data.time = state['qpos'][i], state['qvel'][i], float(state['time'][i])
        mujoco.mj_forward(model, data)
        p,v,r,w = _body_pose_and_twist(model,data,target,scene.grasp_point_target_frame_m)
        cp,cv = scene.continuum_target.sample(data.time)
        original = {'rigid_target_position':p, 'rigid_target_velocity':v,
            'rigid_target_rotation':r@scene.grasp_rotation_target_frame,
            'rigid_target_angular_velocity':w, 'continuum_target_position':cp,
            'continuum_target_velocity':cv, 'continuum_target_rotation':scene.continuum_target_rotation_world,
            'continuum_target_angular_velocity':np.zeros(3),
            'posture_reference_q':spec.planner_zero, 'posture_reference_dq':np.zeros(17)}
        observed = passthrough.sample(data.time)
        for key in original:
            exact[key] = max(exact.get(key,0.), float(np.max(np.abs(observed[key]-original[key]))))
    np.savez_compressed(output/'derived_actual_reference.npz', control_points=controls,
        time=provider.prediction['time'], **{k:provider.prediction[k] for k in ('q','dq','ddq')})
    reconstructed = codec.sample(controls, state['time'])
    acceleration = np.gradient(state['dq'], state['time'], axis=0)
    r2 = {'reference_origin':'derived from R0 actual q; original q_ref was not saved',
        'strict_same_input_comparison':False, 'test_generalization_claim':False,
        'position_rad':vector_error(state['q'], reconstructed['q']),
        'velocity_rad_s':vector_error(state['dq'], reconstructed['dq']),
        'acceleration_rad_s2':vector_error(acceleration, reconstructed['ddq']),
        'actual_acceleration_definition':'offline numpy.gradient of 500Hz fresh dq, one-sided endpoints',
        'fit':'existing codec.fit defaults; position-only constrained least squares; 32x17',
        'cartesian_position_error_m':{}, 'cartesian_rotation_error_rad':{}}
    for arm in ('rigid','continuum'):
        r2['cartesian_position_error_m'][arm] = vector_error(
            state[arm+'_position'][::10],provider.prediction[arm+'_position'])
        delta = np.matmul(np.swapaxes(state[arm+'_rotation'][::10],1,2), provider.prediction[arm+'_rotation'])
        angles = Rotation.from_matrix(delta).magnitude()
        r2['cartesian_rotation_error_rad'][arm] = {'rms':float(np.sqrt(np.mean(angles**2))), 'maximum':float(np.max(angles))}
    report = {'schema':'v64_a1_reference_diagnostic_v1', 'task_sha256':task.sha256(),
        'R0':{'source':str(previous),'result_sha256':sha(previous/'result.json'),
              'fresh_state_source_sha256':sha(source),'task_success':True},
        'R1':{'planning_snapshots':1351,'same_feedback_and_absolute_time':True,
              'component_maximum_absolute_difference':exact,'comparison_threshold':0.,
              'passed':all(x==0. for x in exact.values())},
        'R2':r2, 'new_controller_steps':0,'new_force_replay_steps':0,'new_geometry_queries':0,
        'offline_actual_snapshots_are_online_planner_inputs':False}
    write(output/'report.json',report)
    print(json.dumps({'R1_exact':report['R1']['passed'],'R2_position_RMS_L2':r2['position_rad']['RMS_L2']}),flush=True)


def execute(task_path, output, *, proposal_path=None, positive=False, config=None, attribution=None,
            method='diffusion', terminal_progress_repair=False):
    task=TaskSpec.from_dict(json.loads(Path(task_path).read_text(encoding='utf8')))
    proposal=(TrajectoryProposal.from_dict(json.loads(Path(proposal_path).read_text(encoding='utf8')))
              if proposal_path else None)
    before=record_sources()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    print(json.dumps({'event':'ACTUAL_EXECUTION_STARTED','task':task.task_id,'output':str(output),
        'positive_control':positive,'source_commit':head}),flush=True)
    started=time.perf_counter()
    result=run_attempt(task,output,method='fixed_reference' if positive else method,
        proposal=proposal,require_raw_gate=False if proposal else True,
        attribution=attribution or {},pass_through_reference=positive,
        qp_config_override=config,terminal_progress_repair=terminal_progress_repair)
    after=record_sources()
    receipt={'schema':'v64_a1_actual_execution_receipt_v1','command_argv':sys.argv,
        'source_commit':head,'source_before':before,'source_after':after,
        'sources_unchanged':before==after,'elapsed_wall_s':time.perf_counter()-started,
        'task_sha256':task.sha256(),'proposal_path':str(proposal_path) if proposal_path else None,
        'proposal_sha256':sha(proposal_path) if proposal_path else None,
        'status':result['status'],'task_success':result['task_success'],
        'cli_exit_status':0,'exit_status_means':'saved terminal attempt, not task success',
        'new_controller_trace_is_force_playback':False}
    write(Path(output)/'execution_receipt.json',receipt)
    print(json.dumps({'event':'ACTUAL_EXECUTION_TERMINAL','status':result['status'],
        'task_success':result['task_success'],'sources_unchanged':before==after}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    diagnostic=sub.add_parser('diagnose-references')
    diagnostic.add_argument('--mapping',type=Path,required=True)
    diagnostic.add_argument('--output',type=Path,required=True)
    actual=sub.add_parser('execute')
    actual.add_argument('--task',type=Path,required=True)
    actual.add_argument('--proposal',type=Path)
    actual.add_argument('--positive-control',action='store_true')
    actual.add_argument('--method',choices=('diffusion','derived_reference_diagnostic'),default='diffusion')
    actual.add_argument('--terminal-progress-repair',action='store_true')
    actual.add_argument('--config',type=Path,required=True)
    actual.add_argument('--attribution',type=Path)
    actual.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='diagnose-references':
        diagnose(json.loads(args.mapping.read_text(encoding='utf8')),args.output)
    else:
        if args.positive_control==bool(args.proposal):
            parser.error('choose exactly one positive control or frozen proposal')
        config=HierarchicalQPConfig(**json.loads(args.config.read_text(encoding='utf8')))
        attribution=json.loads(args.attribution.read_text(encoding='utf8')) if args.attribution else {}
        execute(args.task,args.output,proposal_path=args.proposal,positive=args.positive_control,
                config=config,attribution=attribution,method=args.method,
                terminal_progress_repair=args.terminal_progress_repair)


if __name__=='__main__':
    main()
