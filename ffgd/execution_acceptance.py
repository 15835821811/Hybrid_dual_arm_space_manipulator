"""Read actual saved states for task/tracking gates; execute no extra physics."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .contracts import Snapshot


def inspect_actual(robot,bundle,summary,task,profile,replay_report):
    r=robot;m=r.model;dt=m.opt.timestep;tasks=[];states=bundle['states'];n=len(bundle['ctrl'])
    for target in task.poses:
        index=int(round(target.time/dt))
        if abs(index*dt-target.time)>1e-10:
            tasks.append(dict(time=target.time,tip=target.tip,passed=False,reason='TASK_TIME_NOT_ON_NATIVE_GRID'));continue
        if index>=len(states):
            tasks.append(dict(time=target.time,tip=target.tip,passed=False,reason='EXECUTION_DID_NOT_REACH_TIME'));continue
        d=r.restore(Snapshot(r.model_hash,int(bundle['mask']),states[index]));mujoco.mj_forward(m,d)
        body=r.tips[target.tip];R=d.xmat[body].reshape(3,3).copy();pos=d.xpos[body].copy()
        if target.tip==0:pos+=R@np.array([.0475,0,0])
        pe=float(np.linalg.norm(pos-target.position));re=float(Rotation.from_matrix(target.rotation.T@R).magnitude())
        tasks.append(dict(time=target.time,tip=target.tip,position_m=pe,rotation_rad=re,
                          passed=(not target.position_enabled or pe<=target.position_tolerance) and (not target.rotation_enabled or re<=target.rotation_tolerance)))
    gates=dict(native_completion=summary['status']=='FINISHED' and n==summary['planned_steps'],task=all(t['passed'] for t in tasks),
               physical_geometry=summary['min_clearance']>=profile['physical_geometry_clearance_m'],
               raw_torque=summary['raw_torque_ratio'] is not None and summary['raw_torque_ratio']<=1.,
               world_linear_momentum=summary['linear_momentum_residual']<=profile['momentum_linear_kg_m_s'],
               world_angular_momentum=summary['angular_momentum_residual']<=profile['momentum_angular_kg_m2_s'],
               independent_replay=replay_report['max_state_abs']<=profile['replay_state_atol'],
               native_state_only=summary['state_writes_after_restore']==0)
    return dict(status='EXECUTION_VALID' if all(gates.values()) else 'FAILED',gates=gates,task_errors=tasks,
                replay=replay_report,safety_enabled=True,task_based_cartesian_assist=False,
                safety_interventions=dict(clipped_control_steps=summary.get('clipped_control_steps'),safety_abort=summary.get('safety_abort')),
                scope='discrete timed poses; continuous hold/tracking not yet supported')


def execute_selected(path,planning_directory):
    import json
    from pathlib import Path
    from .problem_io import load_problem
    from .contracts import Candidate
    from .api import boundary_codec
    from .physics import DifferentiableTree
    from .propagation import Propagator
    from .geometry import Geometry
    from .validation import validate
    from .execution import track,replay
    from .orchestration import atomic_json
    planning_directory=Path(planning_directory)
    planned=json.loads((planning_directory/'planning_result.json').read_text(encoding='utf-8'))
    if planned['status']!='PLAN_VALID' or planned['selected_index'] is None:raise ValueError('NO_VALID_SELECTED_PLAN')
    index=planned['selected_index'];folder=planning_directory/f'candidate_{index:03d}'
    r,snapshot,task,problem=load_problem(planning_directory/'problem.json')
    manifest=json.loads((planning_directory/'planning_manifest.json').read_text(encoding='utf-8'));profile=manifest['profile']
    with np.load(folder/'candidate.npz') as data:candidate=Candidate(data['control'],float(data['duration']),int(data['seed']))
    if candidate.identity!=planned['records'][index]['identity']:raise ValueError('SELECTED_CANDIDATE_HASH_CHANGED')
    codec=boundary_codec(r,snapshot,task.duration,len(candidate.control));prop=Propagator(DifferentiableTree(r),snapshot)
    validation=validate(candidate,snapshot,task,prop,codec,Geometry(r),profile)
    atomic_json(path/'validation_before_execution.json',validation)
    if validation['status']!='PLAN_VALID':return dict(passed=False,status='PRE_EXECUTION_REJECTED',validation=validation)
    atomic_json(path/'provenance.json',dict(planning_directory=str(planning_directory.resolve()),candidate_hash=candidate.identity,profile=profile))
    bundle,summary=track(r,snapshot,candidate,path,profile)
    if summary['status']=='RESOURCE_BUDGET_ABORT':
        actual=dict(status='INCOMPLETE_BUDGET',independent_replay='NOT_RUN',safety_enabled=True,task_based_cartesian_assist=False)
        atomic_json(path/'acceptance.json',actual)
        return dict(passed=False,execution=summary,acceptance=actual)
    replay_report=replay(r,bundle,snapshot);actual=inspect_actual(r,bundle,summary,task,profile,replay_report)
    atomic_json(path/'acceptance.json',actual)
    return dict(passed=actual['status']=='EXECUTION_VALID',execution=summary,acceptance=actual)
