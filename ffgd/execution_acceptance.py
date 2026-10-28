"""Read actual saved states for task/tracking gates; execute no extra physics."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .contracts import Snapshot


def inspect_state_bounds(robot,bundle,profile=None):
    """Read-only all-state checks, including the final state and applied inputs.

    The 67-axis derivative envelopes are induced by the existing 17-axis
    limits and |B|, not newly asserted hardware speed ratings. Acceleration is
    recomputed natively on scratch data with the recorded control for that step.
    """
    r=robot;m=r.model;limited=m.jnt_limited[r.joints].astype(bool);bounds=m.jnt_range[r.joints]
    from .orchestration import check_budget
    maxima=dict(planner_velocity_ratio=0.,mapped_q67_velocity_ratio=0.,planner_acceleration_ratio=0.,mapped_q67_acceleration_ratio=0.)
    manifold=dict(position_rad=0.,velocity_rad_s=0.)
    tracking=None
    if 'control_points' in bundle:
        from .trajectory import basis
        times=np.arange(len(bundle['states']))*m.opt.timestep
        planned_q=basis(len(bundle['control_points']),times,float(bundle['duration']))@bundle['control_points']
        planned_v=basis(len(bundle['control_points']),times,float(bundle['duration']),1)@bundle['control_points']
        tracking=dict(position_rad=0.,velocity_rad_s=0.,planner_position_rad=0.)
    q17_ok=q67_ok=True
    velocity_envelope=abs(r.B)@r.spec.planner_velocity_limits
    acceleration_envelope=abs(r.B)@r.spec.planner_acceleration_limits
    if len(bundle['states'])!=len(bundle['ctrl'])+1:raise ValueError('INCOMPLETE_INTEGRATION_INPUT_LEDGER')
    for i,state in enumerate(bundle['states']):
        if i%64==0:check_budget()
        d=r.restore(Snapshot(r.model_hash,int(bundle['mask']),state))
        low=d.qpos[r.qpos_ids];q=r.Bplus@low;v=d.qvel[r.dof_ids]
        manifold['position_rad']=max(manifold['position_rad'],float(np.max(abs(low-r.B@q))))
        manifold['velocity_rad_s']=max(manifold['velocity_rad_s'],float(np.max(abs(v-r.B@(r.Bplus@v)))))
        if tracking is not None:
            tracking['position_rad']=max(tracking['position_rad'],float(np.max(abs(low-r.B@planned_q[i]))))
            tracking['velocity_rad_s']=max(tracking['velocity_rad_s'],float(np.max(abs(v-r.B@planned_v[i]))))
            tracking['planner_position_rad']=max(tracking['planner_position_rad'],float(np.max(abs(q-planned_q[i]))))
        q17_ok=q17_ok and bool(np.all(q>=r.lower) and np.all(q<=r.upper))
        q67_ok=q67_ok and bool(np.all(low[limited]>=bounds[limited,0]) and np.all(low[limited]<=bounds[limited,1]))
        if i<len(bundle['ctrl']):
            d.ctrl[:]=bundle['ctrl'][i];d.qfrc_applied[:]=bundle['qfrc_applied'][i];d.xfrc_applied[:]=bundle['xfrc_applied'][i]
        mujoco.mj_forward(m,d);a=d.qacc[r.dof_ids]
        values=dict(planner_velocity_ratio=float(np.max(abs(r.Bplus@v)/r.spec.planner_velocity_limits)),
                    mapped_q67_velocity_ratio=float(np.max(abs(v)/velocity_envelope)),
                    planner_acceleration_ratio=float(np.max(abs(r.Bplus@a)/r.spec.planner_acceleration_limits)),
                    mapped_q67_acceleration_ratio=float(np.max(abs(a)/acceleration_envelope)))
        for key,value in values.items():maxima[key]=max(maxima[key],value) if np.isfinite(value) else float('inf')
    gates=dict(actual_q17=q17_ok,actual_q67=q67_ok,**{key:value<=1. for key,value in maxima.items()})
    thresholds='NOT_FROZEN'
    if profile is not None and 'tracking_error_rad' in profile:
        if 'manifold_residual_rad' not in profile:raise ValueError('INCOMPLETE_TRACKING_PROFILE')
        thresholds={key:profile[key] for key in ('tracking_error_rad','manifold_residual_rad')}
        gates['reference_tracking']=tracking is not None and tracking['planner_position_rad']<=profile['tracking_error_rad']
        gates['position_manifold']=manifold['position_rad']<=profile['manifold_residual_rad']
    return dict(gates=gates,
                maxima=maxima,states_checked=len(bundle['states']),includes_final_state=True,
                derivative_limits_source='existing planner limits; q67 envelopes = abs(B) @ planner limits',
                acceleration_observation='native scratch forward with each recorded applied input; final state uses last applied control',
                manifold_residual_max=manifold,reference_tracking_max=tracking,
                actual_states_modified=False,tracking_manifold_thresholds=thresholds,
                continuation_prediction_domain='NOT_CALIBRATED_BY_THESE_EXECUTION_THRESHOLDS')


def inspect_actual(robot,bundle,summary,task,profile,replay_report):
    r=robot;m=r.model;dt=m.opt.timestep;tasks=[];states=bundle['states'];n=len(bundle['ctrl'])
    from .temporal import interpolate_qpos,inspect_intervals
    times=np.arange(len(states))*dt
    qpos=np.stack([r.restore(Snapshot(r.model_hash,int(bundle['mask']),s)).qpos.copy() for s in states])
    for target in task.poses:
        if target.time>times[-1]:
            tasks.append(dict(time=target.time,tip=target.tip,passed=False,reason='EXECUTION_DID_NOT_REACH_TIME'));continue
        d=mujoco.MjData(m);d.qpos[:],bracket=interpolate_qpos(r,times,qpos,target.time);mujoco.mj_kinematics(m,d)
        body=r.tips[target.tip];R=d.xmat[body].reshape(3,3).copy();pos=d.xpos[body].copy()
        if target.tip==0:pos+=R@np.array([.0475,0,0])
        pe=float(np.linalg.norm(pos-target.position));re=float(Rotation.from_matrix(target.rotation.T@R).magnitude())
        tasks.append(dict(time=target.time,tip=target.tip,position_m=pe,rotation_rad=re,interpolation_bracket_s=bracket,
                          passed=(not target.position_enabled or pe<=target.position_tolerance) and (not target.rotation_enabled or re<=target.rotation_tolerance)))
    bounds=inspect_state_bounds(r,bundle,profile)
    interval_errors=[]
    if task.intervals:
        interval_errors=inspect_intervals(r,task,times,qpos)
    gates=dict(native_completion=summary['status']=='FINISHED' and n==summary['planned_steps'],task=all(t['passed'] for t in tasks),
               physical_geometry=summary['min_clearance']>=profile['physical_geometry_clearance_m'],
               raw_torque=summary['raw_torque_ratio'] is not None and summary['raw_torque_ratio']<=1.,
               world_linear_momentum=summary['linear_momentum_residual']<=profile['momentum_linear_kg_m_s'],
               world_angular_momentum=summary['angular_momentum_residual']<=profile['momentum_angular_kg_m2_s'],
               independent_replay=replay_report['max_state_abs']<=profile['replay_state_atol'],
               native_state_only=summary['state_writes_after_restore']==0)
    gates.update(bounds['gates'])
    if task.intervals:gates['task_intervals']=all(x['passed'] for x in interval_errors)
    return dict(status='EXECUTION_VALID' if all(gates.values()) else 'FAILED',gates=gates,task_errors=tasks,interval_errors=interval_errors,
                replay=replay_report,state_bounds=bounds,safety_enabled=True,task_based_cartesian_assist=False,
                safety_interventions=dict(clipped_control_steps=summary.get('clipped_control_steps'),safety_abort=summary.get('safety_abort')),
                scope='timed poses plus every-native-state hold/target-relative tracking; no continuous-time proof')


def execute_selected(path,planning_directory,config=None):
    import json
    from pathlib import Path
    from .problem_io import load_problem
    from .contracts import Candidate
    from .api import boundary_codec
    from .physics import DifferentiableTree
    from .propagation import Propagator
    from .geometry import Geometry
    from .validation import validate
    from .execution import track,replay,refine_physics
    from .orchestration import atomic_json,ROOT
    planning_directory=Path(planning_directory)
    planned=json.loads((planning_directory/'planning_result.json').read_text(encoding='utf-8'))
    if planned['status']!='PLAN_VALID' or planned['selected_index'] is None:raise ValueError('NO_VALID_SELECTED_PLAN')
    index=planned['selected_index'];folder=planning_directory/f'candidate_{index:03d}'
    r,snapshot,task,problem=load_problem(planning_directory/'problem.json')
    manifest=json.loads((planning_directory/'planning_manifest.json').read_text(encoding='utf-8'));profile=manifest['profile']
    with np.load(folder/'candidate.npz') as data:candidate=Candidate(data['control'],float(data['duration']),int(data['seed']))
    if candidate.identity!=planned['records'][index]['identity']:raise ValueError('SELECTED_CANDIDATE_HASH_CHANGED')
    codec=boundary_codec(r,snapshot,task.duration,len(candidate.control),origin=manifest['budget'].get('boundary_origin','pivot_zero'),reference_start=task.reference_start)
    from .propagation import FullStatePropagation
    prop=(FullStatePropagation if manifest.get('physics_backend')=='full_state' else Propagator)(DifferentiableTree(r),snapshot)
    validation=validate(candidate,snapshot,task,prop,codec,Geometry(r),profile)
    atomic_json(path/'validation_before_execution.json',validation)
    if validation['status']!='PLAN_VALID':return dict(passed=False,status='PRE_EXECUTION_REJECTED',validation=validation)
    # Restore inherited execution gates explicitly. The frozen numerical profile
    # and all saved candidate/old execution verdicts remain unchanged.
    tracking_profile=json.loads((ROOT/'configs/ffgd/execution_tracking_profile_v1.json').read_text(encoding='utf-8'))
    profile=dict(profile,**tracking_profile['thresholds'])
    config={} if config is None else config
    if set(config)-{'physics_substeps'}:raise ValueError('UNKNOWN_EXECUTION_CONFIG_FIELDS')
    substeps=config.get('physics_substeps',1);planning_hash=r.model_hash;control_period=float(r.model.opt.timestep)
    r,snapshot=refine_physics(r,snapshot,substeps)
    atomic_json(path/'provenance.json',dict(planning_directory=str(planning_directory.resolve()),candidate_hash=candidate.identity,profile=profile,
                                          physics_substeps=substeps,physics_dt=float(r.model.opt.timestep),control_period=control_period,
                                          planning_model_hash=planning_hash,execution_model_hash=r.model_hash,initial_state_bytes_unchanged=True,
                                          execution_tracking_profile=tracking_profile,
                                          method=manifest.get('method'),comparison_only=manifest.get('comparison_only',False)))
    bundle,summary=track(r,snapshot,candidate,path,profile,control_substeps=substeps)
    if summary['status']=='RESOURCE_BUDGET_ABORT':
        actual=dict(status='INCOMPLETE_BUDGET',independent_replay='NOT_RUN',safety_enabled=True,task_based_cartesian_assist=False)
        atomic_json(path/'acceptance.json',actual)
        return dict(passed=False,execution=summary,acceptance=actual)
    replay_report=replay(r,bundle,snapshot);actual=inspect_actual(r,bundle,summary,task,profile,replay_report)
    atomic_json(path/'acceptance.json',actual)
    return dict(passed=actual['status']=='EXECUTION_VALID',execution=summary,acceptance=actual)
