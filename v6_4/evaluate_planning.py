"""Fresh-state, same-torque evaluation of new frozen planning tasks.

This evaluator never advances a state by assigning planned qpos/qvel.  It restores
the frozen initial state once, applies every saved 67-channel torque, and refreshes
current kinematics after each mj_step.  New task criteria replace old-curve RMSE;
the original geometric, interval and shared-ramp checks remain independent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from v6_4.task_protocol import TaskSpec,canonical_json
from v6_4.proposal_gate import requirement_results


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _finite(value,shape,name):
    a=np.asarray(value)
    if a.shape!=shape or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return a


def _stats(a):
    a=np.asarray(a,dtype=float)
    if not a.size:return {'count':0,'mean':None,'rms':None,'maximum':None,'p95':None}
    if not np.all(np.isfinite(a)):raise ValueError('nonfinite metric samples')
    return {'count':a.size,'mean':float(np.mean(a)),'rms':float(np.sqrt(np.mean(a*a))),
            'maximum':float(np.max(a)),'p95':float(np.quantile(a,.95))}


def _performance(samples,period):
    a=np.asarray(samples,dtype=float)
    if a.ndim!=1 or not len(a) or not np.all(np.isfinite(a)) or np.any(a<0):
        raise ValueError('performance samples must be finite, nonnegative and nonempty')
    over=a>period;longest=current=0
    for flag in over:
        current=current+1 if flag else 0;longest=max(longest,current)
    return {**_stats(a),'p99':float(np.quantile(a,.99)),'period_s':period,
            'over_period_count':int(np.sum(over)),'longest_over_period_run':longest,
            'is_task_success_gate':False,'hard_realtime_certified':False}


def _shape_error_metrics(actual_q, reference_q, times, coordinate_names, *, reference_scope):
    """Compare aligned 17D state and posture preference, never Cartesian task error."""
    actual=np.asarray(actual_q,dtype=float);reference=np.asarray(reference_q,dtype=float)
    if actual.ndim!=2 or actual.shape[1]!=17 or reference.shape!=actual.shape or not len(actual):
        raise ValueError('shape-error vectors must be matching nonempty Nx17 arrays')
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
        raise ValueError('shape-error vectors must be finite')
    clock=_finite(times,(len(actual),),'shape-error aligned time')
    names=tuple(coordinate_names)
    if len(names)!=17 or len(set(names))!=17:raise ValueError('17 distinct coordinate names required')
    error=actual-reference
    coordinate_rmse=np.sqrt(np.mean(error*error,axis=0))
    return {'definition':'e_k = fresh replay decoded actual q(t_k) - frozen selected posture reference q_ref(t_k)',
            'unit':'rad','reference_scope':reference_scope,'sample_count':len(actual),
            'includes_initial_state':bool(clock[0]==0.),'first_time_s':float(clock[0]),'last_time_s':float(clock[-1]),
            'time_alignment':'both vectors at the same actual simulation time; no time shift or future-state input',
            'RMS_L2_17':float(np.sqrt(np.mean(np.sum(error*error,axis=1)))),
            'RMS_L2_continuum_10':float(np.sqrt(np.mean(np.sum(error[:,:10]**2,axis=1)))),
            'RMS_L2_rigid_7':float(np.sqrt(np.mean(np.sum(error[:,10:]**2,axis=1)))),
            'coordinate_rmse_17':{name:float(value) for name,value in zip(names,coordinate_rmse)},
            'is_end_effector_task_error':False,'is_single_coordinate_rmse':False}


def _save_native_geometry(task,state,whole,verifier,target_pairs,query_counts,witness,output_dir):
    """Persist fresh geometry before later command/reference binding can fail."""
    config=asdict(verifier.config)
    pairs=[{'class':p.pair_class,'geom_a':p.geom_a_name,'geom_b':p.geom_b_name,
            'body_a':p.body_a_name,'body_b':p.body_b_name} for p in verifier.pairs]
    policy={'schema':'v64_a1_declared_whole_body_policy_v1','configuration':config,'pairs':pairs,
            'pair_policy_sha256':whole['pair_policy_sha256'],
            'intentional_target_contact_geom_names':list(verifier.config.intentional_target_contact_geom_names),
            'source_model_contract_sha256':task.model_contract_sha256,
            'verifier_source_sha256':_sha(Path(__file__).resolve().parents[1]/'model_test/whole_body_verifier_v5.py')}
    policy_path=Path(output_dir)/'whole_body_policy.json'
    with policy_path.open('x',encoding='utf-8') as f:
        json.dump(policy,f,indent=2,allow_nan=False);f.write('\n')
    values=np.asarray(state['target_minimum_m'])
    native={'schema':'v64_a1_fresh_native_geometry_v1','evidence_available':True,
        'task_id':task.task_id,'task_sha256':task.sha256(),'model_contract_sha256':task.model_contract_sha256,
        'model_source_bundle_sha256':verifier.robot_spec.source_bundle_sha256(),
        'passed':bool(np.min(values)>=config['minimum_clearance'] and whole['feasible']),
        'robot_target_500hz':{'state_count':len(state['time']),'pair_count':len(target_pairs),
            'query_count':len(state['time'])*len(target_pairs),'minimum_m':float(np.min(values)),
            'minimum_witness':witness,'minimum_clearance_m':config['minimum_clearance'],
            'below_5mm_states':int(np.sum(values<.005)),
            'negative_states':int(np.sum(values<0)),
            'below_threshold_query_count':int(query_counts['below_threshold']),
            'negative_distance_query_count':int(query_counts['negative']),
            'truncated_query_count':int(query_counts['truncated']),
            'minimum_is_censored_lower_bound':bool(state['target_minimum_censored'][np.argmin(values)]),
            'sampling_scope':'all saved native 2ms physical states including declared initial state'},
        'whole_body':whole,
        'whole_body_count_scope':'pair queries over 50Hz boundary states plus configuration-space adaptive subdivisions; not all 500Hz states',
        'whole_body_supplied_period_s':.02,'whole_body_adaptive_subdivisions':config['adaptive_subdivisions'],
        'whole_body_policy':{'path':policy_path.resolve().as_posix(),'sha256':_sha(policy_path)},
        'query_distance_max_m':config['query_distance_max'],'continuous_time_certified':False,
        'offline_state_copy_independent_of_execution':True}
    path=Path(output_dir)/'native_geometry.json'
    with path.open('x',encoding='utf-8') as f:
        json.dump(native,f,indent=2,allow_nan=False);f.write('\n')
    return native


def _execution_trace_checks(task,trace,spec,qp_config):
    """The original per-trace ramp invariants, without a five-scenario gate."""
    from v6_lite.execution_ramp import advance_reference
    n=len(trace['torque']);ticks=n//10
    command=_finite(trace['command_velocity'],(n,17),'command_velocity')
    # A refusal may append an unconsumed NaN command after the saved ramps.
    selected=_finite(np.asarray(trace['task_selected_command'])[:ticks],
                     (ticks,17),'consumed task_selected_command')
    names=('reference_q','reference_velocity','feedforward_acceleration','measured_planner_q_before_servo','measured_velocity_before_servo')
    for k in names:_finite(trace[k],(n,17),k)
    for k in ('reference_joint_limit_clip_count','reference_measured_window_clip_count','acceleration_clip_count','torque_saturation_count'):
        a=_finite(trace[k],(n,),k)
        if np.any(a<0):raise ValueError('negative clip/saturation count')
    position=np.asarray(task.initial_planner_q).copy();max_errors=np.zeros(3)
    for i in range(n):
        tick,within=divmod(i,10)
        if tick>=len(selected):raise ValueError('incomplete executed ramp record')
        old=np.asarray(task.initial_planner_dq) if tick==0 else selected[tick-1]
        step=advance_reference(position,old,selected[tick],within+1,
            trace['measured_planner_q_before_servo'][i],spec.planner_lower,spec.planner_upper)
        position=step.position
        for j,(actual,key) in enumerate(((position,'reference_q'),(step.velocity,'reference_velocity'),(step.feedforward_acceleration,'feedforward_acceleration'))):
            max_errors[j]=max(max_errors[j],float(np.max(np.abs(actual-trace[key][i]))))
    velocity_error=np.linalg.norm(trace['reference_velocity']-trace['measured_velocity_before_servo'],axis=1)
    saved=_finite(trace['reference_velocity_error_norm_rad_s'],(n,),'reference_velocity_error_norm_rad_s')
    reason=np.asarray(trace['task_failure_reason'])[:ticks]
    mode=np.asarray(trace['task_execution_mode'])[:ticks]
    checks={'actual_selected_commands_consumed':n%10==0 and np.array_equal(selected,command[::10]),
            'shared_ten_step_reference_exact':bool(np.max(max_errors)<=1e-12),
            'reference_velocity_error_recorded':bool(np.allclose(saved,velocity_error,atol=1e-12,rtol=0.)),
            'no_uncertified_consumed_steps':len(reason)==ticks and len(mode)==ticks and bool(np.all(reason=='none')) and bool(np.all(np.isin(mode,['TRACK','CAUTION']))),
            'all_actual_torques_within_original_limits':bool(np.all(np.abs(trace['torque'])<=spec.torque_limits+1e-12))}
    return {'passed':all(checks.values()),'checks':checks,'shared_ramp_max_errors':max_errors.tolist(),
            'reference_velocity_error_rad_s':_stats(velocity_error),
            'joint_limit_clips':int(np.sum(trace['reference_joint_limit_clip_count'])),
            'measured_window_clips':int(np.sum(trace['reference_measured_window_clip_count'])),
            'acceleration_clips':int(np.sum(trace['acceleration_clip_count'])),
            'torque_saturations':int(np.sum(trace['torque_saturation_count']))}


def _interval_checker(spec,model,pairs,cfg):
    from v6_lite.pcc_interval_cbf import FixedIntervalCBFEvaluator
    from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit
    from v6_lite.pcc_persistent_interval_query import PersistentIntervalDecisionQuery
    from v6_lite.b2_screened_next_start_rows import ScreenedReplayConstraintBuilder
    evaluator=FixedIntervalCBFEvaluator(spec,model)
    return (evaluator,StateLocalPCCEnvelopeAudit(model,evaluator.shape_spec),
            PersistentIntervalDecisionQuery(evaluator.shape_model),
            ScreenedReplayConstraintBuilder(spec,model,pairs,replace(cfg,enable_pcc_cbf=False,enable_capsule_cbf=True)))


def _interval_boundary(spec,data,trace,tick,ticks,cfg,tools):
    from v6_lite.pcc_interval_cbf import IntervalPartition
    from v6_lite.continuum_shape_model import transform_from_free_qpos
    from v6_lite.shape_clearance import target_box_from_mujoco
    from v6_lite.b2_shadow_feasibility import ramp_velocity_abs_bound,combined_rows
    from v6_lite.shadow_b2_interval_cbf import select_intervals_by_frozen_reach
    from v6_lite.audit_b2_online_interval_recompute import _recomputed_ramp_and_lookahead
    evaluator,envelope,query,builder=tools
    projection=evaluator.shape_spec.project_actual_configuration(data.qpos[evaluator.qpos_ids[:60]])
    base=transform_from_free_qpos(data.qpos[evaluator.base_qpos_slice])
    coverage=envelope.evaluate(data,projection.planner_configuration,base)
    box=target_box_from_mujoco(evaluator.model,data,evaluator.target_geom_id)
    decision=query.evaluate(projection.planner_configuration,base,box,IntervalPartition.uniform(),max_point_evaluations=255)
    previous=np.zeros(17) if tick==0 else trace['task_selected_command'][tick-1]
    q=spec.low_level_to_planner@data.qpos[evaluator.qpos_ids]
    _,_,speed,box_valid=ramp_velocity_abs_bound(spec,cfg,q,previous)
    gmap,_=builder._reaction_map(data)
    ids,_=select_intervals_by_frozen_reach(decision.lower_by_interval_id,decision.partition,evaluator,data,cfg,gmap,velocity_abs_bound=speed)
    batch=evaluator.evaluate_state(data,decision.partition,generalized_map=gmap,derivative_interval_ids=ids)
    matrix,lower,drifts,gains,sources=combined_rows(builder.build(data),batch,ids,cfg)
    if not len(lower) or not all(np.all(np.isfinite(x)) for x in (matrix,lower,drifts,gains)):
        raise ValueError('independent interval/physical rows empty or nonfinite')
    start=float(np.min(matrix@previous-lower));errors=[]
    qpos_error=float(np.max(np.abs(data.qpos-trace['task_qpos'][tick])))
    if qpos_error>1e-9:errors.append('ACTUAL_TASK_STATE_MISMATCH')
    if (not box_valid or not decision.bounds_valid or decision.proxy_clearance_status!='PROXY_CLEARANCE_AT_LEAST_GATE'
        or coverage.status!='COVERED_AT_THIS_STATE' or not batch.interval_well_formed or not batch.coverage_complete
        or not batch.analytic_bound_assumptions_satisfied or any(r.derivative_status!='SUPPORTED' for r in batch.rows if r.interval_id in ids)):
        errors.append('INTERVAL_UNSUPPORTED')
    residuals={}
    if tick<ticks:
        ramp,lookahead=_recomputed_ramp_and_lookahead(matrix,lower,drifts,gains,previous,trace['task_selected_command'][tick],cfg)
        observations=(decision.distance_lower_bound_m,coverage.min_margin_m,ramp,lookahead)
        logged=tuple(float(trace[k][tick]) for k in ('task_interval_proxy_lower_m','task_interval_current_envelope_margin_m','task_ramp_clearance_min_slack_m_s','task_lookahead_min_slack_m_s'))
        if not all(np.isfinite(x) for x in observations+logged):raise ValueError('nonfinite raw/recomputed interval scalar')
        residuals={'query_error_m':abs(observations[0]-logged[0]),'envelope_error_m':abs(observations[1]-logged[1]),
                   'ramp_error_m_s':abs(ramp-logged[2]),'lookahead_error_m_s':abs(lookahead-logged[3]),
                   'ramp_slack_m_s':ramp,'lookahead_slack_m_s':lookahead}
        if max(residuals['query_error_m'],residuals['envelope_error_m'])>1e-8 or len(ids)!=trace['task_interval_selected_rows'][tick]:errors.append('QUERY_SELECTION_MISMATCH')
        if max(residuals['ramp_error_m_s'],residuals['lookahead_error_m_s'])>1e-7:errors.append('ACTION_LOG_MISMATCH')
        if min(ramp,lookahead)<-cfg.clearance_rate_tolerance_m_s:errors.append('ACTION_CONSTRAINT_VIOLATION')
    if tick:
        saved=float(trace['task_interval_realized_next_start_minimum_slack_m_s'][tick-1])
        if not np.isfinite(saved):raise ValueError('nonfinite realized-next-start log')
        residuals['next_start_error_m_s']=abs(start-saved)
        if abs(start-saved)>1e-7:errors.append('NEXT_START_LOG_MISMATCH')
        if start<-cfg.clearance_rate_tolerance_m_s:errors.append('ACTUAL_NEXT_START_VIOLATION')
    rows=[{'tick':tick,'interval_id':r.interval_id,'h_m':r.h_m,'lower_m':r.distance_lower_bound_m,
           'gradient_17d':r.generalized_gradient.tolist(),'target_drift_m_s':r.target_drift_m_s,
           'derivative_status':r.derivative_status} for r in batch.rows if r.interval_id in ids]
    return {'tick':tick,'errors':errors,'actual_qpos_error_max':qpos_error,'selected_interval_count':len(ids),
            'recomputed_clearance_row_count':len(sources),'start_slack_m_s':start,**residuals},rows


def _replay(task,trace,qp_config,output_dir):
    import mujoco
    from model_test.robot_model_spec_v5 import RobotModelSpecV5,default_robot_model_spec_v5
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier,WholeBodyVerificationConfig,WorkspaceSphere
    from v6_lite.hierarchical_qp import joint_addresses,free_joint_slices
    spec=RobotModelSpecV5.from_urdf(default_robot_model_spec_v5().source_urdf,
        continuum_kp=100.,continuum_kd=5.,continuum_torque_limit=4.,rigid_kp=200.,rigid_kd=20.,rigid_torque_limit=80.)
    if spec.runtime_contract_sha256()!=task.model_contract_sha256:raise ValueError('nominal model identity mismatch')
    spheres=tuple(WorkspaceSphere(o['name'],np.asarray(o['center_w']),o['radius_m']) for o in task.scenario['workspace_obstacles'])
    verifier=WholeBodyCollisionVerifier(spec,spheres,WholeBodyVerificationConfig(minimum_clearance=.005,
        query_distance_max=2.5,adaptive_subdivisions=4,self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True))
    model=verifier.model;model.geom_contype[:]=0;model.geom_conaffinity[:]=0
    data=mujoco.MjData(model);data.qpos[:]=trace['initial_qpos'];data.qvel[:]=trace['initial_qvel'];data.ctrl[:]=0.
    qp,qv=joint_addresses(model,spec);bp,_=free_joint_slices(model,spec.base_joint_name)
    target_jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,spec.target_free_joint_name)
    target_body=int(model.jnt_bodyid[target_jid])
    arms={name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,getattr(spec,name+'_tip_body_name')) for name in ('rigid','continuum')}
    if any(v<0 for v in arms.values()) or not verifier.pairs:raise ValueError('empty native geometry/body policy')
    target_pairs=[p for p in verifier.pairs if p.pair_class.endswith('_target')]
    if not target_pairs:raise ValueError('robot-target pair policy empty')
    ticks=len(trace['torque'])//10;tools=_interval_checker(spec,model,verifier.pairs,qp_config)
    values={key:[] for key in ('time','q','dq','qpos','qvel','base_pose','target_position','target_rotation','rigid_position','rigid_rotation','continuum_position','continuum_rotation','target_minimum_m','target_minimum_censored')}
    boundaries=[];intervals=[]
    target_query_counts={'below_threshold':0,'negative':0,'truncated':0}
    target_witness={};target_global_minimum=float('inf')
    for index in range(len(trace['torque'])+1):
        mujoco.mj_forward(model,data)  # new current-state definition, never stale mj_step cache.
        values['time'].append(index*.002);values['q'].append(spec.decode_position(data.qpos[qp]).copy())
        values['dq'].append(spec.decode_velocity(data.qvel[qv]).copy());values['qpos'].append(data.qpos.copy());values['qvel'].append(data.qvel.copy())
        values['base_pose'].append(data.qpos[bp].copy());values['target_position'].append(data.xpos[target_body].copy())
        values['target_rotation'].append(data.xmat[target_body].reshape(3,3).copy())
        for arm,body in arms.items():
            r=data.xmat[body].reshape(3,3).copy();position=data.xpos[body].copy()
            if arm=='continuum':position+=r@np.asarray([.0475,0.,0.])
            values[arm+'_position'].append(position);values[arm+'_rotation'].append(r)
        distances=[]
        for pair in target_pairs:
            fromto=np.zeros(6)
            distance=float(mujoco.mj_geomDistance(model,data,pair.geom_a,pair.geom_b,2.5,fromto))
            distances.append(distance)
            target_query_counts['below_threshold']+=int(distance<verifier.config.minimum_clearance)
            target_query_counts['negative']+=int(distance<0.)
            target_query_counts['truncated']+=int(distance>=2.5-1e-12)
            if distance<target_global_minimum:
                target_global_minimum=distance
                target_witness={'pair_class':pair.pair_class,'geom_a':pair.geom_a_name,'geom_b':pair.geom_b_name,
                    'body_a':pair.body_a_name,'body_b':pair.body_b_name,'state_index':index,
                    'actual_time_s':index*.002,'signed_distance_m':distance,'fromto':fromto.tolist()}
        if not np.all(np.isfinite(distances)):raise ValueError('nonfinite native robot-target signed query')
        values['target_minimum_m'].append(min(distances));values['target_minimum_censored'].append(min(distances)>=2.5)
        if index%10==0:
            row,rows=_interval_boundary(spec,data,trace,index//10,ticks,qp_config,tools)
            boundaries.append(row);intervals.extend(rows)
        if index<len(trace['torque']):
            data.ctrl[:]=trace['torque'][index];mujoco.mj_step(model,data)
    state={k:np.asarray(a) for k,a in values.items()}
    for key,a in state.items():
        if not np.all(np.isfinite(a)):raise ValueError('nonfinite fresh replay '+key)
    np.savez_compressed(output_dir/'fresh_replay.npz',**state)
    whole=verifier.verify_qpos_sequence(state['qpos'][::10]).to_dict()
    native=_save_native_geometry(task,state,whole,verifier,target_pairs,target_query_counts,target_witness,output_dir)
    for filename,rows in (('interval_boundaries.jsonl',boundaries),('interval_rows.jsonl',intervals)):
        with (output_dir/filename).open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    interval={'passed':bool(boundaries) and all(not r['errors'] for r in boundaries),
              'boundary_count':len(boundaries),'row_count':len(intervals),
              'mismatch_count':sum(bool(r['errors']) for r in boundaries),
              'boundary_sha256':_sha(output_dir/'interval_boundaries.jsonl'),'rows_sha256':_sha(output_dir/'interval_rows.jsonl')}
    return state,spec,interval,native


def evaluate_trial(task,trace_path,reference_path,*,scenario_result,qp_config,output_dir):
    """Write an exclusive complete or partial evidence report for one attempt."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    trace_path=Path(trace_path);reference_path=Path(reference_path) if reference_path is not None else None
    report={'schema':'v6_4_planning_task_evaluation_v1','task_id':task.task_id,'task_sha256':task.sha256(),
        'model_contract_sha256':task.model_contract_sha256,'complete':False,'evidence_valid':False,'task_success':False,
        'passed':False,'errors':[],'wall_deployment':'NOT_MET','hardware':'NOT_ESTABLISHED',
        'old_curve_rmse_is_success_gate':False,'source_sha256':_sha(Path(__file__))}
    started=time.perf_counter()
    try:
        report.update({'trace_path':str(trace_path.resolve()),'trace_sha256':_sha(trace_path),
                       'reference_path':str(reference_path.resolve()) if reference_path else None,
                       'reference_sha256':_sha(reference_path) if reference_path else None})
        with np.load(trace_path,allow_pickle=False) as f:trace={k:f[k].copy() for k in f.files}
        if reference_path:
            with np.load(reference_path,allow_pickle=False) as f:reference={k:f[k].copy() for k in f.files}
        else:
            reference=None
        pass_through_home = (reference is not None and
            str(np.asarray(reference.get('reference_mode','')).item()) == 'cartesian_passthrough_home_posture')
        n=len(trace['torque'])
        if n==0 or n%10:raise ValueError('no complete consumed ten-step ramp in saved trace')
        _finite(trace['torque'],(n,67),'torque');times=_finite(trace['time'],(n,),'time')
        if not np.allclose(times,np.arange(1,n+1)*.002,atol=1e-9,rtol=0.):raise ValueError('actual saved physics time grid mismatch')
        if not np.array_equal(trace['initial_qpos'],task.initial_qpos) or not np.array_equal(trace['initial_qvel'],task.initial_qvel):raise ValueError('actual initial state differs from frozen task')
        _finite(trace['task_qpos'],(n//10+1,81),'task_qpos');_finite(trace['planner_q'],(n,17),'planner_q')
        state,spec,interval,native=_replay(task,trace,qp_config,output_dir)
        # This component survives a later execution-contract/reference-binding error.
        geometry_path=output_dir/'native_geometry.json'
        report['native_geometry']=native
        if geometry_path.exists():report['native_geometry_evidence']={'path':geometry_path.resolve().as_posix(),'sha256':_sha(geometry_path)}
        execution=_execution_trace_checks(task,trace,spec,qp_config)
        if np.max(np.abs(state['q'][1:]-trace['planner_q']))>1e-9:raise ValueError('actual saved planner state does not reproduce')
        from v6_4.trajectory_codec import CubicBSplineCodec
        codec=CubicBSplineCodec(task.initial_planner_q,task.initial_planner_dq)
        controls=(_finite(reference['control_points'],(32,17),'control_points') if reference is not None
                  else codec.decode_free(np.tile(task.initial_planner_q,(30,1))))
        codec.encode_free(controls)
        reference_binding={'passed':True,'generated_reference_consumed':reference is not None}
        if reference is not None:
            saved_clock=_finite(trace['generated_reference_time_s'],(n,),'generated_reference_time_s')
            if not np.allclose(saved_clock,times,atol=1e-9,rtol=0.):raise ValueError('generated reference and actual state clocks differ')
            # Only the same <=1ns endpoint roundoff accepted by the provider;
            # never project or clip candidate values.
            if np.any(saved_clock< -1e-9) or np.any(saved_clock>27.+1e-9):raise ValueError('generated reference clock escaped horizon')
            consumed=codec.sample(controls,np.minimum(27.,np.maximum(0.,saved_clock)))
            errors=[]
            for source,key in (('q','generated_reference_q'),('dq','generated_reference_dq')):
                errors.append(float(np.max(np.abs(consumed[source]-_finite(trace[key],(n,17),key)))))
            task_clock=_finite(trace['task_time'],(len(trace['task_selected_command']),),'task_time')[:n//10]
            task_consumed=codec.sample(controls,np.minimum(27.,np.maximum(0.,task_clock)))
            for source,key in (('q','task_generated_reference_q'),('dq','task_generated_reference_dq')):
                a=np.asarray(trace[key])[:n//10];_finite(a,(n//10,17),key)
                errors.append(float(np.max(np.abs(task_consumed[source]-a))))
            actual_qpos=_finite(trace['actual_full_qpos'],(n,81),'actual_full_qpos')
            actual_qvel=_finite(trace['actual_full_qvel'],(n,79),'actual_full_qvel')
            if np.max(np.abs(actual_qpos-state['qpos'][1:]))>1e-9 or np.max(np.abs(actual_qvel-state['qvel'][1:]))>1e-8:
                raise ValueError('saved full actual state does not reproduce from actual torque')
            if max(errors)>1e-12:raise ValueError('executed generated reference differs from frozen selected controls')
            actual_error=np.linalg.norm(state['q'][1:]-consumed['q'],axis=1)
            saved_error=_finite(trace['actual_generated_q_error_norm'],(n,),'actual_generated_q_error_norm')
            if not np.allclose(actual_error,saved_error,atol=1e-12,rtol=0.):raise ValueError('saved actual-to-reference shape metric differs')
            reference_binding.update({'maximum_generated_reference_residual':max(errors),
                'physics_samples_bound':n,'task_samples_bound':n//10,
                'actual_saved_full_state_reproduced':True})
        elif any(k.startswith('generated_reference_') for k in trace):
            raise ValueError('generated execution has no frozen selected reference source')
        ref=codec.sample(controls,state['time'])
        requirements=requirement_results(task,state)
        translation=np.linalg.norm(state['base_pose'][:,:3]-np.asarray(task.base_pose[:3]),axis=1)
        dots=np.abs(state['base_pose'][:,3:]@np.asarray(task.base_pose[3:]))
        angles=2*np.arccos(np.clip(dots,-1.,1.))
        metrics={'actual_to_planned_shape_error_rad':_stats(np.linalg.norm(state['q']-ref['q'],axis=1)),
            'aligned_posture_reference_error':_shape_error_metrics(state['q'],ref['q'],state['time'],
                getattr(spec,'planner_coordinate_names',tuple([f'theta{i+1}' for i in range(10)]+[f'theta_R{i+1}' for i in range(7)])),
                reference_scope='saved_spline_controls' if reference is not None else 'historical_fixed_home_posture_preference'),
            'actual_command_to_planned_velocity_difference_rad_s':_stats(np.linalg.norm(trace['command_velocity']-ref['dq'][:-1],axis=1)),
            'actual_joint_path_length_rad':float(np.sum(np.linalg.norm(np.diff(state['q'],axis=0),axis=1))),
            'actual_tip_path_length_m':{arm:float(np.sum(np.linalg.norm(np.diff(state[arm+'_position'],axis=0),axis=1))) for arm in ('rigid','continuum')},
            'base_translation_drift_m':_stats(translation),'base_orientation_drift_rad':_stats(angles),
            'planned_shape_scope':('historical_fixed_home_posture_preference'
                if reference is None or pass_through_home else 'saved_spline_controls'),
            'cartesian_reference_scope':('original current-feedback Cartesian task formulas, no codec'
                if pass_through_home else 'joint proposal private nominal prediction' if reference is not None else 'original fixed Cartesian task formulas'),
            'actual_torque_abs_max_nm':np.max(np.abs(trace['torque']),axis=0).tolist(),
            'physics_steps':n,'planning_ticks':n//10,'actual_saved_horizon_s':float(times[-1]),
            'planning_algorithm_wall_latency_s':_performance(trace['task_full_latency'],.020),
            'torque_preparation_wall_latency_s':_performance(trace['torque_latency'],.002),
            'planning_latency_trace_field':'task_full_latency',
            'latency_scope':'original saved algorithm/preparation clocks; not an end_to_end_dispatch_guarantee'}
        complete=n==13500
        report.update({'complete':complete,'evidence_valid':True,'task_requirements':requirements,
            'executed_reference_binding':reference_binding,
            'execution_contract':execution,'independent_interval':interval,'native_geometry':native,'metrics':metrics,
            'rejection_count':int(np.sum(np.asarray(trace.get('task_failure_reason',[]))!='none')),
            'runtime_reported_passed':scenario_result.get('passed') if isinstance(scenario_result,dict) else None,
            'runtime_report_is_new_task_gate':False,'fresh_replay_sha256':_sha(output_dir/'fresh_replay.npz'),
            'scope':{'restores_frozen_initial_state_once':True,'actual_saved_torque_replay':True,
                'fresh_mj_forward_after_each_step':True,'state_time':'post_integration_current_state',
                'reference_time':'same_actual_simulation_time','learned_torque_or_qpos_playback':False,
                'future_actual_state_used_for_inference':False,'continuous_time_certified':False}})
        report['task_success']=bool(complete and requirements['passed'] and execution['passed'] and interval['passed'] and native['passed'])
        report['passed']=report['task_success']
    except Exception as e:
        report['errors'].append({'type':type(e).__name__,'message':str(e)})
    report['evaluation_wall_s']=time.perf_counter()-started
    (output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest={p.relative_to(output_dir).as_posix():_sha(p) for p in output_dir.rglob('*') if p.is_file()}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return report


def summarize_trials(trials):
    """Do not credit repair, K8 or fallback as single raw diffusion success."""
    methods={}
    for row in trials:
        key=(row['method'],row.get('candidate_count',1),bool(row.get('repair_used')),bool(row.get('fallback_used')))
        name=f'{key[0]}|K={key[1]}|repair={key[2]}|fallback={key[3]}'
        methods.setdefault(name,[]).append(row)
    report={}
    for method,rows in methods.items():
        ids=[r['task_id'] for r in rows]
        if len(set(ids))!=len(ids):raise ValueError('duplicate task in same method/variant denominator')
        accepted=sum(r.get('proposal_accepted') is True for r in rows)
        success=sum(r.get('evaluation',{}).get('task_success') is True for r in rows)
        raw=sum(r.get('raw_proposal_passed') is True for r in rows)
        fallback_success=sum(r.get('fallback_used') is True and r.get('evaluation',{}).get('task_success') is True for r in rows)
        report[method]={'task_count':len(rows),'raw_proposal_pass_count':raw,'accepted_proposal_count':accepted,
            'task_success_count':success,'raw_proposal_pass_rate':raw/len(rows),
            'conditional_closed_loop_completion_rate':success/accepted if accepted else None,
            'total_task_completion_rate':success/len(rows),'fallback_success_count':fallback_success,
            'fallback_fraction_of_successes':fallback_success/success if success else None,
            'task_ids':ids,'proposal_only_is_task_success':False}
    return report
