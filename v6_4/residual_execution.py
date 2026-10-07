"""New Cartesian residual trials on the frozen nominal physical executor.

No joint trajectory is inferred from an endpoint route. Independent evaluation
reuses the original saved-torque replay, interval and whole-body policy, then
binds every consumed Cartesian reference to the frozen residual definition.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import time
import traceback

import mujoco
import numpy as np

from v6_4.task_protocol import TaskSpec
from v6_4.reference_adapter import CartesianPassThroughReferenceProvider, scenario_from_task
from v6_4.task_anchored_reference import TaskAnchoredResidualReferenceProvider, reference_precheck
from v6_4.evaluate_planning import (
    _replay, _execution_trace_checks, _finite, _stats, _shape_error_metrics, _performance,
)
from v6_4.proposal_gate import requirement_results
from v6_lite.hierarchical_qp import HierarchicalQPConfig
from v6_lite.run_v6_lite import V6LiteRunConfig, default_v6_lite_robot_spec, run_synchronous_scenario
from v6_lite.run_evidence import start_run, finish_run

ROOT=Path(__file__).resolve().parents[1]
REFERENCE_KEYS=(
    'rigid_target_position','rigid_target_velocity','rigid_target_rotation','rigid_target_angular_velocity',
    'continuum_target_position','continuum_target_velocity','continuum_target_rotation',
    'continuum_target_angular_velocity','posture_reference_q','posture_reference_dq')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')

def source_guard(identity_path):
    identity=json.loads(Path(identity_path).read_text(encoding='utf8'))
    hashes={name:sha(ROOT/name) for name in identity['source_sha256']}
    if hashes!=identity['source_sha256']:raise ValueError('source differs from frozen B.2 execution identity')
    for path,digest in identity.get('protected_artifacts',{}).items():
        if sha(path)!=digest:raise ValueError('historical protected artifact changed: '+path)
    return hashes

def consumed_prefix_view(trace_path,output):
    """Retain a missing final boundary using an already-saved actual state only."""
    trace_path=Path(trace_path);output=Path(output)
    with np.load(trace_path,allow_pickle=False) as saved:
        trace={key:saved[key].copy() for key in saved.files}
    n=len(trace['torque']);ticks=n//10
    if not n or n%10:raise ValueError('no complete consumed ten-step ramps')
    before=sha(trace_path);changed=[]
    if len(trace['task_qpos'])==ticks:
        final=_finite(trace['actual_full_qpos'],(n,81),'saved actual_full_qpos')[-1]
        trace['task_qpos']=np.concatenate((trace['task_qpos'],final[None]),axis=0)
        changed=['task_qpos:append_saved_actual_full_qpos_last']
    elif len(trace['task_qpos'])!=ticks+1:
        raise ValueError('saved planning boundaries do not match consumed torque prefix')
    if changed:
        path=output/'consumed_prefix_evidence_view.npz'
        np.savez_compressed(path,**trace)
    else:path=trace_path
    write(output/'trace_view_identity.json',{
        'source_trace':str(trace_path.resolve()),'source_sha256':before,
        'view_trace':str(path.resolve()),'view_sha256':sha(path),
        'changed_fields':changed,'physics_steps_added':0,'torques_changed':False,
        'unconsumed_rows_not_credited':True,'source_unchanged':sha(trace_path)==before})
    return trace,path

def timing_evidence(trace_path):
    """Retain the existing C.1 dispatch timeline without changing its policy."""
    from v6_lite.runtime_timing import latency_summary
    paths=sorted((Path(trace_path).parent.parent/'timing').glob('*.jsonl'))
    evidence=[]
    for path in paths:
        rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
        evidence.append({'path':str(path.resolve()),'sha256':sha(path),'cycle_count':len(rows),
            'dispatch_latency':latency_summary([row['dispatch_latency_s'] for row in rows]),
            'accepted_cycle_count':sum(row.get('accepted') is True for row in rows),
            'scope':'existing instrumented state acquisition to torque publication; research_simulation admission, no real delayed-state deployment validation'})
    return {'status':'MEASURED' if evidence else 'NOT_RUN','records':evidence,
        'wall_20ms_is_research_gate':False,'deployment':'NOT_MET',
        'real_calculation_delay_execution_validity':'NOT_VERIFIED',
        'continuous_time_or_hard_realtime_certification':False}

def reference_consumption_binding(task,plan,trace,state,spec):
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier,WholeBodyVerificationConfig
    scene=scenario_from_task(task)
    verifier=WholeBodyCollisionVerifier(spec,scene.obstacles,WholeBodyVerificationConfig(
        minimum_clearance=.005,query_distance_max=2.5,adaptive_subdivisions=4,
        self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True))
    model=verifier.model;observed=mujoco.MjData(model)
    observed.qpos[:]=task.initial_qpos;observed.qvel[:]=task.initial_qvel
    mujoco.mj_forward(model,observed)
    provider=TaskAnchoredResidualReferenceProvider(task,plan).prepare(spec,model,observed,scene)
    base=CartesianPassThroughReferenceProvider(task).prepare(spec,model,observed,scene)
    n=len(trace['torque']);ticks=n//10;inputs={};errors={};zero_errors={};peaks=[]
    _finite(trace['actual_full_qpos'],(n,81),'actual_full_qpos')
    _finite(trace['actual_full_qvel'],(n,79),'actual_full_qvel')
    for arm in ('rigid','continuum'):
        _finite(trace['generated_'+arm+'_position'],(n,3),'generated_'+arm+'_position')
        _finite(trace['generated_'+arm+'_rotation'],(n,3,3),'generated_'+arm+'_rotation')
    for key in ('q','dq'):
        _finite(trace['generated_reference_'+key],(n,17),'generated_reference_'+key)
    task_times=_finite(np.asarray(trace['task_time'])[:ticks],(ticks,),'task_time')
    if not np.allclose(task_times,state['time'][:-1:10],rtol=0.,atol=1e-9):
        raise ValueError('consumed QP input clock differs from physical execution')
    for key in REFERENCE_KEYS:
        shape=(ticks,3,3) if key.endswith('rotation') else (ticks,17) if key.startswith('posture') else (ticks,3)
        inputs[key]=_finite(np.asarray(trace['task_input_reference_'+key])[:ticks],shape,key)
    _finite(trace['generated_reference_time_s'],(n,),'generated_reference_time_s')
    if not np.allclose(trace['generated_reference_time_s'],state['time'][1:],rtol=0.,atol=1e-9):
        raise ValueError('generated reference clock differs from physical execution')
    if np.max(np.abs(trace['actual_full_qpos']-state['qpos'][1:]))>1e-9:
        raise ValueError('actual qpos does not reproduce from saved physical torque')
    if np.max(np.abs(trace['actual_full_qvel']-state['qvel'][1:]))>1e-8:
        raise ValueError('actual qvel does not reproduce from saved physical torque')
    for index,t in enumerate(state['time']):
        observed.qpos[:]=state['qpos'][index];observed.qvel[:]=state['qvel'][index];observed.time=float(t)
        mujoco.mj_forward(model,observed)
        expected=provider.sample(float(t));original=base.sample(float(t))
        if index%10==0 and index<n:
            for key in REFERENCE_KEYS:
                errors['QP_'+key]=max(errors.get('QP_'+key,0.),float(np.max(np.abs(inputs[key][index//10]-expected[key]))))
            peaks.append(float(np.linalg.norm(inputs['continuum_target_position'][index//10]-original['continuum_target_position'])))
        for key in REFERENCE_KEYS:
            if not np.any(plan.z_m) or key not in ('continuum_target_position','continuum_target_velocity'):
                zero_errors[key]=max(zero_errors.get(key,0.),float(np.max(np.abs(expected[key]-original[key]))))
        if index:
            # Original mj_step leaves pose caches from its forward stage. The
            # poststep rigid log is diagnostic, not a same-time QP input. Do
            # not impose new freshness semantics on the frozen controller.
            for arm in ('continuum',):
                for suffix in ('position','rotation'):
                    key=arm+'_'+suffix
                    errors['generated_'+key]=max(errors.get('generated_'+key,0.),float(np.max(np.abs(
                        trace['generated_'+key][index-1]-expected[arm+'_target_'+suffix]))))
            for key in ('q','dq'):
                errors['posture_'+key]=max(errors.get('posture_'+key,0.),float(np.max(np.abs(
                    trace['generated_reference_'+key][index-1]-expected['posture_reference_'+key]))))
    if max(errors.values())>1e-9 or max(zero_errors.values())!=0.:
        raise ValueError('actual consumed reference differs from frozen Cartesian residual or retained fields')
    return {'passed':True,'component_maximum_absolute_residual':errors,
        'retained_field_maximum_residual':zero_errors,'comparison_threshold':1e-9,
        'planning_inputs_bound':ticks,'QP_reference_fields_bound':list(REFERENCE_KEYS),
        'generated_continuum_and_posture_poststep_samples_bound':n,
        'generated_rigid_poststep_pose_scope':'finite cached MuJoCo diagnostic; not same-time fresh QP input, no freshness claim',
        'zero_residual_all_10_fields_exact_base_parity':bool(not np.any(plan.z_m) and max(zero_errors.values())==0.),
        'consumed_QP_reference_offset_peak_m':max(peaks,default=0.),
        'nonzero_reference_consumed':max(peaks,default=0.)>1e-12,
        'peak_at_least_5mm':max(peaks,default=0.)>=.005,
        'target_feedback_source':'independent same-torque current physical-state replay',
        'new_physics_steps_by_binding':0,'new_distance_queries_by_binding':0,
        'future_actual_trace_is_online_input':False,'joint_codec_check':'N/A/new-representation'}

def evaluate_residual(task,plan,trace_path,*,scenario_result,qp_config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    report={'schema':'v64_b2_residual_independent_evaluation_v1','task_id':task.task_id,
        'task_sha256':task.sha256(),'complete':False,'evidence_valid':False,'task_success':False,
        'full_task_success':False,'errors':[],'joint_codec_checks':'N/A/new-representation',
        'nominal_q_ref_whole_body_precheck':'N/A/no_joint_reference',
        'deployment':'NOT_MET','continuous_time_certified':False}
    try:
        trace,view=consumed_prefix_view(trace_path,output)
        n=len(trace['torque'])
        report.update(complete=n==13500,actual_physics_steps=n,actual_saved_horizon_s=float(trace['time'][-1]))
        _finite(trace['torque'],(n,67),'torque');_finite(trace['planner_q'],(n,17),'planner_q')
        _finite(trace['task_qpos'],(n//10+1,81),'task_qpos')
        if not np.allclose(trace['time'],np.arange(1,n+1)*.002,atol=1e-9,rtol=0.):
            raise ValueError('saved native physical times mismatch')
        if not np.array_equal(trace['initial_qpos'],task.initial_qpos) or not np.array_equal(trace['initial_qvel'],task.initial_qvel):
            raise ValueError('actual initial state differs from frozen Task')
        state,spec,interval,native=_replay(task,trace,qp_config,output)
        # Persist component verdicts before reference or Task metrics can fail.
        report.update(native_geometry=native,independent_interval=interval,
            fresh_replay_sha256=sha(output/'fresh_replay.npz'),replayed_physics_steps=n)
        report['dispatch_timing']=timing_evidence(trace_path)
        execution=_execution_trace_checks(task,trace,spec,qp_config)
        report['execution_contract']=execution
        if np.max(np.abs(state['q'][1:]-trace['planner_q']))>1e-9:
            raise ValueError('saved actual planner state does not reproduce')
        binding=reference_consumption_binding(task,plan,trace,state,spec)
        report['reference_binding']=binding
        requirements=requirement_results(task,state);report['task_requirements']=requirements
        home=np.tile(spec.planner_zero,(len(state['q']),1))
        metrics={'actual_tip_path_length_m':{arm:float(np.sum(np.linalg.norm(np.diff(state[arm+'_position'],axis=0),axis=1))) for arm in ('continuum','rigid')},
            'actual_joint_path_length_rad':float(np.sum(np.linalg.norm(np.diff(state['q'],axis=0),axis=1))),
            'soft_home_preference_error':_shape_error_metrics(state['q'],home,state['time'],
                getattr(spec,'planner_coordinate_names',tuple([f'theta{i+1}' for i in range(10)]+[f'rigid_q{i+1}' for i in range(7)])),
                reference_scope='unchanged Cartesian home posture preference, not generated joint reference'),
            'command_intervention_rad_s':_stats(np.asarray(trace['task_avoidance_intervention'])[:n//10]),
            'physics_steps':n,'planning_ticks':n//10,'actual_saved_horizon_s':float(trace['time'][-1]),
            'planning_algorithm_wall_latency_s':_performance(trace['task_full_latency'],.020),
            'torque_preparation_wall_latency_s':_performance(trace['torque_latency'],.002),
            'path_quality_comparison_eligible':n==13500 and requirements['passed']}
        complete=n==13500
        success=bool(complete and requirements['passed'] and execution['passed'] and interval['passed'] and native['passed'] and binding['passed'])
        metrics['path_quality_comparison_eligible']=success
        report.update(complete=complete,evidence_valid=True,task_success=success,
            full_task_success=success,metrics=metrics,trace_path=str(Path(trace_path).resolve()),
            trace_sha256=sha(trace_path),evidence_view_path=str(view.resolve()),
            runtime_reported_passed=scenario_result.get('passed') if isinstance(scenario_result,dict) else None,
            runtime_report_is_Task_gate=False,old_curve_rmse_is_success_gate=False,
            whole_body_scope='original 50Hz boundary + configuration subdivisions4; robot-target native500Hz')
    except Exception as error:
        report['errors'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    report['evaluation_wall_s']=time.perf_counter()-started
    write(output/'report.json',report)
    write(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    return report

def execute_residual_attempt(task,plan,output,*,qp_config_path,identity_path,slot_id,reuse_completed=False):
    output=Path(output);before=source_guard(identity_path)
    identity=json.loads(Path(identity_path).read_text(encoding='utf8'))
    config_path=str(Path(qp_config_path).resolve());config_sha=sha(qp_config_path)
    if identity.get('protected_artifacts',{}).get(config_path)!=config_sha:
        raise ValueError('execution QP configuration is not frozen in source identity')
    task_hash=task.sha256();plan_dict=plan.to_dict()
    plan_hash=hashlib.sha256(json.dumps(plan_dict,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if output.exists():
        retained=output/'attempt_result.json'
        if not reuse_completed or not retained.exists():raise FileExistsError('retained or unfinished slot: '+str(output))
        result=json.loads(retained.read_text(encoding='utf8'))
        if (result['slot_id'],result['task_sha256'],result['plan_content_sha256'],result['source_identity_sha256'],result['qp_config_sha256'])!=(slot_id,task_hash,plan_hash,sha(identity_path),config_sha):
            raise ValueError('completed slot reuse identity mismatch')
        if sha(output/'plan.json')!=result['plan_file_sha256']:
            raise ValueError('retained plan file differs')
        for key in ('trace','evaluation'):
            if result.get(key+'_path') and sha(result[key+'_path'])!=result[key+'_sha256']:
                raise ValueError('retained '+key+' evidence differs')
        if result.get('evaluation_path'):
            evaluation_dir=Path(result['evaluation_path']).parent
            for name,digest in json.loads((evaluation_dir/'manifest.json').read_text(encoding='utf8')).items():
                if sha(evaluation_dir/name)!=digest:raise ValueError('retained evaluation artifact differs: '+name)
        return result
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    write(output/'task.json',task.to_dict());write(output/'plan.json',plan_dict)
    write(output/'slot_started.json',{'slot_id':slot_id,'argv':[sys.executable,*sys.argv],
        'source_identity_sha256':sha(identity_path),'task_sha256':task_hash,'plan_content_sha256':plan_hash})
    result={'schema':'v64_b2_residual_attempt_v1','slot_id':slot_id,'task_id':task.task_id,
        'task_sha256':task_hash,'split':task.split,'plan_content_sha256':plan_hash,
        'qp_config_path':config_path,'qp_config_sha256':config_sha,
        'source_identity_sha256':sha(identity_path),'status':'STARTED','actual_steps':0,
        'entered_actual':False,'full_task_success':False,'full_27s_success':False,
        'fallback_used':False,'postprocessing':[],'deployment':'NOT_MET','evaluation':None}
    actual=output/'actual';metadata=None
    try:
        precheck=reference_precheck(task,plan,cartesian_speed_limit_m_s=.24)
        write(output/'reference_precheck.json',precheck);result['reference_precheck']=precheck
        if not precheck['passed']:
            result['status']='REFERENCE_PRECHECK_REJECTED'
        else:
            qp=HierarchicalQPConfig(**json.loads(Path(qp_config_path).read_text(encoding='utf8')))
            config=V6LiteRunConfig(pcc_mode='bounded_interval_pcc',dispatch_clock_policy='research_simulation')
            config.validate();spec=default_v6_lite_robot_spec();scene=scenario_from_task(task)
            metadata=start_run(actual,run_config=config,qp_config=qp,spec=spec,scenarios=(scene,))
            provider=TaskAnchoredResidualReferenceProvider(task,plan)
            result['actual_runner_started']=True
            print(json.dumps({'event':'B2_ACTUAL_STARTED','slot_id':slot_id,'task':task.task_id}),flush=True)
            scene_result=None
            try:
                scene_result=run_synchronous_scenario(spec,config,qp,scene,actual/'traces',reference_provider=provider)
                write(actual/'historical_metric_observations.json',scene_result)
                trace_path=actual/'traces'/f'{scene.scenario_id}.npz'
            except Exception as error:
                result['execution_failure']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
                write(actual/'execution_failure.json',result['execution_failure'])
                partials=sorted((actual/'failures').glob('*partial_trace.npz'))
                trace_path=partials[-1] if partials else None
            if trace_path is not None:
                with np.load(trace_path,allow_pickle=False) as trace:steps=len(trace['torque'])
                result['actual_steps']=steps;result['entered_actual']=steps>0
                result['trace_path']=str(trace_path.resolve());result['trace_sha256']=sha(trace_path)
                if steps:
                    evaluation=evaluate_residual(task,plan,trace_path,scenario_result=scene_result,qp_config=qp,output=actual/'evaluation')
                    result['evaluation']=evaluation;result['evaluation_path']=str((actual/'evaluation/report.json').resolve())
                    result['evaluation_sha256']=sha(actual/'evaluation/report.json')
                    success=bool(not result.get('execution_failure') and evaluation['full_task_success'])
                    result.update(full_task_success=success,full_27s_success=success,
                        status='TASK_COMPLETED' if success else 'EXECUTION_REFUSED' if result.get('execution_failure') else 'TASK_OR_EVIDENCE_FAILED')
                else:result['status']='ZERO_STEP_EXECUTION_REFUSAL'
            else:result['status']='EXECUTION_FAILURE_NO_TRACE'
            finish_run(actual,metadata,passed=result['full_task_success'],summary={'slot_id':slot_id,'status':result['status'],'task_success':result['full_task_success']})
    except Exception as error:
        result['status']='PIPELINE_FAILURE';result['pipeline_failure']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    after=source_guard(identity_path)
    result.update(source_sha256_before=before,source_sha256_after=after,sources_unchanged=before==after,
        elapsed_wall_s=time.perf_counter()-started,plan_file_sha256=sha(output/'plan.json'))
    write(output/'attempt_result.json',result)
    print(json.dumps({'event':'B2_ACTUAL_TERMINAL','slot_id':slot_id,'status':result['status'],
        'actual_steps':result['actual_steps'],'full_task_success':result['full_task_success']}),flush=True)
    return result
