"""Reject inadmissible proposals without clipping, optimization or fallback.

Nominal screening is a discrete prediction, not an execution certificate.  Every
accepted reference still passes the existing runtime QP, interval, ramp and torque
checks.  The same screening grid and native pair policy apply to all methods.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from v6_4.task_protocol import TaskSpec


def _array(value, shape, name):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a


def requirement_results(task: TaskSpec, state: dict):
    """Position and orientation must be satisfied at the SAME saved time.

    Terminal criteria use only the actual final state.  An earlier visit does
    not establish task completion at the frozen deadline.
    """
    times=np.asarray(state['time'],dtype=float)
    if times.ndim!=1 or not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError("task evaluation needs nonempty increasing finite times")
    count=len(times)
    target_p=_array(state['target_position'],(count,3),'target_position')
    target_r=_array(state['target_rotation'],(count,3,3),'target_rotation')
    outcomes=[]
    for p in task.requirements:
        position=_array(state[p.arm+'_position'],(count,3),p.arm+'_position')
        rotation=_array(state[p.arm+'_rotation'],(count,3,3),p.arm+'_rotation')
        desired_p=np.asarray(p.position_m)
        desired_r=np.asarray(p.rotation).reshape(3,3)
        if p.frame=='target':
            desired_p=target_p+np.einsum('nij,j->ni',target_r,desired_p)
            desired_r=np.einsum('nij,jk->nik',target_r,desired_r)
        position_error=np.linalg.norm(position-desired_p,axis=1)
        # tr(R_desired.T R_actual) = Frobenius inner product.
        cosine=(np.sum(rotation*desired_r,axis=(1,2))-1.)*.5
        angle=np.arccos(np.clip(cosine,-1.,1.))
        if p.kind=='terminal':
            indices=np.array([count-1],dtype=int) if abs(times[-1]-task.duration_s)<=1e-9 else np.array([],dtype=int)
        else:
            indices=np.flatnonzero((times>=p.time_window_s[0]-1e-9)&(times<=p.time_window_s[1]+1e-9))
        simultaneous=(position_error[indices]<=p.position_tolerance_m)&(angle[indices]<=p.orientation_tolerance_rad)
        best=int(indices[np.argmin(np.maximum(position_error[indices]/p.position_tolerance_m,
             angle[indices]/p.orientation_tolerance_rad))]) if len(indices) else None
        outcomes.append({'point_id':p.point_id,'arm':p.arm,'kind':p.kind,'frame':p.frame,
            'passed':bool(np.any(simultaneous)),'eligible_state_count':len(indices),
            'best_time_s':float(times[best]) if best is not None else None,
            'position_error_m':float(position_error[best]) if best is not None else None,
            'orientation_error_rad':float(angle[best]) if best is not None else None,
            'position_tolerance_m':p.position_tolerance_m,'orientation_tolerance_rad':p.orientation_tolerance_rad})
    return {'passed':all(x['passed'] for x in outcomes),'requirements':outcomes,
            'definition':'same_time_position_and_orientation_with_frozen_windows_terminal_at_final_state',
            'old_curve_rmse_is_success_gate':False}


def gate_nominal_prediction(task, prediction, spec, *, verifier=None):
    """Check a complete private nominal prediction; no future executed trace."""
    from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier,WholeBodyVerificationConfig,WorkspaceSphere
    from v6_lite.continuum_model_spec import default_continuum_model_spec
    checks={}; errors=[]; geometry=None; requirements=None
    try:
        times=_array(prediction['time'],(1351,),'time')
        q=_array(prediction['q'],(1351,17),'q')
        dq=_array(prediction['dq'],(1351,17),'dq')
        qpos=_array(prediction['full_qpos'],(1351,81),'full_qpos')
        _array(prediction['full_qvel'],(1351,79),'full_qvel')
        _array(prediction['base_pose'],(1351,7),'base_pose')
        checks['same_nominal_model']=spec.runtime_contract_sha256()==task.model_contract_sha256
        checks['complete_frozen_50hz_grid']=np.allclose(times,np.arange(1351)*.020,atol=1e-12,rtol=0.)
        checks['fixed_initial_reference']=np.allclose(q[0],task.initial_planner_q,atol=1e-12,rtol=0.) and np.allclose(dq[0],task.initial_planner_dq,atol=1e-12,rtol=0.)
        checks['fixed_full_initial_state']=np.array_equal(qpos[0],task.initial_qpos) and np.array_equal(np.asarray(prediction['full_qvel'])[0],task.initial_qvel)
        checks['raw_reconstructed_position_range']=bool(np.all(q>=spec.planner_lower)&np.all(q<=spec.planner_upper))
        checks['reference_velocity_range']=bool(np.all(np.abs(dq)<=spec.planner_velocity_limits+1e-12))
        domain=default_continuum_model_spec(spec)
        checks['original_pcc_work_domain']=bool(np.all(q[:,:10]>=domain.work_domain_lower_rad)&np.all(q[:,:10]<=domain.work_domain_upper_rad))
        requirements=requirement_results(task,prediction)
        checks['independent_task_points_and_terminal']=requirements['passed']
        if all(checks.values()):
            if verifier is None:
                obstacles=tuple(WorkspaceSphere(o['name'],np.asarray(o['center_w']),o['radius_m']) for o in task.scenario['workspace_obstacles'])
                verifier=WholeBodyCollisionVerifier(spec,obstacles,WholeBodyVerificationConfig(
                    minimum_clearance=.005,query_distance_max=2.5,adaptive_subdivisions=1,
                    self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True))
            if not verifier.pairs:
                raise ValueError('native proposal pair policy is empty')
            # FK is refreshed from the full private free-base prediction.  A
            # forged successful pose log cannot certify a different qpos path.
            import mujoco
            from v6_lite.hierarchical_qp import joint_addresses,free_joint_slices
            model=verifier.model;data=mujoco.MjData(model)
            qids,vids=joint_addresses(model,spec);bs,_=free_joint_slices(model,spec.base_joint_name)
            checks['predicted_full_state_matches_spline_map']=bool(np.allclose(spec.decode_position(qpos[:,qids]),q,atol=1e-12,rtol=0.) and
                np.allclose(spec.decode_velocity(np.asarray(prediction['full_qvel'])[:,vids]),dq,atol=1e-12,rtol=0.) and
                np.array_equal(qpos[:,bs],np.asarray(prediction['base_pose'])))
            bodies={arm:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,getattr(spec,arm+'_tip_body_name')) for arm in ('rigid','continuum')}
            target_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,spec.target_free_joint_name)
            bodies['target']=int(model.jnt_bodyid[target_joint])
            if any(body<0 for body in bodies.values()):raise ValueError('prediction FK body mapping missing')
            residual=0.
            for i in range(len(times)):
                data.qpos[:]=qpos[i];data.qvel[:]=prediction['full_qvel'][i]
                mujoco.mj_forward(model,data)
                for name,body in bodies.items():
                    r=data.xmat[body].reshape(3,3);p=data.xpos[body].copy()
                    if name=='continuum':p+=r@np.asarray([.0475,0.,0.])
                    residual=max(residual,float(np.max(np.abs(p-np.asarray(prediction[name+'_position'])[i]))),
                        float(np.max(np.abs(r-np.asarray(prediction[name+'_rotation'])[i]))))
            checks['fresh_fk_matches_declared_task_pose_prediction']=bool(np.isfinite(residual) and residual<=1e-10)
            if all(checks.values()):
                geometry=verifier.verify_qpos_sequence(qpos).to_dict()
                checks['nominal_free_base_native_geometry']=geometry['feasible'] is True and geometry['minimum_clearance']>=.005
    except (ValueError,KeyError,TypeError,RuntimeError) as e:
        errors.append({'type':type(e).__name__,'message':str(e)})
        checks['prediction_evidence_well_formed']=False
    return {'passed':bool(checks) and all(checks.values()) and not errors,'checks':{k:bool(v) for k,v in checks.items()},
            'errors':errors,'task_requirements':requirements,'geometry':geometry,
            'scope':{'nominal_prediction_only':True,'uses_fixed_base_fk':False,'physics_steps_executed':0,
                     'future_actual_trace_used':False,'geometry_grid_period_s':.020,
                     'geometry_adaptive_subdivisions':1,'runtime_safety_checks_still_required':True,
                     'continuous_time_certified':False}}


def gate_proposal(task, proposal, spec, *, provider=None, output_path=None):
    """Standard gate for fixed, teacher, raw diffusion and attributed variants."""
    started=time.perf_counter(); checks={}; errors=[]; screening=None
    try:
        from v6_4.trajectory_codec import CubicBSplineCodec
        from v6_4.reference_adapter import SplineReferenceProvider
        controls=_array(proposal.controls_free,(30,17),'controls_free')
        checks['proposal_bound_to_frozen_task']=proposal.task_id==task.task_id and proposal.task_sha256==task.sha256()
        codec=CubicBSplineCodec(task.initial_planner_q,task.initial_planner_dq)
        full=codec.decode_free(controls)
        checks['raw_control_point_range']=bool(np.all(full>=spec.planner_lower)&np.all(full<=spec.planner_upper))
        if all(checks.values()):
            if provider is None:
                import mujoco
                from v6_4.reference_adapter import scenario_from_task
                provider=SplineReferenceProvider(task,full)
                scenario=scenario_from_task(task)
                from model_test.whole_body_verifier_v5 import WholeBodyCollisionVerifier,WholeBodyVerificationConfig
                verifier=WholeBodyCollisionVerifier(spec,scenario.obstacles,WholeBodyVerificationConfig(
                    minimum_clearance=.005,query_distance_max=2.5,adaptive_subdivisions=1,
                    self_collision_ancestor_exclusion_depth=3,include_target_satellite_pairs=True))
                verifier.model.geom_contype[:]=0;verifier.model.geom_conaffinity[:]=0
                data=mujoco.MjData(verifier.model)
                data.qpos[:]=task.initial_qpos;data.qvel[:]=task.initial_qvel
                mujoco.mj_forward(verifier.model,data)
                provider.prepare(spec,verifier.model,data,scenario)
                screening=gate_nominal_prediction(task,provider.prediction,spec,verifier=verifier)
            else:
                screening=gate_nominal_prediction(task,provider.prediction,spec)
            samples=codec.sample(full,np.asarray(provider.prediction['time']))
            checks['prediction_bound_to_raw_controls']=np.allclose(samples['q'],provider.prediction['q'],atol=1e-12,rtol=0.) and np.allclose(samples['dq'],provider.prediction['dq'],atol=1e-12,rtol=0.)
            checks['nominal_screen_passed']=screening['passed']
    except (ValueError,KeyError,TypeError,RuntimeError) as e:
        errors.append({'type':type(e).__name__,'message':str(e)});checks['raw_candidate_well_formed']=False
    passed=bool(checks) and all(checks.values()) and not errors
    report={'schema':'v6_4_proposal_gate_v1','task_id':task.task_id,'task_sha256':task.sha256(),
            'passed':passed,'raw_passed':passed and not proposal.postprocessing and proposal.origin!='fallback',
            'origin':proposal.origin,'seed':proposal.seed,'postprocessing':list(proposal.postprocessing),
            'checks':{k:bool(v) for k,v in checks.items()},'errors':errors,'nominal_screen':screening,
            'elapsed_wall_s':time.perf_counter()-started,'clipped_or_repaired':False,
            'fallback_performed':False,'accepted_reference_is_execution_certificate':False,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if output_path is not None:
        path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report
