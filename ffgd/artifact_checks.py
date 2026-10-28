"""Independent CLI consumers of saved candidate and native execution inputs."""
import json
from pathlib import Path
import numpy as np
from .problem_io import load_problem
from .contracts import Candidate,Snapshot
from .api import boundary_codec
from .physics import DifferentiableTree
from .propagation import Propagator
from .geometry import Geometry
from .validation import validate
from .execution import replay
from .orchestration import atomic_json,check_budget


def inspect_chain(output,config):
    """Read-only all-state continuity and momentum about the single root origin.

    Segment reports are necessary but cannot erase inherited momentum error.
    The historical generalized/action-scaled diagnostic is reported separately.
    This inspector does not run physics, select candidates or repair plans.
    """
    import hashlib
    import mujoco
    from .execution import refine_physics
    from .trajectory import basis
    from .propagation import native_subtree_momentum
    sources=config['execution_runs'];replays=config['replay_runs']
    if len(sources)!=len(replays) or not sources:raise ValueError('CHAIN_REQUIRES_MATCHED_EXECUTIONS_AND_REPLAYS')
    rows=[];previous=None;root_H=None;root_time=None;root_model=None;root_profile=None
    source_ids=set();weights=set();world_peaks=np.zeros(2);legacy_peaks=np.zeros(2);legacy_exchange=np.zeros(2)
    for execution,replay_run in zip(sources,replays):
        check_budget();root=Path(execution);provenance=json.loads((root/'provenance.json').read_text())
        planned=Path(provenance['planning_directory']);manifest=json.loads((planned/'planning_manifest.json').read_text())
        unit=json.loads((planned/'manifest.json').read_text());source_ids.add(unit['source_hash']);weights.add(manifest['checkpoint']['sha256'])
        acceptance=json.loads((root/'acceptance.json').read_text());replay_result=json.loads((Path(replay_run)/'result.json').read_text())
        if Path(replay_result['source']).resolve()!=root.resolve():raise ValueError('CHAIN_REPLAY_EXECUTION_MISMATCH')
        r,snapshot,task,_=load_problem(planned/'problem.json');r,snapshot=refine_physics(r,snapshot,provenance['physics_substeps'])
        result=json.loads((planned/'planning_result.json').read_text());index=result['selected_index']
        with np.load(planned/f'candidate_{index:03d}'/'candidate.npz') as data:candidate=Candidate(data['control'],float(data['duration']),int(data['seed']))
        if candidate.identity!=provenance['candidate_hash'] or result['records'][index]['identity']!=candidate.identity:raise ValueError('CHAIN_CANDIDATE_IDENTITY_MISMATCH')
        with np.load(root/'execution.npz') as saved:bundle={k:saved[k] for k in saved.files}
        if not np.array_equal(bundle['control_points'],candidate.control):raise ValueError('CHAIN_EXECUTED_DIFFERENT_REFERENCE')
        d=r.restore(snapshot);start=float(d.time);mask=int(bundle['mask']);dt=float(bundle['physics_dt'])
        if root_H is None:
            root_H=native_subtree_momentum(r,d);root_time=start;root_model=r.model_hash;root_profile=manifest['profile']
        same_contract=r.model_hash==root_model and manifest['profile']==root_profile
        state_exact=np.array_equal(bundle['states'][0],snapshot.state)
        continuity=True if previous is None else np.array_equal(bundle['states'][0],previous['state'])
        reference_error=None
        if previous is not None:
            reference_error=None if task.reference_start is None else float(np.max(abs(task.reference_start-previous['reference'])))
        time_error=0.;local_peaks=np.zeros(2)
        for i,state in enumerate(bundle['states']):
            if i%128==0:check_budget()
            d=r.restore(Snapshot(r.model_hash,mask,state));H=native_subtree_momentum(r,d)
            time_error=max(time_error,abs(float(d.time)-(start+i*dt)))
            delta=H-root_H;peaks=np.array([np.linalg.norm(delta[:3]),np.linalg.norm(delta[3:])])
            local_peaks=np.maximum(local_peaks,peaks);world_peaks=np.maximum(world_peaks,peaks)
            M=r.mass(d);old=(M@d.qvel)[r.base_dofs];exchange=M[np.ix_(r.base_dofs,r.dof_ids)]@d.qvel[r.dof_ids]
            legacy_peaks=np.maximum(legacy_peaks,[np.linalg.norm(old[:3]),np.linalg.norm(old[3:])])
            legacy_exchange=np.maximum(legacy_exchange,[np.linalg.norm(exchange[:3]),np.linalg.norm(exchange[3:])])
        end=float(d.time);complete=abs(len(bundle['ctrl'])*dt-task.duration)<=1e-10 and len(bundle['states'])==len(bundle['ctrl'])+1
        boundary_time=previous is None or abs(start-previous['time'])<=1e-9
        reference_ok=previous is None or reference_error is not None and reference_error<=1e-10
        row_gates=dict(execution_valid=acceptance['status']=='EXECUTION_VALID',separate_replay=bool(replay_result['passed']),same_model_and_profile=same_contract,
            exact_initial_state=state_exact,exact_actual_boundary=continuity,c2_reference_boundary=reference_ok,absolute_boundary_time=boundary_time,
            complete_duration=complete,absolute_clock=time_error<=1e-9)
        rows.append(dict(execution=str(root),replay=str(replay_run),planning=str(planned),candidate_hash=candidate.identity,start=start,end=end,
            gates=row_gates,reference_boundary_max_error=reference_error,maximum_clock_error=time_error,root_momentum_peaks=local_peaks.tolist(),
            execution_npz_sha256=hashlib.sha256((root/'execution.npz').read_bytes()).hexdigest()))
        previous=dict(state=bundle['states'][-1].copy(),time=end,reference=np.stack([basis(len(candidate.control),candidate.duration,candidate.duration,k)@candidate.control for k in (0,1,2)]))
    thresholds=np.array([root_profile['momentum_linear_kg_m_s'],root_profile['momentum_angular_kg_m2_s']])
    gates=dict(segments_and_boundaries=all(all(x['gates'].values()) for x in rows),cumulative_world_momentum=bool(np.all(world_peaks<=thresholds)))
    if config.get('require_uniform_planner_version',True):gates['uniform_planner_source_and_weights']=len(source_ids)==1 and len(weights)==1
    legacy_threshold=1e-5+.002*legacy_exchange
    return dict(passed=all(gates.values()),gates=gates,segments=rows,absolute_start=root_time,absolute_end=previous['time'],
        cumulative_world_momentum_peaks=world_peaks.tolist(),world_momentum_thresholds=thresholds.tolist(),
        historical_generalized_action_scaled=dict(passed=bool(np.all(legacy_peaks<=legacy_threshold)),peaks=legacy_peaks.tolist(),thresholds=legacy_threshold.tolist(),
            scope='Historical zero-generalized-momentum criterion, separate from world momentum about root; no old result relabeling'),
        source_hashes=sorted(source_ids),checkpoint_hashes=sorted(weights),readonly=True,source_reports_modified=False,
        scope='All provided native states, exact actual and C2 boundaries, root momentum and independent replay bindings; final task coverage must also be checked')


def validate_saved(output,planning_directory,candidate_index=None):
    root=Path(planning_directory);r,snapshot,task,problem=load_problem(root/'problem.json')
    manifest=json.loads((root/'planning_manifest.json').read_text(encoding='utf-8'));profile=manifest['profile']
    from .propagation import FullStatePropagation
    prop=(FullStatePropagation(DifferentiableTree(r),snapshot,physics_dt=manifest['budget'].get('prediction_dt',.002),target_force_pulses=task.target_force_pulses) if manifest.get('physics_backend')=='full_state' else Propagator(DifferentiableTree(r),snapshot));geometry=Geometry(r);rows=[]
    folders=[root/f'candidate_{int(candidate_index):03d}'] if candidate_index is not None else sorted(root.glob('candidate_*'))
    for folder in folders:
        check_budget()
        if not (folder/'candidate.npz').exists():
            rows.append(dict(candidate=folder.name,status='NOT_RUN_NO_COMPLETE_CANDIDATE'));continue
        with np.load(folder/'candidate.npz') as data:c=Candidate(data['control'],float(data['duration']),int(data['seed']))
        record=json.loads((folder/'candidate.json').read_text(encoding='utf-8'))
        if c.identity!=record['identity']:raise ValueError('CANDIDATE_IDENTITY_MISMATCH')
        codec=boundary_codec(r,snapshot,task.duration,len(c.control),origin=manifest['budget'].get('boundary_origin','pivot_zero'),reference_start=task.reference_start,terminal_policy=task.terminal_policy);report=validate(c,snapshot,task,prop,codec,geometry,profile)
        atomic_json(output/(folder.name+'.json'),report);rows.append(dict(candidate=folder.name,**report))
    return dict(passed=bool(rows) and all(x['status']=='PLAN_VALID' for x in rows),rows=rows,source=str(root),read_only=True,source_reports_modified=False)


def replay_saved(output,execution_directory):
    root=Path(execution_directory);provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    r,snapshot,task,problem=load_problem(Path(provenance['planning_directory'])/'problem.json')
    from .execution import refine_physics
    r,snapshot=refine_physics(r,snapshot,provenance.get('physics_substeps',1))
    if r.model_hash!=provenance.get('execution_model_hash',r.model_hash):raise ValueError('EXECUTION_MODEL_IDENTITY_MISMATCH')
    filename='execution.npz' if (root/'execution.npz').exists() else 'execution_partial.npz'
    with np.load(root/filename) as data:bundle={k:data[k] for k in data.files}
    if len(bundle['states'])!=len(bundle['ctrl'])+1:raise ValueError('INCOMPLETE_INTEGRATION_INPUT_LEDGER')
    np.testing.assert_array_equal(snapshot.state,bundle['initial_state'])
    report=replay(r,bundle,snapshot)
    from .execution_acceptance import inspect_state_bounds
    bounds=inspect_state_bounds(r,bundle,provenance['profile'])
    return dict(passed=report['max_state_abs']<=provenance['profile']['replay_state_atol'],source=str(root),replay=report,
                actual_state_bounds=bounds,actual_state_bounds_passed=all(bounds['gates'].values()),
                model_hash=r.model_hash,source_reports_modified=False,source_file=filename,
                scope='replay of recorded inputs only; does not establish execution completion or task success')


def validate_execution_states(output,execution_directory,tracking_profile=None):
    """Inspect immutable execution records without integrating or editing them."""
    from .execution_acceptance import inspect_state_bounds
    root=Path(execution_directory);provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    r,snapshot,task,problem=load_problem(Path(provenance['planning_directory'])/'problem.json')
    from .execution import refine_physics
    r,snapshot=refine_physics(r,snapshot,provenance.get('physics_substeps',1))
    if r.model_hash!=provenance.get('execution_model_hash',r.model_hash):raise ValueError('EXECUTION_MODEL_IDENTITY_MISMATCH')
    with np.load(root/'execution.npz') as data:bundle={k:data[k] for k in data.files}
    np.testing.assert_array_equal(snapshot.state,bundle['initial_state'])
    profile=dict(provenance['profile'])
    if tracking_profile is not None:
        thresholds=tracking_profile['thresholds']
        if set(thresholds)!={'tracking_error_rad','manifold_residual_rad'}:raise ValueError('INVALID_TRACKING_PROFILE_FIELDS')
        if any(not np.isfinite(v) or not 0<v<=.02 for v in thresholds.values()):raise ValueError('TRACKING_PROFILE_EXCEEDS_INHERITED_GATES')
        profile.update(thresholds)
    report=inspect_state_bounds(r,bundle,profile)
    return dict(passed=all(report['gates'].values()),source=str(root),state_bounds=report,
                source_reports_modified=False,native_integration_steps=0,tracking_profile_override=tracking_profile,
                scope='saved actual position/velocity/applied-input acceleration bounds only; not a new execution or full M5 acceptance')


def compare_tracking_prediction(output,execution_directory):
    """Measure nominal floating-path versus immutable actual world poses.

    Scratch-state restoration is observation only, never execution or replay.
    Finite measurements do not establish a tracking bound or change any gate.
    """
    import torch
    import mujoco
    from scipy.spatial.transform import Rotation
    from .model_io import file_hash
    from .execution import refine_physics
    torch.set_num_threads(1)
    root=Path(execution_directory)
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    planning=Path(provenance['planning_directory'])
    result=json.loads((planning/'planning_result.json').read_text(encoding='utf-8'))
    if result.get('selected_index') is None:raise ValueError('MISSING_PRESELECTED_PLAN')
    candidate_file=planning/f'candidate_{result["selected_index"]:03d}'/'candidate.npz'
    with np.load(candidate_file) as data:candidate=Candidate(data['control'],float(data['duration']),int(data['seed']))
    if candidate.identity!=provenance['candidate_hash']:raise ValueError('EXECUTED_CANDIDATE_IDENTITY_MISMATCH')
    actual_file=root/'execution.npz'
    before={str(p):file_hash(p) for p in (candidate_file,actual_file,root/'acceptance.json')}
    with np.load(actual_file) as data:bundle={k:data[k] for k in data.files}
    np.testing.assert_array_equal(candidate.control,bundle['control_points'])
    if float(bundle['duration'])!=candidate.duration:raise ValueError('EXECUTED_DURATION_IDENTITY_MISMATCH')
    r,snapshot,task,_=load_problem(planning/'problem.json')
    r,snapshot=refine_physics(r,snapshot,provenance.get('physics_substeps',1))
    if r.model_hash!=provenance.get('execution_model_hash',r.model_hash):raise ValueError('EXECUTION_MODEL_IDENTITY_MISMATCH')
    np.testing.assert_array_equal(snapshot.state,bundle['initial_state'])
    if int(bundle['mask'])!=snapshot.mask:raise ValueError('EXECUTION_STATE_MASK_MISMATCH')
    n=len(bundle['ctrl'])
    if len(bundle['states'])!=n+1:raise ValueError('INCOMPLETE_INTEGRATION_INPUT_LEDGER')
    elapsed=n*r.model.opt.timestep
    if elapsed>candidate.duration+1e-10:raise ValueError('EXECUTION_EXCEEDS_REFERENCE_DURATION')
    if abs(elapsed-candidate.duration)<=1e-10:elapsed=candidate.duration
    times=np.linspace(0,elapsed,n+1)
    manifest=json.loads((planning/'planning_manifest.json').read_text(encoding='utf-8'))
    codec=boundary_codec(r,snapshot,candidate.duration,len(candidate.control),origin=manifest['budget'].get('boundary_origin','pivot_zero'),reference_start=task.reference_start,terminal_policy=task.terminal_policy)
    prop=Propagator(DifferentiableTree(r),snapshot)
    xs,_=prop.native_rollout(codec,codec.encode(candidate.control),n,grid=times)
    position=np.zeros((n+1,2));rotation=np.zeros_like(position)
    base_position=np.zeros(n+1);base_rotation=np.zeros(n+1)
    reference_velocity=codec.evaluate(codec.encode(candidate.control),times,1)
    reference_acceleration=codec.evaluate(codec.encode(candidate.control),times,2)
    velocity_errors=np.zeros((n+1,17));acceleration_errors=np.zeros_like(velocity_errors)
    derivative_maxima={}
    predicted=mujoco.MjData(r.model);task_rows=[]
    for i,state in enumerate(bundle['states']):
        if i%64==0:check_budget()
        actual=r.restore(Snapshot(r.model_hash,int(bundle['mask']),state))
        if i<n:
            actual.ctrl[:]=bundle['ctrl'][i]
            actual.qfrc_applied[:]=bundle['qfrc_applied'][i]
            actual.xfrc_applied[:]=bundle['xfrc_applied'][i]
        # Same applied-input convention as independent execution acceptance.
        mujoco.mj_forward(r.model,actual)
        for name,observed,reference,limits,errors in (
            ('velocity',actual.qvel[r.dof_ids],reference_velocity[i],r.spec.planner_velocity_limits,velocity_errors),
            ('acceleration',actual.qacc[r.dof_ids],reference_acceleration[i],r.spec.planner_acceleration_limits,acceleration_errors)):
            reduced=r.Bplus@observed
            errors[i]=(reduced-reference)/limits
            for space,values,ref,envelope in (('q17',reduced,reference,limits),('q67',observed,r.B@reference,abs(r.B)@limits)):
                ratios=abs(values)/envelope;j=int(np.argmax(ratios));key=f'{space}_{name}'
                if key not in derivative_maxima or ratios[j]>derivative_maxima[key]['actual_ratio']:
                    derivative_maxima[key]=dict(time=float(times[i]),joint=j,actual_ratio=float(ratios[j]),
                        reference_ratio_at_same_time=float(abs(ref[j])/envelope[j]),
                        signed_prediction_error_ratio=float((values[j]-ref[j])/envelope[j]))
        predicted.qpos[:]=xs[i]
        for data in (actual,predicted):mujoco.mj_kinematics(r.model,data)
        base_position[i]=np.linalg.norm(actual.qpos[r.base_qpos[:3]]-predicted.qpos[r.base_qpos[:3]])
        base_rotation[i]=Rotation.from_matrix(predicted.xmat[r.base_body].reshape(3,3).T@actual.xmat[r.base_body].reshape(3,3)).magnitude()
        for tip,body in enumerate(r.tips):
            Ra=actual.xmat[body].reshape(3,3);Rp=predicted.xmat[body].reshape(3,3)
            pa=actual.xpos[body].copy();pp=predicted.xpos[body].copy()
            if tip==0:pa+=Ra@np.array([.0475,0,0]);pp+=Rp@np.array([.0475,0,0])
            position[i,tip]=np.linalg.norm(pa-pp)
            rotation[i,tip]=Rotation.from_matrix(Rp.T@Ra).magnitude()
            for target in task.poses:
                if target.tip==tip and abs(times[i]-target.time)<=1e-10:
                    task_rows.append(dict(time=target.time,tip=tip,prediction_position_error_m=float(position[i,tip]),
                        prediction_rotation_error_rad=float(rotation[i,tip]),nominal_goal_position_error_m=float(np.linalg.norm(pp-target.position)),
                        actual_goal_position_error_m=float(np.linalg.norm(pa-target.position)),
                        nominal_goal_rotation_error_rad=float(Rotation.from_matrix(target.rotation.T@Rp).magnitude()),
                        actual_goal_rotation_error_rad=float(Rotation.from_matrix(target.rotation.T@Ra).magnitude()),
                        position_tolerance_m=target.position_tolerance,rotation_tolerance_rad=target.rotation_tolerance))
    arrays=dict(time=times,tip_position_error_m=position,tip_rotation_error_rad=rotation,
                base_position_error_m=base_position,base_rotation_error_rad=base_rotation,
                q17_velocity_prediction_error_ratio=velocity_errors,q17_acceleration_prediction_error_ratio=acceleration_errors)
    np.savez_compressed(output/'prediction_errors.npz',**arrays)
    unchanged=all(file_hash(Path(p))==digest for p,digest in before.items())
    return dict(passed=unchanged and all(np.isfinite(v).all() for v in arrays.values()),source=str(root),
                source_acceptance_status=json.loads((root/'acceptance.json').read_text(encoding='utf-8'))['status'],
                candidate_hash=candidate.identity,source_files_unchanged=unchanged,states_compared=n+1,
                native_execution_steps=0,nominal_propagation_intervals=n,physics_dt=r.model.opt.timestep,
                initial_state_bytes_unchanged=True,actual_states_projected=False,
                maximum_tip_position_error_m=position.max(0).tolist(),maximum_tip_rotation_error_rad=rotation.max(0).tolist(),
                maximum_base_position_error_m=float(base_position.max()),maximum_base_rotation_error_rad=float(base_rotation.max()),
                derivative_maxima=derivative_maxima,
                maximum_q17_velocity_prediction_error_ratio=float(abs(velocity_errors).max()),
                maximum_q17_acceleration_prediction_error_ratio=float(abs(acceleration_errors).max()),
                task_times=task_rows,unmeasured_task_times=len(task.poses)-len(task_rows),
                engineering_status='NOT_RUN',tracking_domain='NOT_ESTABLISHED',
                scope='read-only measurement on saved executions; not a new execution, acceptance verdict, or certified tracking bound')
