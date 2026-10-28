"""Public JSON problem IO. Only state, world tasks and scene configuration."""
import json
from pathlib import Path
import numpy as np
import mujoco
from .contracts import BoxObstacle,SphereObstacle,SceneSpec,PoseConstraint,PoseInterval,ForcePulse,TaskSpec,Snapshot
from .robot import Robot
from .api import PlanningBudget
from .diffusion import SamplingBudget


def load_problem(path):
    path=Path(path);value=json.loads(path.read_text(encoding='utf-8-sig'))
    allowed={'schema','id','split','initial','task','scene','description'}
    if set(value)-allowed:raise ValueError('unknown public problem keys')
    if set(value.get('scene',{}))-{'boxes','spheres'}:raise ValueError('unknown scene keys')
    if set(value['task'])-{'duration','poses','intervals','reference_start','terminal_policy','target_force_pulses'} or 'duration' not in value['task']:raise ValueError('unknown or missing task keys')
    for box in value.get('scene',{}).get('boxes',[]):
        if set(box)-{'center','half_size','rotation'}:raise ValueError('unknown obstacle keys')
    for sphere in value.get('scene',{}).get('spheres',[]):
        if set(sphere)!={'center','radius'}:raise ValueError('unknown or missing sphere keys')
    scene=SceneSpec(tuple(BoxObstacle(b['center'],b['half_size'],b.get('rotation',np.eye(3))) for b in value.get('scene',{}).get('boxes',[])),
                    tuple(SphereObstacle(s['center'],s['radius']) for s in value.get('scene',{}).get('spheres',[])))
    r=Robot(scene=scene);d=r.new_data();initial=value['initial']
    if set(initial)-{'q17','v17','base_pose','base_velocity','target_pose','target_velocity','snapshot_file'}:raise ValueError('unknown initial-state keys')
    if 'snapshot_file' in initial:
        if len(initial)!=1:raise ValueError('ambiguous snapshot and initial overrides')
        source=path.parent/initial['snapshot_file']
        with np.load(source) as saved:snapshot=Snapshot(str(saved['model_hash']),int(saved['mask']),saved['state'])
        r.restore(snapshot)
    else:
        if 'q17' in initial:d.qpos[r.qpos_ids]=r.B@np.asarray(initial['q17'])
        if 'v17' in initial:d.qvel[r.dof_ids]=r.B@np.asarray(initial['v17'])
        if 'base_pose' in initial:d.qpos[r.base_qpos]=initial['base_pose']
        if 'base_velocity' in initial:d.qvel[r.base_dofs]=initial['base_velocity']
        target=r.model.joint(r.spec.target_free_joint_name);qa=int(target.qposadr[0]);va=int(target.dofadr[0])
        if 'target_pose' in initial:d.qpos[qa:qa+7]=initial['target_pose']
        if 'target_velocity' in initial:d.qvel[va:va+6]=initial['target_velocity']
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise ValueError('nonfinite initial state')
        for adr in (r.base_qpos[3],qa+3):
            if abs(np.linalg.norm(d.qpos[adr:adr+4])-1)>1e-10:raise ValueError('nonunit initial quaternion')
        mujoco.mj_forward(r.model,d);snapshot=r.snapshot(d)
    t=value['task'];task=TaskSpec(float(t['duration']),tuple(PoseConstraint(**p) for p in t.get('poses',[])),
                               tuple(PoseInterval(**p) for p in t.get('intervals',[])),t.get('reference_start'),t.get('terminal_policy','zero_derivatives'),
                               tuple(ForcePulse(**p) for p in t.get('target_force_pulses',[])))
    return r,snapshot,task,value


def planning_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    values=dict(config['budget']);values['sampling']=SamplingBudget(**values.get('sampling',{}))
    return config,PlanningBudget(**values)


def export_continuation(output, execution_directory, task_file, state_index=-1):
    """Export actual integration state, never a projected/planned joint state.

    The next public task supplies only goals/time/scene; its initial is replaced
    explicitly with the recorded state. Current FF+PD is memoryless at control
    update boundaries. Mid-hold handoffs are refused, not silently rescheduled.
    This export does not assert that the reduced planner supports the state.
    """
    import hashlib
    from .execution import refine_physics
    from .orchestration import atomic_json
    root=Path(execution_directory);output=Path(output)
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    acceptance=json.loads((root/'acceptance.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'execution.json').read_text(encoding='utf-8'))
    if summary['controller']!='floating_ff_torque_pd_v1':raise ValueError('CONTINUATION_CONTROLLER_MEMORY_UNSUPPORTED')
    if acceptance['status']!='EXECUTION_VALID':raise ValueError('CONTINUATION_SOURCE_NOT_ACCEPTED')
    planning=Path(provenance['planning_directory'])
    robot,initial,_,_=load_problem(planning/'problem.json')
    native,_=refine_physics(robot,initial,provenance['physics_substeps'])
    if native.model_hash!=provenance['execution_model_hash']:raise ValueError('CONTINUATION_MODEL_MISMATCH')
    target_robot,_,_,problem=load_problem(task_file)
    if target_robot.model_hash!=robot.model_hash:raise ValueError('CONTINUATION_SCENE_OR_MODEL_CHANGED')
    data_path=root/'execution.npz'
    with np.load(data_path,allow_pickle=False) as saved:
        count=len(saved['states']);index=state_index if state_index>=0 else count+state_index
        if index<0 or index>=count:raise ValueError('CONTINUATION_STATE_INDEX_OUT_OF_RANGE')
        substeps=int(saved['control_substeps'])
        if index%substeps:raise ValueError('CONTINUATION_REQUIRES_CONTROL_UPDATE_BOUNDARY')
        state=saved['states'][index].copy();mask=int(saved['mask'])
    snap=Snapshot(native.model_hash,mask,state);actual=native.restore(snap)
    # The only model change is the existing validated integration timestep
    # refinement. Rebinding back to its parent does not alter any state byte.
    rebound=Snapshot(robot.model_hash,mask,state)
    restored=robot.restore(rebound)
    np.testing.assert_array_equal(robot.snapshot(restored).state,state)
    low=actual.qpos[native.qpos_ids];vel=actual.qvel[native.dof_ids]
    q_res=float(np.max(abs(low-native.B@(native.Bplus@low))))
    v_res=float(np.max(abs(vel-native.B@(native.Bplus@vel))))
    problem['initial']={'snapshot_file':'initial_state.npz'}
    # Only the previous reference's local C2 boundary is controller memory.
    # Never expose its future/path controls to the new planner.
    from .trajectory import basis
    from .contracts import Candidate
    planned=json.loads((planning/'planning_result.json').read_text(encoding='utf-8'))
    folder=planning/f"candidate_{planned['selected_index']:03d}"
    with np.load(folder/'candidate.npz',allow_pickle=False) as saved:
        candidate=Candidate(saved['control'],float(saved['duration']),int(saved['seed']))
    if candidate.identity!=provenance['candidate_hash']:raise ValueError('CONTINUATION_REFERENCE_IDENTITY_CHANGED')
    local_time=index*native.model.opt.timestep
    if abs(actual.time-(robot.restore(initial).time+local_time))>1e-8:raise ValueError('CONTINUATION_ABSOLUTE_TIME_MISMATCH')
    if local_time>candidate.duration+1e-10:raise ValueError('CONTINUATION_REFERENCE_EXPIRED')
    local_time=min(local_time,candidate.duration)
    reference=np.stack([basis(len(candidate.control),local_time,candidate.duration,k)@candidate.control for k in (0,1,2)])
    problem['task']['reference_start']=reference.tolist()
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'initial_state.npz',state=state,mask=mask,model_hash=robot.model_hash)
    atomic_json(output/'problem.json',problem)
    result=dict(passed=True,scope='EXACT_STATE_HANDOFF_ONLY_NOT_PLANNING_OR_EXECUTION',
        source_execution=str(root.resolve()),source_execution_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        state_index=index,state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
        planning_model_hash=robot.model_hash,execution_model_hash=native.model_hash,
        state_bytes_unchanged=True,actual_time=float(actual.time),
        controller='floating_ff_torque_pd_v1',controller_memory='STATELESS_AT_CONTROL_UPDATE_BOUNDARY',
        held_ctrl_in_full_state=True,control_phase=0,base_velocity=actual.qvel[native.base_dofs].tolist(),
        reference_boundary='C2_FROM_PREVIOUS_SELECTED_PLAN',reference_candidate_hash=candidate.identity,
        reference_start=reference.tolist(),reference_actual_position_error=float(np.max(abs(reference[0]-native.Bplus@low))),
        reference_actual_velocity_error=float(np.max(abs(reference[1]-native.Bplus@vel))),
        planner_velocity=(native.Bplus@vel).tolist(),position_manifold_residual=q_res,velocity_manifold_residual=v_res,
        reduced_planner_domain='SUPPORTED_MANIFOLD' if max(q_res,v_res)<=1e-10 else 'REQUIRES_FULL_STATE_PREDICTION',
        next_planning_status='NOT_RUN',native_states_modified=False)
    atomic_json(output/'continuation_provenance.json',result)
    return result
