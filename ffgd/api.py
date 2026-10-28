"""Unique main planner: fresh noise -> guided diffusion -> read-only selection.

No expert path, hidden solution, optimizer/search fallback or post-hoc repair is
an accepted input. Model data/labels are only consumed by the training module.
"""
from dataclasses import dataclass,asdict
import hashlib
import json
import time
import os
from pathlib import Path
import numpy as np
import torch
from .contracts import Candidate
from .trajectory import Boundary,BoundaryCodec,basis
from .physics import DifferentiableTree
from .propagation import Propagator
from .geometry import Geometry
from .costs import MultiSpaceEnergy
from .diffusion import sample,SamplingBudget
from .conditioning import encode_condition,CONDITION_SCHEMA
from .model_io import load_model,normalize_condition
from .validation import validate,input_identity,selection_score
from .orchestration import atomic_json,check_budget
from .architecture_guard import ArchitectureViolation


@dataclass(frozen=True)
class PlanningBudget:
    seeds: tuple = (13001,13002,13003,13004,13005)
    sampling: SamplingBudget = SamplingBudget()
    control_points: int = 24
    intervals: int = 32
    max_wall_seconds: float = 1800.
    cpu_threads: int = 1
    boundary_origin: str = 'pivot_zero'

    def __post_init__(self):
        object.__setattr__(self,'seeds',tuple(self.seeds))
        if not self.seeds or len(set(self.seeds))!=len(self.seeds):raise ValueError('candidate seeds must be distinct')
        if self.control_points<6 or self.intervals<1 or self.max_wall_seconds<=0:raise ValueError('invalid planning budget')
        if type(self.cpu_threads) is not int or not 1<=self.cpu_threads<=min(8,max(1,(os.cpu_count() or 1)-1)):
            raise ValueError('invalid bounded CPU thread budget')
        if self.boundary_origin not in ('pivot_zero','initial_position'):raise ValueError('unknown boundary coordinate origin')
        if 0<self.sampling.guide_steps<3:raise ValueError('main guidance must include at least two nonfinal steps')


def boundary_codec(robot,snapshot,duration,n=24,origin='pivot_zero'):
    d=robot.restore(snapshot);q=robot.Bplus@d.qpos[robot.qpos_ids];v=robot.Bplus@d.qvel[robot.dof_ids]
    # Declared baseline template: fixed initial q/v/a; terminal v/a zero,
    # terminal joint position unconstrained. Dynamic a0 policy added with M6.
    bc=[Boundary(0,k,tuple(range(17)),tuple(q if k==0 else v if k==1 else np.zeros(17))) for k in (0,1,2)]
    bc += [Boundary(duration,k,tuple(range(17)),(0.,)*17) for k in (1,2)]
    return BoundaryCodec(n,17,duration,bc,origin=origin)


def plan(robot,initial,task,checkpoint_path,budget,profile,output,device='cpu',disabled_guide_terms=(),guide_weights=None,physics_backend='eager',collision_aggregation='pair_mean',joint_peak_penalty=False,task_objective='center_squared',planning_margins=None):
    """All candidates remain in the denominator, including numerical failures."""
    total_start=time.perf_counter();torch.set_num_threads(budget.cpu_threads);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model,checkpoint,checkpoint_identity=load_model(checkpoint_path,device)
    if checkpoint['dataset_manifest'].get('boundary_origin','pivot_zero')!=budget.boundary_origin:raise ValueError('BOUNDARY_ORIGIN_MISMATCH')
    if checkpoint.get('update',0)<1 or checkpoint.get('training_config',{}).get('scope')=='SOFTWARE_TEST':raise ValueError('MAIN_PLANNER_REQUIRES_TRAINED_PRIOR')
    if checkpoint['dataset_manifest']['source_kind']!='FF_MODEL_PRIOR':raise ValueError('MAIN_PLANNER_DATA_CLASS_MISMATCH')
    if checkpoint['robot_physical_hash']!=robot.physical_hash:raise ValueError('PLANNING_ROBOT_CONTRACT_MISMATCH')
    codec=boundary_codec(robot,initial,task.duration,budget.control_points,origin=budget.boundary_origin)
    from .model_io import bind_free_coordinates
    checkpoint=bind_free_coordinates(checkpoint,task.duration)
    if len(checkpoint['mean'])!=codec.free_dim:raise ValueError('CHECKPOINT_BOUNDARY_TEMPLATE_MISMATCH')
    tree=DifferentiableTree(robot)
    compile_seconds=0.
    if physics_backend=='torchscript':
        from .compiled_physics import compile_tree
        compile_seconds=compile_tree(tree,output/'compiled_tree.pt')['seconds']
    elif physics_backend=='eager_mass_rows':tree.use_momentum_mass_rows=True
    elif physics_backend in ('batched_bodyframe','batched_full_path'):
        tree.use_momentum_mass_rows=True;tree.use_batched_bodyframe_momentum=True
        tree.use_batched_path_physics=physics_backend=='batched_full_path'
    elif physics_backend=='full_state':tree.use_batched_path_physics=True
    elif physics_backend!='eager':raise ValueError('unknown physics backend')
    if physics_backend=='full_state':
        from .propagation import FullStatePropagation
        prop=FullStatePropagation(tree,initial)
    else:prop=Propagator(tree,initial)
    geometry=Geometry(robot)
    if budget.sampling.task_metric and physics_backend not in ('eager','eager_mass_rows','batched_bodyframe','batched_full_path'):raise ValueError('TASK_JACOBIAN_REQUIRES_CALIBRATED_BACKEND')
    energy=MultiSpaceEnergy(prop,codec,geometry,task,intervals=budget.intervals,collect_task_residual=budget.sampling.task_metric,collision_aggregation=collision_aggregation,joint_peak_penalty=joint_peak_penalty,task_objective=task_objective,planning_margins=planning_margins)
    if guide_weights is not None:
        if set(guide_weights)!=set(energy.weights) or any(not np.isfinite(v) or v<=0 for v in guide_weights.values()):raise ValueError('provide positive weights for all five physical energy terms')
        energy.weights.update(guide_weights)
    if any(name not in energy.weights for name in disabled_guide_terms):raise ValueError('unknown ablation term')
    for name in disabled_guide_terms:energy.weights[name]=0.
    tokens=encode_condition(robot,initial,task,prop.H.numpy())
    condition=normalize_condition(tokens[None],np.zeros((1,len(tokens)),bool),checkpoint,device)
    from .guidance_metric import acceleration_metric
    metric=acceleration_metric(codec,checkpoint['scale'],robot.spec.planner_acceleration_limits,energy.grid,budget.sampling.acceleration_metric_weight)
    start=total_start;records=[];valid=[]
    manifest=dict(budget=asdict(budget),checkpoint=checkpoint_identity,robot_model_hash=robot.model_hash,
                  free_coordinate_schema=checkpoint['dataset_manifest'].get('free_coordinate_schema','physical_position_v1'),
                  candidate_scope='FIXED_SEED_DIAGNOSTIC' if len(budget.seeds)<5 else 'MULTI_NOISE_TASK_EVALUATION',
                  minimum_five_candidate_protocol_met=len(budget.seeds)>=5,
                  robot_physical_hash=robot.physical_hash,scene_hash=robot.scene.identity,geometry_policy_hash=geometry.policy_hash,
                  disabled_guide_terms=list(disabled_guide_terms),independent_validation_terms_disabled=[],cold_start=True,
                  guide_weights=energy.weights,
                  collision_aggregation=collision_aggregation,
                  joint_peak_penalty=joint_peak_penalty,
                  task_objective=task_objective,planning_margins=dict(energy.planning_margins),
                  network_condition_schema=CONDITION_SCHEMA,external_scene_in_physics_and_validation=True,
                  physics_backend=physics_backend,compile_seconds=compile_seconds,setup_wall_seconds=time.perf_counter()-total_start,
                  guidance_metric_condition=float(torch.linalg.cond(metric)),
                  method=('GUIDED_DIFFUSION_ABLATION' if disabled_guide_terms else 'GUIDED_DIFFUSION') if budget.sampling.guide_steps else 'LEARNED_PRIOR_WITHOUT_GUIDANCE')
    manifest['profile']=profile
    manifest['profile_hash']=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
    manifest['initial_state_hash']=hashlib.sha256(initial.state.tobytes()).hexdigest()
    atomic_json(output/'planning_manifest.json',manifest)
    np.savez_compressed(output/'initial_state.npz',state=initial.state,mask=initial.mask,model_hash=initial.model_hash)
    for index,seed in enumerate(budget.seeds):
        candidate_start=time.perf_counter()
        folder=output/f'candidate_{index:03d}';folder.mkdir(exist_ok=False)
        record=dict(index=index,seed=seed,status='STARTED',sampled=False,validated=False)
        atomic_json(folder/'candidate.json',record)
        if time.perf_counter()-start>=budget.max_wall_seconds:
            record.update(status='NOT_RUN_BUDGET_EXHAUSTED');records.append(record);atomic_json(folder/'candidate.json',record);continue
        try:
            energy_calls=0
            query_counters=dict(completed_energy_evaluations=0,mass_evaluations=0,support_pair_evaluations=0,broadphase_pair_evaluations=0,
                                full_energy_gradients=0,task_jacobian_rows=0,autograd_engine_calls=0)
            def observe(kind,value):
                if kind=='initial':
                    np.save(folder/'initial_noise.npy',value)
                    record['initial_noise_hash']=hashlib.sha256(value.tobytes()).hexdigest()
                elif kind=='before_step':
                    np.savez_compressed(folder/'current_step.npz',input=value['input'],index=value['index'],noise_level=value['noise_level'])
                else:
                    updates=value['updates'];query_counters['full_energy_gradients']+=len(updates);query_counters['autograd_engine_calls']+=len(updates)
                    if updates and updates[0].get('task_metric'):
                        rows=updates[0]['task_metric']['residual_dimension'];query_counters['task_jacobian_rows']+=rows
                        query_counters['autograd_engine_calls']+=int(np.ceil(rows/budget.sampling.jacobian_batch_size))
                    arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
                    np.savez_compressed(folder/f'step_{value["index"]:03d}.npz',**arrays)
                    atomic_json(folder/f'step_{value["index"]:03d}.json',{k:v for k,v in value.items() if k not in arrays})
            def checked_energy(z):
                nonlocal energy_calls
                energy_calls+=1
                check_budget()
                if time.perf_counter()-start>budget.max_wall_seconds:raise RuntimeError('PLANNING_WALL_BUDGET_EXHAUSTED')
                terms=energy.evaluate(z);checked_energy.diagnostics=energy.last_diagnostics
                checked_energy.task_residual=energy.last_task_residual
                query_counters['completed_energy_evaluations']+=1
                for key in ('mass_evaluations','support_pair_evaluations','broadphase_pair_evaluations'):query_counters[key]+=energy.last_diagnostics[key]
                return terms
            actual_inputs=[]
            hook=model.register_forward_pre_hook(lambda model,args:actual_inputs.append(args[0].detach().cpu().numpy().copy()))
            try:
                result=sample(model,condition,codec.free_dim,seed,budget.sampling,checked_energy,checkpoint['mean'],checkpoint['scale'],observer=observe,
                              control_jacobian=torch.tensor(codec.N*checkpoint['scale'][None,:],dtype=torch.float64),audit_inactive_energy=True,metric=metric)
            finally:
                hook.remove();record['observed_network_calls']=len(actual_inputs)
                record['physical_energy_evaluations']=energy_calls
                record['query_counters']=query_counters
                if actual_inputs:np.save(folder/'observed_network_inputs.npy',np.stack(actual_inputs))
            if len(actual_inputs)!=budget.sampling.steps:raise RuntimeError('ACTUAL_NETWORK_CALL_COUNT_MISMATCH')
            for i in range(1,len(actual_inputs)):
                if not np.array_equal(actual_inputs[i][0],result['trace'][i-1]['output']):raise RuntimeError('NETWORK_DID_NOT_CONSUME_GUIDED_CHAIN')
            candidate=Candidate(codec.decode(result['free']),task.duration,seed)
            trace=result.pop('trace');trace_arrays={name:np.stack([row[name] for row in trace]) for name in ('input','clean','guided','output')}
            np.savez_compressed(folder/'denoising.npz',**trace_arrays)
            atomic_json(folder/'denoising.json',[{k:v for k,v in row.items() if k not in trace_arrays} for row in trace])
            np.savez_compressed(folder/'candidate.npz',control=candidate.control,free=result['free'],duration=candidate.duration,seed=seed,
                                initial_state=initial.state,mask=initial.mask,model_hash=initial.model_hash)
            record.update(sampled=True,network_calls=result['network_calls'],initial_noise_hash=result['initial_noise_hash'],
                          post_sampling_trajectory_repair_calls=result['post_sampling_trajectory_repair_calls'],identity=candidate.identity)
            report=validate(candidate,initial,task,prop,codec,geometry,profile)
            atomic_json(folder/'validation.json',report);record.update(validated=True,status=report['status'],validation=report)
            # Validation belongs to total planning time. Preserve its physical
            # verdict, but never select a result obtained after the wall cap.
            if time.perf_counter()-start>budget.max_wall_seconds:raise RuntimeError('PLANNING_WALL_BUDGET_EXHAUSTED_DURING_VALIDATION')
            if report['status']=='PLAN_VALID':
                score=selection_score(report,task)
                record['selection_score']=score;valid.append((score,index,candidate))
                # Save low-rate model prediction only; native execution is separate.
                grid=np.unique(np.r_[np.linspace(0,task.duration,65),[p.time for p in task.poses]])
                xs,vs=prop.native_rollout(codec,result['free'],64,grid)
                np.savez_compressed(folder/'prediction.npz',time=grid,qpos=xs,qvel=vs,q=codec.evaluate(result['free'],grid),
                                    qd=codec.evaluate(result['free'],grid,1),qdd=codec.evaluate(result['free'],grid,2))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            if isinstance(error,ArchitectureViolation):raise
            import traceback
            (folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            record.update(status='FAILED_NUMERICAL' if 'BUDGET' not in str(error) else 'INCOMPLETE_BUDGET',error=str(error))
        record['candidate_wall_seconds']=time.perf_counter()-candidate_start
        record['elapsed_seconds']=time.perf_counter()-start
        record['cumulative_wall_seconds']=record['elapsed_seconds']
        records.append(record);atomic_json(folder/'candidate.json',record)
        atomic_json(output/'progress.json',dict(completed=len(records),requested=len(budget.seeds),valid=len(valid)))
        print(json.dumps(dict(candidate=index,seed=seed,status=record['status'],elapsed_seconds=record['elapsed_seconds'])),flush=True)
    valid.sort(key=lambda value:(value[0],value[1]))
    selected=valid[0][2] if valid else None
    result=dict(status='PLAN_VALID' if selected is not None else 'NO_VALID_CANDIDATE',requested_candidates=len(budget.seeds),
                sampled=sum(r['sampled'] for r in records),valid=len(valid),records=records,selected_index=valid[0][1] if valid else None,
                selection='lowest fixed normalized task error plus nominal torque ratio among read-only valid candidates',
                wall_seconds=time.perf_counter()-start,repair_calls=0)
    atomic_json(output/'planning_result.json',result)
    return selected,result
