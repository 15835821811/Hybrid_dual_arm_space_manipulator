"""C/D comparison entry: explicit post-optimization, never a main fallback.

Uses the same public problem, full variable codec, physics and read-only
validator. Main A and no-guide B remain separate `plan` invocations. Running
this module establishes no learning-benefit claim.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from ..api import boundary_codec
from ..contracts import Candidate
from ..diffusion import sample, SamplingBudget
from ..model_io import load_model, normalize_condition
from ..conditioning import encode_condition
from ..problem_io import load_problem, planning_config
from ..physics import DifferentiableTree, tensor
from ..propagation import Propagator
from ..geometry import Geometry
from ..costs import MultiSpaceEnergy
from ..guidance_metric import acceleration_metric, residual_jacobian
from ..validation import validate
from ..orchestration import atomic_json, check_budget


class PhysicsBudgetExhausted(RuntimeError):
    pass


def optimize(initial, objective, metric, control_jacobian, sampling, limits, progress=None):
    """Bounded optimizer used ONLY by comparison C/D, explicitly after sampling."""
    x=tensor(initial);trace=[];factor=None;calls=0;jacobians=0;gradients=0
    def evaluate(value):
        nonlocal calls
        check_budget()
        if calls>=limits['energy_evaluations']:raise PhysicsBudgetExhausted('BASELINE_ENERGY_BUDGET_EXHAUSTED')
        calls+=1
        return objective(value)
    try:
        for index in range(limits['gradient_evaluations']):
            variable=x.detach().requires_grad_();terms=evaluate(variable)
            if index % sampling.updates_per_step==0:
                if jacobians>=limits['jacobian_refreshes']:raise PhysicsBudgetExhausted('BASELINE_JACOBIAN_BUDGET_EXHAUSTED')
                J=residual_jacobian(objective.task_residual,variable,batch_size=sampling.jacobian_batch_size)
                factor=torch.linalg.cholesky(metric+J.T@J);jacobians+=1
            gradient=torch.autograd.grad(terms['total'],variable)[0];gradients+=1
            if not torch.isfinite(gradient).all() or not torch.isfinite(terms['total']):raise ValueError('NONFINITE_BASELINE_GRADIENT')
            direction=torch.cholesky_solve(gradient[:,None],factor).squeeze(1)
            coefficient=sampling.step_size
            if sampling.step_rule=='polyak':coefficient*=max(float(terms['total'].detach()),0.)/max(float(gradient@direction),1e-30)
            if sampling.max_step_coefficient is not None:coefficient=min(coefficient,sampling.max_step_coefficient)
            delta=coefficient*direction
            delta*=min(1.,sampling.max_update_norm/max(float(torch.linalg.vector_norm(delta)),1e-30))
            delta*=min(1.,sampling.max_control_change/max(float(torch.max(abs(control_jacobian@delta))),1e-30))
            accepted=False;before=float(terms['total'].detach());after=before
            for trial in range(sampling.line_trials):
                proposal=(x-(2.**(-trial))*delta).detach()
                with torch.no_grad():after_terms=evaluate(proposal)
                value=float(after_terms['total'])
                if np.isfinite(value) and value<=before:x=proposal;after=value;accepted=True;break
            trace.append(dict(index=index,accepted=accepted,before=before,after=after,trials=trial+1,
                              gradient_norm=float(torch.linalg.vector_norm(gradient)),step_coefficient=coefficient,
                              terms={k:float(v.detach()) for k,v in terms.items()}))
            if progress:progress(x.numpy(),trace)
    except PhysicsBudgetExhausted as error:
        return x.numpy(),dict(status='INCOMPLETE_PHYSICS_BUDGET',error=str(error),trace=trace,energy_evaluations=calls,gradient_evaluations=gradients,jacobian_refreshes=jacobians)
    return x.numpy(),dict(status='FINISHED',trace=trace,energy_evaluations=calls,gradient_evaluations=gradients,jacobian_refreshes=jacobians)


def nonlearned_initial(codec,q0,joint_ranges,seed,index):
    """Several public smooth initializations; no targets solved or paths read."""
    rng=np.random.default_rng(seed);u=np.linspace(0,1,codec.n)
    amplitude=(0.,.025,.05,.1,.2)[index%5]
    controls=np.tile(q0,(codec.n,1))
    for frequency in range(1,5):
        controls+=amplitude*np.asarray(joint_ranges)*rng.normal(size=codec.dof)*np.sin(np.pi*frequency*u)[:,None]/frequency**2
    # Only the common analytic boundary is projected; all task DOFs stay free.
    return (controls.ravel()-codec.offset)@codec.N


def run(output,config):
    torch.set_num_threads(1);output=Path(output);start=time.perf_counter()
    method=config['method']
    if method not in ('C_NONLEARNED','D_POST_OPTIMIZATION'):raise ValueError('comparison entry supports only explicit C/D baselines')
    planning,budget=planning_config(config['planning_config'])
    if not budget.sampling.task_metric:raise ValueError('comparison requires the same task-metric contract')
    limits=config['physical_budget']
    expected=dict(energy_evaluations=budget.sampling.steps-budget.sampling.guide_steps+budget.sampling.guide_steps*budget.sampling.updates_per_step*(1+budget.sampling.line_trials),
                  gradient_evaluations=budget.sampling.guide_steps*budget.sampling.updates_per_step,jacobian_refreshes=budget.sampling.guide_steps)
    if limits!=expected:raise ValueError('comparison physical caps must match the declared A worst-case budget')
    r,snapshot,task,problem=load_problem(config['task']);profile=json.loads(Path(planning['profile']).read_text())
    codec=boundary_codec(r,snapshot,task.duration,budget.control_points)
    model,checkpoint,identity=load_model(planning['checkpoint'],planning.get('device','cpu'))
    if checkpoint['robot_physical_hash']!=r.physical_hash or len(checkpoint['mean'])!=codec.free_dim:raise ValueError('BASELINE_MODEL_CONTRACT_MISMATCH')
    tree=DifferentiableTree(r)
    if planning.get('physics_backend','eager')=='eager_mass_rows':tree.use_momentum_mass_rows=True
    elif planning.get('physics_backend')=='batched_bodyframe':
        tree.use_momentum_mass_rows=True;tree.use_batched_bodyframe_momentum=True
    elif planning.get('physics_backend','eager')!='eager':raise ValueError('uncalibrated comparison backend')
    prop=Propagator(tree,snapshot);geometry=Geometry(r)
    energy=MultiSpaceEnergy(prop,codec,geometry,task,budget.intervals,collect_task_residual=True,collision_aggregation=planning.get('collision_aggregation','pair_mean'))
    energy.weights.update(planning.get('guide_weights',{}))
    mean=tensor(checkpoint['mean']);scale=tensor(checkpoint['scale'])
    metric=acceleration_metric(codec,checkpoint['scale'],r.spec.planner_acceleration_limits,energy.grid,budget.sampling.acceleration_metric_weight)
    jac=tensor(codec.N*checkpoint['scale'][None,:]);tokens=encode_condition(r,snapshot,task,prop.H.numpy())
    condition=normalize_condition(tokens[None],np.zeros((1,len(tokens)),bool),checkpoint,planning.get('device','cpu'))
    from dataclasses import asdict
    atomic_json(output/'problem.json',problem)
    np.savez_compressed(output/'initial_state.npz',state=snapshot.state,mask=snapshot.mask,model_hash=snapshot.model_hash)
    atomic_json(output/'planning_manifest.json',dict(method=method,comparison_only=True,budget=asdict(budget),physical_budget=limits,
        checkpoint=identity,shared_train_scaler=True,learned_network_used=method=='D_POST_OPTIMIZATION',profile=profile,
        guide_weights=energy.weights,collision_aggregation=energy.collision_aggregation,physics_backend=planning.get('physics_backend','eager'),
        robot_model_hash=r.model_hash,robot_physical_hash=r.physical_hash,scene_hash=r.scene.identity,
        timing_includes_setup_and_network=True,learning_benefit_status='NOT_RUN'))
    records=[];valid=[]
    for index,seed in enumerate(budget.seeds):
        folder=output/f'candidate_{index:03d}';folder.mkdir();candidate_start=time.perf_counter()
        row=dict(index=index,seed=seed,status='NOT_RUN',sampled=False,validated=False,observed_network_calls=0)
        atomic_json(folder/'candidate.json',row)
        if time.perf_counter()-start>=budget.max_wall_seconds:
            row['status']='NOT_RUN_BUDGET_EXHAUSTED';records.append(row);atomic_json(folder/'candidate.json',row);continue
        counts=dict(attempted_energy_evaluations=0,completed_energy_evaluations=0,mass_evaluations=0,support_pair_evaluations=0,broadphase_pair_evaluations=0)
        def objective(x):
            if time.perf_counter()-start>budget.max_wall_seconds:raise RuntimeError('BASELINE_WALL_BUDGET_EXHAUSTED')
            counts['attempted_energy_evaluations']+=1
            terms=energy.evaluate(mean+scale*x);objective.task_residual=energy.last_task_residual
            counts['completed_energy_evaluations']+=1
            for key in ('mass_evaluations','support_pair_evaluations','broadphase_pair_evaluations'):counts[key]+=energy.last_diagnostics[key]
            return terms
        def progress(x,trace):
            np.savez_compressed(folder/'optimization_partial.npz',normalized_free=x)
            atomic_json(folder/'optimization.json',trace)
        try:
            if method=='D_POST_OPTIMIZATION':
                actual=[];hook=model.register_forward_pre_hook(lambda model,args:actual.append(args[0].detach().cpu().numpy().copy()))
                try:
                    naked=sample(model,condition,codec.free_dim,seed,SamplingBudget(steps=budget.sampling.steps,guide_steps=0),mean=checkpoint['mean'],scale=checkpoint['scale'])
                finally:
                    hook.remove();row['observed_network_calls']=len(actual)
                    if actual:np.save(folder/'observed_network_inputs.npy',np.stack(actual))
                initial=(naked['free']-checkpoint['mean'])/checkpoint['scale']
                np.savez_compressed(folder/'unguided.npz',**{k:np.stack([s[k] for s in naked['trace']]) for k in ('input','clean','output')})
            else:
                d=r.restore(snapshot);q0=r.Bplus@d.qpos[r.qpos_ids]
                z=nonlearned_initial(codec,q0,r.upper-r.lower,seed,index)
                initial=(z-checkpoint['mean'])/checkpoint['scale']
            np.save(folder/'initial_normalized_free.npy',initial)
            row['initial_hash']=hashlib.sha256(initial.tobytes()).hexdigest()
            final,optimization=optimize(initial,objective,metric,jac,budget.sampling,limits,progress)
            atomic_json(folder/'optimization_result.json',optimization)
            row['optimization']= {k:v for k,v in optimization.items() if k!='trace'}
            if optimization['status']!='FINISHED':row['status']=optimization['status']
            else:
                z=checkpoint['mean']+checkpoint['scale']*final;candidate=Candidate(codec.decode(z),task.duration,seed)
                np.savez_compressed(folder/'candidate.npz',control=candidate.control,free=z,duration=task.duration,seed=seed)
                row.update(sampled=True,identity=candidate.identity)
                report=validate(candidate,snapshot,task,prop,codec,geometry,profile);atomic_json(folder/'validation.json',report)
                row.update(sampled=True,validated=True,status=report['status'],validation=report,identity=candidate.identity)
                if time.perf_counter()-start>budget.max_wall_seconds:raise RuntimeError('BASELINE_WALL_BUDGET_EXHAUSTED_DURING_VALIDATION')
                if report['status']=='PLAN_VALID':
                    score=max(max(e['position_m']/.005,e['rotation_rad']/(np.pi/180)) for e in report['task_errors'])+report['max_nominal_torque_ratio']
                    row['selection_score']=score;valid.append((score,index))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            import traceback
            (folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            row.update(status='INCOMPLETE_BUDGET' if 'BUDGET' in str(error) else 'FAILED_NUMERICAL',error=str(error))
        row.update(query_counters=counts,candidate_wall_seconds=time.perf_counter()-candidate_start,cumulative_wall_seconds=time.perf_counter()-start)
        records.append(row);atomic_json(folder/'candidate.json',row)
        atomic_json(output/'progress.json',dict(completed=len(records),requested=len(budget.seeds),valid=len(valid)))
    valid.sort();result=dict(status='PLAN_VALID' if valid else 'NO_VALID_CANDIDATE',method=method,comparison_only=True,requested_candidates=len(budget.seeds),
        records=records,sampled=sum(row['sampled'] for row in records),valid=len(valid),selected_index=valid[0][1] if valid else None,
        wall_seconds=time.perf_counter()-start,learning_benefit_status='NOT_RUN')
    atomic_json(output/'planning_result.json',result)
    return dict(passed=bool(valid),**result)
