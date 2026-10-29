"""Guided DDIM on every free scalar; no sampling-after repair stage.

Network and physics alternate. Provenance preserves the actual network inputs,
clean estimates and guided estimates for counterfactual/sentinel checks.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch


@dataclass(frozen=True)
class SamplingBudget:
    steps: int = 60
    guide_steps: int = 18
    updates_per_step: int = 1
    line_trials: int = 2
    step_size: float = .01
    max_update_norm: float = .1
    max_control_change: float = .02
    step_rule: str = 'constant'
    acceleration_metric_weight: float = 0.
    task_metric: bool = False
    jacobian_batch_size: int = 1
    max_step_coefficient: float | None = None
    rejection_backoff: bool = False
    guidance_transport: str = 'reconstruct_noise'
    line_rule: str = 'halving'
    local_metric: str = 'none'

    def __post_init__(self):
        if not 1<=self.steps<=1000 or not 0<=self.guide_steps<=self.steps:raise ValueError('invalid DDIM schedule')
        if not 1<=self.updates_per_step<=3 or not 1<=self.line_trials<=2:raise ValueError('invalid guidance budget')
        if self.guide_steps and (self.step_size<=0 or self.max_update_norm<=0 or self.max_control_change<=0):raise ValueError('invalid guidance scale')
        if self.step_rule not in ('constant','polyak') or self.acceleration_metric_weight<0:raise ValueError('invalid guidance rule/metric')
        if not 1<=self.jacobian_batch_size<=12:raise ValueError('invalid Jacobian VJP batch size')
        if self.max_step_coefficient is not None and (not np.isfinite(self.max_step_coefficient) or self.max_step_coefficient<=0):raise ValueError('invalid step coefficient cap')
        if type(self.rejection_backoff) is not bool:raise ValueError('invalid rejection backoff')
        if self.guidance_transport not in ('reconstruct_noise','retain_network_noise'):raise ValueError('invalid guidance transport')
        if self.line_rule not in ('halving','quadratic_rejection'):raise ValueError('invalid bounded line rule')
        if self.local_metric not in ('none','instantaneous_multispace'):raise ValueError('invalid local guidance metric')
        if self.task_metric and self.local_metric!='none':raise ValueError('select one task curvature implementation')


def rejected_trial_scale(before,after,directional_delta,scale,rule):
    """Choose the next already-budgeted trial; never evaluates an energy.

    E(s) is approximated by E(0)-s*g.delta+c*s*s. Only a rejected
    observation is used, and the next trial is at most half as large.
    Native energy acceptance remains authoritative for nonsmooth costs.
    """
    if rule=='halving':return scale*.5
    if rule!='quadratic_rejection':raise ValueError('invalid bounded line rule')
    denominator=2*(after-before+scale*directional_delta)
    if not np.isfinite(denominator) or not np.isfinite(directional_delta) or directional_delta<=0 or denominator<=0:
        return scale*.5
    proposed=directional_delta*scale*scale/denominator
    return float(np.clip(proposed,.05*scale,.5*scale))


def sample(model,condition,free_dim,seed,budget,energy=None,mean=None,scale=None,observer=None,control_jacobian=None,audit_inactive_energy=False,metric=None,metric_callback=None):
    """One candidate, exactly the requested network calls; never adds seeds."""
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn(free_dim,generator=generator,device=device,dtype=torch.float32)
    if observer:observer('initial',x.detach().cpu().numpy())
    alpha=torch.cumprod(1-torch.linspace(.0001,.02,1000,dtype=torch.float64),0)
    schedule=np.rint(np.linspace(999,0,budget.steps)).astype(int)
    if len(set(schedule))!=budget.steps:raise ValueError('schedule contains duplicate network calls')
    mean=torch.zeros(free_dim,dtype=torch.float64) if mean is None else torch.as_tensor(mean,dtype=torch.float64)
    scale=torch.ones(free_dim,dtype=torch.float64) if scale is None else torch.as_tensor(scale,dtype=torch.float64)
    if mean.shape!=(free_dim,) or scale.shape!=(free_dim,) or bool((scale<=0).any()):raise ValueError('invalid reversible scaler')
    if budget.guide_steps and energy is None:raise ValueError('missing physical energy')
    if budget.local_metric!='none' and metric_callback is None:raise ValueError('missing declared local metric callback')
    trace=[];calls=0
    metric_factor=None if metric is None else torch.linalg.cholesky(metric)
    feedback_cap=None
    model.eval()
    for index,k in enumerate(schedule):
        input_x=x.detach().clone()
        if observer:observer('before_step',dict(index=index,noise_level=int(k),input=input_x.cpu().numpy()))
        with torch.no_grad():v=model(x[None,:],torch.tensor([k],device=device),condition)[0]
        calls+=1
        a=float(torch.sqrt(alpha[k]));s=float(torch.sqrt(1-alpha[k]))
        clean=a*x-s*v;original=clean.detach().cpu().double()
        guided=original.clone();updates=[]
        active=index>=budget.steps-budget.guide_steps
        step_terms=None
        if not active and audit_inactive_energy:
            if energy is None:raise ValueError('missing energy for inactive-step audit')
            with torch.no_grad():diagnostic=energy(mean+scale*guided)
            if not all(torch.isfinite(v) for v in diagnostic.values()):raise ValueError('NONFINITE_ENERGY_AUDIT')
            step_terms={name:float(value) for name,value in diagnostic.items()}
        if active:
            step_metric_factor=metric_factor
            metric_diagnostics=None
            local_metric_diagnostics=None
            for update in range(budget.updates_per_step):
                variable=guided.detach().requires_grad_()
                terms=energy(mean+scale*variable)
                if step_terms is None:step_terms={name:float(value.detach()) for name,value in terms.items()}
                if budget.local_metric!='none' and update==0:
                    import time
                    metric_start=time.perf_counter()
                    addition,local_metric_diagnostics=metric_callback()
                    if addition.shape!=(free_dim,free_dim) or not torch.isfinite(addition).all():raise ValueError('INVALID_LOCAL_METRIC')
                    base_metric=torch.eye(free_dim,dtype=variable.dtype) if metric is None else metric
                    step_metric_factor=torch.linalg.cholesky(base_metric+addition)
                    local_metric_diagnostics=dict(local_metric_diagnostics,refresh_step=index,refresh_update=0,seconds=time.perf_counter()-metric_start)
                if budget.task_metric and update==0:
                    # Refresh at every guided noise level, reuse only within its
                    # bounded (at most three) updates. Full physical gradient is
                    # preconditioned; no free variable or constraint is removed.
                    from .guidance_metric import residual_jacobian
                    import time
                    metric_start=time.perf_counter()
                    J=residual_jacobian(getattr(energy,'task_residual',None),variable,batch_size=budget.jacobian_batch_size)
                    base_metric=torch.eye(free_dim,dtype=variable.dtype) if metric is None else metric
                    local_metric=base_metric+J.T@J
                    step_metric_factor=torch.linalg.cholesky(local_metric)
                    singular=torch.linalg.svdvals(J)
                    metric_diagnostics=dict(refresh_step=index,residual_dimension=len(J),full_free_dimension=free_dim,
                                            condition=float(torch.linalg.cond(local_metric)),jacobian_singular_values=singular.tolist(),
                                            seconds=time.perf_counter()-metric_start)
                grad=torch.autograd.grad(terms['total'],variable)[0]
                if not torch.isfinite(grad).all() or not torch.isfinite(terms['total']):raise ValueError('NONFINITE_GUIDANCE')
                direction=grad if step_metric_factor is None else torch.cholesky_solve(grad[:,None],step_metric_factor).squeeze(1)
                directional_descent=float(grad@direction)
                coefficient=budget.step_size
                if budget.step_rule=='polyak':
                    # All five energies are nonnegative. Their common lower bound
                    # zero supplies a conservative energy-scaled step, with the
                    # same two bounded acceptance trials as constant guidance.
                    coefficient*=max(float(terms['total'].detach()),0.)/max(directional_descent,1e-30)
                uncapped_coefficient=coefficient
                if budget.max_step_coefficient is not None:coefficient=min(coefficient,budget.max_step_coefficient)
                if budget.rejection_backoff and feedback_cap is not None:coefficient=min(coefficient,feedback_cap)
                delta=coefficient*direction
                norm_factor=min(1.,budget.max_update_norm/max(float(torch.linalg.vector_norm(delta)),1e-30))
                delta=delta*norm_factor
                effective_coefficient=coefficient*norm_factor
                before=float(terms['total'].detach());accepted=False;after=before;chosen=0.
                physical_change=None
                if control_jacobian is not None:
                    physical_change=float(torch.max(abs(control_jacobian@delta)))
                    factor=min(1.,budget.max_control_change/max(physical_change,1e-30));delta=delta*factor;physical_change*=factor
                    effective_coefficient*=factor
                diagnostics=dict(getattr(energy,'diagnostics',{}))
                step=1.;trial_evaluations=[]
                directional_delta=float(grad@delta)
                for trial in range(budget.line_trials):
                    proposal=(guided-step*delta).detach()
                    with torch.no_grad():after_terms=energy(mean+scale*proposal)
                    trial_value=float(after_terms['total'])
                    trial_evaluations.append(dict(scale=step,energy=trial_value if np.isfinite(trial_value) else None))
                    if np.isfinite(trial_value) and trial_value<=before:
                        guided=proposal;after=trial_value;accepted=True;chosen=step;break
                    if trial+1<budget.line_trials:
                        step=rejected_trial_scale(before,trial_value,directional_delta,step,budget.line_rule)
                # Spend only the next ALREADY budgeted update on a smaller
                # physical step instead of repeating an identical rejection.
                # Carry the scalar within this candidate, never across seeds.
                # Use the effective coefficient after both trust caps: merely
                # halving the uncapped Polyak scale can repeat the same delta.
                previous_feedback_cap=feedback_cap
                if budget.rejection_backoff:
                    if not accepted:feedback_cap=effective_coefficient*step*.5
                    elif feedback_cap is not None:feedback_cap=2.*effective_coefficient*chosen
                updates.append(dict(gradient_norm=float(torch.linalg.vector_norm(grad)),before=before,after=after,accepted=accepted,
                                    preconditioned_gradient_norm=float(torch.linalg.vector_norm(direction)),directional_descent=directional_descent,step_coefficient=coefficient,step_rule=budget.step_rule,
                                    uncapped_step_coefficient=uncapped_coefficient,
                                    effective_step_coefficient=effective_coefficient,
                                    feedback_cap_before=previous_feedback_cap,feedback_cap_after=feedback_cap,
                                    line_rule=budget.line_rule,trial_evaluations=trial_evaluations,
                                    line_scale=chosen,trials=trial+1,terms={n:float(value.detach()) for n,value in terms.items()},
                                    terms_after={n:float(value.detach()) for n,value in (after_terms if accepted else terms).items()},
                                    proposed_maximum_control_change=physical_change,
                                    accepted_maximum_control_change=physical_change*chosen if physical_change is not None else None,diagnostics=diagnostics))
                updates[-1]['task_metric']=metric_diagnostics
                updates[-1]['local_metric']=local_metric_diagnostics
        gc=guided.to(device=device,dtype=x.dtype)
        # Explicit guided DDIM intervention: retain the current network noise
        # estimate so changing clean is not simultaneously cancelled in noise.
        # Historical reconstruction remains an explicit compatibility policy.
        # Both consume fresh x_k and current network output; neither restores
        # a historical path, optimizes after sampling, or adds network calls.
        decomposition_clean=clean if budget.guidance_transport=='retain_network_noise' else gc
        noise=(x-a*decomposition_clean)/s
        next_alpha=float(alpha[schedule[index+1]]) if index+1<len(schedule) else 1.
        x=(next_alpha**.5)*gc+((1-next_alpha)**.5)*noise
        if not torch.isfinite(x).all():raise ValueError('NONFINITE_SAMPLE')
        trace.append(dict(index=index,noise_level=int(k),input=input_x.cpu().numpy(),clean=original.numpy(),guided=guided.detach().numpy(),
                          output=x.detach().cpu().numpy(),active=active,updates=updates,
                          energies_before_guidance=step_terms,energy_status='MEASURED' if step_terms is not None else 'NOT_RUN',
                          guidance_transport=budget.guidance_transport,
                          decomposition_reference='network_clean' if budget.guidance_transport=='retain_network_noise' else 'guided_clean',
                          clean_to_next_state_gain=next_alpha**.5 if budget.guidance_transport=='retain_network_noise' else next_alpha**.5-(1-next_alpha)**.5*a/s,
                          decomposition_error=float(torch.max(abs(input_x-a*decomposition_clean-s*noise)))))
        if observer:observer('completed_step',trace[-1])
    return dict(free=(mean+scale*x.detach().cpu().double()).numpy(),seed=seed,network_calls=calls,trace=trace,
                post_sampling_trajectory_repair_calls=0,initial_noise_hash=hashlib.sha256(trace[0]['input'].tobytes()).hexdigest())
