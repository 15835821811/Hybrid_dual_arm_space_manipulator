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

    def __post_init__(self):
        if not 1<=self.steps<=1000 or not 0<=self.guide_steps<=self.steps:raise ValueError('invalid DDIM schedule')
        if not 1<=self.updates_per_step<=3 or not 1<=self.line_trials<=2:raise ValueError('invalid guidance budget')
        if self.guide_steps and (self.step_size<=0 or self.max_update_norm<=0 or self.max_control_change<=0):raise ValueError('invalid guidance scale')


def sample(model,condition,free_dim,seed,budget,energy=None,mean=None,scale=None,observer=None,control_jacobian=None,audit_inactive_energy=False):
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
    trace=[];calls=0
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
            for update in range(budget.updates_per_step):
                variable=guided.detach().requires_grad_()
                terms=energy(mean+scale*variable)
                if step_terms is None:step_terms={name:float(value.detach()) for name,value in terms.items()}
                grad=torch.autograd.grad(terms['total'],variable)[0]
                if not torch.isfinite(grad).all() or not torch.isfinite(terms['total']):raise ValueError('NONFINITE_GUIDANCE')
                delta=budget.step_size*grad
                delta=delta*min(1.,budget.max_update_norm/max(float(torch.linalg.vector_norm(delta)),1e-30))
                before=float(terms['total'].detach());accepted=False;after=before;chosen=0.
                physical_change=None
                if control_jacobian is not None:
                    physical_change=float(torch.max(abs(control_jacobian@delta)))
                    factor=min(1.,budget.max_control_change/max(physical_change,1e-30));delta=delta*factor;physical_change*=factor
                diagnostics=dict(getattr(energy,'diagnostics',{}))
                for trial in range(budget.line_trials):
                    step=2.**(-trial);proposal=(guided-step*delta).detach()
                    with torch.no_grad():after_terms=energy(mean+scale*proposal)
                    trial_value=float(after_terms['total'])
                    if np.isfinite(trial_value) and trial_value<=before:
                        guided=proposal;after=trial_value;accepted=True;chosen=step;break
                updates.append(dict(gradient_norm=float(torch.linalg.vector_norm(grad)),before=before,after=after,accepted=accepted,
                                    line_scale=chosen,trials=trial+1,terms={n:float(value.detach()) for n,value in terms.items()},
                                    terms_after={n:float(value.detach()) for n,value in (after_terms if accepted else terms).items()},
                                    proposed_maximum_control_change=physical_change,
                                    accepted_maximum_control_change=physical_change*chosen if physical_change is not None else None,diagnostics=diagnostics))
        gc=guided.to(device=device,dtype=x.dtype)
        # Recompute noise from the same actual x_k, never mix old noise with guided clean.
        noise=(x-a*gc)/s
        next_alpha=float(alpha[schedule[index+1]]) if index+1<len(schedule) else 1.
        x=(next_alpha**.5)*gc+((1-next_alpha)**.5)*noise
        if not torch.isfinite(x).all():raise ValueError('NONFINITE_SAMPLE')
        trace.append(dict(index=index,noise_level=int(k),input=input_x.cpu().numpy(),clean=original.numpy(),guided=guided.detach().numpy(),
                          output=x.detach().cpu().numpy(),active=active,updates=updates,
                          energies_before_guidance=step_terms,energy_status='MEASURED' if step_terms is not None else 'NOT_RUN',
                          decomposition_error=float(torch.max(abs(input_x-a*gc-s*noise)))))
        if observer:observer('completed_step',trace[-1])
    return dict(free=(mean+scale*x.detach().cpu().double()).numpy(),seed=seed,network_calls=calls,trace=trace,
                post_sampling_trajectory_repair_calls=0,initial_noise_hash=hashlib.sha256(trace[0]['input'].tobytes()).hexdigest())
