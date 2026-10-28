"""Optional exact-graph TorchScript experiment; disabled in the main planner.

This changes execution of the fixed robot tree graph, not its physical model.
It must pass native values and whole-path gradient regressions before enabling.
"""
import time
import torch
from .physics import tensor


KEYS=('position','rotation','com','Jcom','Jangular','omega','alpha','acom','inertia')


class TreeKernel(torch.nn.Module):
    def __init__(self,tree):
        super().__init__();self.function=tree.kinematics

    def forward(self,qpos,velocity,acceleration):
        values=self.function(qpos,velocity,acceleration)
        return tuple(values[k] for k in KEYS)


def compile_tree(tree,output=None):
    start=time.perf_counter();r=tree.robot;d=r.new_data();x=tensor(d.qpos);v=torch.zeros(r.model.nv,dtype=torch.float64)
    compiled=torch.jit.trace(TreeKernel(tree).eval(),(x,v,v),check_trace=False)
    compiled=torch.jit.freeze(compiled.eval())
    if output is not None:torch.jit.save(compiled,str(output))
    original=tree.kinematics
    def kinematics(qpos,velocity=None,acceleration=None):
        vel=torch.zeros(tree.nv,dtype=qpos.dtype) if velocity is None else velocity
        acc=torch.zeros_like(vel) if acceleration is None else acceleration
        return dict(zip(KEYS,compiled(qpos,vel,acc)))
    tree.kinematics=kinematics
    return dict(seconds=time.perf_counter()-start,model_hash=r.model_hash,original=original,compiled=compiled)


def compile_full_state_step(predictor,dt,output=None):
    """Opt-in exact graph of a full-state discrete step, including its adjoint.

    Fixed model, dt and update/hold branches; no larger integration steps,
    reduced coordinates, changed controller or approximated gradients.
    Calibration must precede use in a main run.
    """
    original=predictor.step;r=predictor.robot;d=r.restore(predictor.initial_state)
    x=tensor(d.qpos);v=tensor(d.qvel);u=tensor(d.ctrl[r.actuator_ids])
    q=tensor(r.Bplus@d.qpos[r.qpos_ids]);qd=tensor(r.Bplus@d.qvel[r.dof_ids]);qdd=torch.zeros(17,dtype=x.dtype)
    class StepKernel(torch.nn.Module):
        def __init__(self,update):super().__init__();self.update=update
        def forward(self,x,v,u,raw,q,qd,qdd,external):return original(x,v,u,raw,q,qd,qdd,dt,self.update,external)
    start=time.perf_counter();kernels={}
    for update in (True,False):
        kernel=torch.jit.trace(StepKernel(update).eval(),(x,v,u,u.clone(),q,qd,qdd,torch.zeros_like(v)),check_trace=False)
        kernels[update]=torch.jit.freeze(kernel.eval())
        if output is not None:
            from pathlib import Path
            torch.jit.save(kernels[update],str(Path(output)/('step_update.pt' if update else 'step_hold.pt')))
    def step(x,v,u,raw,q,qd,qdd,step_dt,update,external=None):
        if step_dt!=dt:raise ValueError('COMPILED_STEP_TIMESTEP_MISMATCH')
        if external is None:external=torch.zeros_like(v)
        # Large traced robot graphs otherwise spend minutes in speculative JIT
        # optimization before their first differentiated call. Interpret this
        # exact traced graph; native and whole-path AD comparisons still apply.
        with torch.jit.optimized_execution(False):
            return kernels[bool(update)](x,v,u,raw,q,qd,qdd,external)
    comparisons=[]
    for update in (True,False):
        values=(x,v+.001,u+.02,u+.03,q+.002,qd+.001,qdd+.01)
        external=torch.zeros_like(v)
        from .forcing import target_indices
        _,adr=target_indices(r);external[adr:adr+3]=tensor([.012,-.023,.034])
        def evaluate(function):
            inputs=[a.detach().clone().requires_grad_() for a in values]
            outputs=function(*inputs,dt,update,external)
            loss=sum(a.square().sum() for a in outputs)
            grads=torch.autograd.grad(loss,inputs,allow_unused=True)
            return tuple(a.detach() for a in outputs),tuple(torch.zeros_like(a) if b is None else b.detach() for a,b in zip(inputs,grads))
        expected,ge=evaluate(original);actual,ga=evaluate(step)
        forward=max(float((a-b).abs().max()) for a,b in zip(expected,actual))
        derivative=max(float((a-b).abs().max())/max(1.,float(a.abs().max()),float(b.abs().max())) for a,b in zip(ge,ga))
        comparisons.append(dict(update=update,forward_max_abs=forward,gradient_max_relative=derivative))
        if output is not None:
            from .orchestration import atomic_json
            atomic_json(Path(output)/'compile_progress.json',dict(seconds=time.perf_counter()-start,local_comparisons=comparisons))
        if forward>1e-9 or derivative>1e-8:raise ValueError('COMPILED_FULL_STEP_LOCAL_CALIBRATION_FAILED')
    predictor.step=step
    return dict(seconds=time.perf_counter()-start,model_hash=r.model_hash,physics_dt=dt,
                mode='EXACT_DISCRETE_GRAPH_COMPILE_NOT_MODEL_REDUCTION',jit_speculative_optimization=False,kernel_count=2,local_comparisons=comparisons)


def compile_multirate(predictor,dt,substeps,refresh,total_steps,output=None):
    """Compile the declared refresh approximation without changing any step."""
    from .multirate import macro_step
    from pathlib import Path
    from .orchestration import atomic_json
    r=predictor.robot;d=r.restore(predictor.initial_state)
    x=tensor(d.qpos);v=tensor(d.qvel);u=tensor(d.ctrl[r.actuator_ids])
    q=tensor(r.Bplus@d.qpos[r.qpos_ids]);qd=tensor(r.Bplus@d.qvel[r.dof_ids])
    lengths=sorted({refresh,total_steps%refresh}-{0});kernels={};comparisons=[];start_time=time.perf_counter()
    class Kernel(torch.nn.Module):
        def __init__(self,phase):super().__init__();self.phase=phase
        def forward(self,x,v,u,raw,q,qd,qdd,forces):
            return macro_step(predictor,x,v,u,raw,q,qd,qdd,forces,dt,substeps,self.phase)
    for length in lengths:
        for phase in range(substeps):
            sample=(x,v,u,u.clone(),q.repeat(length,1),qd.repeat(length,1),torch.zeros(length,17,dtype=x.dtype),torch.zeros(length,r.model.nv,dtype=x.dtype))
            kernel=torch.jit.trace(Kernel(phase).eval(),sample,check_trace=False)
            kernel=torch.jit.freeze(kernel.eval());kernels[length,phase]=kernel
            if output is not None:torch.jit.save(kernel,str(Path(output)/f'macro_{length}_{phase}.pt'))
            values=tuple(value+.001 if i else value for i,value in enumerate(sample))
            def evaluate(compiled):
                inputs=[value.detach().clone().requires_grad_() for value in values]
                with torch.jit.optimized_execution(False):
                    outputs=kernel(*inputs) if compiled else macro_step(predictor,*inputs,dt,substeps,phase)
                loss=sum(value.square().sum() for value in outputs)
                grads=torch.autograd.grad(loss,inputs,allow_unused=True)
                return tuple(value.detach() for value in outputs),tuple(torch.zeros_like(a) if g is None else g.detach() for a,g in zip(inputs,grads))
            expected,ge=evaluate(False);actual,ga=evaluate(True)
            error=max(float((a-b).abs().max()) for a,b in zip(expected,actual))
            derivative=max(float((a-b).abs().max())/max(1.,float(a.abs().max()),float(b.abs().max())) for a,b in zip(ge,ga))
            comparisons.append(dict(length=length,phase=phase,forward_max_abs=error,gradient_max_relative=derivative))
            if error>1e-9 or derivative>1e-8:raise ValueError('COMPILED_MULTIRATE_LOCAL_MISMATCH')
            if output is not None:atomic_json(Path(output)/'compile_progress.json',dict(seconds=time.perf_counter()-start_time,local_comparisons=comparisons))
    def compiled(x,v,u,raw,q,qd,qdd,forces,step_dt,control_substeps,first):
        if step_dt!=dt or control_substeps!=substeps:raise ValueError('COMPILED_MULTIRATE_GRID_MISMATCH')
        with torch.jit.optimized_execution(False):return kernels[len(q),first%substeps](x,v,u,raw,q,qd,qdd,forces)
    predictor.compiled_macro_step=compiled
    return dict(seconds=time.perf_counter()-start_time,model_hash=r.model_hash,physics_dt=dt,mass_refresh_steps=refresh,control_period=dt*substeps,
                mode='EXACT_GRAPH_OF_DECLARED_INERTIA_REFRESH_APPROXIMATION',kernel_count=len(kernels),local_comparisons=comparisons)
