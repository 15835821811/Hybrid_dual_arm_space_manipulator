"""Native full-state force/gradient calibration before planner integration."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import mujoco
from .problem_io import load_problem
from .physics import DifferentiableTree,tensor
from .execution import command
from .orchestration import atomic_json,check_budget


def run(output,config):
    torch.set_num_threads(1)
    r,snapshot,_,_=load_problem(config['problem'])
    original=r.restore(snapshot);tree=DifferentiableTree(r)
    rng=np.random.default_rng(8127);rows=[]
    # Actual handoff; generic off-manifold perturbation; nonzero target twist.
    for kind in ('actual','residual_perturbed','moving_target'):
        d=r.restore(snapshot)
        if kind!='actual':
            d.qpos[r.qpos_ids]+=rng.normal(0,1e-4,67)
            d.qvel[r.dof_ids]+=rng.normal(0,.001,67)
        if kind=='moving_target':
            j=r.model.joint(r.spec.target_free_joint_name);a=int(j.dofadr[0])
            d.qvel[a:a+6]=[.01,-.02,.01,.02,-.03,.04]
        x0=d.qpos.copy();v0=d.qvel.copy()
        qr=r.Bplus@x0[r.qpos_ids];vr=r.Bplus@v0[r.dof_ids]
        ar=rng.normal(0,.01,17)
        z0=np.r_[x0[r.qpos_ids],v0,qr,vr,ar]
        n=r.model.nv
        probe=rng.normal(size=n);probe/=np.linalg.norm(probe)
        def native(z):
            nd=r.restore(snapshot);nd.qpos[:]=x0;nd.qpos[r.qpos_ids]=z[:67];nd.qvel[:]=z[67:67+n]
            q,v,a=z[67+n:].reshape(3,17)
            applied,raw=command(r,nd,q,v,a)
            nd.ctrl[r.actuator_ids]=applied;mujoco.mj_forward(r.model,nd)
            if nd.nefc:raise ValueError('FULL_STATE_CALIBRATION_REQUIRES_CONTACT_FREE_STATE')
            return nd.qacc.copy(),applied,raw
        def differentiable(z):
            x=tensor(x0).index_copy(0,torch.as_tensor(r.qpos_ids,dtype=torch.long),z[:67]);v=z[67:67+n]
            q,qd,qdd=z[67+n:].reshape(3,17)
            return tree.tracking_acceleration(x,v,q,qd,qdd)
        z=tensor(z0).requires_grad_(True);pred,applied,raw=differentiable(z);actual,na,nr=native(z0)
        errors=dict(acceleration=float(np.max(abs(pred.detach().numpy()-actual))),
                    applied_torque=float(np.max(abs(applied.detach().numpy()-na))),raw_torque=float(np.max(abs(raw.detach().numpy()-nr))))
        grad=torch.autograd.grad(pred@tensor(probe),z)[0].detach().numpy()
        directions=[]
        for index in range(3):
            direction=rng.normal(size=len(z0));direction/=np.linalg.norm(direction)
            ad=float(grad@direction)
            for eps in (1e-5,1e-6,1e-7):
                check_budget()
                fd=float(probe@(native(z0+eps*direction)[0]-native(z0-eps*direction)[0])/(2*eps))
                relative=abs(ad-fd)/max(1.,abs(ad),abs(fd))
                directions.append(dict(direction=index,epsilon=eps,ad=ad,native_fd=fd,relative_error=relative,passed=relative<2e-4))
        row=dict(case=kind,errors=errors,directions=directions,
                 passed=errors['acceleration']<1e-7 and errors['raw_torque']<1e-9 and errors['applied_torque']<1e-9 and all(x['passed'] for x in directions))
        rows.append(row);atomic_json(output/'progress.json',dict(cases=rows))
    path_report=None
    if config.get('path_check',False):
        path_report=check_path(r,snapshot,tree,output,config.get('path_duration',.02))
    np.testing.assert_array_equal(r.snapshot(original).state,snapshot.state)
    result=dict(passed=all(x['passed'] for x in rows) and (path_report is None or path_report['passed']),cases=rows,path_report=path_report,initial_state_unchanged=True,
                initial_state_sha256=hashlib.sha256(snapshot.state.tobytes()).hexdigest(),
                scope='Instantaneous full-state contact-free FF+PD dynamics and native directional derivatives; whole-path integration/guide NOT_RUN',
                controller='floating_ff_torque_pd_v1',projection_calls=0)
    atomic_json(output/'full_state_calibration.json',result)
    return result


def check_path(r,snapshot,tree,output,path_duration=.02):
    from .api import boundary_codec
    from .propagation import FullStatePredictor
    from .execution import refine_physics
    codec=boundary_codec(r,snapshot,.8,origin='initial_position')
    z=np.zeros(codec.free_dim);predictor=FullStatePredictor(tree,snapshot)
    native,initial=refine_physics(r,snapshot,4)
    def native_path(value,duration):
        d=native.restore(initial);n=int(round(duration/.0005));times=np.arange(n+1)*.0005
        q=codec.evaluate(value,times);v=codec.evaluate(value,times,1);a=codec.evaluate(value,times,2)
        xs=[d.qpos.copy()];vs=[d.qvel.copy()]
        for i in range(n):
            if i%4==0:applied,_=command(native,d,q[i],v[i],a[i])
            d.ctrl[native.actuator_ids]=applied;mujoco.mj_step(native.model,d)
            if d.nefc:raise ValueError('PATH_LEFT_CONTACT_FREE_DOMAIN')
            xs.append(d.qpos.copy());vs.append(d.qvel.copy())
        return np.array(xs),np.array(vs)
    with torch.no_grad():prediction=predictor.rollout(codec,tensor(z),duration=path_duration)
    nx,nv=native_path(z,path_duration)
    pe=float(np.max(abs(prediction['qpos'].numpy()-nx)));ve=float(np.max(abs(prediction['velocity'].numpy()-nv)))
    np.testing.assert_array_equal(prediction['qpos'][0].numpy(),r.restore(snapshot).qpos)
    np.testing.assert_array_equal(prediction['velocity'][0].numpy(),r.restore(snapshot).qvel)
    np.savez_compressed(output/'path_comparison.npz',predicted_qpos=prediction['qpos'].numpy(),predicted_velocity=prediction['velocity'].numpy(),native_qpos=nx,native_velocity=nv)
    rng=np.random.default_rng(6137);probe=rng.normal(size=r.model.nv);probe/=np.linalg.norm(probe)
    variable=tensor(z).requires_grad_(True)
    differentiable=predictor.rollout(codec,variable,duration=.004)
    g=torch.autograd.grad(differentiable['velocity'][-1]@tensor(probe),variable)[0].detach().numpy()
    checks=[]
    for _ in range(2):
        direction=rng.normal(size=codec.free_dim);direction/=np.linalg.norm(direction);eps=1e-7
        fd=float(probe@(native_path(z+eps*direction,.004)[1][-1]-native_path(z-eps*direction,.004)[1][-1])/(2*eps))
        ad=float(g@direction);relative=abs(ad-fd)/max(1.,abs(ad),abs(fd))
        checks.append(dict(ad=ad,native_fd=fd,relative_error=relative,passed=relative<.01))
    return dict(passed=pe<1e-5 and ve<.001 and all(c['passed'] for c in checks),
                max_qpos_error=pe,max_velocity_error=ve,gradient_checks=checks,
                state_preserved_at_origin=True,projection_calls=0,duration=path_duration,gradient_duration=.004,
                scope='Short contact-free integration calibration only; full planning horizons and guidance NOT_RUN')
