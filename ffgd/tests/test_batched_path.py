"""Full body/actuator batching agrees with native dynamics and original VJP."""
import numpy as np
import torch
import mujoco
from ffgd.robot import Robot
from ffgd.physics import DifferentiableTree,tensor


def test_batched_path_all_bodies_actuators_and_derivatives():
    torch.set_num_threads(1);r=Robot();tree=DifferentiableTree(r);rng=np.random.default_rng(7234)
    d=r.new_data();xs=[];vs=[];acc=[];native=[]
    for _ in range(3):
        d.qpos[r.qpos_ids]=r.B@(r.spec.planner_zero+rng.normal(0,.05,17))
        d.qvel[:]=rng.normal(0,.02,r.model.nv);qdd=rng.normal(0,.2,17)
        d.qpos[r.base_qpos[:3]]=rng.normal(0,.2,3)
        quat=rng.normal(size=4);d.qpos[r.base_qpos[3:]]=quat/np.linalg.norm(quat)
        mujoco.mj_forward(r.model,d);M=r.mass(d);h=d.qfrc_bias-d.qfrc_passive;b,a=r.base_dofs,r.dof_ids
        aa=r.B@qdd;ab=-np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,a)]@aa+h[b])
        native.append(M[np.ix_(a,b)]@ab+M[np.ix_(a,a)]@aa+h[a])
        xs.append(d.qpos.copy());vs.append(d.qvel.copy());acc.append(qdd)
    x=tensor(xs).requires_grad_();v=tensor(vs).requires_grad_();qdd=tensor(acc).requires_grad_()
    k,tau=tree.path_kinematics_actuation(x,v,qdd)
    assert tau.shape==(3,67) and k['position'].shape==(3,r.model.nbody,3)
    sequential=torch.stack([tree.nominal(x[i],v[i],qdd[i])[0] for i in range(3)])
    torch.testing.assert_close(tau,sequential,atol=1e-11,rtol=1e-10)
    np.testing.assert_allclose(tau.detach().numpy(),native,atol=1e-10,rtol=1e-9)
    weight=tensor(rng.normal(size=tau.shape))
    batch_grad=torch.autograd.grad((tau*weight).sum(),(x,v,qdd))
    serial_grad=torch.autograd.grad((sequential*weight).sum(),(x,v,qdd))
    for left,right in zip(batch_grad,serial_grad):torch.testing.assert_close(left,right,atol=1e-9,rtol=1e-8)
