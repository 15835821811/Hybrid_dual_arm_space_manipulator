"""Exact block extraction and VJP tests for the calibrated base-row backend."""
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from ffgd.robot import Robot
from ffgd.physics import DifferentiableTree,tensor
from ffgd.propagation import Propagator


def test_batched_bodyframe_rows_preserve_world_momentum_and_gradients():
    torch.set_num_threads(1);r=Robot();tree=DifferentiableTree(r);tree.use_momentum_mass_rows=True
    d=r.new_data();rng=np.random.default_rng(8107)
    d.qvel[r.dof_ids]=r.B@rng.normal(0,.01,17);d.qvel[r.base_dofs]=rng.normal(0,.01,6)
    d.qpos[r.base_qpos[:3]]=[.7,-.3,.5]
    quat=Rotation.from_rotvec([.3,-.2,.4]).as_quat();d.qpos[r.base_qpos[3:]]=quat[[3,0,1,2]]
    prop=Propagator(tree,r.snapshot(d))
    q=tensor(np.tile(prop.initial_q,(3,1))+rng.normal(0,.03,(3,17))).requires_grad_()
    bases=tensor(np.tile(d.qpos[r.base_qpos],(3,1))).requires_grad_()
    qd=tensor(rng.normal(0,.02,(3,17)));rows=prop.bodyframe_mass_rows(q)
    direct=[];batched=[];native=[]
    for i in range(3):
        x,v=prop.velocity(bases[i],q[i],qd[i]);direct.append(v)
        _,other=prop.velocity_from_rows(bases[i],q[i],qd[i],rows[i]);batched.append(other)
        scratch=r.new_data();scratch.qpos[:]=x.detach().numpy();mujoco.mj_forward(r.model,scratch)
        M=r.mass(scratch);b,a=r.base_dofs,r.dof_ids;R=scratch.xmat[r.base_body].reshape(3,3)
        P,L=prop.H.numpy()[:3],prop.H.numpy()[3:];va=r.B@qd[i].numpy()
        h=np.r_[P,R.T@(L-np.cross(scratch.qpos[r.base_qpos[:3]],P))]
        nv=np.zeros(r.model.nv);nv[a]=va;nv[b]=np.linalg.solve(M[np.ix_(b,b)],h-M[np.ix_(b,a)]@va);native.append(nv)
    direct=torch.stack(direct);batched=torch.stack(batched)
    torch.testing.assert_close(direct,batched,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(batched.detach().numpy(),native,rtol=1e-10,atol=1e-11)
    weight=tensor(rng.normal(size=direct.shape))
    a=torch.autograd.grad((direct*weight).sum(),(q,bases),retain_graph=True)
    b=torch.autograd.grad((batched*weight).sum(),(q,bases))
    for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=1e-9,atol=1e-10)


def test_base_mass_rows_match_full_matrix_and_vjp_with_all_columns():
    torch.set_num_threads(1);r=Robot();tree=DifferentiableTree(r);d=r.new_data()
    rng=np.random.default_rng(291);d.qpos[r.qpos_ids]+=r.B@rng.normal(0,.07,17)
    x=tensor(d.qpos).requires_grad_();rows=r.base_dofs
    full=tree.mass(x)[rows];block=tree.mass_rows(x,rows)
    assert block.shape==(6,r.model.nv)
    torch.testing.assert_close(full,block,rtol=1e-12,atol=1e-12)
    cotangent=tensor(rng.normal(size=block.shape))
    a=torch.autograd.grad((full*cotangent).sum(),x)[0]
    b=torch.autograd.grad((block*cotangent).sum(),x)[0]
    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
