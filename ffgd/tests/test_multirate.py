"""Recurrence/adjoint regressions; native accuracy is a separate diagnostic."""
import numpy as np
import torch
from ffgd.robot import Robot
from ffgd.physics import DifferentiableTree,tensor
from ffgd.propagation import FullStatePredictor,FullStateAdjoint
from ffgd.multirate import macro_step,MultirateAdjoint


def test_refresh_one_preserves_full_step_and_held_control_gradient():
    torch.set_num_threads(1);r=Robot();d=r.new_data();p=FullStatePredictor(DifferentiableTree(r),r.snapshot(d),integrator='implicit_damping')
    torch.manual_seed(29);n=5
    values=[tensor(d.qpos),torch.randn(r.model.nv,dtype=torch.float64)*1e-4]
    home=tensor(r.Bplus@d.qpos[r.qpos_ids])
    values += [home.repeat(n+1,1)+torch.randn(n+1,17,dtype=torch.float64)*1e-5,
               torch.randn(n+1,17,dtype=torch.float64)*1e-4,torch.randn(n+1,17,dtype=torch.float64)*1e-3]
    forces=torch.zeros(n,r.model.nv,dtype=torch.float64)
    def evaluate(refresh):
        inputs=[v.clone().requires_grad_() for v in values]
        if refresh is None:out=FullStateAdjoint.apply(*inputs,forces,p,.001,2)
        else:out=MultirateAdjoint.apply(*inputs,forces,p,.001,2,refresh)
        loss=sum((value**2).sum() for value in out)
        return out,torch.autograd.grad(loss,inputs)
    expected,ge=evaluate(None);actual,ga=evaluate(1)
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)
    for a,b in zip(ge,ga):torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-8)
    # A multi-step refresh must propagate gradients through the frozen M/h
    # back to its original actual state and every reference microstep.
    inputs=[v.clone().requires_grad_() for v in values];x,v,q,qd,qdd=inputs
    u=torch.zeros(67,dtype=x.dtype);raw=u;rows=[[x[None]],[v[None]],[],[]]
    for first in range(0,n,3):
        last=min(first+3,n)
        out=macro_step(p,x,v,u,raw,q[first:last],qd[first:last],qdd[first:last],forces[first:last],.001,2,first)
        for row,value in zip(rows,out):row.append(value)
        x,v,u,raw=(value[-1] for value in out)
    out=tuple(torch.cat(row) for row in rows);g=torch.autograd.grad(sum(value.square().sum() for value in out),inputs)
    actual,ga=evaluate(3)
    for a,b in zip(out,actual):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(g,ga):torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-8)


def test_multirate_native_coefficients_and_forced_full_state():
    from ffgd.api import boundary_codec
    from ffgd.contracts import ForcePulse
    from ffgd.multirate import rollout,native_prediction
    torch.set_num_threads(1);r=Robot();d=r.new_data()
    d.qvel[:]=np.random.default_rng(43).normal(0,1e-5,r.model.nv)
    snapshot=r.snapshot(d);before=snapshot.state.copy()
    pulses=(ForcePulse(0.,.2,np.array([.01,-.02,.03])),)
    p=FullStatePredictor(DifferentiableTree(r),snapshot,integrator='implicit_damping',target_force_pulses=pulses)
    codec=boundary_codec(r,snapshot,.2,n=8,origin='initial_position',terminal_policy='free')
    z=np.random.default_rng(51).normal(0,1e-6,codec.free_dim)
    with torch.no_grad():got=rollout(p,codec,tensor(z),physics_dt=.001,control_substeps=2,refresh=5)
    native=native_prediction(r,snapshot,codec,z,dt=.001,refresh=5,substeps=2,target_force_pulses=pulses)
    for key in ('qpos','velocity','applied_torque'):
        np.testing.assert_allclose(got[key].numpy(),native[key],atol=1e-10,rtol=1e-8)
    np.testing.assert_allclose(got['raw_torque'].numpy(),native['raw'][:-1],atol=1e-10,rtol=1e-8)
    np.testing.assert_array_equal(snapshot.state,before)


def test_contact_free_oracle_never_disables_native_default_or_geometry():
    from ffgd.contracts import SphereObstacle,SceneSpec
    from ffgd.api import boundary_codec
    from ffgd.full_state_validation import native_prediction
    base=Robot();d=base.new_data();point=d.xpos[base.tips[0]].copy()
    r=Robot(scene=SceneSpec(spheres=(SphereObstacle(point+[.015,.02,.01],.025),)))
    initial=r.snapshot(r.new_data());before=initial.state.copy();identity=r.model_hash
    codec=boundary_codec(r,initial,.2,n=8,origin='initial_position',terminal_policy='free');z=np.zeros(codec.free_dim)
    physical=native_prediction(r,initial,codec,z,.0005)
    oracle=native_prediction(r,initial,codec,z,.0005,contact_free=True)
    assert not physical['contact_solver_disabled_for_guide_oracle']
    assert oracle['contact_solver_disabled_for_guide_oracle']
    assert physical['native_model_hash']!=oracle['native_model_hash']
    assert physical['minimum_clearance']<0 and oracle['minimum_clearance']<0
    assert np.max(abs(physical['velocity']))>1e-3 and np.max(abs(oracle['velocity']))<1e-8
    np.testing.assert_array_equal(initial.state,before)
    assert r.model_hash==identity
    r.verify_identity()
