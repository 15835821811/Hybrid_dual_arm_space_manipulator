"""Software recurrence tests; native physics evidence is separate calibration."""
import torch
from ffgd.propagation import FullStateAdjoint


class NonlinearHeldInput:
    def step(self,x,v,u,raw,q,qd,qdd,dt,update,external):
        if update:
            raw=torch.cat((q+.2*qd+.03*qdd,torch.zeros(50,dtype=q.dtype)))+.1*x[:67]-.2*v[:67]
            u=torch.tanh(raw)
        accel=torch.cat((u,torch.zeros(12,dtype=v.dtype)))+.01*torch.sin(x[:79])+external
        vv=v+dt*accel
        xx=x+dt*torch.cat((vv,torch.zeros(2,dtype=v.dtype)))
        return xx,vv,u,raw


def test_adjoint_preserves_initial_state_and_held_input_derivatives():
    torch.manual_seed(13);predictor=NonlinearHeldInput();dt=.02;steps=9
    values=[torch.randn(n,dtype=torch.float64)*.1 for n in (81,79)]
    values += [torch.randn(steps+1,17,dtype=torch.float64)*.1 for _ in range(3)]
    forces=torch.randn(steps,79,dtype=torch.float64)*.03
    direct=[x.clone().requires_grad_() for x in values]
    x,v,q,qd,qdd=direct;u=torch.zeros(67,dtype=x.dtype);raw=u
    xs=[x];vs=[v];us=[];rs=[]
    for i in range(steps):
        x,v,u,raw=predictor.step(x,v,u,raw,q[i],qd[i],qdd[i],dt,i%4==0,forces[i])
        xs.append(x);vs.append(v);us.append(u);rs.append(raw)
    expected=tuple(torch.stack(a) for a in (xs,vs,us,rs))
    weights=[torch.randn_like(a) for a in expected]
    true_grad=torch.autograd.grad(sum((a*w).sum() for a,w in zip(expected,weights)),direct)
    inputs=[x.clone().requires_grad_() for x in values]
    actual=FullStateAdjoint.apply(*inputs,forces,predictor,dt,4)
    got_grad=torch.autograd.grad(sum((a*w).sum() for a,w in zip(actual,weights)),inputs)
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(true_grad,got_grad):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
