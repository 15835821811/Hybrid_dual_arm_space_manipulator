"""Fixed calibration profile, native torque execution, separate control replay."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from .robot import Robot
from .physics import DifferentiableTree,tensor,quat_mul
from .propagation import Propagator
from .trajectory import Boundary,BoundaryCodec
from .contracts import Candidate
from .execution import track,replay


def run(path,count=1):
    torch.set_num_threads(1)
    profile_path=Path(__file__).resolve().parents[1]/'configs/ffgd/numerical_profile_v1.json'
    profile=json.loads(profile_path.read_text());profile_hash=hashlib.sha256(profile_path.read_bytes()).hexdigest()
    r=Robot();m=r.model;d=r.new_data();snapshot=r.snapshot(d);tree=DifferentiableTree(r);p=Propagator(tree,snapshot)
    T=2.;q0=r.spec.planner_zero;rng=np.random.default_rng(921)
    bc=[Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else (0.,)*17) for j in (0,1,2)]+[Boundary(T,j,tuple(range(17)),(0.,)*17) for j in (1,2)]
    codec=BoundaryCodec(8,17,T,bc)
    C=np.tile(q0,(8,1));end=q0+rng.uniform(-.02,.02,17)
    C[3]=q0+(end-q0)*.3;C[4]=q0+(end-q0)*.7;C[5:]=end
    z=codec.encode(C);candidate=Candidate(C,T,921)
    comparisons=[];previous=None
    for intervals in (32,64,128):
        xs,vs=p.native_rollout(codec,z,intervals)
        body=[];rot=[]
        for x in xs:
            d.qpos[:]=x;mujoco.mj_kinematics(m,d)
            body.append(d.geom_xpos.copy());rot.append(d.geom_xmat.reshape(-1,3,3).copy())
        body=np.asarray(body);rot=np.asarray(rot)
        if previous is not None:
            delta=np.linalg.norm(body[::2]-previous[0],axis=-1)
            rd=np.linalg.norm(rot[::2]-previous[1],axis=(-1,-2))
            extent=float(np.max(delta+rd*m.geom_rbound))
            comparisons.append(dict(coarse=intervals//2,fine=intervals,volume_error_m=extent,passed=extent<=profile['prop_finite_volume_m']))
        previous=(body,rot)
    nominal=[]
    for index in (0,17,51,89,128):
        x=tensor(xs[index]);v=tensor(vs[index]);qdd=tensor(codec.evaluate(z,index*T/128,2))
        tau,ab=tree.nominal(x,v,qdd)
        d.qpos[:]=xs[index];d.qvel[:]=vs[index];d.qacc[:]=0
        d.qacc[r.base_dofs]=ab.detach().numpy();d.qacc[r.dof_ids]=r.B@qdd.numpy()
        mujoco.mj_inverse(m,d)
        inv=d.qfrc_inverse+d.qfrc_constraint
        nominal.append(dict(index=index,torque_abs=float(np.max(abs(tau.detach().numpy()-inv[r.dof_ids]))),
                            base_residual=float(np.max(abs(inv[r.base_dofs]))),raw_ratio=float(np.max(abs(tau.detach().numpy())/r.spec.torque_limits))))
    # Independent differentiable directional derivative of base velocity along its own flow.
    index=51;t=index*T/128;base=tensor(xs[index,r.base_qpos]);q=tensor(codec.evaluate(z,t));qd=tensor(codec.evaluate(z,t,1));qdd=tensor(codec.evaluate(z,t,2))
    vb=tensor(vs[index,r.base_dofs]);bdot=torch.cat((vb[:3],.5*quat_mul(base[3:],torch.cat((torch.zeros(1,dtype=torch.float64),vb[3:])))))
    _,acc=torch.autograd.functional.jvp(lambda b,q,v:p.velocity(b,q,v)[1][r.base_dofs],(base,q,qd),(bdot,qd,qdd))
    _,ab=tree.nominal(tensor(xs[index]),tensor(vs[index]),qdd)
    base_acceleration_error=float(torch.max(abs(acc-ab)))
    bundle,execution=track(r,snapshot,candidate,path,profile)
    replay_result=replay(r,bundle,snapshot)
    replay_result['passed']=replay_result['max_state_abs']<=profile['replay_state_atol']
    (path/'replay.json').write_text(json.dumps(replay_result,indent=2),encoding='utf-8')
    passed=(all(x['passed'] for x in comparisons) and max(x['torque_abs'] for x in nominal)<1e-6 and max(x['base_residual'] for x in nominal)<1e-8
            and base_acceleration_error<1e-6 and execution['status']=='FINISHED' and execution['raw_torque_ratio']<=1
            and execution['linear_momentum_residual']<=profile['momentum_linear_kg_m_s'] and execution['angular_momentum_residual']<=profile['momentum_angular_kg_m2_s'] and replay_result['passed'])
    result=dict(passed=passed,profile_hash=profile_hash,convergence=comparisons,nominal=nominal,base_acceleration_abs=base_acceleration_error,
                execution=execution,replay=replay_result,scope='fixed small spline calibration; NOT a Diffusion result; momentum dt sweep pending')
    print(json.dumps(result,indent=2),flush=True)
    return result
