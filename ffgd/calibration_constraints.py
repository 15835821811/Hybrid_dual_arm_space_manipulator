"""Active geometry/actuation calibration; every attempted case is retained."""
import json
import time
import numpy as np
import mujoco
import torch
from .robot import Robot
from .physics import DifferentiableTree,tensor
from .geometry import Geometry,witness_gaps
from .propagation import Propagator
from .trajectory import Boundary,BoundaryCodec
from .contracts import PoseConstraint,TaskSpec
from .costs import MultiSpaceEnergy
from .orchestration import check_budget


def geometry_audit(path,count=32):
    torch.set_num_threads(1)
    r=Robot();tree=DifferentiableTree(r);g=Geometry(r);rng=np.random.default_rng(187)
    d=r.new_data();target=r.model.joint(r.spec.target_free_joint_name).qposadr[0]
    (path/'pair_catalog.json').write_text(json.dumps(g.catalog(),indent=2),encoding='utf-8')
    rows=[]
    for i in range(count):
        check_budget()
        q=r.spec.planner_zero+rng.uniform(-.3,.3,17)
        d.qpos[r.qpos_ids]=r.B@q;mujoco.mj_kinematics(r.model,d)
        if i%3 != 0:d.qpos[target:target+3]=d.xpos[r.tips[i%2]]+np.array([.15 if i%3==1 else .205,.01,.013])
        else:d.qpos[target:target+3]=[1.2,0,.05]
        base=tensor(d.qpos)
        def calculate(z):
            x=base.clone();x[r.qpos_ids]=tree.B@z[:17];x[target:target+3]=z[17:]
            k=tree.kinematics(x)
            values=witness_gaps(g,k,x,cap=.05)
            return torch.relu((.02-values)/.02).square().sum(),values,k,x
        z0=np.r_[q,d.qpos[target:target+3]];z=tensor(z0).requires_grad_()
        val,dist,k,x=calculate(z)
        grad=torch.autograd.grad(val,z)[0].numpy()
        exact=np.minimum(g.native_distances(x.detach().numpy(),.05),.05)
        agreement=float(np.max(abs(exact-dist.detach().numpy())))
        box=g.gaps(k).detach().numpy()
        false_safe=int(np.sum((box>=1e-7)&(exact < -1e-7)))
        directions=[]
        # First three states get 8 directions x 3 eps, remaining pressure set forward checks.
        if i<3:
            for j in range(8):
                direction=rng.normal(size=len(z0));direction/=np.linalg.norm(direction)
                ad=float(grad@direction);fds=[]
                for eps in (1e-4,1e-5,1e-6):
                    with torch.no_grad():fd=float((calculate(tensor(z0+eps*direction))[0]-calculate(tensor(z0-eps*direction))[0])/(2*eps))
                    fds.append(dict(epsilon=eps,fd=fd,ad=ad,error=abs(fd-ad),passed=abs(fd-ad)<=1e-6+.005*max(abs(fd),abs(ad))))
                directions.append(fds)
        row=dict(case=i,energy=float(val.detach()),gradient_norm=float(np.linalg.norm(grad)),native_min=float(exact.min()),
                 proxy_false_safe=false_safe,witness_forward_abs=agreement,directions=directions,
                 passed=agreement<1e-10 and false_safe==0 and all(f['passed'] for fs in directions for f in fs))
        rows.append(row)
        (path/f'geometry_{i:03}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='directions'}),flush=True)
    return dict(passed=all(x['passed'] for x in rows),rows=rows,coverage='all native collision geoms and eligible pairs',
                production_guide='native_witness_VJP',proxy='diagnostic only; false-unsafe home overlaps retained',
                limitations='piecewise differentiable; zero-distance degenerate witness explicitly rejected; feature switches not smooth')


def energy_gradients(path,count=3,control_points=8,compiled_backend=False):
    torch.set_num_threads(1)
    r=Robot();tree=DifferentiableTree(r);geometry=Geometry(r);rng=np.random.default_rng(32026)
    if compiled_backend:
        from .compiled_physics import compile_tree
        metadata=compile_tree(tree,path/'tree.pt')
        (path/'compile.json').write_text(json.dumps({k:v for k,v in metadata.items() if k not in ('original','compiled')},indent=2))
    rows=[]
    for case in range(count):
        check_budget()
        d=r.new_data();q0=r.spec.planner_zero+rng.uniform(-.05,.05,17)
        d.qpos[r.qpos_ids]=r.B@q0;mujoco.mj_forward(r.model,d)
        target=int(r.model.joint(r.spec.target_free_joint_name).qposadr[0])
        d.qpos[target:target+3]=d.xpos[r.tips[0]]+np.array([.2,.013,.017])
        p=Propagator(tree,r.snapshot(d));T=(.3 if control_points==8 else .6)+(case%3)*.05
        bc=[Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else (0.,)*17) for j in (0,1,2)]
        if control_points==24:bc += [Boundary(T,j,tuple(range(17)),(0.,)*17) for j in (1,2)]
        codec=BoundaryCodec(control_points,17,T,bc)
        C=np.tile(q0,(control_points,1))+rng.normal(0,.1,(control_points,17));z0=(C.ravel()-codec.offset)@codec.N
        k=tree.kinematics(tensor(d.qpos))
        tasks=tuple(PoseConstraint(T,j,k['position'][r.tips[j]].numpy()+np.array([.02,-.01,.03]),k['rotation'][r.tips[j]].numpy()) for j in (0,1))
        energy=MultiSpaceEnergy(p,codec,geometry,TaskSpec(T,tasks),intervals=8)
        z=tensor(z0).requires_grad_();start=time.perf_counter();terms,pathout=energy.evaluate(z,True)
        grads={name:torch.autograd.grad(value,z,retain_graph=True)[0].detach().numpy() for name,value in terms.items()}
        initial={k:float(v.detach()) for k,v in terms.items()};graph_wall=time.perf_counter()-start
        native=energy.native_evaluate(z0)
        forward_errors={k:abs(initial[k]-native[k]) for k in initial}
        tau_max=float(pathout['raw_torque'].detach().abs().max())
        del terms,pathout
        directions=[]
        for j in range(8):
            direction=rng.normal(size=len(z0));direction/=np.linalg.norm(direction)
            checks=[]
            for eps in (1e-4,1e-5,1e-6):
                plus=energy.native_evaluate(z0+eps*direction);minus=energy.native_evaluate(z0-eps*direction)
                for name in grads:
                    fd=float((plus[name]-minus[name])/(2*eps));ad=float(grads[name]@direction)
                    checks.append(dict(term=name,epsilon=eps,fd=fd,ad=ad,error=abs(fd-ad),passed=abs(fd-ad)<=1e-6+.005*max(abs(fd),abs(ad))))
            directions.append(checks)
        row=dict(case=case,free_dim=codec.free_dim,control_points=control_points,energies=initial,gradient_norms={k:float(np.linalg.norm(v)) for k,v in grads.items()},
                 max_raw_torque=tau_max,graph_wall_seconds=graph_wall,native_forward_errors=forward_errors,directions=directions,
                 passed=all(v['passed'] for checks in directions for v in checks) and all(np.linalg.norm(v)>1e-10 for v in grads.values()) and all(forward_errors[k]<=1e-6+1e-8*abs(native[k]) for k in native))
        rows.append(row);(path/f'case_{case}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='directions'}),flush=True)
    return dict(passed=all(r['passed'] for r in rows),rows=rows,scope='active five terms and total; full-convex-support whole-path AD against independent native/NumPy FD')
