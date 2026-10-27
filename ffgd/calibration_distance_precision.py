"""Bounded root-cause experiment for frozen failures 20 and 51 (no reseeding)."""
import json
import numpy as np
import mujoco
import torch
from .robot import Robot
from .physics import DifferentiableTree,tensor
from .geometry import Geometry
from .propagation import Propagator
from .trajectory import Boundary,BoundaryCodec
from .contracts import PoseConstraint,TaskSpec
from .costs import MultiSpaceEnergy


def run(path):
    torch.set_num_threads(1);r=Robot();m=r.model;tree=DifferentiableTree(r);g=Geometry(r);rng=np.random.default_rng(32026)
    fixtures=[]
    for case in range(52):
        q0=r.spec.planner_zero+rng.uniform(-.05,.05,17);T=.3+(case%3)*.05
        C=np.tile(q0,(8,1))+rng.normal(0,.1,(8,17))
        # All initial derivative constraints have rank 3 x 17.
        directions=[]
        for j in range(8):
            direction=rng.normal(size=85);direction/=np.linalg.norm(direction);directions.append(direction)
        if case in (20,51):fixtures.append((case,q0,T,C,directions))
    rows=[];original=float(m.opt.ccd_tolerance);original_graph=m.mesh_graphadr.copy();original_iterations=int(m.opt.ccd_iterations)
    for case,q0,T,C,directions in fixtures:
        d=r.new_data();d.qpos[r.qpos_ids]=r.B@q0;mujoco.mj_forward(m,d)
        target=int(m.joint(r.spec.target_free_joint_name).qposadr[0]);d.qpos[target:target+3]=d.xpos[r.tips[0]]+[.2,.013,.017]
        p=Propagator(tree,r.snapshot(d))
        bc=[Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else (0.,)*17) for j in (0,1,2)]
        codec=BoundaryCodec(8,17,T,bc);z0=(C.ravel()-codec.offset)@codec.N
        k=tree.kinematics(tensor(d.qpos));task=TaskSpec(T,tuple(PoseConstraint(T,j,k['position'][r.tips[j]].numpy()+[.02,-.01,.03],k['rotation'][r.tips[j]].numpy()) for j in (0,1)))
        energy=MultiSpaceEnergy(p,codec,g,task,intervals=8)
        for mode,tol in (('original',original),('exhaustive_support',original),('more_iterations',original)):
            # Numerical oracle experiment only; no physical integration under changed options.
            m.opt.ccd_tolerance=tol
            m.mesh_graphadr[:]=-1 if mode=='exhaustive_support' else original_graph
            m.opt.ccd_iterations=500 if mode=='more_iterations' else original_iterations
            z=tensor(z0).requires_grad_();value=energy.evaluate(z)['collision'];grad=torch.autograd.grad(value,z)[0].numpy()
            checks=[]
            for j,direction in enumerate(directions):
                for eps in (1e-4,1e-5,1e-6):
                    plus=energy.native_evaluate(z0+eps*direction)['collision'];minus=energy.native_evaluate(z0-eps*direction)['collision']
                    ad=float(grad@direction);fd=(plus-minus)/(2*eps);threshold=1e-6+.005*max(abs(ad),abs(fd))
                    checks.append(dict(direction=j,epsilon=eps,ad=ad,fd=fd,error=abs(fd-ad),ratio=abs(fd-ad)/threshold,passed=abs(fd-ad)<=threshold))
            row=dict(case=case,mode=mode,ccd_tolerance=tol,energy=float(value.detach()),checks=checks,passed=all(x['passed'] for x in checks))
            rows.append(row);print(json.dumps(dict(case=case,mode=mode,tolerance=tol,passed=row['passed'],worst_ratio=max(x['ratio'] for x in checks))),flush=True)
    m.opt.ccd_tolerance=original
    m.mesh_graphadr[:]=original_graph;m.opt.ccd_iterations=original_iterations
    return dict(passed=any(all(r['passed'] for r in rows if r['mode']==mode) for mode in ('exhaustive_support','more_iterations')),rows=rows,
                hypothesis='convex support graph or CCD iteration ceiling creates finite-difference noise',intervention='one mechanism at a time; frozen failing cases and seeds unchanged')
