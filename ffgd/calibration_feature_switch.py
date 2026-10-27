"""Retained large-epsilon failure: demonstrate feature change and local limit."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import mujoco
import torch
from .robot import Robot
from .physics import DifferentiableTree,tensor
from .geometry import Geometry
from .propagation import Propagator
from .contracts import PoseConstraint,TaskSpec
from .costs import MultiSpaceEnergy


def run(path):
    torch.set_num_threads(1)
    original=Path(__file__).resolve().parents[1]/'artifacts/ffgd/runs/20261027T032216_energy_4d19ad/source.zip'
    with zipfile.ZipFile(original) as z:source=z.read('ffgd/trajectory.py')
    target_file=path/'frozen_trajectory.py';target_file.write_bytes(source)
    spec=importlib.util.spec_from_file_location('frozen_boundary_diagnostic',target_file);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    r=Robot();tree=DifferentiableTree(r);g=Geometry(r);rng=np.random.default_rng(32026)
    for case in range(52):
        q0=r.spec.planner_zero+rng.uniform(-.05,.05,17);T=.3+(case%3)*.05
        C=np.tile(q0,(8,1))+rng.normal(0,.1,(8,17));directions=[]
        for j in range(8):
            direction=rng.normal(size=85);direction/=np.linalg.norm(direction);directions.append(direction)
    d=r.new_data();d.qpos[r.qpos_ids]=r.B@q0;mujoco.mj_forward(r.model,d)
    tj=int(r.model.joint(r.spec.target_free_joint_name).qposadr[0]);d.qpos[tj:tj+3]=d.xpos[r.tips[0]]+[.2,.013,.017]
    p=Propagator(tree,r.snapshot(d));bc=[module.Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else (0.,)*17) for j in (0,1,2)]
    codec=module.BoundaryCodec(8,17,T,bc);z0=(C.ravel()-codec.offset)@codec.N
    k=tree.kinematics(tensor(d.qpos));task=TaskSpec(T,tuple(PoseConstraint(T,j,k['position'][r.tips[j]].numpy()+[.02,-.01,.03],k['rotation'][r.tips[j]].numpy()) for j in (0,1)))
    e=MultiSpaceEnergy(p,codec,g,task,intervals=8);direction=directions[3]
    z=tensor(z0).requires_grad_();cost=e.evaluate(z)['collision'];ad=float(torch.autograd.grad(cost,z)[0]@tensor(direction))
    def features(z):
        xs,_=p.native_rollout(codec,z,len(e.grid)-1,grid=e.grid);d=mujoco.MjData(r.model);out=[]
        for x in xs:
            d.qpos[:]=x;mujoco.mj_kinematics(r.model,d)
            _,identity=e.collision_guide.numpy_gaps(d.geom_xpos[g.ids],d.geom_xmat[g.ids].reshape(-1,3,3),features=True);out.append(identity)
        return out
    rows=[]
    for eps in (1e-3,1e-4,3e-5,1e-5,3e-6,1e-6):
        plus=z0+eps*direction;minus=z0-eps*direction
        fd=(e.native_evaluate(plus)['collision']-e.native_evaluate(minus)['collision'])/(2*eps)
        fp,fm=features(plus),features(minus);changed=[]
        for t,(a,b) in enumerate(zip(fp,fm)):
            for pair in set(a)|set(b):
                if a.get(pair)!=b.get(pair):changed.append(dict(time=float(e.grid[t]),pair=g.ids[g.pairs[pair]].tolist(),plus=a.get(pair),minus=b.get(pair)))
        rows.append(dict(epsilon=eps,ad=ad,fd=fd,error=abs(ad-fd),feature_changes=changed,passed=abs(ad-fd)<=1e-6+.005*max(abs(ad),abs(fd))))
    result=dict(passed=all(x['passed'] for x in rows[-3:]) and any(x['feature_changes'] for x in rows if x['epsilon']==1e-4),
                case=51,direction=3,source_run='20261027T032216_energy_4d19ad',rows=rows,
                interpretation='large-step finite difference crosses finite-axis max features; original failed check retained, local derivative tested at multiple smaller scales')
    print(json.dumps(result,indent=2),flush=True);return result
