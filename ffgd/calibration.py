"""Native independent calibration. Results are scoped by tested mechanism."""
import json
import platform
import subprocess
import sys
import time
import mujoco
import numpy as np
import torch
from .robot import Robot
from .physics import DifferentiableTree,tensor


def doctor(path):
    robot=Robot();d=robot.new_data();snap=robot.snapshot(d)
    restored=robot.restore(snap)
    error=float(np.max(abs(robot.snapshot(restored).state-snap.state)))
    meta=robot.metadata()
    (path/'robot_spec.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    np.savez_compressed(path/'initial_state.npz',state=snap.state,mask=snap.mask,model_hash=snap.model_hash)
    result=dict(passed=error==0,python=sys.executable,python_version=sys.version,platform=platform.platform(),
                torch=torch.__version__,cuda=torch.cuda.is_available(),numpy=np.__version__,mujoco=mujoco.__version__,
                snapshot_roundtrip_max=error,model=meta,
                independent_architecture='NOT_RUN',physics_calibration='NOT_RUN')
    return result


def gradients(path,count=3):
    from .trajectory import Boundary,BoundaryCodec
    from .propagation import Propagator,native_momentum
    torch.set_num_threads(1)
    r=Robot();tree=DifferentiableTree(r);rng=np.random.default_rng(27102026)
    rows=[]
    for case in range(count):
        d=r.new_data();q0=r.spec.planner_zero+rng.uniform(-.1,.1,17)
        d.qpos[r.qpos_ids]=r.B@q0
        if case%3:
            d.qvel[r.dof_ids]=r.B@rng.normal(0,.01,17)
            d.qvel[r.base_dofs]=rng.normal(0,.003,6)
        snapshot=r.snapshot(d);p=Propagator(tree,snapshot)
        bc=[Boundary(0,0,tuple(range(17)),tuple(q0)),Boundary(0,1,tuple(range(17)),tuple(p.initial_v)),Boundary(0,2,tuple(range(17)),(0.,)*17)]
        codec=BoundaryCodec(8,17,2.,bc)
        C=np.tile(q0,(8,1))+rng.normal(0,.02,(8,17))
        # Build a full free-space curve while exactly enforcing initial state.
        z0=(C.ravel()-codec.offset)@codec.N
        z=tensor(z0).requires_grad_()
        start=time.perf_counter();out=p.rollout(codec,z,16)
        k=tree.kinematics(out['qpos'][-1]);tips=k['position'][r.tips]
        weights=tensor([[.4,-.3,.2],[-.2,.6,.1]])
        cost=(tips*weights).sum()+.01*out['qpos'][:,:3].square().sum()
        grad=torch.autograd.grad(cost,z)[0].detach().numpy()
        graph_wall=time.perf_counter()-start
        xn,vn=p.native_rollout(codec,z0,16)
        native_error=float(np.max(abs(out['qpos'].detach().numpy()-xn)))
        v_error=float(np.max(abs(out['velocity'].detach().numpy()-vn)))
        momentum_error=float(np.max(abs(tree.momentum(out['qpos'][-1],out['velocity'][-1]).detach().numpy()-p.H.numpy())))
        # Higher-resolution independent propagation, finite-volume error checked later.
        xfine,_=p.native_rollout(codec,z0,64)
        conv=float(np.max(abs(xfine[::4]-xn)))
        del cost,out,k,tips
        def value(zz):
            with torch.no_grad():
                o=p.rollout(codec,tensor(zz),16)
                kk=tree.kinematics(o['qpos'][-1])
                return float((kk['position'][r.tips]*weights).sum()+.01*o['qpos'][:,:3].square().sum())
        directions=[]
        for j in range(8):
            direction=rng.normal(size=len(z0));direction/=np.linalg.norm(direction)
            ad=float(grad@direction);fds=[]
            for eps in (1e-4,1e-5,1e-6):
                fd=(value(z0+eps*direction)-value(z0-eps*direction))/(2*eps)
                fds.append(dict(epsilon=eps,fd=fd,error=abs(ad-fd),passed=abs(ad-fd)<=1e-6+.005*max(abs(ad),abs(fd))))
            directions.append(dict(ad=ad,steps=fds,passed=all(f['passed'] for f in fds)))
        row=dict(case=case,nonzero_initial_momentum=case%3!=0,native_qpos_abs=native_error,native_velocity_abs=v_error,
                 momentum_abs=momentum_error,coarse_fine_qpos_abs=conv,graph_wall_seconds=graph_wall,directions=directions,
                 passed=all(x['passed'] for x in directions) and native_error<1e-8 and v_error<1e-8 and momentum_error<1e-8)
        rows.append(row)
        (path/f'case_{case}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='directions'}),flush=True)
    return dict(passed=all(row['passed'] for row in rows),rows=rows,scope='whole-path FK plus base accumulation direction derivatives; other energies NOT_RUN',
                full_PROP_gate='NOT_RUN',full_GRAD_gate='NOT_RUN')


def physics(path,count=32):
    torch.set_num_threads(1)
    robot=Robot(); tree=DifferentiableTree(robot);m=robot.model;d=robot.new_data()
    rng=np.random.default_rng(27102026)
    rows=[];start=time.perf_counter()
    for i in range(count):
        q=robot.spec.planner_zero+rng.uniform(-.25,.25,17)
        d.qpos[robot.qpos_ids]=robot.B@q
        quat=rng.normal(size=4);quat/=np.linalg.norm(quat)
        d.qpos[robot.base_qpos[3:]]=quat
        d.qpos[robot.base_qpos[:3]]=rng.uniform(-.2,.2,3)
        d.qvel[:]=rng.normal(0,.1,m.nv)
        mujoco.mj_forward(m,d)
        x=tensor(d.qpos);v=tensor(d.qvel)
        k=tree.kinematics(x); M=tree.mass(x,k).numpy()
        native=robot.mass(d)
        pos=float(np.max(np.linalg.norm(k['position'].numpy()-d.xpos,axis=1)))
        # Matrix Frobenius bound is stricter than angle for small errors.
        rot=float(np.max(np.linalg.norm(k['rotation'].numpy()-d.xmat.reshape(-1,3,3),axis=(1,2))))
        mass=float(np.linalg.norm(M-native)/np.linalg.norm(native))
        qacc=rng.normal(0,.2,m.nv);d.qacc[:]=qacc;mujoco.mj_inverse(m,d)
        expected=d.qfrc_inverse.copy()
        # mj_inverse includes -constraint forces; compare contact-free inverse components.
        expected+=d.qfrc_constraint
        computed=tree.inverse(x,v,tensor(qacc)).detach().numpy()
        inv_abs=float(np.max(abs(computed-expected)))
        inv_rel=float(np.linalg.norm(computed-expected)/max(1,np.linalg.norm(expected)))
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
        jerr=0.
        for b in range(1,m.nbody):
            mujoco.mj_jacBodyCom(m,d,jp,jr,b)
            jerr=max(jerr,float(np.max(abs(jp-k['Jcom'][b].numpy()))),float(np.max(abs(jr-k['Jangular'][b].numpy()))))
        rows.append(dict(index=i,position_m=pos,rotation_matrix=rot,mass_relative=mass,jacobian_abs=jerr,inverse_abs=inv_abs,inverse_relative=inv_rel,
                         passed=bool(pos<=1e-6 and rot<=1e-6 and mass<=1e-6 and inv_abs<=1e-6+1e-6*max(abs(expected)))))
    result=dict(passed=all(r['passed'] for r in rows),samples=count,seed=27102026,rows=rows,wall_seconds=time.perf_counter()-start,
                scope='all bodies, full native nv mass, bias and inverse including armature/damping; contacts excluded from nominal inverse',
                model_hash=robot.model_hash,propagation='NOT_RUN',geometry='NOT_RUN')
    print(json.dumps({k:max(r[k] for r in rows) for k in ('position_m','rotation_matrix','mass_relative','jacobian_abs','inverse_abs','inverse_relative')}))
    return result
