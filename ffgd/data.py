"""Forward prior data, independent native checks and complete attempt ledger."""
import hashlib
import json
import os
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from .robot import Robot
from .physics import DifferentiableTree
from .propagation import Propagator
from .geometry import Geometry
from .trajectory import Boundary,BoundaryCodec,knots
from .contracts import PoseConstraint,TaskSpec,Candidate
from .conditioning import encode_condition
from .orchestration import atomic_json,check_budget
from .validation import validate


def generate(path,config,resume=None):
    if resume is not None:
        path=Path(resume)
        if json.loads((path/'generation_config.json').read_text())!=config:raise ValueError('resume configuration mismatch')
    else:atomic_json(path/'generation_config.json',config)
    (path/'attempts').mkdir(exist_ok=True)
    torch.set_num_threads(1);r=Robot();m=r.model;tree=DifferentiableTree(r);geometry=Geometry(r)
    profile=json.loads((Path(__file__).resolve().parents[1]/'configs/ffgd/numerical_profile_v1.json').read_text())
    n=config['control_points'];accepted=[];attempts=[];cache={}
    started=time.perf_counter()
    for attempt in range(config['max_attempts']):
        check_budget();group=attempt//config['group_size'];split='VAL' if group%config['validation_group_modulus']==0 else 'TRAIN'
        result_path=path/'attempts'/f'{attempt:06d}.json';array_path=result_path.with_suffix('.npz')
        if result_path.exists():
            record=json.loads(result_path.read_text());attempts.append(record)
            if record['accepted']:
                with np.load(array_path) as saved:accepted.append({**{k:saved[k] for k in saved.files},'split':split,'group':group,'attempt':attempt})
            if len(accepted)>=config['valid_target']:break
            continue
        # Attempt-local streams give exact resumption without a half-written RNG checkpoint.
        rng=np.random.default_rng(np.random.SeedSequence([config['seed'],attempt,1]))
        family_rng=np.random.default_rng(np.random.SeedSequence([config['seed'],group,2]))
        q0=r.spec.planner_zero+family_rng.uniform(-config['initial_variation'],config['initial_variation'],17)
        T=float(rng.choice(config['duration_choices']));d=r.new_data();d.qpos[r.qpos_ids]=r.B@q0
        snapshot=r.snapshot(d);prop=Propagator(tree,snapshot)
        key=(group,T)
        if key not in cache:
            if len(cache)>4:cache.clear()
            bc=[Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else (0.,)*17) for j in (0,1,2)]+[Boundary(T,j,tuple(range(17)),(0.,)*17) for j in (1,2)]
            cache[key]=BoundaryCodec(n,17,T,bc)
        codec=cache[key]
        phase=np.linspace(0,1,n);delta=rng.normal(0,config['motion_scale'],17)
        C=q0+phase[:,None]*delta
        C+=np.sin(np.pi*phase)[:,None]*rng.normal(0,config['motion_scale']*.4,17)
        C+=rng.normal(0,config['high_frequency_control_std'],C.shape)
        C[:3]=q0;C[-3:]=q0+delta
        z=codec.encode(C);reason=None
        if np.any(C<r.lower) or np.any(C>r.upper):reason='Q17_BOUNDS'
        grid=np.unique(np.r_[np.linspace(0,T,config['intervals']+1),knots(n)*T,np.array([.25,.5,.75])*T])
        qd=codec.evaluate(z,grid,1);qdd=codec.evaluate(z,grid,2)
        if np.max(abs(qd)/r.spec.planner_velocity_limits)>1:reason='VELOCITY'
        if np.max(abs(qdd)/r.spec.planner_acceleration_limits)>1:reason='ACCELERATION'
        poses={};min_clearance=10.;max_torque=0.
        validation_report=None
        if reason is None:
            xs,vs=prop.native_rollout(codec,z,len(grid)-1,grid=grid);scratch=mujoco.MjData(m)
            for i,t in enumerate(grid):
                scratch.qpos[:]=xs[i];scratch.qvel[:]=vs[i];mujoco.mj_forward(m,scratch)
                clearance=geometry.native_scan(scratch)['minimum'];min_clearance=min(min_clearance,clearance)
                if clearance<0:reason='COLLISION';break
                M=r.mass(scratch);h=scratch.qfrc_bias-scratch.qfrc_passive;a,b=r.dof_ids,r.base_dofs;aa=r.B@qdd[i]
                ab=-np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,a)]@aa+h[b]);tau=M[np.ix_(a,b)]@ab+M[np.ix_(a,a)]@aa+h[a]
                max_torque=max(max_torque,float(np.max(abs(tau)/r.spec.torque_limits)))
                if max_torque>config['torque_fraction']:reason='ACTUATOR';break
                if t in (T,.25*T,.5*T,.75*T):
                    for tip,body in enumerate(r.tips):
                        R=scratch.xmat[body].reshape(3,3).copy();pos=scratch.xpos[body].copy()
                        if tip==0:pos+=R@np.array([.0475,0,0])
                        poses[(t,tip)]=PoseConstraint(t,tip,pos,R)
        tokens=np.empty((0,32),np.float32)
        if reason is None:
            constraints=[poses[(T,0)],poses[(T,1)]]
            for t in rng.choice([.25*T,.5*T,.75*T],size=int(rng.integers(0,4)),replace=False):constraints.append(poses[(float(t),int(rng.integers(0,2)))])
            task=TaskSpec(T,tuple(constraints));tokens=encode_condition(r,snapshot,task,prop.H.numpy())
            validation_report=validate(Candidate(C,T,attempt),snapshot,task,prop,codec,geometry,profile)
            if validation_report['status']!='PLAN_VALID':reason=validation_report['status']+':'+','.join(validation_report.get('failures',[]))
        record=dict(attempt=attempt,group=group,split=split,accepted=reason is None,reason=reason,minimum_clearance=min_clearance,max_torque_ratio=max_torque,validation=validation_report)
        item=dict(z=z,control=C,initial_state=snapshot.state,tokens=tokens,duration=T)
        # Every proposed trajectory, including rejected inputs, is recoverable.
        with array_path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**item)
        os.replace(array_path.with_suffix('.tmp'),array_path)
        record['input_sha256']=hashlib.sha256(array_path.read_bytes()).hexdigest()
        atomic_json(result_path,record);attempts.append(record)
        if reason is None:accepted.append(dict(**item,split=split,group=group,attempt=attempt))
        atomic_json(path/'progress.json',dict(attempts=len(attempts),accepted=len(accepted),last_completed_attempt=attempt,elapsed_seconds=time.perf_counter()-started))
        if attempt%5==0 or len(accepted)==config['valid_target']:print(json.dumps(dict(attempts=len(attempts),accepted=len(accepted),last_reason=reason,elapsed_seconds=time.perf_counter()-started)),flush=True)
        if len(accepted)>=config['valid_target']:break
    if not accepted:return dict(passed=False,attempts=len(attempts),accepted=0,dataset_path=str(path))
    max_tokens=max(len(x['tokens']) for x in accepted);tokens=np.zeros((len(accepted),max_tokens,32),np.float32);padding=np.ones(tokens.shape[:2],bool)
    for i,x in enumerate(accepted):tokens[i,:len(x['tokens'])]=x['tokens'];padding[i,:len(x['tokens'])]=False
    arrays=dict(z=np.stack([x['z'] for x in accepted]),control=np.stack([x['control'] for x in accepted]),
                initial_state=np.stack([x['initial_state'] for x in accepted]),tokens=tokens,padding=padding,
                duration=np.array([x['duration'] for x in accepted]),is_train=np.array([x['split']=='TRAIN' for x in accepted]),
                group=np.array([x['group'] for x in accepted]),attempt=np.array([x['attempt'] for x in accepted]))
    np.savez_compressed(path/'prior.npz',**arrays)
    manifest=dict(schema='ffgd_prior_v1',source_kind='FF_MODEL_PRIOR',attempts=len(attempts),accepted=len(accepted),
                  train=int(arrays['is_train'].sum()),val=int((~arrays['is_train']).sum()),model_hash=r.model_hash,config=config,
                  data_hash=hashlib.sha256((path/'prior.npz').read_bytes()).hexdigest(),state_mask=snapshot.mask,
                  execution_verified=False,geometry_policy_hash=geometry.policy_hash,control_points=n,free_dim=int(arrays['z'].shape[1]),
                  dataset_path=str(path),profile=profile,reason_counts={reason:sum(str(x['reason'])==reason for x in attempts) for reason in sorted(set(str(x['reason']) for x in attempts))},
                  limitations='native independently refined model-valid prior; zero velocity/momentum starts; not execution-verified or continuous collision proof')
    atomic_json(path/'dataset_manifest.json',manifest)
    return dict(passed=len(accepted)==config['valid_target'],**manifest)
