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


def sample_initial_position(robot,config,group,rng):
    """Public mechanical-domain sampling; no DEV task or target is consulted."""
    q=robot.spec.planner_zero+rng.uniform(-config['initial_variation'],config['initial_variation'],17)
    policy=config.get('initial_policy','home_box')
    if policy=='home_box':return q
    if policy!='factorized_limits':raise ValueError('UNKNOWN_INITIAL_PRIOR_POLICY')
    fraction=float(config['initial_limit_fraction']);bins=int(config['initial_strata'])
    if not 0<fraction<1 or bins<2:raise ValueError('INVALID_INITIAL_DOMAIN')
    # Each group sweeps a different coordinate. The next cycle sweeps the next
    # range stratum, including both signs for every coordinate, independent of
    # rejection and family split. Rejected initial states are never resampled.
    j=group%17;stratum=(group//17)%bins
    u=(stratum+rng.uniform())/bins
    midpoint=(robot.lower[j]+robot.upper[j])/2
    halfwidth=(robot.upper[j]-robot.lower[j])*fraction/2
    q[j]=midpoint+(2*u-1)*halfwidth
    return q


def merge_prior(path,config):
    """Concatenate validated prior batches without changing physical examples.

    Family ids are namespaced, never resplit. Parent ledgers retain every failed
    attempt, including ancestors of coordinate-only dataset transformations.
    """
    sources=config['source_datasets']
    if len(sources)<2 or len(set(map(str,sources)))!=len(sources):raise ValueError('DISTINCT_PRIOR_SOURCES_REQUIRED')
    parents=[];batches=[];next_group=0;max_tokens=0;reference=None
    keys=('model_hash','state_mask','control_points','free_dim','geometry_policy_hash','boundary_origin','profile')
    for index,directory in enumerate(sources):
        check_budget();directory=Path(directory)
        manifest=json.loads((directory/'dataset_manifest.json').read_text(encoding='utf-8'))
        data_hash=hashlib.sha256((directory/'prior.npz').read_bytes()).hexdigest()
        if data_hash!=manifest['data_hash'] or manifest['source_kind']!='FF_MODEL_PRIOR':raise ValueError('SOURCE_DATASET_IDENTITY_MISMATCH')
        contract={k:manifest.get(k,'pivot_zero' if k=='boundary_origin' else None) for k in keys}
        if any(v is None for v in contract.values()):raise ValueError('INCOMPLETE_PRIOR_CONTRACT')
        if reference is None:reference=contract
        if contract!=reference:raise ValueError('PRIOR_MERGE_CONTRACT_MISMATCH')
        if any(p['data_hash']==data_hash for p in parents):raise ValueError('DUPLICATE_PRIOR_DATA')
        with np.load(directory/'prior.npz') as source:arrays={k:source[k] for k in source.files}
        if 'source_dataset_index' in arrays:raise ValueError('NESTED_MERGE_REQUIRES_FLAT_SOURCE_LIST')
        if len(arrays['z'])!=manifest['accepted']:raise ValueError('SOURCE_COUNT_MISMATCH')
        train=arrays['is_train']
        if set(arrays['group'][train]) & set(arrays['group'][~train]):raise ValueError('FAMILY_SPLIT_LEAK')
        original_groups=arrays['group'].copy()
        unique,inverse=np.unique(original_groups,return_inverse=True)
        arrays['group']=inverse+next_group;next_group+=len(unique)
        arrays['source_dataset_index']=np.full(len(train),index,dtype=np.int64)
        arrays['source_group']=original_groups
        max_tokens=max(max_tokens,arrays['tokens'].shape[1]);batches.append(arrays)
        parents.append(dict(dataset=str(directory.resolve()),data_hash=data_hash,
            manifest_hash=hashlib.sha256((directory/'dataset_manifest.json').read_bytes()).hexdigest(),
            attempts=manifest['attempts'],accepted=manifest['accepted'],
            attempt_directory=manifest.get('parent_attempt_directory',str((directory/'attempts').resolve())),
            reason_counts=manifest.get('reason_counts',{})))
    for arrays in batches:
        extra=max_tokens-arrays['tokens'].shape[1]
        arrays['tokens']=np.pad(arrays['tokens'],((0,0),(0,extra),(0,0)))
        arrays['padding']=np.pad(arrays['padding'],((0,0),(0,extra)),constant_values=True)
    if any(set(a)!=set(batches[0]) for a in batches):raise ValueError('PRIOR_ARRAY_SCHEMA_MISMATCH')
    merged={k:np.concatenate([a[k] for a in batches]) for k in batches[0]}
    # Equality is checked again against files, including all unpadded conditions.
    start=0
    for parent in parents:
        with np.load(Path(parent['dataset'])/'prior.npz') as source:
            stop=start+len(source['z'])
            for key in source.files:
                actual=merged['source_group' if key=='group' else key][start:stop]
                if key in ('tokens','padding'):actual=actual[:,:source[key].shape[1]]
                np.testing.assert_array_equal(actual,source[key])
            start=stop
    np.savez_compressed(path/'prior.npz',**merged)
    from collections import Counter
    reasons=Counter()
    for parent in parents:reasons.update(parent['reason_counts'])
    manifest=dict(schema='ffgd_prior_union_v1',source_kind='FF_MODEL_PRIOR',**reference,
        attempts=sum(p['attempts'] for p in parents),accepted=len(merged['z']),
        train=int(merged['is_train'].sum()),val=int((~merged['is_train']).sum()),
        data_hash=hashlib.sha256((path/'prior.npz').read_bytes()).hexdigest(),dataset_path=str(path),
        parents=parents,reason_counts=dict(reasons),transformation_config=config,
        new_physical_trajectories=0,execution_verified=False,
        physical_validation='Inherited native results from byte-identical source examples; family split unchanged')
    atomic_json(path/'dataset_manifest.json',manifest)
    return dict(passed=True,**manifest)


def reencode_origin(path,config):
    """Relabel immutable model-prior paths in an invertible affine coordinate system.

    No new trajectories or labels derived from evaluation tasks. Physical input
    arrays and TRAIN/VAL membership remain bit-identical; old rejected attempts
    remain available through the explicit parent dependency.
    """
    from .contracts import Snapshot
    torch.set_num_threads(1)
    source=Path(config['source_dataset']);origin=config['boundary_origin']
    if origin!='initial_position':raise ValueError('UNSUPPORTED_REENCODING_ORIGIN')
    parent=json.loads((source/'dataset_manifest.json').read_text(encoding='utf-8'))
    parent_hash=hashlib.sha256((source/'prior.npz').read_bytes()).hexdigest()
    if parent_hash!=parent['data_hash'] or parent['source_kind']!='FF_MODEL_PRIOR':raise ValueError('SOURCE_DATASET_IDENTITY_MISMATCH')
    if parent.get('boundary_origin','pivot_zero')!='pivot_zero':raise ValueError('SOURCE_ORIGIN_NOT_SUPPORTED')
    with np.load(source/'prior.npz') as z:arrays={k:z[k] for k in z.files}
    r=Robot()
    if r.model_hash!=parent['model_hash']:raise ValueError('REENCODING_MODEL_MISMATCH')
    labels=[];cache={};maximum=0.
    preserved={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items() if k!='z'}
    for i,(C,T,state) in enumerate(zip(arrays['control'],arrays['duration'],arrays['initial_state'])):
        if i%16==0:check_budget()
        d=r.restore(Snapshot(r.model_hash,parent['state_mask'],state))
        if np.any(abs(d.qvel)>1e-12):raise ValueError('REENCODING_NONZERO_BOUNDARY_TEMPLATE_NOT_IMPLEMENTED')
        q=r.Bplus@d.qpos[r.qpos_ids];key=(float(T),q.tobytes())
        if key not in cache:
            if len(cache)>8:cache.clear()
            bc=[Boundary(0,j,tuple(range(17)),tuple(q) if j==0 else (0.,)*17) for j in (0,1,2)]+[Boundary(T,j,tuple(range(17)),(0.,)*17) for j in (1,2)]
            cache[key]=(BoundaryCodec(len(C),17,T,bc),BoundaryCodec(len(C),17,T,bc,origin=origin))
        old,new=cache[key]
        np.testing.assert_allclose(old.decode(arrays['z'][i]),C,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(old.N,new.N)
        label=new.encode(C);maximum=max(maximum,float(np.max(abs(new.decode(label)-C))));labels.append(label)
        if (i+1)%200==0:print(json.dumps(dict(reencoded=i+1,total=len(arrays['z']),maximum_decode_error=maximum)),flush=True)
    arrays['z']=np.stack(labels)
    for k,h in preserved.items():
        if hashlib.sha256(np.ascontiguousarray(arrays[k]).tobytes()).hexdigest()!=h:raise ValueError('PHYSICAL_DATA_OR_SPLIT_CHANGED')
    if maximum>1e-10:raise ValueError('REENCODING_PHYSICAL_PATH_MISMATCH')
    train=arrays['z'][arrays['is_train']];std=train.std(0)
    normalized=(train-train.mean(0))/np.maximum(std,1e-8)
    singular=np.linalg.svd(normalized,compute_uv=False)
    diagnostics=dict(empirical_train_rank=int(np.linalg.matrix_rank(normalized)),free_dim=arrays['z'].shape[1],
        train_std_min=float(std.min()),train_std_max=float(std.max()),
        train_correlation_condition=float((singular[0]/singular[-1])**2),
        standardized_constant_initial_reference_norm=float(np.linalg.norm(train.mean(0)/np.maximum(std,1e-8))))
    np.savez_compressed(path/'prior.npz',**arrays)
    manifest=dict(parent,schema='ffgd_prior_affine_origin_v2',boundary_origin=origin,
        data_hash=hashlib.sha256((path/'prior.npz').read_bytes()).hexdigest(),dataset_path=str(path),
        parent_dataset=str(source.resolve()),parent_data_hash=parent_hash,
        parent_manifest_hash=hashlib.sha256((source/'dataset_manifest.json').read_bytes()).hexdigest(),
        parent_attempt_directory=str((source/'attempts').resolve()),transformation_config=config,
        unchanged_array_hashes=preserved,maximum_decode_error=maximum,new_physical_trajectories=0,
        label_diagnostics=diagnostics,
        physical_validation='Inherited parent native results on bit-identical control/initial/time arrays; no new physical validation claimed')
    atomic_json(path/'dataset_manifest.json',manifest)
    return dict(passed=diagnostics['empirical_train_rank']==diagnostics['free_dim'],**manifest)


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
        q0=sample_initial_position(r,config,group,family_rng)
        T=float(rng.choice(config['duration_choices']));d=r.new_data();d.qpos[r.qpos_ids]=r.B@q0
        snapshot=r.snapshot(d);prop=Propagator(tree,snapshot)
        key=(group,T)
        if key not in cache:
            if len(cache)>4:cache.clear()
            bc=[Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else (0.,)*17) for j in (0,1,2)]+[Boundary(T,j,tuple(range(17)),(0.,)*17) for j in (1,2)]
            cache[key]=BoundaryCodec(n,17,T,bc,origin=config.get('boundary_origin','pivot_zero'))
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
                  boundary_origin=config.get('boundary_origin','pivot_zero'),
                  train=int(arrays['is_train'].sum()),val=int((~arrays['is_train']).sum()),model_hash=r.model_hash,config=config,
                  data_hash=hashlib.sha256((path/'prior.npz').read_bytes()).hexdigest(),state_mask=snapshot.mask,
                  execution_verified=False,geometry_policy_hash=geometry.policy_hash,control_points=n,free_dim=int(arrays['z'].shape[1]),
                  dataset_path=str(path),profile=profile,reason_counts={reason:sum(str(x['reason'])==reason for x in attempts) for reason in sorted(set(str(x['reason']) for x in attempts))},
                  limitations='native independently refined model-valid prior; zero velocity/momentum starts; not execution-verified or continuous collision proof')
    manifest['initial_position_coverage']={split:dict(count=int(mask.sum()),
        minimum=arrays['tokens'][mask,0,5:22].min(0).tolist(),
        maximum=arrays['tokens'][mask,0,5:22].max(0).tolist())
        for split,mask in [('TRAIN',arrays['is_train']),('VAL',~arrays['is_train'])] if mask.any()}
    atomic_json(path/'dataset_manifest.json',manifest)
    return dict(passed=len(accepted)==config['valid_target'],**manifest)
