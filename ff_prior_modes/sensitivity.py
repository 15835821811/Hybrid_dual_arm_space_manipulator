"""Full free-floating propagation, 34-direction local training surrogate only."""
import time
from pathlib import Path
import numpy as np
from ff_prior.storage import atomic_json,atomic_npz,read_json,file_digest,digest
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import predict_coupled
from ff_prior_fixed.correction import pose_vector
from ff_prior.contracts import assert_frozen

CONTRACT={'directions':34,'directions_kind':'17 smooth terminal + 17 zero-terminal sine bows',
    'phase_intervals':128,'difference_rad':1e-4,'half_step_relative_tolerance':.02,
    'mixed_direction_relative_tolerance':.05,'queries_per_anchor':39,'max_anchors':180,
    'cache':'one existing qualified TRAIN anchor per exact task hash',
    'not_full_latent_gradient':True,'not_a_physical_validator':True,'VAL_queries':0}


def directions():
    s=np.r_[np.linspace(0,1,20)[1:-1],1.];h=s*s*(3-2*s);bow=np.sin(np.pi*s);bow[-1]=0.
    return np.concatenate([np.eye(17)[:,None,:]*h[None,:,None],np.eye(17)[:,None,:]*bow[None,:,None]]),h,bow


def coefficient_map(basis):
    _,h,bow=directions();end=basis[-1];inner=(bow@basis-(bow@h)*end)/(bow@bow)
    return np.vstack([np.kron(end[None,:],np.eye(17)),np.kron(inner[None,:],np.eye(17))])


def build(task,z,robot,initial,basis,deadline):
    assert_frozen(task);start=time.perf_counter();before=robot.calls.copy();ds,_,_=directions();eps=1e-4
    def evaluate(value):
        p=predict_coupled(initial,Reference(value,task['condition']['initial_features'],task['condition']['duration']),robot,128,deadline)
        return pose_vector(p['ee'][-1],task['condition']['goals'])
    e=evaluate(z);jac=np.stack([(evaluate(z+eps*d)-e)/eps for d in ds],axis=1)
    checks=[]
    for j in (0,17):
        half=(evaluate(z+.5*eps*ds[j])-e)/(.5*eps)
        error=float(np.linalg.norm(half-jac[:,j])/max(np.linalg.norm(half),1e-8))
        checks.append({'direction':j,'relative_difference':error,'passed':error<=.02})
    rng=np.random.default_rng(20261097);coeff=rng.normal(size=34);delta=np.einsum('i,ijk->jk',coeff,ds)
    coeff*=.001/max(abs(delta).ravel());delta=np.einsum('i,ijk->jk',coeff,ds)
    actual=evaluate(z+delta)-e;approx=jac@coeff
    mixed=float(np.linalg.norm(actual-approx)/max(np.linalg.norm(actual),1e-8))
    # A separate held-out temporal direction exposes the approximation's limit;
    # it is diagnostic, never silently represented as an exact 119D derivative.
    latent=rng.normal(size=(basis.shape[1],17));delta=basis@latent;latent*=.001/max(abs(delta).ravel())
    actual_full=evaluate(z+basis@latent)-e;effective=jac@coefficient_map(basis)
    omitted=float(np.linalg.norm(actual_full-effective@latent.ravel())/max(np.linalg.norm(actual_full),1e-8))
    calls={k:robot.calls[k]-before[k] for k in before}
    passed=bool(np.isfinite(effective).all() and all(v['passed'] for v in checks) and mixed<=.05 and calls['coupled']==39)
    return effective,{'status':'PASS' if passed else 'BLOCKED','task_hash':task['task_hash'],'reference_hash':digest(z),
        'anchor_pose_error':e,'half_step_checks':checks,'mixed_direction_relative_error':mixed,
        'unrepresented_temporal_direction_relative_error':omitted,'calls':calls,'wall_s':time.perf_counter()-start,
        'surrogate_scope':CONTRACT,'geometry_validation_or_execution_claim':False}


def prepare(out,arrays,rows,codec,cache,ledger,deadline,progress):
    out=Path(out);z,bases,conditions,ids,splits,states=arrays;anchors={}
    for i,(r,split) in enumerate(zip(rows,splits)):
        if split=='TRAIN':anchors.setdefault(r['task']['task_hash'],i)
    if len(anchors)>180:raise ValueError('sensitivity anchor quota exceeded')
    result={};failures=[]
    for j,(key,i) in enumerate(anchors.items()):
        work='sensitivity_'+key;path=out/'sensitivity'/f'{key}.json';arr=path.with_suffix('.npz')
        if work not in ledger.data['completed']:
            ledger.begin(work,'sensitivity',5400,180);rr,ii,cc,_=cache.get(rows[i]);before=rr.calls.copy()
            progress('TRAIN_coupled_sensitivities',anchor=j+1,total=len(anchors),sample=ids[i])
            try:
                g,report=build(rows[i]['task'],z[i],rr,ii,codec.basis,deadline)
                report['anchor_sample_id']=ids[i]
                if report['status']=='PASS':atomic_npz(arr,G=g)
            except Exception as error:report={'status':'BLOCKED','reason':repr(error),'anchor_sample_id':ids[i],'task_hash':key}
            report['calls']={k:rr.calls[k]-before[k] for k in before};atomic_json(path,report)
            ledger.finish({'status':report['status']},report['calls'])
        if not path.exists() or read_json(path)['status']!='PASS' or not arr.exists():failures.append(key)
        else:
            with np.load(arr) as a:result[key]=a['G'].copy()
    complete=not failures
    report={'status':'PASS' if complete else 'BLOCKED','anchors':len(anchors),'failed_tasks':failures,
        'contract':CONTRACT,'TRAIN_only':True,'TEST_arrays_read':False,
        'jacobian_files':{k:file_digest(out/'sensitivity'/f'{k}.npz') for k in result}}
    atomic_json(out/'sensitivity_summary.json',report)
    if not complete:return None,report
    # Matrices are bound to the exact TRAIN row's task, never a nearest VAL task.
    matrices=np.stack([result[r['task']['task_hash']] for r,s in zip(rows,splits) if s=='TRAIN'])
    return matrices,report
