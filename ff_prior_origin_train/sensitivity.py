"""Full free-floating propagation, 34-direction local training surrogate only."""
import time
import traceback
import shutil
from functools import lru_cache
from pathlib import Path
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_coverage_train.storage import atomic_json,atomic_npz
from ff_prior_coverage_train.storage import retry_write
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import predict_coupled
from ff_prior_fixed.correction import pose_vector
from ff_prior.contracts import assert_frozen

CONTRACT={'directions':34,'directions_kind':'17 smooth terminal + 17 zero-terminal sine bows',
    'phase_intervals':128,'difference_rad':1e-4,'half_step_relative_tolerance':.02,
    'mixed_direction_relative_tolerance':.05,'queries_per_anchor':39,'max_anchors':13000,
    'cache':'one existing qualified TRAIN anchor per exact task hash',
    'not_full_latent_gradient':True,'not_a_physical_validator':True,'VAL_queries':0}


from ff_prior_pilot_train.sensitivity import directions,coefficient_map,build


@lru_cache(maxsize=1)
def prior_cache():
    from .learning import FROZEN
    source=FROZEN.parent.parent
    summary=read_json(source/'sensitivity_summary.json')
    if summary['status']!='PASS' or not summary['TRAIN_only']:raise ValueError('invalid parent sensitivity summary')
    return source,summary,read_json(FROZEN/'codec.json')


def import_anchor(out,key,i,arrays,codec,ledger):
    source,summary,state=prior_cache()
    if digest(codec.state)!=digest(state):raise ValueError('exact cache requires identical frozen codec')
    if key not in summary['jacobian_files']:return False
    z,_,_,ids,splits,_=arrays;path=source/'sensitivity'/f'{key}.json';arr=path.with_suffix('.npz')
    proof=read_json(path)
    if not (splits[i]=='TRAIN' and proof['status']=='PASS' and proof['task_hash']==key
        and proof['anchor_sample_id']==ids[i] and proof['reference_hash']==digest(z[i])
        and proof['calls']['coupled']==39 and proof['mixed_direction_relative_error']<=.05
        and all(c['passed'] for c in proof['half_step_checks'])
        and file_digest(arr)==summary['jacobian_files'][key]):raise ValueError('cached task/anchor/proof mismatch')
    with np.load(arr) as a:
        if a['G'].shape!=(12,codec.dim) or not np.isfinite(a['G']).all():raise ValueError('invalid cached matrix')
    ledger.begin('sensitivity_'+key,'cached_sensitivity',9000,13000)
    for p in (path,arr):
        dst=Path(out)/'sensitivity'/p.name;dst.parent.mkdir(parents=True,exist_ok=True)
        retry_write(shutil.copyfile,p,dst)
        if file_digest(dst)!=file_digest(p):raise ValueError('cache copy differs')
    ledger.finish({'status':'PASS','reused_from':str(source),'source_proof_sha256':file_digest(path)},
                  {'coupled':39,'new_coupled':0})
    return True


def prepare(out,arrays,rows,codec,cache,ledger,deadline,progress):
    out=Path(out);z,bases,conditions,ids,splits,states=arrays;anchors={}
    for i,(r,split) in enumerate(zip(rows,splits)):
        assert_frozen(r['task'])
        if split=='TRAIN':anchors.setdefault(r['task']['task_hash'],i)
    if len(anchors)>CONTRACT['max_anchors']:raise ValueError('sensitivity anchor quota exceeded')
    result={};failures=[]
    for j,(key,i) in enumerate(anchors.items()):
        if time.time()>=deadline:
            failures.extend(list(anchors)[j:]);break
        if j%100==0 and sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=20*1024**3:
            failures.extend(list(anchors)[j:]);break
        work='sensitivity_'+key;path=out/'sensitivity'/f'{key}.json';arr=path.with_suffix('.npz')
        if work not in ledger.data['completed']:
            if import_anchor(out,key,i,arrays,codec,ledger):
                pass
            else:
                compute_anchor(out,key,i,arrays,rows,codec,cache,ledger,deadline,progress,j,len(anchors))
        if not path.exists() or read_json(path)['status']!='PASS' or not arr.exists():failures.append(key)
        else:
            with np.load(arr) as a:result[key]=a['G'].copy()
    complete=not failures
    report={'status':'PASS' if complete else 'BLOCKED','anchors':len(anchors),'failed_tasks':failures,
        'contract':CONTRACT,'TRAIN_only':True,'TEST_arrays_read':False,
        'jacobian_files':{k:file_digest(out/'sensitivity'/f'{k}.npz') for k in result}}
    atomic_json(out/'sensitivity_summary.json',report)
    if not complete:return None,report
    matrices=np.stack([result[r['task']['task_hash']] for r,s in zip(rows,splits) if s=='TRAIN'])
    return matrices,report


def compute_anchor(out,key,i,arrays,rows,codec,cache,ledger,deadline,progress,j,total):
    z,bases,conditions,ids,splits,states=arrays
    work='sensitivity_'+key;path=out/'sensitivity'/f'{key}.json';arr=path.with_suffix('.npz')
    ledger.begin(work,'sensitivity',9000,CONTRACT['max_anchors']);rr,ii,cc,_=cache.get(rows[i]);before=rr.calls.copy()
    try:
        progress('TRAIN_coupled_sensitivities',anchor=j+1,total=total,sample=ids[i])
        g,report=build(rows[i]['task'],z[i],rr,ii,codec.basis,deadline)
        report['anchor_sample_id']=ids[i]
        if report['status']=='PASS':atomic_npz(arr,G=g)
    except Exception as error:report={'status':'BLOCKED','reason':repr(error),'traceback':traceback.format_exc(),'anchor_sample_id':ids[i],'task_hash':key}
    report['calls']={k:rr.calls[k]-before[k] for k in before}
    try:
        atomic_json(path,report)
    except Exception as error:
        ledger.finish({'status':'BLOCKED','reason':'sensitivity proof write failed: '+repr(error)},report['calls'])
        raise
    ledger.finish({'status':report['status']},report['calls'])
