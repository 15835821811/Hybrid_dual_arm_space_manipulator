import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_coverage_train.storage import atomic_json
from .protocol import ROOT,SOURCE,WARM

def load(out,index,codec,progress):
    proof=read_json(SOURCE/'training_input_audit.json')
    if proof['records']!=index or proof['actual_array_count']!=len(index):raise ValueError('input proof mismatch')
    with np.load(SOURCE/'loaded_training_arrays.npz') as a:z=a['Z'].copy();bases=a['bases'].copy()
    rows=[];conditions=[];ids=[];splits=[];states=[];anchors={}
    for i,e in enumerate(index):
        if e['split'] not in ('TRAIN','VAL') or e['execution_status']=='FAIL':raise ValueError('invalid split/row')
        for k in ('record','arrays'):
            if file_digest(ROOT/e[k])!=e[k+'_sha256']:raise ValueError('source row changed')
        row=read_json(ROOT/e['record'])
        with np.load(ROOT/e['arrays']) as a:
            if not np.array_equal(z[i],a['Z']):raise ValueError('cached Z mismatch')
        old=proof['runtime_proofs'][i]
        if old['sample_id']!=e['sample_id'] or old['initial_state_id']!=row['task']['contract']['initial_state_id']:
            raise ValueError('complete initial state mismatch')
        rows.append(row);conditions.append(row['condition']);ids.append(e['sample_id'])
        splits.append(e['split']);states.append(old['initial_state_id'])
        if e['split']=='TRAIN':anchors.setdefault(row['task']['task_hash'],i)
        if i%500==0:progress('verify_same_training_inputs',loaded=i,total=len(index))
    summary=read_json(WARM/'sensitivity_summary.json');oldcodec=read_json(WARM/'learning/goal_loss/codec.json')
    if summary['status']!='PASS' or digest(codec.basis)!=digest(np.asarray(oldcodec['basis'])):raise ValueError('local G basis mismatch')
    matrices={};bindings={}
    for j,(key,i) in enumerate(anchors.items()):
        path=WARM/'sensitivity'/f'{key}.json';arr=path.with_suffix('.npz');p=read_json(path)
        if (p['status']!='PASS' or p['anchor_sample_id']!=ids[i] or p['reference_hash']!=digest(z[i])
            or file_digest(arr)!=summary['jacobian_files'][key]):raise ValueError('local anchor changed')
        with np.load(arr) as a:g=a['G'].copy()
        if g.shape!=(12,102) or not np.isfinite(g).all():raise ValueError('invalid local G')
        matrices[key]=g*codec.latent_scale[None,:]
        bindings[key]={'anchor':ids[i],'proof_sha256':file_digest(path),'array_sha256':file_digest(arr)}
        if j%500==0:progress('verify_frozen_local_G',loaded=j,total=len(anchors))
    G=np.stack([matrices[r['task']['task_hash']] for r,s in zip(rows,splits) if s=='TRAIN'])
    atomic_json(out/'input_verification.json',{'status':'PASS','records':index,'actual_array_count':len(index),
        'TRAIN':splits.count('TRAIN'),'VAL':splits.count('VAL'),'new_data':0,'new_physical_executions':0,
        'parent_array_sha256':file_digest(SOURCE/'loaded_training_arrays.npz'),
        'parent_input_audit_sha256':file_digest(SOURCE/'training_input_audit.json'),
        'G_bindings':bindings,'TEST_read':False})
    return (z,bases,conditions,ids,splits,states),rows,G
