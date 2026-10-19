"""Reuse byte-bound prior Z/bases and exact local anchors; load all TRAIN rows."""
import time
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.guidance import public_goal_initializer
from .protocol import ROOT,SOURCE,CACHE,OLD,CACHE_HASHES,PROTOCOL


def same_entry(a,b):
    return {k:v for k,v in a.items() if k!='new_in_this_run'}=={k:v for k,v in b.items() if k!='new_in_this_run'}


def load(out,index,codec,cache,ledger,deadline,progress):
    prior=read_json(CACHE/'training_input_audit.json');old_index=read_json(CACHE/'training_index.json')['records'];n=len(old_index)
    if n!=14014 or prior['records']!=old_index or prior['actual_rows_loaded']!=n:
        raise ValueError('parent cache audit mismatch')
    if not all(same_entry(a,b) for a,b in zip(index[:n],old_index)):
        raise ValueError('inherited ordering or identity changed')
    with np.load(CACHE/'loaded_training_arrays.npz') as a:
        inherited_z=a['Z'].copy();inherited_bases=a['bases'].copy()
    z=[];bases=[];rows=[];conditions=[];ids=[];splits=[];states=[];proofs=[];base_cache={}
    for i,e in enumerate(index):
        if time.time()>=deadline:raise RuntimeError('TRAIN_INPUT_TIME_LIMIT')
        if file_digest(ROOT/e['record'])!=e['record_sha256']:raise ValueError('training metadata changed')
        r=read_json(ROOT/e['record']);task=r['task']
        if i<n:
            zz,b=inherited_z[i],inherited_bases[i]
            old=prior['runtime_proofs'][i]
            if old['sample_id']!=e['sample_id'] or old['initial_state_id']!=task['contract']['initial_state_id']:
                raise ValueError('cached complete state identity changed')
            proof={'sample_id':e['sample_id'],'initial_state_id':old['initial_state_id'],'reused_bound_cache_slot':i}
        else:
            if file_digest(ROOT/e['arrays'])!=e['arrays_sha256']:raise ValueError('new training arrays changed')
            with np.load(ROOT/e['arrays']) as a:zz=a['Z'].copy()
            key=task['task_hash']
            if key not in base_cache:
                rr,ii,cc,transfer=cache.get(r)
                base_cache[key]=(public_goal_initializer(task,rr,ii),ii.state_id,transfer)
            b,state,transfer=base_cache[key]
            proof={'sample_id':e['sample_id'],'initial_state_id':state,'new_public_initializer':True,'transfer':transfer}
        z.append(zz);bases.append(b);rows.append(r);conditions.append(r['condition']);ids.append(e['sample_id']);splits.append(e['split']);states.append(proof['initial_state_id']);proofs.append(proof)
        if (i+1)%1000==0:progress('consume_bound_TRAIN_VAL_arrays',rows=i+1,total=len(index))
    arrays=(np.stack(z),np.stack(bases),conditions,ids,splits,states)
    atomic_npz(out/'loaded_training_arrays.npz',Z=arrays[0],bases=arrays[1])
    atomic_json(out/'training_input_audit.json',{'status':'PASS','records':index,'actual_rows_loaded':len(index),
        'reused_byte_bound_array_rows':n,'new_individual_arrays_read':len(index)-n,'parent_cache_hashes':CACHE_HASHES,
        'runtime_proofs':proofs,'TEST_arrays_read':False})
    old_training=read_json(OLD/'learning/goal_loss/training.json')
    inherited_hash=digest({'z':arrays[0][:n],'bases':arrays[1][:n],'conditions':conditions[:n],
        'ids':ids[:n],'splits':splits[:n],'states':states[:n]})
    if inherited_hash!=old_training['metadata']['data_hash']:raise ValueError('cache not bound to warm model data hash')
    anchors={}
    for i,(r,s) in enumerate(zip(rows,splits)):
        if s=='TRAIN':anchors.setdefault(r['task']['task_hash'],i)
    old_summary=read_json(OLD/'sensitivity_summary.json')
    if digest(old_summary)!=old_training['metadata']['sensitivity_summary_hash'] or old_summary['status']!='PASS':
        raise ValueError('warm G summary identity changed')
    if digest(codec.state)!=digest(read_json(OLD/'learning/goal_loss/codec.json')):raise ValueError('codec changed')
    matrices={};bindings={}
    for j,(key,i) in enumerate(anchors.items()):
        if time.time()>=deadline:raise RuntimeError('TRAIN_ANCHOR_TIME_LIMIT')
        b=old_summary['bindings'].get(key)
        if b is None:raise ValueError('unexpected new task anchor; cannot silently query new G')
        path=ROOT/b['proof'];arr=ROOT/b['array'];p=read_json(path)
        if (b['anchor']!=ids[i] or b['reference_hash']!=digest(arrays[0][i])
            or file_digest(path)!=b['proof_sha256'] or file_digest(arr)!=b['array_sha256']
            or p['status']!='PASS' or p['anchor_sample_id']!=ids[i] or p['reference_hash']!=b['reference_hash']
            or p['calls']['coupled']!=39 or p['mixed_direction_relative_error']>.05 or not all(c['passed'] for c in p['half_step_checks'])):
            raise ValueError('local G reuse identity/derivative checks mismatch')
        with np.load(arr) as a:g=a['G'].copy()
        if g.shape!=(12,codec.dim) or not np.isfinite(g).all():raise ValueError('invalid G')
        matrices[key]=g;bindings[key]=dict(b,reused=True)
        if (j+1)%1000==0:progress('reuse_exact_local_G',anchors=j+1,total=len(anchors))
    G=np.stack([matrices[r['task']['task_hash']] for r,s in zip(rows,splits) if s=='TRAIN'])
    if digest(G[:13966])!=old_training['metadata']['sensitivity_matrix_hash']:raise ValueError('warm G matrix data hash mismatch')
    report={'status':'PASS','anchors':len(anchors),'reused':len(anchors),'new':0,'new_coupled_queries':0,
        'TRAIN_only':True,'not_full_current_prediction_gradient':True,'bindings':bindings,
        'inherited_cache_data_hash_verified':True,'inherited_G_matrix_hash_verified':True}
    atomic_json(out/'sensitivity_summary.json',report)
    return arrays,rows,G,report
