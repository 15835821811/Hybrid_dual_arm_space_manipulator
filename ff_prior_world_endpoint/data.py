from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_coverage_train.storage import atomic_json,atomic_npz
from ff_prior_coverage.data import generate_mode,check_new_record
from ff_prior_generalization.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior.guidance import public_goal_initializer
from ff_prior_loop.data import assert_disjoint
from .protocol import ROOT,SOURCE,PROTOCOL as P

def rows(out,ledger):
    paths=sorted((Path(out)/'records').glob('*.json'))
    return [check_new_record(out,p,ledger) for p in paths
            if ledger.data['completed'].get(p.stem,{}).get('status')=='ADMITTED']

def generate(out,targets,index,cache,profile,ledger,deadline,progress):
    out=Path(out)
    atomic_json(out/'fixed_TRAIN_targets.json',{'entries':targets,'before_generation':True,'TRAIN_only':True})
    for entry in targets:
        for k in range(P['fixed_variants']):
            key=entry['sample_id']+f'_v{k}'
            if key in ledger.data['completed']:continue
            ledger.begin(key,'fixed_generation',1200,64)
            progress('fixed_TRAIN_target_correction',target=entry['slot']+1,variant=k+1)
            r=generate_mode(entry,k,cache,out,profile,deadline)
            atomic_json(out/'generation'/f'{key}.json',r);ledger.finish({'status':r['status']},r['calls'])
    selected=out/'data_audit_selection.json'
    if not selected.exists():
        population=rows(out,ledger);chosen=[]
        for target in targets:
            candidates=[r for r in population if r['parent_sample_id']==target['parent_sample_id']]
            if candidates:chosen.append(min(candidates,key=lambda r:digest({'endpoint_witness':r['sample_id']}))['sample_id'])
        atomic_json(selected,{'ids':chosen,'before_execution':True,'replacement_after_failure':False})
    for sid in read_json(selected)['ids']:
        key='TRAIN_execution_'+sid
        if key in ledger.data['completed']:continue
        r=check_new_record(out,out/'records'/f'{sid}.json',ledger);rr,ii,cc,_=cache.get(r)
        with np.load(out/r['arrays']) as a:z=a['Z'].copy()
        before=rr.calls.copy();ledger.begin(key,'TRAIN_execution',1200,16)
        progress('corrected_TRAIN_native_replay',sample=sid)
        outcome=execute(out/'TRAIN_executions'/sid,r['task'],z,rr,ii,cc,profile,deadline)
        r.update(execution_status='PASS' if outcome['status']=='EXECUTION_VERIFIED' else 'FAIL',
                 training_eligible=outcome['status']=='EXECUTION_VERIFIED')
        if outcome['status']=='EXECUTION_VERIFIED':r['validation_level']='EXECUTION_VERIFIED'
        atomic_json(out/'records'/f'{sid}.json',r)
        ledger.finish(outcome,{k:rr.calls[k]-before[k] for k in before})
    population=rows(out,ledger)
    targets_seen={r['parent_sample_id'] for r in population}
    outcomes=[read_json(p) for p in (out/'TRAIN_executions').glob('*/outcome.json')]
    errors=[k for k,v in ledger.data['completed'].items() if v['status'] in ('ERROR','BLOCKED','INTERRUPTED')]
    passed=(len(population)>=32 and len(targets_seen)==16 and len(outcomes)==16 and
            all(r['status']=='EXECUTION_VERIFIED' for r in outcomes) and not errors)
    new=[]
    for r in population:
        if r['execution_status']=='FAIL':continue
        p=out/'records'/f'{r["sample_id"]}.json';a=out/r['arrays']
        new.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'split':'TRAIN',
                    'record':p.relative_to(ROOT).as_posix(),'arrays':a.relative_to(ROOT).as_posix(),
                    'record_sha256':file_digest(p),'arrays_sha256':file_digest(a),
                    'validation_level':r['validation_level'],'execution_status':r['execution_status'],
                    'producer_run_id':out.name,'new_in_this_run':True})
    merged=[dict(e) for e in index]+new;assert_disjoint(merged)
    report={'status':'PASS' if passed else 'BLOCKED','new_TRAIN':len(new),'fixed_targets_covered':len(targets_seen),
            'selected_native':len(outcomes),'native_pass':sum(o['status']=='EXECUTION_VERIFIED' for o in outcomes),
            'total_TRAIN':13774+len(new),'VAL':48,'errors':errors,'TEST_read':False}
    atomic_json(out/'data_summary.json',report)
    atomic_json(out/'training_index.json',{'records':merged,'actual_training_consumed':False,'TEST_arrays_read':False})
    return merged,[e['sample_id'] for e in new],report

def load_arrays(out,index,cache,progress):
    """Reuse only byte-bound, previously audited public initializers; read all arrays."""
    out=Path(out);old_index=read_json(SOURCE/'training_index.json')['records'];n=len(old_index)
    if index[:n]!=old_index:raise ValueError('inherited array order/record changed')
    prior=read_json(SOURCE/'training_input_audit.json')
    if prior['records']!=old_index or prior['actual_array_count']!=n:raise ValueError('parent input proof changed')
    with np.load(SOURCE/'loaded_training_arrays.npz') as a:oldz=a['Z'].copy();oldbase=a['bases'].copy()
    z=[];bases=[];conditions=[];ids=[];splits=[];states=[];records=[];proofs=[]
    for i,e in enumerate(index):
        if e['split'] not in ('TRAIN','VAL') or e['execution_status']=='FAIL':raise ValueError('invalid training row')
        if any(file_digest(ROOT/e[k])!=e[k+'_sha256'] for k in ('record','arrays')):raise ValueError('training source changed')
        row=read_json(ROOT/e['record'])
        with np.load(ROOT/e['arrays']) as a:zz=a['Z'].copy()
        if i<n:
            if not np.array_equal(zz,oldz[i]):raise ValueError('cached trajectory changed')
            proof=prior['runtime_proofs'][i]
            if proof['sample_id']!=e['sample_id'] or proof['initial_state_id']!=row['task']['contract']['initial_state_id']:
                raise ValueError('cached complete state changed')
            b=oldbase[i];state=proof['initial_state_id']
            proof={'sample_id':e['sample_id'],'initial_state_id':state,'reused_parent_slot':i,
                   'source_input_audit_sha256':file_digest(SOURCE/'training_input_audit.json') if i==0 else None}
        else:
            rr,ii,cc,transfer=cache.get(row);b=public_goal_initializer(row['task'],rr,ii);state=ii.state_id
            proof={'sample_id':e['sample_id'],'initial_state_id':state,'transfer':transfer,'new_runtime':True}
        z.append(zz);bases.append(b);conditions.append(row['condition']);ids.append(e['sample_id'])
        splits.append(e['split']);states.append(state);records.append(row);proofs.append(proof)
        if (i+1)%500==0:progress('load_bound_training_arrays',loaded=i+1,total=len(index))
    arrays=(np.stack(z),np.stack(bases),conditions,ids,splits,states)
    atomic_npz(out/'loaded_training_arrays.npz',Z=arrays[0],bases=arrays[1])
    atomic_json(out/'training_input_audit.json',{'records':index,'actual_array_count':len(index),
        'reused_verified_initializers':n,'new_initializers':len(index)-n,'runtime_proofs':proofs,
        'parent_loaded_arrays_sha256':file_digest(SOURCE/'loaded_training_arrays.npz'),
        'parent_input_audit_sha256':file_digest(SOURCE/'training_input_audit.json'),'TEST_arrays_read':False})
    return arrays,records
