"""Admission preserves world tasks and validation grades; failed executions stay out."""
import numpy as np
from ff_prior.storage import read_json, file_digest
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from ff_prior.spline import Reference
from ff_prior_coverage.data import check_new_record
from ff_prior_loop.data import assert_disjoint
from .protocol import ROOT


def admit(out,entry,sid,candidate,initial,known_hashes):
    reference_hash=candidate.get('reference_hash')
    admitted=bool(candidate['status']=='DONE' and candidate['eligible'] and reference_hash not in known_hashes)
    reason=candidate.get('reason')
    if candidate.get('eligible') and reference_hash in known_hashes:
        reason='DUPLICATE_REFERENCE'
    result={'sample_id':sid,'target_id':entry['target_id'],'task_hash':entry['task']['task_hash'],
            'status':'ADMITTED' if admitted else 'ERROR' if candidate['status']=='ERROR' else 'REJECTED',
            'admitted':admitted,'reason':reason,'reference_hash':reference_hash,'candidate':candidate,
            'calls':candidate['model_calls'],'goal_relabelled':False,
            'validation_result':{k:candidate[k] for k in ('eligible','frozen_goal_pass','validation','score','reference_hash') if k in candidate}}
    if admitted:
        c=entry['task']['condition']
        ref=Reference(candidate['Z'],c['initial_features'],c['duration'])
        if ref.reference_hash!=reference_hash:
            raise ValueError('candidate reference changed on admission')
        task_dir=out/'tasks'/entry['target_id']
        state_path=task_dir/'initial_state.npz'
        if not state_path.exists():
            atomic_npz(state_path,values=initial.values)
        with np.load(state_path) as snapshot:
            if not np.array_equal(snapshot['values'],initial.values):
                raise ValueError('saved full initial state differs from runtime')
        atomic_json(out/'tasks'/sid/'task.json',entry['task'])
        array_path=out/'arrays'/f'{sid}.npz'
        atomic_npz(array_path,**ref.arrays())
        row={'sample_id':sid,'parent_sample_id':entry['parent_sample_id'],'family_id':entry['family_id'],
             'target_id':entry['target_id'],'split':'TRAIN','task':entry['task'],'condition':c,'scene':entry['scene'],
             'context_id':entry['origin_hash'],'origin_hash':entry['origin_hash'],
             'validation_level':'MODEL_VALID','execution_status':'NOT_RUN','training_eligible':True,
             'validation':candidate['validation'],'reference_hash':reference_hash,
             'arrays':array_path.relative_to(out).as_posix(),'arrays_sha256':file_digest(array_path),
             'complete_initial_state':state_path.relative_to(out).as_posix(),'initial_state_sha256':file_digest(state_path),
             'generation_proof':f'generation/{sid}.json','goal_source':'inherited fixed TRAIN world target',
             'goal_relabelled':False,'producer_run_id':out.name,'data_role':'FIXED_WORLD_GUIDED_DIFFUSION_PRIOR',
             'private_teacher_trajectory_used_for_inference':False}
        atomic_json(out/'records'/f'{sid}.json',row)
        known_hashes.add(reference_hash)
    return result


def population(out,ledger):
    return [check_new_record(out,p,ledger) for p in sorted((out/'records').glob('*.json'))
            if ledger.data['completed'].get(p.stem,{}).get('status')=='ADMITTED']


def apply_execution_grade(out,ledger,target_id):
    directory=out/'executions'/target_id
    if not (directory/'selection.json').exists():
        return
    chosen=read_json(directory/'selection.json')['selected_id']
    if chosen is None:
        return
    status=ledger.data['completed'].get('execution_'+target_id,{}).get('status','UNKNOWN')
    row=read_json(out/'records'/f'{chosen}.json')
    passed=status=='EXECUTION_VERIFIED'
    row.update(execution_status='PASS' if passed else 'UNKNOWN' if status in ('UNKNOWN','INTERRUPTED') else 'FAIL',
               training_eligible=passed,validation_level='EXECUTION_VERIFIED' if passed else 'MODEL_VALID')
    atomic_json(out/'records'/f'{chosen}.json',row)


def index_for(out,index,ledger):
    new=[]
    for row in population(out,ledger):
        if not row['training_eligible'] or row['execution_status'] not in ('NOT_RUN','PASS'):
            continue
        p=out/'records'/f'{row["sample_id"]}.json'
        new.append({'sample_id':row['sample_id'],'family_id':row['family_id'],'split':'TRAIN',
                    'record':p.relative_to(ROOT).as_posix(),'arrays':(out/row['arrays']).relative_to(ROOT).as_posix(),
                    'record_sha256':file_digest(p),'arrays_sha256':file_digest(out/row['arrays']),
                    'validation_level':row['validation_level'],'execution_status':row['execution_status'],
                    'producer_run_id':out.name,'new_in_this_run':True})
    merged=[dict(e,new_in_this_run=False) for e in index]+new
    assert_disjoint(merged)
    if len({e['sample_id'] for e in merged})!=len(merged):
        raise ValueError('duplicate dataset sample id')
    return merged,new


def diversity(out,rows):
    result=[]
    for target_id in sorted({r['target_id'] for r in rows}):
        group=[r for r in rows if r['target_id']==target_id and r['training_eligible']]
        views=[]
        for r in group:
            with np.load(out/r['arrays']) as a:
                views.append(a['view_q'].copy())
        distances=[float(np.sqrt(np.mean((views[i]-views[j])**2))) for i in range(len(views)) for j in range(i)]
        result.append({'target_id':target_id,'qualified_slots':len(group),
                       'pairwise_reference_rms_rad':distances,'distinct_modes_claimed':False})
    return result
