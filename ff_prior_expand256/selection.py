"""Freeze 64 existing TRAIN world targets and all possible seeds before generation."""
from collections import Counter
import hashlib
import json
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.contracts import assert_frozen
from ff_prior.spline import from_arrays
from ff_prior_loop.data import assert_disjoint
from .protocol import ROOT,SOURCE,MODEL_RUN,MODEL_DIR,SELECTION,POPULATION,PROTOCOL,proposal_id,proposal_seed


def heldout_registry(index):
    families={e['family_id'] for e in index if e['split']!='TRAIN'}
    tasks=set();bindings={}
    def walk(x,hint=None):
        if isinstance(x,list):
            for v in x:walk(v,hint)
        elif isinstance(x,dict):
            split=x.get('split',hint)
            if split in ('DEV','VAL','NEW_VAL','TEST'):
                if x.get('family_id'):families.add(x['family_id'])
                task=x.get('task',x)
                if isinstance(task,dict) and task.get('task_hash'):tasks.add(task['task_hash'])
            for k,v in x.items():
                walk(v,k if k in ('DEV','VAL','NEW_VAL','TEST') else split)
    for p in sorted((ROOT/'artifacts/v7_1').glob('*/frozen_tasks.json')):
        if 'test' in p.parent.name.lower():continue
        bindings[p.relative_to(ROOT).as_posix()]=file_digest(p)
        walk(read_json(p))
    return families,tasks,bindings


def choose(pool,origin_counts,excluded_families,excluded_tasks,previous_origins):
    chosen=[]
    for obstacle in (False,True):
        for _ in range(32):
            used_tasks={r['task']['task_hash'] for r in chosen}
            used_families={r['family_id'] for r in chosen}
            used_origins={r['origin_hash'] for r in chosen}
            candidates=[r for r in pool if bool(r['scene']['objects'])==obstacle
                and r['family_id'] not in excluded_families|used_families
                and r['task']['task_hash'] not in excluded_tasks|used_tasks]
            novel=[r for r in candidates if r['origin_hash'] not in used_origins]
            if novel:candidates=novel
            if not candidates:raise ValueError('insufficient TRAIN coverage; never backfill from heldout')
            features=np.asarray([r['coverage_features'] for r in chosen])
            def rank(r):
                distance=float(np.min(np.mean((features-r['coverage_features'])**2,axis=1))) if len(chosen) else 1.
                return (r['origin_hash'] in previous_origins,-distance/np.sqrt(origin_counts[r['origin_hash']]),digest({'expand256':r['parent_sample_id']}))
            chosen.append(min(candidates,key=rank))
    return chosen


def freeze():
    if SELECTION.exists() or POPULATION.exists():raise ValueError('already frozen; do not reselect after outcomes')
    index=read_json(SOURCE/'training_index.json')['records'];assert_disjoint(index)
    assert len(index)==14014 and sum(e['split']=='TRAIN' for e in index)==13966
    released=read_json(ROOT/'ff_prior_delivery_release/evidence/model_manifest.json')
    assert released['selected_method']=='expanded_diffusion' and released['engineering_gate']['status']=='PASS'
    for n,b in released['files'].items():assert file_digest(MODEL_DIR/n)==b['sha256']
    families,tasks,bindings=heldout_registry(index)
    previous=read_json(SOURCE/'frozen_tasks.json')['entries']
    excluded_tasks=tasks|{e['task']['task_hash'] for e in previous}
    prior_origins={e['origin_hash'] for e in previous}
    origin_counts=Counter();pool=[];references=set();metadata_sha=hashlib.sha256();fallback_arrays=0
    for e in index:
        if e['split']!='TRAIN':continue
        raw=(ROOT/e['record']).read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if sha!=e['record_sha256']:raise ValueError('TRAIN metadata changed')
        metadata_sha.update(e['sample_id'].encode()+sha.encode());row=json.loads(raw.decode('utf-8-sig'))
        ref_hash=row.get('reference_hash')
        if ref_hash is None:
            if file_digest(ROOT/e['arrays'])!=e['arrays_sha256']:raise ValueError('legacy arrays changed')
            with np.load(ROOT/e['arrays']) as a:ref_hash=from_arrays(a).reference_hash
            fallback_arrays+=1
        references.add(ref_hash)
        condition=row.get('condition',row.get('task',{}).get('condition'))
        if condition is None:continue
        q0=np.asarray(condition['initial_features']);origin=digest(q0);origin_counts[origin]+=1
        if 'task' not in row or e['execution_status'] not in ('NOT_RUN','PASS') or not row.get('training_eligible',True):continue
        task=row['task'];assert_frozen(task)
        if condition!=task['condition']:raise ValueError('public condition mismatch')
        world=np.asarray(condition['goals'])[:,:3,3].reshape(-1)
        pool.append({'parent_sample_id':e['sample_id'],'family_id':e['family_id'],'split':'TRAIN','task':task,
            'scene':row.get('scene',{'objects':[]}), 'origin_hash':origin,'coverage_features':np.r_[q0/.1,world/.1].tolist(),'source_entry':e})
    chosen=choose(pool,origin_counts,families,excluded_tasks,prior_origins);entries=[]
    for i,r in enumerate(chosen):
        if file_digest(ROOT/r['source_entry']['arrays'])!=r['source_entry']['arrays_sha256']:raise ValueError('selected arrays changed')
        entries.append(dict(r,target_id=f'expand256_target_{i:02d}',target_slot=i,
            source_origin_rows=origin_counts[r['origin_hash']],frozen_before_generation=True,
            goal_relabelled=False,inference_uses_teacher=False,
            proposals=[{'slot':s,'attempt':a,'sample_id':proposal_id(i,s,a),'seed':proposal_seed(i,s,a)} for s in range(4) for a in range(2)]))
    selected=[]
    for obstacle in (False,True):
        group=[e for e in entries if bool(e['scene']['objects'])==obstacle]
        selected += [e['target_id'] for e in sorted(group,key=lambda e:digest({'expand256_execution':e['target_id']}))[:16]]
    POPULATION.parent.mkdir(parents=True,exist_ok=False)
    atomic_json(POPULATION,{'existing_reference_hashes':sorted(references),'TRAIN_metadata_digest':metadata_sha.hexdigest(),
        'metadata_rows_checked':13966,'legacy_arrays_read_for_missing_reference_hash':fallback_arrays,'origin_counts':dict(origin_counts),
        'excluded_heldout_families':sorted(families),'excluded_task_hashes':sorted(excluded_tasks),'heldout_metadata_bindings':bindings,
        'TEST_arrays_or_results_read':False,'selection_used_candidate_results':False})
    proof={'protocol':PROTOCOL,'entries':entries,'entries_digest':digest(entries),'execution_target_ids':selected,
        'selection_before_generation':True,'source_index_sha256':file_digest(SOURCE/'training_index.json'),
        'population_path':POPULATION.relative_to(ROOT).as_posix(),'population_sha256':file_digest(POPULATION),
        'counts':{'targets':len(entries),'obstacle':sum(bool(e['scene']['objects']) for e in entries),
            'distinct_initial_states':len({e['origin_hash'] for e in entries}),'origins_not_in_previous32':sum(e['origin_hash'] not in prior_origins for e in entries),
            'pool':len(pool),'execution_units':len(selected),'frozen_possible_proposals':512},
        'sampler':'late4_diffusion','model_sha256':file_digest(MODEL_DIR/'model.pt'),
        'scope':'existing TRAIN fixed world targets; no new world goal labeling; no heldout arrays or outcomes'}
    atomic_json(SELECTION,proof);print(json.dumps(proof['counts']))


if __name__=='__main__':freeze()
