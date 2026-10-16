"""Bounded forward-derived TRAIN coverage, immutable goals, fail-closed admission."""
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import atomic_json,atomic_npz,read_json,file_digest,digest
from ff_prior.contracts import assert_frozen,MODEL_GATES,all_pass
from ff_prior.proposal_generator import propose,quick_gates,motion_valid
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior_complex.tasks import task_for
from ff_prior_loop.runtime import at_origin,origin_audit
from ff_prior_loop import data as old_data
from ff_prior_modes.data import RuntimeCache
from ff_prior_feasible.validation import validate
from ff_prior.scenes import SceneSpec
from ff_prior.spline import Reference

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'artifacts/v7_1/qualified_modes_smoke_001'
CONTRACT={'seed':20261201,'max_attempts':1200,'cumulative_TRAIN_target':1000,
    'root_families':600,'proposals_per_family':2,'empty_origin_contexts':6,
    'TRAIN_obstacle_contexts':3,'origin_jitter_rad':[.015,.025,.035,.045,.055,.065],
    'motion_scales':[1.,.75,1.25,1.5],'new_empty_family_target':150,
    'goal_prediction_intervals':256,'goal_frozen_before_independent_validation':True,
    'maximum_generation_queries':4,'maximum_generation_intervals':2048,'candidate_wall_s':15.,
    'random_audits':8,'risk_audits':8,'known_execution_failures_excluded':True,
    'scene_scope':'6 stationary origins and 3 existing TRAIN static obstacle scenes',
    'no_target_search_or_repair_in_main_generator':True,'minimum_same_world_goal_candidates':None,
    'final_TEST_arrays_read':False,'strict_multimodal':False}


def old_index():
    index=read_json(PREVIOUS/'training_index.json')['records']
    if sum(e['split']=='TRAIN' for e in index)!=256 or sum(e['split']=='VAL' for e in index)!=48:
        raise ValueError('unexpected original population')
    old_data.assert_disjoint(index)
    for e in index:
        if e['execution_status']=='FAIL' or e['validation_level'] not in ('MODEL_VALID','EXECUTION_VERIFIED'):
            raise ValueError('invalid inherited admission')
        for k in ('record','arrays'):
            if file_digest(ROOT/e[k])!=e[k+'_sha256']:raise ValueError('inherited bytes changed')
    return index


def contexts(canonical,initial,profile,index):
    values=[];cache=RuntimeCache(canonical,initial,profile)
    for slot in range(6):
        rng=np.random.default_rng(CONTRACT['seed']+slot*173)
        scale=CONTRACT['origin_jitter_rad'][slot]
        q0=canonical.spec.planner_zero+rng.uniform(-scale,scale,17)
        rr,ii,cc=at_origin(canonical,q0)
        values.append(({'context_id':f'origin_{slot}','scene':SceneSpec().to_dict(),
            'family_id':None,'origin_audit':origin_audit(canonical,rr,ii)},rr,ii,cc))
    parents=sorted((e for e in index if e['split']=='TRAIN' and e['sample_id'].startswith('complex_train_')
        and e['sample_id'].endswith('static_detour')),key=lambda e:e['sample_id'])
    if len(parents)!=3:raise ValueError('expected 3 existing TRAIN obstacle contexts')
    for slot,e in enumerate(parents):
        row=read_json(ROOT/e['record']);rr,ii,cc,proof=cache.get(row)
        values.append(({'context_id':f'obstacle_{slot}','scene':row['scene'],'family_id':e['family_id'],
            'parent_sample_id':e['sample_id'],'origin_audit':proof},rr,ii,cc))
    return values


def heldout_goals(index,dev,source=None):
    # Reserved TEST goal metadata is used solely by the split-isolation guard;
    # no TEST arrays, training gradients, checkpoint selection or evaluation.
    goals=[(e['sample_id'],read_json(ROOT/e['record'])['condition']['goals']) for e in index if e['split']=='VAL']+[
        (f'DEV_{j}',e['task']['condition']['goals']) for j,e in enumerate(dev)]
    if source is not None:
        source=Path(source)
        for sid,r in read_json(source/'splits.json')['records'].items():
            if r['split']=='TEST':goals.append(('RESERVED_'+sid,read_json(source/'records'/f'{sid}.json')['condition']['goals']))
    return goals


def goal_neighbours(task,heldout):
    hits=[]
    for sid,goal in heldout:
        p,r=pose_errors(task['condition']['goals'],goal)
        if max(p)<.01 and max(r)<np.deg2rad(1):hits.append(sid)
    return hits


def generate(attempt,context,out,profile,heldout,existing_hashes,deadline):
    meta,rr,ii,cc=context;out=Path(out);sid=f'expand_train_{attempt:05d}'
    family=attempt//2;variant=(family+attempt%2)%5
    family_id=meta['family_id'] or f'expand_family_{family:04d}'
    before=rr.calls.copy();started=time.perf_counter();deadline=min(deadline,time.time()+15.)
    result={'sample_id':sid,'attempt':attempt,'family_id':family_id,'split':'TRAIN',
        'context_id':meta['context_id'],'status':'REJECTED','admitted':False,'goal_relabelled':False}
    try:
        ref,provenance=propose(rr,100000+family,variant,CONTRACT['seed'])
        motion_scale=CONTRACT['motion_scales'][(family//9)%4]
        provenance['motion_scale']=motion_scale;result['provenance']=provenance
        if ref is None:result['reason']='DURATION_SET_EXHAUSTED';return result
        z=ref.q0+motion_scale*(ref.z-ref.q0)
        ref=next((v for duration in (10.,15.,20.,25.,29.5)
            if motion_valid(rr,v:=Reference(z,rr.spec.planner_zero,duration))),None)
        if ref is None:result['reason']='SCALED_DURATION_SET_EXHAUSTED';return result
        result['quick_gates']=quick_gates(rr,ref)
        if not all(result['quick_gates'].values()):result['reason']='QUICK_GATES';return result
        # Derive exactly once before independent model/goal validation. Failed
        # paths retain this task; never substitute the validator's final pose.
        pred=predict_coupled(ii,ref,rr,256,deadline)
        task=task_for(rr,ii,cc,profile,pred['ee'][-1],ref.duration)
        atomic_json(out/'tasks'/sid/'task.json',task)
        atomic_npz(out/'tasks'/sid/'initial_state.npz',values=ii.values)
        result.update(task_hash=task['task_hash'],reference_hash=ref.reference_hash,
            complete_initial_state_sha256=file_digest(out/'tasks'/sid/'initial_state.npz'))
        hits=goal_neighbours(task,heldout);result['cross_split_goal_neighbours']=hits
        if hits:result['reason']='HELDOUT_GOAL_NEIGHBOUR';return result
        if ref.reference_hash in existing_hashes:result['reason']='DUPLICATE_REFERENCE';return result
        checks=validate(ref.z,task,rr,ii,cc,profile,deadline)
        result['validation_result']=checks
        if not checks['eligible']:result['reason']='INDEPENDENT_VALIDATION_REJECTED';return result
        if time.time()>deadline:raise RuntimeError('GENERATION_WALL_BUDGET_EXHAUSTED')
        calls={k:rr.calls[k]-before[k] for k in before}
        if calls['coupled']>4 or any(calls[k] for k in ('rrt','sqp','torque_rollout','physics_steps')):
            raise RuntimeError('FORBIDDEN_MAIN_GENERATOR_CALLS')
        atomic_npz(out/'arrays'/f'{sid}.npz',**ref.arrays())
        row={'sample_id':sid,'family_id':family_id,'root_id':sid,'split':'TRAIN',
            'condition':task['condition'],'task':task,'scene':meta['scene'],'context_id':meta['context_id'],
            'validation_level':'MODEL_VALID','execution_status':'NOT_RUN','training_eligible':True,
            'validation':checks['validation'],'reference_hash':ref.reference_hash,
            'arrays':f'arrays/{sid}.npz','complete_initial_state':f'tasks/{sid}/initial_state.npz',
            'initial_state_sha256':file_digest(out/'tasks'/sid/'initial_state.npz'),
            'arrays_sha256':file_digest(out/'arrays'/f'{sid}.npz'),
            'goal_source':'coupled_forward_256_before_independent_validation','goal_relabelled':False,
            'data_role':'BOUNDED_EXPANDED_FF_MODEL_PRIOR','generation_proof':f'generation/{sid}.json'}
        atomic_json(out/'records'/f'{sid}.json',row)
        result.update(status='ADMITTED',admitted=True)
        assert_frozen(task)
    except Exception as error:result.update(status='ERROR',admitted=False,reason=repr(error))
    finally:
        result.update(calls={k:rr.calls[k]-before[k] for k in before},wall_s=time.perf_counter()-started)
    return result


def check_new_record(out,path,ledger=None):
    out=Path(out);r=read_json(path);assert_frozen(r['task'])
    proof=read_json(out/r['generation_proof'])
    if ledger and ledger.data['completed'].get(r['sample_id'],{}).get('status')!='ADMITTED':
        raise ValueError('uncommitted/unknown generation cannot enter training')
    if not (proof['admitted'] and proof['status']=='ADMITTED' and proof['task_hash']==r['task']['task_hash']
        and proof['validation_result']['eligible'] and proof['validation_result']['frozen_goal_pass']
        and all_pass(proof['validation_result']['validation']['gates'],MODEL_GATES)
        and r['validation']==proof['validation_result']['validation'] and not r['goal_relabelled']):
        raise ValueError('admission proof missing or inconsistent')
    for key,hkey in [('arrays','arrays_sha256'),('complete_initial_state','initial_state_sha256')]:
        if file_digest(out/r[key])!=r[hkey]:raise ValueError('admitted bytes changed')
    with np.load(out/r['arrays']) as a:
        ref=Reference(a['Z'],r['condition']['initial_features'],r['condition']['duration'])
    if ref.reference_hash!=r['reference_hash'] or ref.reference_hash!=proof['reference_hash']:
        raise ValueError('reference mismatch')
    if read_json(out/'tasks'/r['sample_id']/'task.json')!=r['task']:raise ValueError('frozen task changed')
    return r


def combined(out,ledger):
    out=Path(out);index=old_index().copy();seen={e['sample_id'] for e in index}
    for p in sorted((out/'records').glob('*.json')):
        r=check_new_record(out,p,ledger)
        if r['execution_status']=='FAIL':continue
        if r['sample_id'] in seen:raise ValueError('duplicate sample id')
        seen.add(r['sample_id']);a=out/r['arrays']
        index.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'split':'TRAIN',
            'record':p.relative_to(ROOT).as_posix(),'arrays':a.relative_to(ROOT).as_posix(),
            'record_sha256':file_digest(p),'arrays_sha256':file_digest(a),
            'validation_level':r['validation_level'],'execution_status':r['execution_status']})
    old_data.assert_disjoint(index)
    return index


def select_audits(out,ledger):
    out=Path(out);path=out/'audit_selection.json'
    if path.exists():return read_json(path)
    rows=[check_new_record(out,p,ledger) for p in sorted((out/'records').glob('*.json'))]
    # Balanced hash-based random selection over contexts; separate risk cohort.
    chosen=[];used=set()
    pools={c:sorted([r for r in rows if r['context_id']==c],key=lambda r:digest({'audit':r['sample_id']}))
        for c in sorted({r['context_id'] for r in rows})}
    while len(chosen)<8:
        changed=False
        for context,pool in pools.items():
            r=next((r for r in pool if r['family_id'] not in used),None)
            if r is None:continue
            chosen.append(r['sample_id']);used.add(r['family_id']);changed=True
            if len(chosen)==8:break
        if not changed:break
    risk=[]
    for r in sorted(rows,key=lambda r:(r['validation']['geometry']['minimum_clearance_m'],r['sample_id'])):
        if r['family_id'] in used:continue
        risk.append(r['sample_id']);used.add(r['family_id'])
        if len(risk)==8:break
    selection={'random_stratified':chosen,'risk':risk,'frozen_before_execution':True,'replacement_after_failure':False}
    atomic_json(path,selection);return selection
