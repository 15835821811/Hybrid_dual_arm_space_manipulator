"""Distinct qualified labels for pre-existing frozen TRAIN world targets."""
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,atomic_json,atomic_npz,digest,file_digest
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior.spline import Reference
from ff_prior.guidance import public_goal_initializer
from ff_prior_loop import data as old_data
from ff_prior_feasible.guidance import FeasibleProjector
from ff_prior_feasible.validation import validate

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'artifacts/v7_1/qualified_guidance_loop_001'
CONTRACT={'version':'fixed_train_world_modes_1','targets':8,'variants':6,'max_attempts':48,
    'amplitudes_rad':[.03,-.03,.06,-.06,.10,-.10],'seed':20261093,'minimum_control_rms_distance_rad':.005,
    'duration':'unchanged parent','target':'pre-existing TRAIN world goal; never relabel',
    'witness_slots':[[0,0],[1,2],[2,4],[3,5]],'family_selection':'3 paired complex families + 2 other origin families',
    'strict_multimodal_claim':False,'new_VAL':False,'new_TEST':False}


def select(index):
    rows={e['sample_id']:e for e in index if e['split']=='TRAIN'}
    chosen=[]
    for kind in ('static_detour','larger_motion'):
        chosen.extend(sorted((e for k,e in rows.items() if k.startswith('complex_train_') and k.endswith(kind)),key=lambda e:e['sample_id']))
    if len(chosen)!=6: raise ValueError('expected six frozen complex TRAIN targets')
    families={e['family_id'] for e in chosen}
    for e in sorted((e for k,e in rows.items() if k.startswith('loop_train_') and k.endswith('_v0')),key=lambda e:digest(e['sample_id'])):
        if e['family_id'] in families: continue
        chosen.append(e); families.add(e['family_id'])
        if len(chosen)==8: break
    if len(chosen)!=8: raise ValueError('insufficient independent TRAIN parent slots')
    entries=[]
    for slot,e in enumerate(chosen):
        r=read_json(ROOT/e['record'])
        if file_digest(ROOT/e['record'])!=e['record_sha256'] or file_digest(ROOT/e['arrays'])!=e['arrays_sha256']:
            raise ValueError('parent source changed')
        if e['execution_status']=='FAIL': raise ValueError('failed parent is ineligible')
        assert_frozen(r['task'])
        with np.load(ROOT/e['arrays']) as a:z=a['Z'].copy()
        entries.append({'sample_id':f'mode_train_{slot:02d}','parent_sample_id':e['sample_id'],
            'family_id':e['family_id'],'split':'TRAIN','slot':slot,'task':r['task'],'scene':r.get('scene',{}),
            'source_record':e['record'],'source_arrays':e['arrays'],'Z':z})
    return entries


class RuntimeCache:
    def __init__(self,canonical,initial,profile):
        self.canonical,self.initial,self.profile=canonical,initial,profile
        self.cache={}
    def get(self,row):
        key=digest({'initial':row['task']['contract']['initial_state_id'],'robot':row['task']['contract']['robot_hash'],
            'scene':row.get('scene',{}),'q0':row['task']['condition']['initial_features']})
        if key not in self.cache:
            entry=dict(row)
            if not entry.get('scene'): entry.pop('scene',None)
            self.cache[key]=old_data.runtime(entry,self.canonical,self.initial)
        rr,ii,cc,proof=self.cache[key]
        assert_task_compatible(row['task'],rr,ii,cc,self.profile)
        return rr,ii,cc,proof


def proposal(entry,k):
    if not 0<=k<6: raise ValueError('variant outside frozen quota')
    rng=np.random.default_rng(CONTRACT['seed']+entry['slot']*101+(k//2)*997)
    direction=rng.normal(size=17);direction[:10]*=.5;direction/=max(abs(direction))
    s=np.r_[np.linspace(0,1,20)[1:-1],1.];bow=np.sin(np.pi*s);bow[-1]=0.
    return np.asarray(entry['Z'])+CONTRACT['amplitudes_rad'][k]*bow[:,None]*direction


def generate(entry,k,cache,out,profile,deadline):
    rr,ii,cc,_=cache.get(entry);before=rr.calls.copy();start=time.perf_counter();task=entry['task']
    deadline=min(deadline,time.time()+15.);sid=entry['sample_id']+f'_v{k}';out=Path(out)
    projector=FeasibleProjector(task,rr,ii,cc,profile,deadline)
    r={'sample_id':sid,'task_hash':task['task_hash'],'parent_sample_id':entry['parent_sample_id'],
        'status':'ERROR','physical_eligible':False,'admitted':False,'variant':k,'split':'TRAIN'}
    try:
        z,guide=projector(proposal(entry,k),15)
        checks=validate(z,task,rr,ii,cc,profile,deadline)
        references=[np.asarray(entry['Z'])]
        for f in sorted((out/'records').glob(entry['sample_id']+'_v*.json')):
            row=read_json(f)
            if row['execution_status']=='FAIL':continue
            with np.load(out/row['arrays']) as a:references.append(a['Z'].copy())
        distance=min(float(np.sqrt(np.mean((z-v)**2))) for v in references)
        admitted=bool(checks['eligible'] and distance>=CONTRACT['minimum_control_rms_distance_rad'])
        r.update(Z=z,guidance=guide,validation_result=checks,physical_eligible=checks['eligible'],
            admitted=admitted,minimum_reference_rms_rad=distance,status='ADMITTED' if admitted else 'REJECTED')
        if time.time()>deadline:raise RuntimeError('GENERATION_CANDIDATE_WALL_EXCEEDED')
        if admitted:
            ref=Reference(z,task['condition']['initial_features'],task['condition']['duration'])
            atomic_npz(out/'arrays'/f'{sid}.npz',**ref.arrays())
            atomic_npz(out/'tasks'/entry['sample_id']/'initial_state.npz',values=ii.values)
            row={'sample_id':sid,'family_id':entry['family_id'],'parent_sample_id':entry['parent_sample_id'],
                'split':'TRAIN','condition':task['condition'],'task':task,'scene':entry['scene'],
                'validation_level':'MODEL_VALID','execution_status':'NOT_RUN','validation':checks['validation'],
                'reference_hash':ref.reference_hash,'training_eligible':True,'data_role':'FIXED_TRAIN_WORLD_MODES',
                'arrays':f'arrays/{sid}.npz','complete_initial_state':f'tasks/{entry["sample_id"]}/initial_state.npz',
                'goal_relabelled':False,'goal_source':'unchanged previously frozen TRAIN world target',
                'minimum_reference_rms_rad':distance}
            atomic_json(out/'records'/f'{sid}.json',row)
    except Exception as error:r.update(status='ERROR',admitted=False,reason=repr(error))
    r.update(calls={n:rr.calls[n]-before[n] for n in before},wall_s=time.perf_counter()-start,repair_queries=projector.used)
    assert_frozen(task)
    return r


def combined(out):
    out=Path(out);index=read_json(PREVIOUS/'training_index.json')['records'].copy()
    for p in sorted((out/'records').glob('*.json')):
        r=read_json(p)
        if r['execution_status']=='FAIL':continue
        a=out/r['arrays']
        index.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'split':'TRAIN',
            'record':p.relative_to(ROOT).as_posix(),'arrays':a.relative_to(ROOT).as_posix(),
            'record_sha256':file_digest(p),'arrays_sha256':file_digest(a),
            'validation_level':r['validation_level'],'execution_status':r['execution_status']})
    old_data.assert_disjoint(index);return index


def arrays(index,cache,out,progress):
    z=[];bases=[];conditions=[];ids=[];splits=[];states=[];tasks=[];rows=[];proofs=[]
    for j,e in enumerate(index):
        if e['split'] not in ('TRAIN','VAL') or e['execution_status']=='FAIL':raise ValueError('invalid training population')
        if any(file_digest(ROOT/e[k])!=e[k+'_sha256'] for k in ('record','arrays')):raise ValueError('training input changed')
        r=read_json(ROOT/e['record']);rr,ii,cc,proof=cache.get(r)
        with np.load(ROOT/e['arrays']) as a:z.append(a['Z'].copy())
        bases.append(public_goal_initializer(r['task'],rr,ii));conditions.append(r['condition'])
        ids.append(r['sample_id']);splits.append(e['split']);states.append(ii.state_id);tasks.append(r['task']);rows.append(r)
        proofs.append({'sample_id':r['sample_id'],'initial_state_id':ii.state_id,'transfer':proof})
        if (j+1)%40==0:progress('load_actual_training_arrays',loaded=j+1,total=len(index))
    atomic_json(Path(out)/'training_input_audit.json',{'records':index,'runtime_proofs':proofs,'runtime_cache_entries':len(cache.cache),
        'actual_array_count':len(z),'TEST_arrays_read':False,'split_families_disjoint':old_data.assert_disjoint(index)})
    values=(np.stack(z),np.stack(bases),conditions,ids,splits,states)
    atomic_npz(Path(out)/'loaded_training_arrays.npz',Z=values[0],bases=values[1])
    return values,rows
