"""Lineage-preserving multi-origin fixed-world-goal augmentation and real loading."""
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,atomic_json,atomic_npz,digest,file_digest
from ff_prior.scenes import SceneSpec
from ff_prior.spline import Reference
from ff_prior.model_gates import evaluate_reference
from ff_prior.coupled_predictor import pose_errors
from ff_prior.guidance import public_goal_initializer
from ff_prior_complex.tasks import task_for,scene_runtime,transfer_audit
from ff_prior_cost.experiment import validate
from .runtime import at_origin,origin_audit
from .guidance import JointProjector

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/v7_1/complex_train_val_001'
CONTRACT={'TRAIN_families':12,'VAL_families':4,'variants_per_goal':3,'max_attempts':48,
    'origin_shift_rad':.02,'motion_scale':.9,'bow_rad':.03,'duration_s':15.,'seed':20261059,
    'variant0':'forward-derived world goal frozen before any repair',
    'variant1_2':'opposite smooth bows, repair to same frozen world goal',
    'split':'inherit parent family; reject cross-split 1cm/1deg goal neighbours',
    'old_complex_rows_consumed':True,'full_initial_state':True,'initial_velocity':'zero only',
    'relabel_after_failure':False,'strict_multimodal_claim':False}


def old_index():
    index=read_json(OLD/'combined_train_val_index.json')['records']
    for e in index:
        if e['split'] not in ('TRAIN','VAL'): raise ValueError('non-development partition')
        if file_digest(ROOT/e['record'])!=e['record_sha256'] or file_digest(ROOT/e['arrays'])!=e['arrays_sha256']:
            raise ValueError('source dataset hash changed')
        if e['execution_status']=='FAIL' or e['validation_level'] not in ('MODEL_VALID','EXECUTION_VERIFIED'):
            raise ValueError('ineligible source row')
    assert_disjoint(index)
    return index


def assert_disjoint(index):
    groups={}
    for e in index:
        if e['family_id'] in groups and groups[e['family_id']]!=e['split']: raise ValueError('split family leakage')
        groups[e['family_id']]=e['split']
    return True


def select(index,expansion=False):
    selected=[]
    for split,count in (('TRAIN',12),('VAL',4)):
        pool=sorted((e for e in index if e['split']==split and e['sample_id'].startswith('main_')),
            key=lambda e:digest({'loop_v1':e['sample_id']}))
        families=set(); unique=[]
        for e in pool:
            if e['family_id'] in families: continue
            families.add(e['family_id']); unique.append(e)
        chosen=unique[12:16] if expansion and split=='TRAIN' else [] if expansion else unique[:count]
        selected.extend(chosen)
    if len(selected)!=(4 if expansion else 16): raise ValueError('insufficient frozen parent slots')
    return selected


def runtime(entry,canonical,canonical_initial):
    scene=SceneSpec.from_dict(entry.get('scene',SceneSpec().to_dict()))
    if scene.objects:
        base,bi,_=scene_runtime(scene); transfer=transfer_audit(canonical,canonical_initial,base,bi)
    else: base=canonical; transfer={'status':'SAME_CANONICAL_MODEL'}
    rr,ii,cc=at_origin(base,entry['task']['condition']['initial_features'])
    check=origin_audit(base,rr,ii)
    if ii.state_id!=entry['task']['contract']['initial_state_id']: raise ValueError('complete initial state mismatch')
    return rr,ii,cc,{'scene_transfer':transfer,'origin':check}


def neighbours(task,split,index,dev):
    hits=[]
    for e in index:
        if e['split']==split: continue
        p,r=pose_errors(task['condition']['goals'],read_json(ROOT/e['record'])['condition']['goals'])
        if max(p)<.01 and max(r)<np.deg2rad(1): hits.append(e['sample_id'])
    for j,d in enumerate(dev):
        target=d.get('task',d)
        p,r=pose_errors(task['condition']['goals'],target['condition']['goals'])
        if max(p)<.01 and max(r)<np.deg2rad(1): hits.append('DEV_'+str(j))
    return hits


def freeze_parent(parent,slot,canonical,profile,index,dev,out,deadline,expansion=False):
    row=read_json(ROOT/parent['record']); rng=np.random.default_rng(CONTRACT['seed']+slot+1000*expansion)
    q0=canonical.spec.planner_zero+rng.uniform(-.02,.02,17)
    rr,ii,cc=at_origin(canonical,q0); check=origin_audit(canonical,rr,ii)
    with np.load(ROOT/parent['arrays']) as a: z=q0+.9*(a['Z']-np.asarray(row['condition']['initial_features']))
    sid=('expand' if expansion else 'loop')+f'_{parent["split"].lower()}_{slot:02d}'
    result={'sample_id':sid,'parent_sample_id':parent['sample_id'],'family_id':parent['family_id'],
        'split':parent['split'],'group':'new_origin','status':'BLOCKED','Z':z,'origin_audit':check,
        'scene':SceneSpec().to_dict(),'frozen_before_training_and_comparison':True}
    ref=Reference(z,q0,15.); pred,gates=evaluate_reference(rr,ii,ref,cc,profile,deadline)
    result['anchor_validation']=gates
    if gates['validation_level']!='MODEL_VALID': return result,rr.calls
    task=task_for(rr,ii,cc,profile,pred['ee'][-1],15.)
    result['task']=task; result['cross_split_goal_neighbours']=neighbours(task,parent['split'],index,dev)
    if result['cross_split_goal_neighbours']: return result,rr.calls
    result['status']='FROZEN'
    atomic_npz(Path(out)/'tasks'/sid/'initial_state.npz',values=ii.values)
    atomic_json(Path(out)/'tasks'/sid/'task.json',task)
    return result,rr.calls


def generate_variant(entry,k,rr,ii,cc,profile,out,deadline):
    task=entry['task']; sid=entry['sample_id']+f'_v{k}'; z=np.asarray(entry['Z']).copy()
    guide=None
    if k:
        rng=np.random.default_rng(int(digest(entry['sample_id'])[:8],16))
        direction=rng.normal(size=17); direction/=max(abs(direction))
        s=np.r_[np.linspace(0,1,20)[1:-1],1.]
        z+=(1 if k==1 else -1)*.03*np.sin(np.pi*s)[:,None]*direction
        projector=JointProjector(task,rr,ii,cc,profile,deadline)
        z,guide=projector(z,15)
    validated=validate(z,task,rr,ii,cc,profile,deadline)
    result={'sample_id':sid,'task_hash':task['task_hash'],'variant':k,'guidance':guide,**validated}
    if validated['eligible']:
        directory=Path(out); ref=Reference(z,task['condition']['initial_features'],15.)
        atomic_npz(directory/'arrays'/f'{sid}.npz',**ref.arrays())
        row={'sample_id':sid,'family_id':entry['family_id'],'parent_sample_id':entry['parent_sample_id'],
            'split':entry['split'],'condition':task['condition'],'task':task,'scene':entry['scene'],
            'validation_level':'MODEL_VALID','execution_status':'NOT_RUN','validation':validated['validation'],
            'reference_hash':ref.reference_hash,'training_eligible':entry['split']=='TRAIN',
            'data_role':'FIXED_GOAL_MULTI_ORIGIN','arrays':f'arrays/{sid}.npz',
            'complete_initial_state':f'tasks/{entry["sample_id"]}/initial_state.npz',
            'goal_relabelled':False,'goal_source':'variant0_full_coupled_forward_before_freeze'}
        atomic_json(directory/'records'/f'{sid}.json',row)
    return result


def combined(out,include_expansion=True):
    index=old_index()
    for p in sorted((Path(out)/'records').glob('*.json')):
        r=read_json(p)
        if not include_expansion and r['sample_id'].startswith('expand_'): continue
        if r['execution_status']=='FAIL': continue
        a=Path(out)/r['arrays']
        index.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'split':r['split'],
            'record':p.relative_to(ROOT).as_posix(),'arrays':a.relative_to(ROOT).as_posix(),
            'record_sha256':file_digest(p),'arrays_sha256':file_digest(a),
            'validation_level':r['validation_level'],'execution_status':r['execution_status']})
    assert_disjoint(index)
    return index


def training_arrays(index,canonical,canonical_initial,profile,out,progress):
    z=[]; bases=[]; conditions=[]; ids=[]; splits=[]; states=[]; proofs=[]
    for j,e in enumerate(index):
        if e['split'] not in ('TRAIN','VAL') or e['execution_status']=='FAIL': raise ValueError('invalid training partition')
        r=read_json(ROOT/e['record'])
        if file_digest(ROOT/e['record'])!=e['record_sha256'] or file_digest(ROOT/e['arrays'])!=e['arrays_sha256']:
            raise ValueError('training source changed')
        rr,ii,cc,proof=runtime(r,canonical,canonical_initial)
        from ff_prior.contracts import assert_task_compatible
        assert_task_compatible(r['task'],rr,ii,cc,profile)
        with np.load(ROOT/e['arrays']) as a: value=a['Z'].copy()
        z.append(value); bases.append(public_goal_initializer(r['task'],rr,ii)); conditions.append(r['condition'])
        ids.append(r['sample_id']); splits.append(e['split']); states.append(ii.state_id)
        proofs.append({'sample_id':r['sample_id'],'initial_state_id':ii.state_id,'transfer':proof})
        if (j+1)%40==0: progress('load_actual_training_arrays',loaded=j+1,total=len(index))
    atomic_json(Path(out)/'training_input_audit.json',{'records':index,'runtime_proofs':proofs,
        'split_families_disjoint':assert_disjoint(index),'TEST_arrays_read':False,'actual_array_count':len(z)})
    return np.stack(z),np.stack(bases),conditions,ids,splits,states
