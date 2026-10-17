"""Fresh initial states, one forward goal, independent admission, immutable splits."""
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.proposal_generator import propose,quick_gates,motion_valid
from ff_prior.coupled_predictor import predict_coupled
from ff_prior.spline import Reference
from ff_prior.scenes import SceneSpec,Obstacle
from ff_prior_complex.tasks import task_for
from ff_prior_scene_guard.runtime import at_scene_origin,InitialGeometryRejected
from ff_prior_feasible.validation import validate
from ff_prior_generalization.data import neighbours
from ff_prior_coverage.data import check_new_record,generate_mode as fixed_generate
from .protocol import ROOT,PROTOCOL as P


def proposal_origin(canonical,split,slot,attempt):
    seed=P['TRAIN_seed'] if split=='TRAIN' else P['validation_seed']
    rng=np.random.default_rng(seed+slot*1019+attempt*100003)
    jitter=P['origin_jitter_rad'][slot%6]
    return canonical.spec.planner_zero+rng.uniform(-jitter,jitter,17)


def scratch_scene(prediction,slot,attempt,family,split):
    if slot%4!=3:return SceneSpec()
    seed=P['TRAIN_seed'] if split=='TRAIN' else P['validation_seed']
    rng=np.random.default_rng(seed+slot*997+attempt*104729)
    direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    center=prediction['ee'][128,slot%2,:3,3]+P['scene_offset_m'][(slot//3)%3]*direction
    radius=P['scene_radii_m'][slot%3]
    return SceneSpec((Obstacle(f'batch_v1_sphere_{split}_{slot}_{attempt}','sphere',tuple(center),(radius,0.,0.)),),family)


def generate(split,slot,attempt,canonical,canonical_initial,profile,guard,old_origins,validation_origins,out,deadline):
    if split not in ('TRAIN','NEW_VAL'):raise ValueError('invalid split')
    slots=P['forward_TRAIN_slots'] if split=='TRAIN' else P['new_validation_slots']
    quota=P['attempts_per_TRAIN_slot'] if split=='TRAIN' else P['attempts_per_validation_slot']
    if not 0<=slot<slots or not 0<=attempt<quota:raise ValueError('outside frozen proposal quota')
    out=Path(out);sid=f'batch_v1_{split.lower()}_{slot:04d}_a{attempt}'
    family=f'batch_v1_{split.lower()}_family_{slot:04d}'
    started=time.perf_counter();deadline=min(deadline,time.time()+P['candidate_wall_s'])
    r={'sample_id':sid,'family_id':family,'split':split,'slot':slot,'attempt':attempt,
       'status':'REJECTED','admitted':False,'goal_relabelled':False,'parent_sample_id':None,
       'method_results_used':False,'scene_parent':None,'origin_policy':P['origin_policy'],
       'initialization_costs_in_wall_time':True,
       'initialization_counter_scope':'returned runtimes; internal parent initialization not aggregated'}
    tracked=[]
    try:
        q0=proposal_origin(canonical,split,slot,attempt);r['q17_origin']=q0;r['origin_hash']=digest(q0)
        r['jitter_rad']=P['origin_jitter_rad'][slot%6]
        if r['origin_hash'] in old_origins:
            r['reason']='EXISTING_ORIGIN';return r
        if split=='TRAIN' and any(np.max(abs(q0-v))<P['minimum_new_validation_initial_linf_separation_rad'] for v in validation_origins):
            r['reason']='FROZEN_VALIDATION_INITIAL_NEIGHBOUR';return r
        rr,ii,cc,origin=at_scene_origin(canonical,canonical_initial,SceneSpec(),q0)
        tracked.append((rr,rr.calls.copy()));r['empty_origin_audit']=origin
        variant=(slot//6)%5;seed=P['TRAIN_seed'] if split=='TRAIN' else P['validation_seed']
        offset=P['proposal_family_offset']+(0 if split=='TRAIN' else 10000)
        ref,provenance=propose(rr,offset+slot*8+attempt,variant,seed)
        r['provenance']=provenance
        if ref is None:r['reason']='NO_MOTION_DURATION';return r
        scale=P['motion_scales'][(slot//30 if split=='TRAIN' else slot)%6];z=q0+scale*(ref.z-q0)
        ref=next((v for duration in (10.,15.,20.,25.,29.5)
                  if motion_valid(rr,v:=Reference(z,q0,duration))),None)
        r['motion_scale']=scale
        if ref is None:r['reason']='SCALED_DURATION_SET_EXHAUSTED';return r
        r['quick_gates']=quick_gates(rr,ref)
        if not all(r['quick_gates'].values()):r['reason']='QUICK_GATES';return r
        pred=predict_coupled(ii,ref,rr,P['prediction_intervals'],deadline)
        scene=scratch_scene(pred,slot,attempt,family,split);r['scene']=scene.to_dict()
        if scene.objects:
            rr,ii,cc,origin=at_scene_origin(canonical,canonical_initial,scene,q0)
            tracked.append((rr,rr.calls.copy()));r['scene_origin_audit']=origin
        task=task_for(rr,ii,cc,profile,pred['ee'][-1],ref.duration)
        atomic_json(out/'tasks'/sid/'task.json',task)
        atomic_npz(out/'tasks'/sid/'initial_state.npz',values=ii.values)
        r.update(task_hash=task['task_hash'],reference_hash=ref.reference_hash,
                 complete_initial_state_sha256=file_digest(out/'tasks'/sid/'initial_state.npz'))
        r['protected_goal_neighbours']=neighbours(task['condition']['goals'],guard)
        if r['protected_goal_neighbours']:r['reason']='PROTECTED_WORLD_GOAL_NEIGHBOUR';return r
        checks=validate(ref.z,task,rr,ii,cc,profile,deadline);r['validation_result']=checks
        if not checks['eligible']:r['reason']='INDEPENDENT_VALIDATION_REJECTED';return r
        calls={k:sum(robot.calls[k]-before[k] for robot,before in tracked) for k in rr.calls}
        if calls['coupled']>P['forward_coupled_ceiling'] or any(calls[k] for k in ('rrt','sqp','physics_steps','torque_rollout')):
            raise RuntimeError('FORBIDDEN_FORWARD_GENERATION_CALLS')
        if time.time()>deadline:raise RuntimeError('GENERATION_WALL_EXHAUSTED')
        atomic_npz(out/'arrays'/f'{sid}.npz',**ref.arrays())
        row={'sample_id':sid,'family_id':family,'split':split,'condition':task['condition'],'task':task,
             'scene':scene.to_dict(),'context_id':r['origin_hash'],'origin_hash':r['origin_hash'],
             'slot':slot,'attempt':attempt,'parent_sample_id':None,'scene_parent':None,
             'validation_level':'MODEL_VALID','execution_status':'NOT_RUN','training_eligible':split=='TRAIN',
             'validation':checks['validation'],'reference_hash':ref.reference_hash,
             'arrays':f'arrays/{sid}.npz','arrays_sha256':file_digest(out/'arrays'/f'{sid}.npz'),
             'complete_initial_state':f'tasks/{sid}/initial_state.npz',
             'initial_state_sha256':r['complete_initial_state_sha256'],'goal_relabelled':False,
             'goal_source':'single coupled forward 256 before independent validation',
             'data_role':'INDEPENDENT_ORIGIN_TRAIN' if split=='TRAIN' else 'FROZEN_FRESH_VALIDATION',
             'generation_proof':f'generation/{sid}.json','producer_run_id':out.name}
        atomic_json(out/'records'/f'{sid}.json',row)
        r.update(status='ADMITTED',admitted=True)
    except InitialGeometryRejected as error:
        r.update(status='REJECTED',reason='INITIAL_GEOMETRY_REJECTED',initial_geometry=error.geometry)
    except ValueError as error:
        r.update(reason=repr(error),status='ERROR')
    except Exception as error:r.update(status='ERROR',reason=repr(error))
    finally:
        r['calls']={k:sum(robot.calls[k]-before[k] for robot,before in tracked) for k in canonical.calls}
        r['wall_s']=time.perf_counter()-started
    return r


def rows(out,ledger):
    paths=sorted((Path(out)/'records').glob('*.json'))
    # Interrupted/uncommitted files remain as evidence but never enter a split.
    if ledger:paths=[p for p in paths if ledger.data['completed'].get(p.stem,{}).get('status')=='ADMITTED']
    return [check_new_record(out,p,ledger) for p in paths]


def validation_entries(out,ledger):
    return [{'sample_id':r['sample_id'],'family_id':r['family_id'],'split':'NEW_VAL',
             'task':r['task'],'scene':r['scene'],'origin_hash':r['origin_hash'],
             'source_record':f'records/{r["sample_id"]}.json','source_arrays':r['arrays'],
             'initial_state_sha256':r['initial_state_sha256'],'frozen_before_training':True,
             'training_eligible':False,'parent_sample_id':None,'method_results_used':False}
            for r in rows(out,ledger) if r['split']=='NEW_VAL']


def select_modes(out,ledger):
    out=Path(out);path=out/'mode_selection.json'
    if path.exists():return read_json(path)
    forward=[r for r in rows(out,ledger) if r['data_role']=='INDEPENDENT_ORIGIN_TRAIN']
    chosen=[]
    for obstacle in (False,True):
        pool=sorted((r for r in forward if bool(r['scene']['objects'])==obstacle),
                    key=lambda r:digest({'origin_mode_target':r['sample_id']}))
        chosen+=pool[:P['fixed_TRAIN_targets']//2]
    entries=[]
    for j,r in enumerate(chosen):
        with np.load(out/r['arrays']) as a:z=a['Z'].copy()
        entries.append({'sample_id':f'batch_v1_mode_{j:02d}','slot':j,'split':'TRAIN',
            'parent_sample_id':r['sample_id'],'family_id':r['family_id'],'task':r['task'],'scene':r['scene'],
            'context_id':r['context_id'],'origin_hash':r['origin_hash'],'Z':z,'frozen_before_mode_generation':True})
    atomic_json(path,entries);return entries


def generate_mode(entry,k,cache,out,profile,deadline):
    result=fixed_generate(entry,k,cache,out,profile,deadline)
    if result['admitted']:
        path=Path(out)/'records'/f'{result["sample_id"]}.json';r=read_json(path)
        r.update(origin_hash=entry['origin_hash'],scene_parent=None,data_role='FIXED_NEW_ORIGIN_TRAIN_WORLD_MODES')
        atomic_json(path,r)
    return result


def selection(rows):
    train=[r for r in rows if r['split']=='TRAIN']
    risk=[r['sample_id'] for r in sorted(train,key=lambda r:(r['validation']['geometry']['minimum_clearance_m'],r['sample_id']))[:P['risk_TRAIN_audits']]]
    modes=[r['sample_id'] for r in sorted(train,key=lambda r:digest({'origin_mode_audit':r['sample_id']}))
           if r['data_role']=='FIXED_NEW_ORIGIN_TRAIN_WORLD_MODES' and r['sample_id'] not in risk][:P['fixed_TRAIN_audits']]
    other=[r['sample_id'] for r in sorted(train,key=lambda r:digest({'origin_audit':r['sample_id']}))
           if r['sample_id'] not in risk+modes][:P['hash_TRAIN_audits']]
    val=[r['sample_id'] for r in rows if r['split']=='NEW_VAL']
    return {'risk':risk,'fixed_modes':modes,'hash_ordered':other,'validation':val,
            'frozen_before_execution':True,'replacement_after_failure':False}


def index_for(out,old,ledger):
    out=Path(out);index=[dict(e,new_in_this_run=False) for e in old]
    for r in rows(out,ledger):
        if r['split']!='TRAIN' or r['execution_status']=='FAIL':continue
        witness=ledger.data['completed'].get('witness_'+r['sample_id']) if ledger else None
        if witness and witness['status']!='EXECUTION_VERIFIED':continue
        p=out/'records'/f'{r["sample_id"]}.json'
        index.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'split':'TRAIN',
            'record':p.relative_to(ROOT).as_posix(),'arrays':(out/r['arrays']).relative_to(ROOT).as_posix(),
            'record_sha256':file_digest(p),'arrays_sha256':file_digest(out/r['arrays']),
            'validation_level':r['validation_level'],'execution_status':r['execution_status'],
            'producer_run_id':out.name,'new_in_this_run':True})
    from ff_prior_loop.data import assert_disjoint
    assert_disjoint(index)
    return index
