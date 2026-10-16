"""Every retry is a new immutable task; never relabel a failed original task."""
from pathlib import Path
import time
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from ff_prior.proposal_generator import propose,motion_valid,quick_gates
from ff_prior.coupled_predictor import predict_coupled
from ff_prior.spline import Reference
from ff_prior.scenes import SceneSpec,Obstacle
from ff_prior_complex.tasks import task_for
from ff_prior_scene_guard.runtime import at_scene_origin,InitialGeometryRejected
from ff_prior_feasible.validation import validate
from ff_prior_generalization.data import neighbours
from ff_prior_coverage.data import check_new_record
from .protocol import ROOT,SLOTS,PROTOCOL as P


def origin(canonical,slot,attempt):
    rng=np.random.default_rng(P['seed']+slot*1019+attempt*100003)
    jitter=P['origin_jitter_rad'][slot%6]
    return canonical.spec.planner_zero+rng.uniform(-jitter,jitter,17)


def scene_for(prediction,slot,attempt,family):
    rng=np.random.default_rng(P['seed']+slot*997+attempt*104729)
    direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    center=prediction['ee'][128,slot%2,:3,3]+P['scene_offset_m'][(slot//3)%3]*direction
    radius=P['scene_radii_m'][slot%3]
    return SceneSpec((Obstacle(f'fill_sphere_{slot}_{attempt}','sphere',tuple(center),(radius,0.,0.)),),family)


def generate(slot,attempt,canonical,canonical_initial,profile,guard,existing_origins,valq,out,deadline):
    if slot not in SLOTS or not 0<=attempt<8:raise ValueError('outside frozen supplementary quota')
    out=Path(out);sid=f'origin_fill_train_{slot:03d}_a{attempt}';family=f'origin_fill_family_{slot:03d}'
    started=time.perf_counter();deadline=min(deadline,time.time()+P['candidate_wall_s']);tracked=[]
    r={'sample_id':sid,'family_id':family,'split':'TRAIN','slot':slot,'attempt':attempt,
       'status':'REJECTED','admitted':False,'goal_relabelled':False,'parent_sample_id':None,
       'coverage_source_slot':slot,'source_failed_task_reused':False,'scene_parent':None,
       'initialization_costs_in_wall_time':True,'initialization_counter_scope':'returned runtimes; internal parent initialization not aggregated'}
    try:
        q0=origin(canonical,slot,attempt);r['q17_origin']=q0;r['origin_hash']=digest(q0)
        if r['origin_hash'] in existing_origins:r['reason']='DUPLICATE_ORIGIN';return r
        if any(np.max(abs(q0-v))<.02 for v in valq):r['reason']='FROZEN_VALIDATION_INITIAL_NEIGHBOUR';return r
        rr,ii,cc,proof=at_scene_origin(canonical,canonical_initial,SceneSpec(),q0)
        tracked.append((rr,rr.calls.copy()));r['empty_origin_audit']=proof
        ref,provenance=propose(rr,P['proposal_family_offset']+slot*8+attempt,(slot//6)%5,P['seed'])
        r['provenance']=provenance
        if ref is None:r['reason']='NO_MOTION_DURATION';return r
        scale=P['motion_scales'][(slot//30)%6];z=q0+scale*(ref.z-q0);r['motion_scale']=scale
        ref=next((v for duration in (10.,15.,20.,25.,29.5)
                  if motion_valid(rr,v:=Reference(z,q0,duration))),None)
        if ref is None:r['reason']='SCALED_DURATION_SET_EXHAUSTED';return r
        r['quick_gates']=quick_gates(rr,ref)
        if not all(r['quick_gates'].values()):r['reason']='QUICK_GATES';return r
        pred=predict_coupled(ii,ref,rr,256,deadline)
        scene=scene_for(pred,slot,attempt,family);r['scene']=scene.to_dict()
        rr,ii,cc,proof=at_scene_origin(canonical,canonical_initial,scene,q0)
        tracked.append((rr,rr.calls.copy()));r['scene_origin_audit']=proof
        task=task_for(rr,ii,cc,profile,pred['ee'][-1],ref.duration)
        atomic_json(out/'tasks'/sid/'task.json',task);atomic_npz(out/'tasks'/sid/'initial_state.npz',values=ii.values)
        r.update(task_hash=task['task_hash'],reference_hash=ref.reference_hash,
            complete_initial_state_sha256=file_digest(out/'tasks'/sid/'initial_state.npz'))
        r['protected_goal_neighbours']=neighbours(task['condition']['goals'],guard)
        if r['protected_goal_neighbours']:r['reason']='PROTECTED_WORLD_GOAL_NEIGHBOUR';return r
        checks=validate(ref.z,task,rr,ii,cc,profile,deadline);r['validation_result']=checks
        if not checks['eligible']:r['reason']='INDEPENDENT_VALIDATION_REJECTED';return r
        calls={k:sum(robot.calls[k]-before[k] for robot,before in tracked) for k in rr.calls}
        if calls['coupled']>4 or any(calls[k] for k in ('physics_steps','torque_rollout','rrt','sqp')):raise RuntimeError('FORBIDDEN_FORWARD_CALLS')
        if time.time()>deadline:raise RuntimeError('GENERATION_WALL_EXHAUSTED')
        atomic_npz(out/'arrays'/f'{sid}.npz',**ref.arrays())
        row={'sample_id':sid,'family_id':family,'split':'TRAIN','slot':slot,'attempt':attempt,
            'coverage_source_slot':slot,'condition':task['condition'],'task':task,'scene':scene.to_dict(),
            'origin_hash':r['origin_hash'],'context_id':r['origin_hash'],'parent_sample_id':None,'scene_parent':None,
            'validation_level':'MODEL_VALID','execution_status':'NOT_RUN','training_eligible':True,
            'validation':checks['validation'],'reference_hash':ref.reference_hash,'arrays':f'arrays/{sid}.npz',
            'arrays_sha256':file_digest(out/'arrays'/f'{sid}.npz'),'complete_initial_state':f'tasks/{sid}/initial_state.npz',
            'initial_state_sha256':r['complete_initial_state_sha256'],'goal_source':'single forward before independent validation',
            'goal_relabelled':False,'source_failed_task_reused':False,'producer_run_id':out.name,
            'data_role':'ORIGIN_COVERAGE_FILL_FORWARD','generation_proof':f'generation/{sid}.json'}
        atomic_json(out/'records'/f'{sid}.json',row);r.update(status='ADMITTED',admitted=True)
    except InitialGeometryRejected as error:
        r.update(status='REJECTED',reason='INITIAL_GEOMETRY_REJECTED',initial_geometry=error.geometry)
    except Exception as error:
        import traceback
        r.update(status='ERROR',reason=repr(error),traceback=traceback.format_exc())
    finally:
        r['calls']={k:sum(robot.calls[k]-before[k] for robot,before in tracked) for k in canonical.calls}
        r['wall_s']=time.perf_counter()-started
    return r


def rows(out,ledger=None):
    paths=sorted((Path(out)/'records').glob('*.json'))
    if ledger:paths=[p for p in paths if ledger.data['completed'].get(p.stem,{}).get('status')=='ADMITTED']
    return [check_new_record(out,p,ledger) for p in paths]


def merged_index(out,old,ledger):
    out=Path(out)
    index=[dict(e,new_in_this_run=False) for e in old]
    for r in rows(out,ledger):
        if r['execution_status']=='FAIL':continue
        witness=ledger.data['completed'].get('witness_'+r['sample_id'])
        if witness and witness['status']!='EXECUTION_VERIFIED':continue
        path=out/'records'/f'{r["sample_id"]}.json'
        index.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'split':'TRAIN',
            'record':path.relative_to(ROOT).as_posix(),'arrays':(out/r['arrays']).relative_to(ROOT).as_posix(),
            'record_sha256':file_digest(path),'arrays_sha256':file_digest(out/r['arrays']),
            'validation_level':r['validation_level'],'execution_status':r['execution_status'],
            'producer_run_id':out.name,'new_in_this_run':True})
    from ff_prior_loop.data import assert_disjoint
    assert_disjoint(index);return index
