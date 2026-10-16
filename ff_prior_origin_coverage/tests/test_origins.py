"""Actual physics, frozen split isolation and reserved-work semantics."""
import time
import numpy as np
import pytest
from ff_prior.storage import read_json,digest
from ff_prior_pilot_train.storage import atomic_json,Ledger
from ff_prior.pipeline import acceptance
from ff_prior_origin_scene.runtime import TaskOriginSceneCache
from ff_prior_generalization.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_origin_coverage import data


def profile():return acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])


def generate(physical,tmp,split='TRAIN',slot=0,attempt=0,guard=(),valq=()):
    robot,initial,_=physical
    return data.generate(split,slot,attempt,robot,initial,profile(),guard,set(),valq,tmp,time.time()+60)


def commit(tmp,result):
    ledger=Ledger(tmp,{})
    ledger.begin(result['sample_id'],'forward_generation',60,4)
    atomic_json(tmp/'generation'/f'{result["sample_id"]}.json',result)
    ledger.finish({'status':result['status']},result['calls'])
    return ledger


def test_every_forward_slot_has_a_new_origin(physical):
    robot,_,_=physical
    q=[data.proposal_origin(robot,'TRAIN',slot,0) for slot in range(512)]
    assert len({digest(v) for v in q})==512
    assert len({digest(data.proposal_origin(robot,'NEW_VAL',slot,0)) for slot in range(8)})==8
    assert not {digest(v) for v in q}&{digest(data.proposal_origin(robot,'NEW_VAL',slot,0)) for slot in range(8)}


def test_real_forward_admission_and_fixed_goal_mode(physical,tmp_path):
    r=generate(physical,tmp_path);assert r['admitted'],r
    ledger=commit(tmp_path,r);row=data.rows(tmp_path,ledger)[0]
    assert r['calls']['coupled']<=4 and r['calls']['physics_steps']==0
    with np.load(tmp_path/row['arrays']) as a:z=a['Z'].copy()
    entry={'sample_id':'origin_mode_00','slot':0,'split':'TRAIN','parent_sample_id':row['sample_id'],
           'family_id':row['family_id'],'task':row['task'],'scene':row['scene'],
           'context_id':row['context_id'],'origin_hash':row['origin_hash'],'Z':z}
    robot,initial,_=physical;cache=TaskOriginSceneCache(robot,initial,profile())
    mode=data.generate_mode(entry,0,cache,tmp_path,profile(),time.time()+60)
    assert mode['admitted'],mode
    atomic_json(tmp_path/'generation'/f'{mode["sample_id"]}.json',mode)
    fixed=data.check_new_record(tmp_path,tmp_path/'records'/f'{mode["sample_id"]}.json')
    assert fixed['task']==row['task'] and mode['minimum_reference_rms_rad']>=.005
    assert mode['calls']['coupled']<=113


def test_scratch_scene_is_actually_qualified(physical,tmp_path):
    r=generate(physical,tmp_path,slot=3)
    assert r['admitted'],r
    row=read_json(tmp_path/'records'/f'{r["sample_id"]}.json')
    assert row['scene']['objects'] and row['scene_parent'] is None
    assert r['scene_origin_audit']['complete_state_fields_preserved']
    assert r['validation_result']['eligible'] and r['calls']['coupled']<=4


def test_validation_goal_initial_neighbour_and_tampered_task_refused(physical,tmp_path):
    first=generate(physical,tmp_path/'first',split='NEW_VAL');assert first['admitted'],first
    row=read_json(tmp_path/'first/records'/f'{first["sample_id"]}.json')
    assert not row['training_eligible'] and row['split']=='NEW_VAL'
    guard=[{'sample_id':'reserved','goals':row['condition']['goals']}]
    blocked=generate(physical,tmp_path/'guard',split='NEW_VAL',guard=guard)
    assert blocked['reason']=='PROTECTED_WORLD_GOAL_NEIGHBOUR' and not blocked['admitted']
    assert read_json(tmp_path/'guard/tasks'/first['sample_id']/'task.json')==row['task']
    q=data.proposal_origin(physical[0],'TRAIN',0,0)
    blocked=generate(physical,tmp_path/'initial',valq=[q])
    assert blocked['reason']=='FROZEN_VALIDATION_INITIAL_NEIGHBOUR' and not blocked['admitted']
    commit(tmp_path/'first',first)
    row['task']['condition']['goals'][0][0][3]+=.01
    atomic_json(tmp_path/'first/records'/f'{first["sample_id"]}.json',row)
    with pytest.raises(ValueError):data.rows(tmp_path/'first',Ledger(tmp_path/'first',{}))


def test_actual_native_torque_and_independent_replay(physical,tmp_path):
    r=generate(physical,tmp_path);assert r['admitted'],r
    row=read_json(tmp_path/'records'/f'{r["sample_id"]}.json')
    cache=TaskOriginSceneCache(physical[0],physical[1],profile());rr,ii,cc,_=cache.get(row)
    with np.load(tmp_path/row['arrays']) as a:z=a['Z'].copy()
    result=execute(tmp_path/'native',row['task'],z,rr,ii,cc,profile(),time.time()+60)
    assert result['status']=='EXECUTION_VERIFIED',result
    proof,steps=audit_physics(tmp_path/'native',row['task'],row['reference_hash'])
    assert proof['level']=='EXECUTION_VERIFIED' and steps>0


def test_interrupted_work_and_live_duplicate_are_not_replayed(tmp_path):
    from ff_prior_origin_coverage.job import launch
    from ff_prior_coverage_train.job import Monitor
    ledger=Ledger(tmp_path,{})
    ledger.begin('unknown','forward_generation',60,4)
    recovered=Ledger(tmp_path,{})
    assert recovered.data['completed']['unknown']['status']=='INTERRUPTED'
    assert recovered.data['unresolved_call_accounting']==['unknown']
    assert not recovered.begin('unknown','forward_generation',60,4)
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
