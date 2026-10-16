import time
import numpy as np
import pytest
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_pilot_train.storage import Ledger,atomic_json
from ff_prior.pipeline import acceptance
from ff_prior_generalization.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_scene_guard.runtime import at_scene_origin
from ff_prior_origin_fill import data
from ff_prior_origin_fill.protocol import SOURCE,SLOTS


def profile():return acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])


def propose(physical,out,slot,attempt,guard=(),valq=()):
    return data.generate(slot,attempt,physical[0],physical[1],profile(),guard,set(),valq,out,time.time()+60)


@pytest.fixture(scope='module')
def filled(physical,tmp_path_factory):
    out=tmp_path_factory.mktemp('actual_fill')
    for attempt in range(8):
        r=propose(physical,out,171,attempt)
        if r['admitted']:
            atomic_json(out/'generation'/f'{r["sample_id"]}.json',r)
            return out,r,data.check_new_record(out,out/'records'/f'{r["sample_id"]}.json')
    pytest.fail('frozen test slot produced no physically qualified proposal')


def test_new_namespace_and_actual_forward_admission(filled):
    out,r,row=filled
    assert row['sample_id'].startswith('origin_fill_train_') and row['scene']['objects']
    assert row['scene_parent'] is None and not row['source_failed_task_reused']
    assert r['calls']['coupled']<=4 and r['calls']['physics_steps']==0
    assert r['scene_origin_audit']['initial_geometry_checked_before_transfer']
    assert all(r['scene_origin_audit']['scene_transfer']['checks'].values())
    assert row['execution_status']=='NOT_RUN' and row['validation_level']=='MODEL_VALID'


def test_actual_initial_collision_is_a_physical_rejection(physical,tmp_path):
    for attempt in range(8):
        for slot in SLOTS:
            r=propose(physical,tmp_path/str(slot)/str(attempt),slot,attempt)
            if r.get('reason')=='INITIAL_GEOMETRY_REJECTED':
                assert r['status']=='REJECTED' and not r['admitted']
                assert r['initial_geometry']['minimum_clearance_m']<0
                assert r['calls']['coupled']==1 and r['calls']['physics_steps']==0
                return
    pytest.fail('no actual initial collision exercised within frozen test quota')


def test_frozen_goal_initial_neighbour_and_tamper_are_refused(physical,filled,tmp_path):
    out,r,row=filled;slot,attempt=r['slot'],r['attempt']
    guard=[{'sample_id':'reserved','goals':row['condition']['goals']}]
    rejected=propose(physical,tmp_path/'goal',slot,attempt,guard)
    assert rejected['reason']=='PROTECTED_WORLD_GOAL_NEIGHBOUR' and not rejected['admitted']
    assert read_json(tmp_path/'goal/tasks'/r['sample_id']/'task.json')==row['task']
    q=data.origin(physical[0],slot,attempt)
    rejected=propose(physical,tmp_path/'initial',slot,attempt,valq=[q])
    assert rejected['reason']=='FROZEN_VALIDATION_INITIAL_NEIGHBOUR' and rejected['calls']['coupled']==0
    bad=dict(row);bad['reference_hash']='tampered'
    path=out/'records'/f'{r["sample_id"]}.json';original=path.read_bytes()
    try:
        atomic_json(path,bad)
        with pytest.raises(ValueError,match='reference mismatch'):
            data.check_new_record(out,path)
    finally:path.write_bytes(original)


def test_actual_new_native_execution_and_independent_replay(physical,filled):
    out,r,row=filled;rr,ii,cc,_=at_scene_origin(physical[0],physical[1],row['scene'],row['condition']['initial_features'])
    with np.load(out/row['arrays']) as a:z=a['Z'].copy()
    result=execute(out/'native',row['task'],z,rr,ii,cc,profile(),time.time()+60)
    assert result['status']=='EXECUTION_VERIFIED',result
    proof,steps=audit_physics(out/'native',row['task'],row['reference_hash'])
    assert proof['level']=='EXECUTION_VERIFIED' and steps>0


def test_parent_failed_gate_is_not_silently_unblocked(tmp_path):
    from ff_prior_origin_fill.experiment import report
    before=file_digest(SOURCE/'summary.json');source=read_json(SOURCE/'summary.json')
    result=report(tmp_path,Ledger(tmp_path,{}),'FINISHED')
    assert source['ready_for_training_stage'] is False
    assert result['parent_original_gate_passed'] is False and not result['ready_for_training_stage']
    assert result['merged_forward']==503 and file_digest(SOURCE/'summary.json')==before


def test_reserved_unknown_work_and_live_duplicate_are_not_replayed(tmp_path):
    from ff_prior_origin_fill.job import launch
    from ff_prior_coverage_train.job import Monitor
    ledger=Ledger(tmp_path,{})
    ledger.begin('unknown','generation',1200,72)
    recovered=Ledger(tmp_path,{})
    assert recovered.data['completed']['unknown']['status']=='INTERRUPTED'
    assert not recovered.begin('unknown','generation',1200,72)
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(SOURCE,tmp_path,True)
        monitor.finish({'status':'FINISHED'})
