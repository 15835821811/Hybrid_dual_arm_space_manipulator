import time
import shutil
import pytest
from ff_prior.storage import read_json,atomic_json,Ledger
from ff_prior.pipeline import acceptance
from ff_prior_pilot import data


def test_new_population_and_forward_admission(physical,tmp_path):
    canonical,initial,_=physical
    profile=acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])
    index=data.old_index();assert len(index)==1048
    contexts=data.contexts(canonical,initial,profile,index)
    assert len(contexts)==15 and len({c[2].state_id for c in contexts})==15
    prior=read_json(data.PREVIOUS/'frozen_tasks.json')['contexts']
    assert not {c[2].state_id for c in contexts[:12]} & {c['initial_state_id'] for c in prior}
    r=data.generate(0,contexts[0],tmp_path,profile,[],set(),time.time()+60)
    assert r['admitted'],r
    atomic_json(tmp_path/'generation'/f'{r["sample_id"]}.json',r)
    p=tmp_path/'records'/f'{r["sample_id"]}.json';row=data.check_new_record(tmp_path,p)
    assert row['sample_id'].startswith('pilot_train_') and row['family_id'].startswith('pilot_family_')
    assert row['validation_level']=='MODEL_VALID' and r['calls']['physics_steps']==0
    assert row['producer_run_id']==tmp_path.name and not row['goal_relabelled']
    near=data.generate(0,contexts[0],tmp_path/'near',profile,[('reserved',row['condition']['goals'])],set(),time.time()+60)
    assert near['reason']=='HELDOUT_GOAL_NEIGHBOUR' and not near['admitted']
    assert read_json(tmp_path/'near/tasks'/r['sample_id']/'task.json')==row['task']
    row['task']['condition']['goals'][0][0][3]+=.01;atomic_json(p,row)
    with pytest.raises(ValueError):data.check_new_record(tmp_path,p)


def test_risk_selection_covers_actual_lowest_obstacle_rows(tmp_path,monkeypatch):
    # Actual saved qualified references, not synthetic experiment outcomes.
    for name in ('records','generation','arrays','tasks'):
        shutil.copytree(data.PREVIOUS/name,tmp_path/name)
    with pytest.raises(ValueError,match='admitted context'):
        data.select_audits(tmp_path,None)
    # This saved fixture has six origins, unlike the production pilot's twelve.
    monkeypatch.setitem(data.CONTRACT,'empty_origin_contexts',6)
    chosen=data.select_audits(tmp_path,None)
    assert len(set(chosen['risk']+chosen['random_stratified']))==48
    rows=[read_json(p) for p in (tmp_path/'records').glob('*.json')]
    for context in {r['context_id'] for r in rows}:
        worst=min((r for r in rows if r['context_id']==context),key=lambda r:(r['validation']['geometry']['minimum_clearance_m'],r['sample_id']))
        assert worst['sample_id'] in chosen['risk']
    assert data.select_audits(tmp_path,None)==chosen


def test_empty_and_partial_outputs_cannot_pass_gate(tmp_path):
    from ff_prior_pilot.experiment import report
    r=report(tmp_path,Ledger(tmp_path,{}),'FINISHED')
    assert not r['ready_for_training_stage'] and not r['full_goal_achieved']
    assert not r['training_executed'] and not r['TEST_executed']


def test_live_worker_cannot_be_relaunched(tmp_path):
    from ff_prior_pilot.job import Monitor,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
