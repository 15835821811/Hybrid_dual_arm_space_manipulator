"""Real bounded generation, frozen-target integrity, and recovery regressions."""
import time
import numpy as np
import pytest
from ff_prior.storage import atomic_json,read_json,Ledger,digest
from ff_prior.pipeline import acceptance
from ff_prior_expand import data


def test_real_forward_generation_freezes_target_and_admits_with_full_proof(physical,tmp_path):
    canonical,initial,_=physical
    profile=acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])
    index=data.old_index();contexts=data.contexts(canonical,initial,profile,index)
    assert len(contexts)==9 and sum(bool(c[0]['scene']['objects']) for c in contexts)==3
    r=data.generate(0,contexts[0],tmp_path,profile,[],set(),time.time()+60)
    assert r['admitted'],r
    atomic_json(tmp_path/'generation'/f'{r["sample_id"]}.json',r)
    path=tmp_path/'records'/f'{r["sample_id"]}.json'
    row=data.check_new_record(tmp_path,path)
    assert r['calls']['coupled'] in (3,4) and r['calls']['physics_steps']==0
    assert row['execution_status']=='NOT_RUN' and row['validation_level']=='MODEL_VALID'
    assert all(row['validation']['gates'].values()) and max(row['validation']['frozen_goal_position_error_m'])<=.005
    before=digest(row['task']);proof=read_json(tmp_path/'tasks'/row['sample_id']/'task.json')
    assert digest(proof)==before and not row['goal_relabelled']
    row['task']['condition']['goals'][0][0][3]+=.01;atomic_json(path,row)
    with pytest.raises(ValueError,match='relabel'):data.check_new_record(tmp_path,path)


def test_admission_requires_committed_generation_not_only_record_marker(physical,tmp_path):
    canonical,initial,_=physical;profile=acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])
    ctx=data.contexts(canonical,initial,profile,data.old_index())[0]
    r=data.generate(0,ctx,tmp_path,profile,[],set(),time.time()+60);assert r['admitted']
    atomic_json(tmp_path/'generation'/f'{r["sample_id"]}.json',r)
    with pytest.raises(ValueError,match='unknown generation'):
        data.check_new_record(tmp_path,tmp_path/'records'/f'{r["sample_id"]}.json',Ledger(tmp_path,{}))


def test_duplicate_and_heldout_guards_reject_without_retarget(physical,tmp_path):
    canonical,initial,_=physical;profile=acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])
    ctx=data.contexts(canonical,initial,profile,data.old_index())[0]
    r=data.generate(0,ctx,tmp_path/'first',profile,[],set(),time.time()+60);assert r['admitted']
    task=read_json(tmp_path/'first/tasks'/r['sample_id']/'task.json')
    dup=data.generate(0,ctx,tmp_path/'dup',profile,[],{r['reference_hash']},time.time()+60)
    near=data.generate(0,ctx,tmp_path/'near',profile,[('reserved',task['condition']['goals'])],set(),time.time()+60)
    assert not dup['admitted'] and dup['reason']=='DUPLICATE_REFERENCE'
    assert not near['admitted'] and near['reason']=='HELDOUT_GOAL_NEIGHBOUR'
    assert read_json(tmp_path/'near/tasks'/r['sample_id']/'task.json')==task


def test_numeric_finite_and_frozen_checkpoint_still_usable(physical):
    from ff_prior_expand.sampling import candidate
    from ff_prior_modes.learning import load
    from ff_prior_modes.data import RuntimeCache
    from ff_prior.guidance import public_goal_initializer
    canonical,initial,_=physical;profile=acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])
    e=next(e for e in data.old_index() if e['sample_id']=='complex_train_00_static_detour')
    row=read_json(data.ROOT/e['record']);rr,ii,cc,_=RuntimeCache(canonical,initial,profile).get(row)
    r=candidate('frozen_diffusion',0,row['task'],public_goal_initializer(row['task'],rr,ii),20261203,load(data.PREVIOUS/'learning/goal_loss'),rr,ii,cc,profile,time.time()+60)
    assert r['status']=='DONE' and r['actual_network_calls']==16
    assert np.isfinite(r['Z']).all() and np.all(r['Z']>=rr.lower) and np.all(r['Z']<=rr.upper)
    assert max(s['decomposition_residual_after'] for s in r['sampling']['steps'])<1e-12


def test_empty_output_never_authorizes_pilot(tmp_path):
    from ff_prior_expand.reporting import report
    r=report(tmp_path,Ledger(tmp_path,{}),'FINISHED')
    assert not r['PILOT_ELIGIBLE'] and not r['data_gate_passed'] and not r['full_goal_achieved']


def test_existing_worker_blocks_duplicate_launch(tmp_path):
    from ff_prior_expand.job import Monitor,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
