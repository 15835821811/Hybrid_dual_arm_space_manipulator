import time
from pathlib import Path
import numpy as np
import pytest
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.pipeline import acceptance
from ff_prior_coverage_train.storage import Ledger,atomic_json
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_coverage.data import generate_mode,check_new_record
from ff_prior_generalization.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_world_endpoint.protocol import ROOT,SOURCE
from ff_prior_world_endpoint.identity import select_targets
from ff_prior_world_endpoint.data import load_arrays

@pytest.fixture(scope='module')
def population():
    index=read_json(SOURCE/'training_index.json')['records']
    return index,select_targets(index)

def test_target_selection_only_uses_existing_TRAIN(population):
    index,targets=population;byid={e['sample_id']:e for e in index}
    held={e['task']['task_hash'] for e in read_json(SOURCE/'frozen_tasks.json')['entries']}
    assert len(targets)==16 and len({e['task']['task_hash'] for e in targets})==16
    assert sum(bool(e['scene']['objects']) for e in targets)==8
    for e in targets:
        assert byid[e['parent_sample_id']]['split']=='TRAIN'
        assert e['task']['task_hash'] not in held
        assert e['task']==read_json(ROOT/e['source_record'])['task']

@pytest.mark.parametrize('slot',[0,8])
def test_actual_fixed_world_correction_and_native_replay(physical,population,tmp_path,slot):
    target=population[1][slot];profile=acceptance(read_json(ROOT/'artifacts/v7_1/smoke_002/run.json')['config'])
    cache=TaskOriginSceneCache(physical[0],physical[1],profile)
    for k in range(4):
        r=generate_mode(target,k,cache,tmp_path,profile,time.time()+90)
        atomic_json(tmp_path/'generation'/f'{r["sample_id"]}.json',r)
        if r['admitted']:break
        assert r['status']=='REJECTED',r
    assert r['admitted'],r
    row=check_new_record(tmp_path,tmp_path/'records'/f'{r["sample_id"]}.json')
    assert row['task']==target['task'] and row['family_id']==target['family_id']
    assert r['minimum_reference_rms_rad']>=.005 and r['calls']['coupled']<=113
    rr,ii,cc,_=cache.get(row)
    with np.load(tmp_path/row['arrays']) as a:z=a['Z'].copy()
    outcome=execute(tmp_path/'native',row['task'],z,rr,ii,cc,profile,time.time()+120)
    assert outcome['status']=='EXECUTION_VERIFIED',outcome
    proof,steps=audit_physics(tmp_path/'native',row['task'],row['reference_hash'])
    assert proof['level']=='EXECUTION_VERIFIED' and steps>0

def test_inherited_array_and_complete_state_binding(population,tmp_path):
    index=population[0]
    arrays,rows=load_arrays(tmp_path,index,None,lambda *args,**kwargs:None)
    z,bases,conditions,ids,splits,states=arrays
    assert len(z)==13822 and sum(s=='TRAIN' for s in splits)==13774
    with np.load(SOURCE/'loaded_training_arrays.npz') as a:
        assert np.array_equal(a['Z'],z) and np.array_equal(a['bases'],bases)
    assert all(state==r['task']['contract']['initial_state_id'] for state,r in zip(states,rows))
    proof=read_json(tmp_path/'training_input_audit.json')
    assert proof['reused_verified_initializers']==13822 and proof['new_initializers']==0
    assert not proof['TEST_arrays_read']

def test_empty_report_cannot_pass_or_authorize_TEST(tmp_path):
    from ff_prior_world_endpoint.reporting import report
    result=report(tmp_path,'FINISHED')
    assert not result['world_endpoint_closure_passed'] and not result['eligible_for_TEST_protocol_review']
    assert not result['TEST_executed']
    assert '\\ff_prior_world_endpoint\\job.ps1' in (tmp_path/'REPORT.md').read_text(encoding='utf-8')
    assert '\x0c' not in (tmp_path/'REPORT.md').read_text(encoding='utf-8')

def test_unknown_work_and_running_duplicate_remain_protected(tmp_path):
    from ff_prior_world_endpoint.job import launch
    from ff_prior_coverage_train.job import Monitor
    ledger=Ledger(tmp_path,{});ledger.begin('world_gradient_000','world_gradient',60,120)
    recovered=Ledger(tmp_path,{})
    assert recovered.data['completed']['world_gradient_000']['status']=='INTERRUPTED'
    assert recovered.data['unresolved_call_accounting']==['world_gradient_000']
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})

def test_tampered_frozen_world_target_is_rejected(physical,population):
    import copy
    from ff_prior_world_endpoint.endpoint import evaluate
    from ff_prior_origin_learn_bounded.learning import load
    model,codec=load(SOURCE/'learning/goal_loss');target=copy.deepcopy(population[1][0])
    target['task']['condition']['goals'][0][0][3]+=.01
    with pytest.raises(ValueError):
        evaluate(np.zeros(codec.dim),codec,np.zeros(codec.dim),target['Z'],target['task'],
                 physical[0],physical[1],time.time()+30)
