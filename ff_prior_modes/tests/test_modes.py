"""Real TRAIN checks plus explicitly algebra-only software fixtures."""
import time
import numpy as np
import pytest
import torch
from ff_prior.storage import read_json,digest,atomic_json,Ledger
from ff_prior.pipeline import acceptance
from ff_prior_modes import data,sensitivity,learning


def entries():return data.select(read_json(data.PREVIOUS/'training_index.json')['records'])
def profile():return acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])


def test_train_parent_selection_freezes_world_goals_and_separates_family():
    a=entries();idx=read_json(data.PREVIOUS/'training_index.json')['records']
    assert digest(a)==digest(data.select(list(reversed(idx)))) and len(a)==8
    assert len({e['family_id'] for e in a})==5 and all(e['split']=='TRAIN' for e in a)
    for e in a:
        task_hash=digest(e['task'])
        for k in range(6):
            z=data.proposal(e,k)
            assert np.array_equal(z[-1],np.asarray(e['Z'])[-1]) and digest(e['task'])==task_hash
        assert np.max(abs(data.proposal(e,0)+data.proposal(e,1)-2*np.asarray(e['Z'])))<1e-14


def test_34_direction_coefficient_projection_is_identity_on_its_own_span():
    ds,_,_=sensitivity.directions();T=sensitivity.coefficient_map(np.eye(19))
    assert np.max(abs(T@ds.reshape(34,-1).T-np.eye(34)))<1e-12


def test_real_free_floating_training_sensitivity_and_step_halving(physical):
    robot,initial,_=physical;e=entries()[3];cache=data.RuntimeCache(robot,initial,profile());rr,ii,_,_=cache.get(e)
    G,r=sensitivity.build(e['task'],np.asarray(e['Z']),rr,ii,np.eye(19),time.time()+90)
    assert r['status']=='PASS',r
    assert r['calls']['coupled']==39 and G.shape==(12,323) and np.isfinite(G).all()
    assert not r['geometry_validation_or_execution_claim'] and r['surrogate_scope']['not_full_latent_gradient']


def test_actual_train_variant_is_qualified_distinct_and_goal_not_relabelled(physical,tmp_path):
    robot,initial,_=physical;e=entries()[3];cache=data.RuntimeCache(robot,initial,profile());before=digest(e['task'])
    r=data.generate(e,0,cache,tmp_path,profile(),time.time()+60)
    assert r['admitted'] and r['physical_eligible'],r
    assert r['minimum_reference_rms_rad']>=.005 and r['calls']['coupled']<=113
    row=read_json(tmp_path/'records'/(e['sample_id']+'_v0.json'))
    assert row['validation_level']=='MODEL_VALID' and row['execution_status']=='NOT_RUN'
    assert row['task']['task_hash']==e['task']['task_hash'] and digest(e['task'])==before
    assert max(row['validation']['frozen_goal_position_error_m'])<=.005
    assert row['validation']['geometry']['minimum_clearance_m']>=0


def test_local_pose_penalty_has_correct_scaling_and_zero_on_exact_clean_prediction():
    # Algebra fixture only; never contributes an experimental success count.
    torch.manual_seed(2);x=torch.randn(4,7);noise=torch.randn_like(x);a=torch.full((4,1),.02)
    noisy=a.sqrt()*x+(1-a).sqrt()*noise;v=a.sqrt()*noise-(1-a).sqrt()*x;G=torch.randn(4,12,7)
    loss,mse,pose=learning.objective(v,v,noisy,a,x,G,.05)
    assert float(loss)<1e-12 and float(pose)<1e-12
    p=(v+.1).requires_grad_();l,_,q=learning.objective(p,v,noisy,a,x,G,.05);l.backward()
    assert float(q)>0 and torch.isfinite(p.grad).all() and p.grad.abs().max()>0
    l0,m0,_=learning.objective(p,v,noisy,a,x,G,0.)
    assert torch.equal(l0,m0)


def test_incomplete_evidence_never_opens_data_or_advantage_gate(tmp_path):
    from ff_prior_modes.reporting import report
    atomic_json(tmp_path/'frozen_tasks.json',{'evaluation_entries':[],'generation_entries':[]})
    r=report(tmp_path,Ledger(tmp_path,{}),'RUNNING')
    assert not r['dataset']['quality_gate_passed'] and not r['stability_gate_passed'] and not r['exploratory_advantage_gate_passed']


def test_live_worker_cannot_be_duplicated(tmp_path):
    from ff_prior_modes.job import Monitor,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
