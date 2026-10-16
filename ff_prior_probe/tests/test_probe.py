"""Real frozen checkpoint/physics regressions, plus labeled algebra checks."""
import time
import numpy as np
import pytest
from ff_prior.storage import read_json,digest,atomic_json,Ledger
from ff_prior.pipeline import acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_modes import learning,data
from ff_prior_latent.learning import sample as old_sample
from ff_prior_probe.sampling import attenuate,propose,candidate
from ff_prior_probe.experiment import targets,PREVIOUS


def setup(physical,slot=3):
    robot,initial,_=physical
    profile=acceptance(read_json('artifacts/v7_1/smoke_002/run.json')['config'])
    e=targets()[slot];cache=data.RuntimeCache(robot,initial,profile);rr,ii,cc,_=cache.get(e)
    base=public_goal_initializer(e['task'],rr,ii)
    bundle=learning.load(PREVIOUS/'learning/goal_loss')
    return e,rr,ii,cc,profile,base,bundle


def test_frozen_train_scope_and_no_teacher_controls_in_entry():
    es=targets();assert len(es)==8 and all(e['split']=='TRAIN' for e in es)
    assert len({e['family_id'] for e in es})==5 and all('Z' not in e and 'source_arrays' not in e for e in es)


def test_attenuation_algebra_and_float64_endpoints():
    # Pure algebra fixture, not an experimental result.
    c=np.ones((19,17));r=c+.04
    assert np.array_equal(attenuate(r,c,0.),c) and np.array_equal(attenuate(r,c,1.),r)
    assert np.allclose(attenuate(r,c,.25)-c,.25*(r-c),atol=1e-16)
    with pytest.raises(ValueError):attenuate(r,c,.7)
    with pytest.raises(ValueError):attenuate(r*np.nan,c,.5)


def test_frozen_checkpoint_ddim_unscaled_samples_match_across_amplitudes(physical):
    e,rr,ii,cc,profile,base,bundle=setup(physical)
    # Capture algebraic sampler output only; no physical-success assertion.
    identity=lambda z,i:(z,{'iteration':i})
    original,old=old_sample(bundle,base,e['task'],20261119,identity)
    probes=[]
    for method,a in [('g025',.25),('g050',.5),('g100',1.)]:
        z,r=propose(method,0,e['task'],base,20261119,bundle,identity)
        p=r['residual_probe'];probes.append(p)
        assert np.array_equal(p['raw_unscaled'],original)
        assert np.array_equal(z,attenuate(original,p['mean'],a))
        assert r['actual_network_calls']==16 and max(s['decomposition_residual_after'] for s in r['steps'])<1e-12
    assert len({p['raw_unscaled_hash'] for p in probes})==1
    before=bundle[0].calls
    z,_=propose('mean_zero',3,e['task'],base,999,bundle,identity)
    assert bundle[0].calls==before and np.array_equal(z,probes[0]['mean'])


def test_real_attenuated_candidate_keeps_frozen_goal_and_all_physical_model_gates(physical):
    e,rr,ii,cc,profile,base,bundle=setup(physical)
    frozen=digest(e['task'])
    r=candidate('g050',0,e['task'],base,20261119,bundle,rr,ii,cc,profile,time.time()+60)
    assert r['status']=='DONE' and r['eligible'],r
    assert r['model_calls']['coupled']<=113 and r['integration_intervals']<=15872
    assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']
    assert r['actual_network_calls']==16 and digest(e['task'])==frozen
    assert max(r['validation']['frozen_goal_position_error_m'])<=.005
    assert r['validation']['geometry']['minimum_clearance_m']>=0


def test_repeated_mean_is_not_counted_as_stochastic_diversity():
    from ff_prior_probe.reporting import diversity
    a=np.zeros((19,17));r=diversity([a]*5)
    assert r=={'distinct':1,'pairwise_rms_rad':0.}
    assert diversity([a,a+.006])['distinct']==2


def test_incomplete_records_never_become_a_mechanism_signal(tmp_path):
    from ff_prior_probe.reporting import report
    atomic_json(tmp_path/'frozen_tasks.json',{'evaluation_entries':targets()})
    r=report(tmp_path,Ledger(tmp_path,{}),'RUNNING')
    assert not any(r['attenuation_signal'].values()) and not r['full_goal_achieved']


def test_live_probe_worker_refuses_duplicate_launch(tmp_path):
    from ff_prior_probe.job import Monitor,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
