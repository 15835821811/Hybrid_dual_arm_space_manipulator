"""Real saved models/runtime and fail-closed identity checks; no mock physics."""
import copy
import time
import numpy as np
import pytest
from ff_prior.storage import read_json, digest
from ff_prior.pipeline import audit, acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_expand.sampling import candidate as original_candidate
from ff_prior_modes.data import RuntimeCache
from ff_prior_coverage_train.learning import load
from ff_prior_generalization.protocol import ROOT, SOURCE, BASE
from ff_prior_generalization import data
from ff_prior_generalization.sampling import candidate
from ff_prior_generalization.audit_results import check_candidate
from ff_prior_generalization.reporting import paired_quality
from ff_prior_generalization.job import status


@pytest.fixture(scope='module')
def runtime():
    config=read_json(BASE/'run.json')['config']
    _,rr,ii,_=audit(config);profile=acceptance(config)
    return rr,ii,profile


def test_new_teacher_actual_physics_and_saved_state(runtime,tmp_path):
    rr,ii,p=runtime
    r=data.generate(0,0,rr,ii,p,[],tmp_path,time.time()+60)
    assert r['status']=='FROZEN',r.get('reason')
    assert r['witness_model']['eligible'] and not r['training_eligible']
    assert r['calls']['coupled']>0 and not r['calls']['torque_rollout']
    assert data.assert_isolated(r,[])
    assert data.neighbours(r['task']['condition']['goals'],[{'sample_id':'old','goals':r['task']['condition']['goals']}])==['old']
    bad=copy.deepcopy(r);bad['parent_sample_id']='TRAIN_source'
    with pytest.raises(ValueError):data.assert_isolated(bad,[])


def test_new_scratch_scene_with_unchanged_robot(runtime,tmp_path):
    rr,ii,p=runtime;deadline=time.time()+120;attempts=[]
    for k in range(8):
        r=data.generate(6,k,rr,ii,p,[],tmp_path,deadline);attempts.append(r)
        if r['status']=='FROZEN':break
        if time.time()>=deadline:break
    assert r['status']=='FROZEN',[v.get('reason') for v in attempts]
    assert r['scene_transfer']['status']=='PASS' and r['witness_model']['eligible']
    assert r['scene']['objects'] and not r['unknown_screen_call_accounting']
    assert all(r['scene_transfer']['checks'].values())


@pytest.mark.parametrize('method',['structured','expanded_diffusion'])
def test_recording_preserves_original_sampler_and_audit(runtime,method):
    rr,ii,p=runtime;entry=read_json(SOURCE/'frozen_tasks.json')['DEV'][0]
    cache=RuntimeCache(rr,ii,p);robot,initial,collision,_=cache.get(entry)
    base=public_goal_initializer(entry['task'],robot,initial)
    bundle=None if method=='structured' else load(SOURCE/'learning/goal_loss')
    a=original_candidate(method,0,entry['task'],base,20261619,bundle,robot,initial,collision,p,time.time()+30)
    b=candidate(method,0,entry['task'],base,20261619,bundle,robot,initial,collision,p,time.time()+30)
    assert a['status']==b['status']=='DONE'
    assert np.array_equal(a['Z'],b['Z']) and a['eligible']==b['eligible']
    assert a['model_calls']==b['model_calls'] and a['actual_network_calls']==b['actual_network_calls']
    check_candidate(b,entry['task'],20261619,method,0)
    bad=copy.deepcopy(b);bad['eligible']=not b['eligible']
    with pytest.raises(AssertionError):check_candidate(bad,entry['task'],20261619,method,0)


def test_task_bootstrap_and_live_status(tmp_path):
    # Arithmetic-only synthetic inputs test aggregation, never experimental passes.
    rows={str(i):{m:{'pre_median_norm':v,'completed':15} for m,v in
          [('expanded_diffusion',1.),('structured',2.),('learned_mean',1.),('frozen_diffusion',1.5)]} for i in range(12)}
    result=paired_quality(rows)
    assert result['structured']['pre_registered_quality_pass']
    assert not result['learned_mean']['pre_registered_quality_pass']
    import os
    atomic_json(tmp_path/'status.json',{'status':'RUNNING','pid':os.getpid(),'heartbeat':time.time()})
    assert status(tmp_path)['owner_alive'] is True
