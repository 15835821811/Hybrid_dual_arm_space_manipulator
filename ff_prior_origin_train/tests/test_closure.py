"""Actual checkpoint, propagation, optimizer, DDIM and native replay checks."""
import copy
import os
import time
from pathlib import Path
import numpy as np
import pytest
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_coverage_train.storage import atomic_json,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_origin_scene import TaskOriginSceneCache
from ff_prior_modes import data as loader
from ff_prior.diffusion import weight_hash
from ff_prior_generalization.sampling import candidate
from ff_prior_generalization.audit_results import check_candidate
from ff_prior_generalization.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_origin_train.protocol import ROOT,BASE,SOURCE,OLD,PRODUCERS
from ff_prior_origin_train import learning,sensitivity,mean
from ff_prior_origin_train.experiment import require_data
from ff_prior_origin_train.job import launch,status


@pytest.fixture(scope='module')
def actual(tmp_path_factory):
    out=tmp_path_factory.mktemp('actual_origin_training')
    physical,rr,ii,_=audit(read_json(BASE/'run.json')['config']);profile=acceptance(read_json(BASE/'run.json')['config'])
    cache=TaskOriginSceneCache(rr,ii,profile)
    oldindex=read_json(OLD/'training_index.json')['records']
    oldrows=[read_json(ROOT/e['record']) for e in oldindex]
    with np.load(OLD/'loaded_training_arrays.npz') as a:oz,ob=a['Z'].copy(),a['bases'].copy()
    oldarrays=(oz,ob,[r['condition'] for r in oldrows],[e['sample_id'] for e in oldindex],
               [e['split'] for e in oldindex],[r['task']['contract']['initial_state_id'] for r in oldrows])
    codec=learning.fit_codec(out/'codec',oldarrays)
    index,coverage,validation=require_data(SOURCE)
    selected=[e for e in oldindex if e['split']=='TRAIN'][:6]
    selected += [e for e in index if e['split']=='TRAIN' and e.get('producer_run_id') in PRODUCERS and e['sample_id'] not in set(oldarrays[3]) and read_json(ROOT/e['record'])['data_role']=='INDEPENDENT_ORIGIN_TRAIN'][:6]
    selected += [e for e in oldindex if e['split']=='VAL'][:2]
    arrays,rows=loader.arrays(selected,cache,out/'small',lambda *a,**k:None)
    newids=[e['sample_id'] for e in selected if e.get('producer_run_id') in PRODUCERS]
    return out,cache,profile,arrays,rows,codec,newids


def test_full_544_group_is_not_nine_new_flags():
    index,coverage,validation=require_data(SOURCE)
    assert len(coverage)==544 and sum(e['new_in_this_run'] for e in index)==9
    assert not set(coverage)&{e['sample_id'] for e in validation['entries']}
    old=read_json(ROOT/'artifacts/v7_1/qualified_origin_coverage_001/summary.json')
    assert not old['ready_for_training_stage']


def test_real_balanced_updates_cache_and_resumable_counts(actual,monkeypatch):
    out,cache,profile,arrays,rows,codec,newids=actual;ledger=Ledger(out/'sensitivity',{})
    matrices,proof=sensitivity.prepare(out/'sensitivity',arrays,rows,codec,cache,ledger,time.time()+180,lambda *a,**k:None)
    assert proof['status']=='PASS' and len(matrices)==12
    assert ledger.data['calls']['cached_sensitivity.coupled']==6*39
    assert ledger.data['calls']['sensitivity.coupled']==6*39
    monkeypatch.setitem(learning.CONTRACT,'updates',6);monkeypatch.setitem(learning.CONTRACT,'batch',4)
    trained=learning.train(out/'trained',arrays,codec,matrices,'goal_loss',proof,lambda *a,**k:None,time.time()+120,newids)
    old,_=learning.frozen_learning.load(learning.FROZEN)
    assert trained['status']=='PASS' and trained['initial_weight_hash']==weight_hash(old)
    assert trained['trained_weight_hash']!=weight_hash(old)
    assert trained['group_draw_counts']=={'old':12,'coverage':12} and trained['all_TRAIN_rows_seen']
    assert set(trained['train_draw_counts'].values())=={2} and trained['VAL_gradient_updates']==0
    before={str(p):file_digest(p) for p in (out/'trained').rglob('*') if p.is_file()}
    again=learning.train(out/'trained',arrays,codec,matrices,'goal_loss',proof,lambda *a,**k:None,time.time()+120,newids)
    assert again==trained and before=={str(p):file_digest(p) for p in (out/'trained').rglob('*') if p.is_file()}


def test_cache_rejects_real_reference_tamper(actual,tmp_path):
    out,cache,p,arrays,rows,codec,newids=actual;i=0;key=rows[i]['task']['task_hash']
    modified=list(arrays);modified[0]=arrays[0].copy();modified[0][0,0,0]+=1e-5
    with pytest.raises(ValueError,match='anchor/proof mismatch'):
        sensitivity.import_anchor(tmp_path,key,i,modified,codec,Ledger(tmp_path,{}))


def test_weighted_mean_train_only_and_same_basis(actual,tmp_path):
    out,cache,p,arrays,rows,codec,newids=actual
    fitted=mean.fit(tmp_path,arrays,codec,newids)
    assert len(fitted.state['fit_ids'])==12 and not set(fitted.state['fit_ids'])&set(arrays[3][-2:])
    assert np.array_equal(fitted.basis,codec.basis)
    perturbed=list(arrays);perturbed[0]=arrays[0].copy();perturbed[0][-2:]+=1.
    again=mean.fit(tmp_path,perturbed,codec,newids)
    assert digest(again.state)==digest(fitted.state)
    with pytest.raises(ValueError,match='not TRAIN'):
        mean.fit(tmp_path,arrays,codec,newids+[arrays[3][-1]])


def test_actual_v_ddim_world_goal_guidance_and_native_replay(actual,tmp_path):
    out,cache,p,arrays,rows,codec,newids=actual
    entry=read_json(OLD/'frozen_tasks.json')['DEV'][0];task=entry['task']
    rr,ii,cc,_=cache.get(entry);base=public_goal_initializer(task,rr,ii)
    bundle=learning.frozen_learning.load(learning.FROZEN)
    r=candidate('expanded_diffusion',0,task,base,20261919,bundle,rr,ii,cc,p,time.time()+30)
    assert r['status']=='DONE' and r['eligible'],r.get('reason')
    check_candidate(r,task,20261919,'expanded_diffusion',0)
    assert r['actual_network_calls']==16 and np.isfinite(r['Z']).all()
    result=execute(tmp_path,task,r['Z'],rr,ii,cc,p,time.time()+60)
    assert result['status']=='EXECUTION_VERIFIED'
    report,steps=audit_physics(tmp_path,task,r['reference_hash'])
    assert report['level']=='EXECUTION_VERIFIED' and steps>0


def test_missing_dependency_and_live_duplicate_do_not_pass(tmp_path):
    r=learning.train(tmp_path,None,None,None,'goal_loss',{},lambda *a,**k:None,time.time()+10,[])
    assert r['status']=='BLOCKED' and not (tmp_path/'model.pt').exists()
    atomic_json(tmp_path/'run.json',{'identity':{}})
    atomic_json(tmp_path/'status.json',{'status':'RUNNING','pid':os.getpid(),'heartbeat':time.time()})
    with pytest.raises(RuntimeError,match='duplicate refused'):launch(SOURCE,tmp_path,True)
    assert status(tmp_path)['owner_alive']
