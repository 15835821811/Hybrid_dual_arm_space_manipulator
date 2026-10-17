"""Real data, numerical equivalence, actual updates, coupled physics and replay."""
import copy
import os
import time
import numpy as np
import pytest
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_coverage_train.storage import atomic_json,Ledger
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_modes import data as loader
from ff_prior.diffusion import weight_hash
from ff_prior_generalization.sampling import candidate
from ff_prior_generalization.audit_results import check_candidate
from ff_prior_generalization.experiment import execute
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_origin_learn_bounded.protocol import ROOT,BASE,SOURCE,OLD
from ff_prior_origin_learn_bounded import learning,sensitivity,mean
from ff_prior_origin_learn_bounded.calibration import calibrate,rebase,equivalence
from ff_prior_origin_learn_bounded.distribution import population,CoverageSampler
from ff_prior_origin_learn_bounded.experiment import require_data
from ff_prior_origin_learn_bounded.job import launch,status


@pytest.fixture(scope='module')
def actual(tmp_path_factory):
    torch.set_num_threads(2)
    out=tmp_path_factory.mktemp('actual_origin_learn')
    config=read_json(BASE/'run.json')['config'];physical,rr,ii,_=audit(config);profile=acceptance(config)
    cache=TaskOriginSceneCache(rr,ii,profile)
    oldindex=read_json(OLD/'training_index.json')['records']
    index,coverage,validation=require_data(SOURCE)
    selected=[e for e in oldindex if e['split']=='TRAIN'][:6]
    selected += [e for e in index if e['split']=='TRAIN' and e['sample_id'].startswith('batch_v1_train_')][:6]
    selected += [e for e in oldindex if e['split']=='VAL'][:2]
    arrays,rows=loader.arrays(selected,cache,out/'small',lambda *a,**k:None)
    parent,oldcodec=learning.frozen_learning.load(learning.FROZEN)
    codec=calibrate(oldcodec,arrays)
    return out,cache,profile,arrays,rows,parent,oldcodec,codec,[e['sample_id'] for e in selected[6:12]]


def test_complete_qualified_population_and_holdout():
    index,coverage,validation=require_data(SOURCE)
    rows=[read_json(ROOT/e['record']) for e in index]
    slots,keys,w,p=population([r['condition'] for r in rows],[e['sample_id'] for e in index],[e['split'] for e in index])
    assert p['rows']==13774 and p['origins']==1578 and len(coverage)==1086
    assert len({rows[i]['task']['task_hash'] for i in slots})==13495
    assert abs(sum(w)-13774)<1e-8
    assert not set(coverage)&{e['sample_id'] for e in validation['entries']}
    assert not read_json(ROOT/'artifacts/v7_1/qualified_origin_coverage_001/summary.json')['ready_for_training_stage']


def test_affine_recalibration_preserves_actual_old_network(actual):
    out,cache,p,arrays,rows,parent,old,codec,newids=actual
    rebased=rebase(parent,old,codec);proof=equivalence(parent,rebased,old,codec,arrays)
    assert proof['status']=='PASS' and proof['TRAIN_conditions_tested']==12
    assert proof['old_weight_hash']!=proof['rebased_weight_hash']
    for c,b in zip(arrays[2],arrays[1]):
        np.testing.assert_allclose(old.center(c,b),codec.center(c,b),atol=1e-9,rtol=0)
    for key in ('basis','latent_mean','latent_scale'):assert np.array_equal(getattr(old,key),getattr(codec,key))


def test_calibration_and_mean_are_train_only(actual,tmp_path):
    out,cache,p,arrays,rows,parent,old,codec,newids=actual
    altered=list(arrays);altered[0]=arrays[0].copy();altered[0][-2:]+=1.
    altered[1]=arrays[1].copy();altered[1][-2:]+=1.
    altered[2]=copy.deepcopy(arrays[2]);altered[2][-1]['initial_features']=(np.asarray(altered[2][-1]['initial_features'])+3).tolist()
    assert digest(calibrate(old,altered).state)==digest(codec.state)
    fitted=mean.fit(tmp_path,arrays,codec,newids)
    assert digest(mean.fit(tmp_path,altered,codec,newids).state)==digest(fitted.state)
    assert len(fitted.state['fit_ids'])==12 and not set(arrays[3][-2:])&set(fitted.state['fit_ids'])
    with pytest.raises(ValueError,match='not TRAIN'):mean.fit(tmp_path,arrays,codec,newids+[arrays[3][-1]])


def test_origin_row_sampler_resume_and_exhaustive_balance(actual):
    arrays=actual[3];_,keys,w,p=population(arrays[2],arrays[3],arrays[4])
    uninterrupted=CoverageSampler(keys,918);resumed=CoverageSampler(keys,918)
    for i in range(40):
        assert uninterrupted.take(8)==resumed.take(8)
        if i==16:resumed=CoverageSampler(keys,918,resumed.state())
    assert digest(uninterrupted.state())==digest(resumed.state())
    assert min(resumed.row_counts)>0 and max(resumed.row_counts)-min(resumed.row_counts)<=1
    assert max(resumed.origin_counts)-min(resumed.origin_counts)<=1
    for key,members in resumed.members.items():
        assert max(resumed.origin_row_counts[members])-min(resumed.origin_row_counts[members])<=1


def test_real_updates_cache_and_completed_resume(actual,monkeypatch):
    out,cache,p,arrays,rows,parent,old,codec,newids=actual;ledger=Ledger(out/'sensitivity',{})
    matrices,proof=sensitivity.prepare(out/'sensitivity',arrays,rows,codec,cache,ledger,time.time()+240,lambda *a,**k:None)
    assert proof['status']=='PASS' and len(matrices)==12
    assert ledger.data['calls']['cached_sensitivity.coupled']==6*39
    assert ledger.data['calls']['sensitivity.coupled']==6*39
    monkeypatch.setitem(learning.CONTRACT,'updates',20);monkeypatch.setitem(learning.CONTRACT,'batch',4)
    trained=learning.train(out/'trained',arrays,codec,matrices,'goal_loss',proof,lambda *a,**k:None,time.time()+120,newids)
    assert trained['status']=='PASS' and trained['initial_weight_hash']==weight_hash(rebase(parent,old,codec))
    assert trained['trained_weight_hash']!=trained['initial_weight_hash']
    assert trained['stream_draw_counts']=={'uniform_rows':40,'uniform_origins':40}
    assert trained['all_TRAIN_rows_seen'] and trained['VAL_gradient_updates']==0
    before={str(p):file_digest(p) for p in (out/'trained').rglob('*') if p.is_file()}
    assert learning.train(out/'trained',arrays,codec,matrices,'goal_loss',proof,lambda *a,**k:None,time.time()+120,newids)==trained
    assert before=={str(p):file_digest(p) for p in (out/'trained').rglob('*') if p.is_file()}


def test_cache_rejects_reference_or_basis_tamper(actual,tmp_path):
    out,cache,p,arrays,rows,parent,old,codec,newids=actual;key=rows[0]['task']['task_hash']
    altered=list(arrays);altered[0]=arrays[0].copy();altered[0][0,0,0]+=1e-5
    with pytest.raises(ValueError,match='anchor/proof mismatch'):
        sensitivity.import_anchor(tmp_path,key,0,altered,codec,Ledger(tmp_path,{}))
    state=copy.deepcopy(codec.state);state['basis']=codec.basis[:,::-1]
    changed=learning.Codec(state)
    with pytest.raises(ValueError,match='basis changed'):
        sensitivity.import_anchor(tmp_path,key,0,arrays,changed,Ledger(tmp_path,{}))


def test_rebased_ddim_guidance_and_real_67_torque_replay(actual,tmp_path):
    out,cache,p,arrays,rows,parent,old,codec,newids=actual
    entry=read_json(OLD/'frozen_tasks.json')['entries'][0];task=entry['task']
    rr,ii,cc,_=cache.get(entry);base=public_goal_initializer(task,rr,ii)
    bundle=(rebase(parent,old,codec),codec)
    r=candidate('expanded_diffusion',0,task,base,20262119,bundle,rr,ii,cc,p,time.time()+30)
    assert r['status']=='DONE' and r['eligible'],r.get('reason')
    check_candidate(r,task,20262119,'expanded_diffusion',0)
    assert r['actual_network_calls']==16 and np.isfinite(r['Z']).all()
    outcome=execute(tmp_path,task,r['Z'],rr,ii,cc,p,time.time()+60)
    assert outcome['status']=='EXECUTION_VERIFIED'
    proof,steps=audit_physics(tmp_path,task,r['reference_hash'])
    assert proof['level']=='EXECUTION_VERIFIED' and steps>0


def test_missing_dependency_and_live_duplicate_do_not_pass(tmp_path):
    r=learning.train(tmp_path,None,None,None,'goal_loss',{},lambda *a,**k:None,time.time()+10,[])
    assert r['status']=='BLOCKED' and not (tmp_path/'model.pt').exists()
    atomic_json(tmp_path/'run.json',{'identity':{}})
    atomic_json(tmp_path/'status.json',{'status':'RUNNING','pid':os.getpid(),'heartbeat':time.time()})
    with pytest.raises(RuntimeError,match='duplicate refused'):launch(SOURCE,tmp_path,True)
    assert status(tmp_path)['owner_alive']
