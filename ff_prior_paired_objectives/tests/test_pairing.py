import copy,time
import numpy as np
import pytest
import torch
from ff_prior.storage import read_json,digest
from ff_prior.diffusion import schedule,weight_hash
from ff_prior.pipeline import acceptance
from ff_prior_origin_learn_bounded.learning import load
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_paired_objectives.protocol import ROOT,WARM,ARMS
from ff_prior_paired_objectives.objectives import random_batch,terms,vector_gradient,get_vector,set_vector
from ff_prior_paired_objectives.learning import select
from ff_prior_world_endpoint.endpoint import evaluate,gradient,attach_gradient

def test_noise_pairing_independent_of_global_rng_and_resume():
    a=random_batch(41,64,102);torch.manual_seed(123);torch.randn(1000);b=random_batch(41,64,102)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert not torch.equal(a[1],random_batch(42,64,102)[1])

def test_TRAIN_selection_rejects_regression_and_never_allows_TEST():
    def probe(loss):return {'cases':[{'loss':loss,'sample_id':str(i),'t':500,'seed':i} for i in range(32)]}
    data={k:probe(v) for k,v in {'warm':10,'A_v':12,'B_local':5,'C_world':8,'D_both':9}.items()}
    assert select(data)['selected_arm'] is None
    data['D_both']=probe(4)
    r=select(data);assert r['selected_arm']=='D_both' and not r['TEST_allowed']
    data['D_both']['cases'][0]['seed']+=1
    with pytest.raises(ValueError,match='unpaired'):select(data)

def test_real_parameter_VJP_descent_and_four_optimizer_objectives(physical):
    index=read_json(WARM/'training_index.json')['records']
    i=next(i for i,e in enumerate(index) if e['sample_id']=='batch_v1_train_0000_a0')
    row=read_json(ROOT/index[i]['record']);model,codec=load(WARM/'learning/goal_loss')
    with np.load(WARM/'loaded_training_arrays.npz') as a:z,base=a['Z'][i],a['bases'][i]
    center=codec.center(row['condition'],base)
    clean=torch.tensor(codec.normalized(codec.encode(z,base),center)[None],dtype=torch.float32)
    cond=torch.tensor(codec.condition(row['condition'],base)[None],dtype=torch.float32)
    with np.load(WARM/'sensitivity'/f'{row["task"]["task_hash"]}.npz') as a:G=torch.tensor((a['G']*codec.latent_scale)[None],dtype=torch.float32)
    t=torch.tensor([999]);noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(20262303));alpha=schedule()
    v,l,p=terms(model,clean,cond,t,noise,alpha,G)
    profile=acceptance(read_json(ROOT/'artifacts/v7_1/smoke_002/run.json')['config'])
    cache=TaskOriginSceneCache(physical[0],physical[1],profile);rr,ii,_,_=cache.get(row);deadline=time.time()+240
    f,g,proof,_=gradient(p.detach().numpy()[0].astype(float),codec,center,base,row['task'],rr,ii,deadline)
    assert proof['status']=='PASS'
    w=attach_gradient(p[0],f,g);vec=vector_gradient(w,model);basevec=get_vector(model);h=weight_hash(model)
    direction=vec/vec.norm();set_vector(model,basevec-.0003*direction)
    with torch.no_grad():_,_,q=terms(model,clean,cond,t,noise,alpha,G)
    after,_=evaluate(q.numpy()[0].astype(float),codec,center,base,row['task'],rr,ii,deadline)
    assert after<f
    outcomes={}
    for arm,(lw,ww) in ARMS.items():
        set_vector(model,basevec);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
        v,l,p=terms(model,clean,cond,t,noise,alpha,G);w=attach_gradient(p[0],f,g)
        optimizer.zero_grad();(v+lw*l+ww*w).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        outcomes[arm]=weight_hash(model)
    assert len(set(outcomes.values()))==4
    set_vector(model,basevec);assert weight_hash(model)==h

def test_empty_report_cannot_authorize_stage_progression(tmp_path):
    from ff_prior_paired_objectives.reporting import report
    r=report(tmp_path,'RUNNING')
    assert r['single_step_status']=='PENDING' and not r['training'] and not r['TEST_executed']
    assert r['TEST_decision']=='BLOCKED_BY_CURRENT_USER_SCOPE' and not r['full_goal_achieved']

def test_running_duplicate_and_unknown_work_are_not_replayed(tmp_path):
    from ff_prior_paired_objectives.job import launch
    from ff_prior_coverage_train.job import Monitor
    from ff_prior_coverage_train.storage import Ledger,atomic_json
    ledger=Ledger(tmp_path,{});ledger.begin('C_world_gradient_000','world_gradient',60,240)
    recovered=Ledger(tmp_path,{})
    assert recovered.data['completed']['C_world_gradient_000']['status']=='INTERRUPTED'
    assert recovered.data['unresolved_call_accounting']==['C_world_gradient_000']
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as m:
        with pytest.raises(RuntimeError,match='alive'):launch(tmp_path/'source',tmp_path,True)
        m.finish({'status':'FINISHED'})
