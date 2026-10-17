import time
import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.pipeline import acceptance
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from ff_prior_origin_learn_bounded.learning import load
from ff_prior.diffusion import schedule,prediction_to_clean_noise
from ff_prior_world_endpoint.protocol import ROOT,SOURCE
from ff_prior_world_endpoint.endpoint import evaluate,gradient,attach_gradient
from ff_prior_coverage_train.storage import atomic_json,atomic_npz

def test_real_full102_gradient_and_true_endpoint_descent(physical,tmp_path):
    index=read_json(SOURCE/'training_index.json')['records']
    i=next(i for i,e in enumerate(index) if e['sample_id']=='batch_v1_train_0000_a0')
    row=read_json(ROOT/index[i]['record']);model,codec=load(SOURCE/'learning/goal_loss')
    with np.load(SOURCE/'loaded_training_arrays.npz') as a:z,base=a['Z'][i],a['bases'][i]
    profile=acceptance(read_json(ROOT/'artifacts/v7_1/smoke_002/run.json')['config'])
    cache=TaskOriginSceneCache(physical[0],physical[1],profile);rr,ii,cc,_=cache.get(row)
    center=codec.center(row['condition'],base);encoded=codec.encode(z,base)
    clean=torch.tensor(codec.normalized(encoded,center)[None],dtype=torch.float32)
    c=torch.tensor(codec.condition(row['condition'],base)[None],dtype=torch.float32)
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(20262303))
    alpha=schedule()[999];noisy=alpha.sqrt()*clean+(1-alpha).sqrt()*noise
    predicted,_=prediction_to_clean_noise(noisy,model(noisy,torch.tensor([999]),c),alpha,'v')
    x=predicted.detach().numpy()[0].astype(float);deadline=time.time()+240
    f,g,proof,arrays=gradient(x,codec,center,base,row['task'],rr,ii,deadline)
    assert proof['status']=='PASS',proof
    assert proof['calls']['coupled']==212 and proof['dimensions']==102
    attached=attach_gradient(predicted[0],f,g)
    attached.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.square().sum()) for p in model.parameters())>0
    probe=torch.tensor(x,requires_grad=True);attach_gradient(probe,f,g).backward()
    assert np.array_equal(probe.grad.numpy(),g)
    moved=x-.01*g/max(np.linalg.norm(g),1.)
    after,_=evaluate(moved,codec,center,base,row['task'],rr,ii,deadline)
    assert after<f,(f,after)
    assert digest(row['task'])==digest(read_json(ROOT/index[i]['record'])['task'])
    atomic_npz(tmp_path/'proof.npz',**arrays)
    proof['arrays_sha256']=file_digest(tmp_path/'proof.npz')
    atomic_json(tmp_path/'proof.json',dict(proof,loss_after_latent_descent=after))
    from ff_prior_world_endpoint.audit_results import verify_gradient
    verify_gradient(tmp_path/'proof.json',codec,base,center,row)
    print({'true_before':f,'true_after':after,'half_errors':[h['relative_error'] for h in proof['half_checks']],
           'direction_error':proof['direction_relative_error']})

