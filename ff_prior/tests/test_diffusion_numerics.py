import copy
import numpy as np
import torch
import pytest
from ff_prior.diffusion import schedule,prediction_to_clean_noise,ddim,NoiseTransformer
from ff_prior.dataset import Scaler
from ff_prior.contracts import frozen_task
from ff_prior.clean_prior import TemporalPrior
from ff_prior.storage import read_json,atomic_json


def test_prediction_rotations_and_high_noise_error_bound():
    g=torch.Generator().manual_seed(5)
    x0=torch.randn((2,19,17),generator=g,dtype=torch.float64)
    eps=torch.randn(x0.shape,generator=g,dtype=torch.float64)
    for t in (0,250,750,999):
        a=schedule()[t].double()
        x=a.sqrt()*x0+(1-a).sqrt()*eps
        v=a.sqrt()*eps-(1-a).sqrt()*x0
        for kind,p in (('epsilon',eps),('v',v)):
            clean,noise=prediction_to_clean_noise(x,p,a,kind)
            torch.testing.assert_close(clean,x0,atol=1e-12,rtol=0)
            torch.testing.assert_close(noise,eps,atol=1e-12,rtol=0)
        perturbed,_=prediction_to_clean_noise(x,v+.1,a,'v')
        assert (perturbed-x0).abs().max()<=.100000000001
    unstable,_=prediction_to_clean_noise(x,eps+.1,a,'epsilon')
    assert (unstable-x0).abs().max()>15
    with pytest.raises(ValueError): prediction_to_clean_noise(x,eps,a,'unknown')


def test_ddim_oracle_recovers_clean_at_exact_requested_steps():
    condition={'initial_features':np.zeros(17),'goals':np.tile(np.eye(4),(2,1,1)),
        'duration':10.,'pose_mask':np.ones((2,6))}
    scaler=Scaler.fit(np.zeros((2,19,17)),[condition]*2,['TRAIN']*2,'robot','initial')
    class Oracle:
        calls=0
        prediction_type='epsilon'
        def __call__(self,x,t,c):
            self.calls+=1
            return (x-schedule()[t].sqrt()[:,None,None]*.2)/(1-schedule()[t]).sqrt()[:,None,None]
    for steps in (1,7,16,51):
        z,log=ddim(Oracle(),scaler,frozen_task(condition,{}),steps,8)
        np.testing.assert_allclose(z,.2,atol=4e-5,rtol=0)
        assert log['actual_network_calls']==steps


def test_guided_bound_survives_affine_round_trip_and_consistent_ddim():
    torch.set_num_threads(2)
    condition={'initial_features':np.zeros(17),'goals':np.tile(np.eye(4),(2,1,1)),
        'duration':10.,'pose_mask':np.ones((2,6))}
    # Non-binary scale/offset recreates the old float32 overshoot at an exact bound.
    scaler=Scaler(np.full(17,.0123456789),np.full(17,.0532109876),np.zeros(48),np.ones(48),{})
    model=NoiseTransformer(48,hidden=32,layers=1,heads=4)
    target=np.broadcast_to(np.array([1.]*10+[np.pi]*7),(19,17)).copy()
    seen=[]
    def guide(z,i):
        seen.append((i,model.calls))
        return target.copy(),{'iteration':i}
    result,log=ddim(model,scaler,frozen_task(condition,{}),16,9,guide)
    np.testing.assert_array_equal(result,target)
    assert seen==[(14,15),(15,16)]
    assert max(r['decomposition_residual_after'] for r in log['steps'])<1e-12


def test_temporal_prior_train_only_reconstruction_and_tamper(tmp_path):
    s=np.linspace(0,1,19)
    z=np.stack([(s[:,None]**2)*np.full((1,17),i*.02) for i in range(1,5)])
    prior=TemporalPrior.fit(z,np.zeros(17),list('abcd'),['TRAIN']*4,'robot','initial')
    assert prior.metadata['rank']==1
    np.testing.assert_allclose(prior.project(z[0]),z[0],atol=1e-15,rtol=0)
    with pytest.raises(ValueError,match='TRAIN-only'):
        TemporalPrior.fit(z,np.zeros(17),list('abcd'),['TRAIN','TRAIN','VAL','TRAIN'],'robot','initial')
    file=tmp_path/'prior.json'; prior.save(file)
    value=read_json(file)
    TemporalPrior.load(file,value['hash'],'robot','initial')
    value['projector'][0][0]+=.1; atomic_json(file,value)
    with pytest.raises(ValueError,match='hash'):
        TemporalPrior.load(file,value['hash'],'robot','initial')


def test_clean_prior_projection_is_inside_ddim_with_exact_calls():
    torch.set_num_threads(2)
    condition={'initial_features':np.zeros(17),'goals':np.tile(np.eye(4),(2,1,1)),
        'duration':10.,'pose_mask':np.ones((2,6))}
    s=np.linspace(0,1,19)
    z=np.stack([s[:,None]*np.full((1,17),i*.02) for i in range(1,5)])
    prior=TemporalPrior.fit(z,np.zeros(17),list('abcd'),['TRAIN']*4,'robot','initial')
    scaler=Scaler.fit(z,[condition]*4,['TRAIN']*4,'robot','initial')
    model=NoiseTransformer(48,hidden=32,layers=1,heads=4)
    model.prediction_type='v'; model.clean_prior=prior
    result,log=ddim(model,scaler,frozen_task(condition,{}),16,9)
    np.testing.assert_allclose(prior.project(result),result,atol=1e-14,rtol=0)
    assert log['actual_network_calls']==16 and len(log['clean_prior_projection'])==16
    assert log['clean_prior_fit_split']=='TRAIN'
