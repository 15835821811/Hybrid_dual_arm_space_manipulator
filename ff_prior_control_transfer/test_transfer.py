import numpy as np
import pytest
import torch
from ff_prior.storage import read_json
from ff_prior_sample_alignment.learning import load as old_load
from ff_prior_endpoint_jacobian.comparison import unguided_sample as old_sample
from .data import load
from .model import TransferDenoiser,initial_noise,rollout,teacher_loss
from .sampling import unguided_sample,FullProjector
from .protocol import SOURCE,P,PACKAGE
from .codec import FullCodec
from .experiment import assert_resume_safe

@pytest.fixture(scope='module')
def dataset():
    torch.set_num_threads(2);return load()

def test_full_rank_roundtrip(dataset):
    c,a,r,d=dataset
    assert c.dim==323 and np.allclose(c.basis.T@c.basis,np.eye(19),atol=1e-12)
    for k in range(323):
        z=np.zeros((19,17));z.flat[k]=.1
        assert np.max(abs(c.decode(c.encode(z,np.zeros_like(z)),np.zeros_like(z))-z))<1e-12

def test_no_dev_fit():
    with pytest.raises(ValueError,match='TRAIN'):FullCodec.fit([],[],[],[],['VAL'])

def test_noise_preserves_old_stream():
    x=initial_noise([77]);old=torch.randn((1,102),generator=torch.Generator().manual_seed(77)).double()
    assert torch.equal(x[:,:102],old) and x.shape==(1,323)

def test_functional_full16_transfer(dataset):
    c,a,r,d=dataset;m=TransferDenoiser(c);old=old_load(SOURCE/'learning/aligned');i=d['tr'][0]
    for seed in (7,91):
        z,steps=unguided_sample((m,c),a[1][i],r[i]['task'],seed)
        oz,_=old_sample(old,a[1][i],r[i]['task'],seed)
        assert np.max(abs(z-oz))<=P['migration_error_rad'] and len(steps)==16

def test_all_detail_outputs_bounded_and_live(dataset):
    c,a,r,d=dataset;m=TransferDenoiser(c);x=initial_noise([7]);t=torch.tensor([333]);cond=d['c'][d['tr'][:1]]
    _,z=m.components(x,t,cond);assert torch.count_nonzero(z)==0
    with torch.no_grad():m.detail[-1].bias.fill_(.1)
    _,z=m.components(x,t,cond)
    assert (z>0).all() and (z.abs()<=m.detail_limit).all()

def test_rollout_deployment_and_gradient(dataset):
    c,a,r,d=dataset;m=TransferDenoiser(c);i=d['tr'][0]
    x=rollout(m,d['c'][i:i+1],[7]);z,_=unguided_sample((m,c),a[1][i],r[i]['task'],7)
    assert np.array_equal(z,c.decode(c.coefficients(x.detach().numpy()[0],c.center(r[i]['condition'],a[1][i])),a[1][i]))
    loss=teacher_loss(m,c,d['c'][i:i+1],d['x'][i:i+1],[7]);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.detail[-1].bias.grad.abs().sum()>0

def test_preserve_and_nonfinite(monkeypatch):
    from . import sampling
    monkeypatch.setattr(sampling,'quick_gates',lambda *a:{'software_stub':True})
    p=FullProjector.__new__(FullProjector);p.task={'condition':{'initial_features':np.zeros(17),'duration':27.}};p.robot=object();p.base=np.zeros((19,17))
    z=np.arange(323).reshape(19,17)*.001;o,proof=p.prepare(z);assert np.array_equal(o,z)
    with pytest.raises(ValueError,match='nonfinite'):p.prepare(np.full((19,17),np.nan))

def test_unknown_and_budget(tmp_path):
    from ff_prior_pilot_train.storage import atomic_json
    atomic_json(tmp_path/'ledger.json',{'inflight':{'unit':'x'}})
    with pytest.raises(ValueError,match='UNKNOWN'):assert_resume_safe(tmp_path)
    assert not P['TEST'] and not P['expand'] and P['TRAIN_added']==0
    assert P['new_DEV_candidates']==180 and P['candidate_coupled_cap']==113
