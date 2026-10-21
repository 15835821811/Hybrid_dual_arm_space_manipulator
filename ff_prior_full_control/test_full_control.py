from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ff_prior.storage import read_json
from ff_prior_latent.learning import LatentDenoiser
from ff_prior_endpoint_jacobian.comparison import unguided_sample
from .codec import FullCodec
from .learning import rollout
from .sampling import FullProjector
from .protocol import ROOT,P
from .experiment import assert_resume_safe

def codec():
    return FullCodec({'basis':np.eye(19),'feature_mean':np.zeros(65),'feature_scale':np.ones(65),
        'ridge':np.zeros((66,323)),'latent_mean':np.zeros(323),'latent_scale':np.ones(323)})

def test_full_codec_preserves_all_control_directions():
    c=codec();base=np.zeros((19,17));z=np.arange(323,dtype=float).reshape(19,17)/1000
    assert np.array_equal(c.decode(c.encode(z,base),base),z)
    assert c.dim==323
    quantized=c.encode(z,base).astype(np.float32).astype(np.float64)
    assert np.max(abs(c.decode(quantized,base)-z))<=1e-6

def test_dev_fit_rejected():
    with pytest.raises(ValueError,match='TRAIN'):FullCodec.fit([],[],[],[],['VAL'])

def test_preserve_is_bit_exact(monkeypatch):
    from . import sampling
    monkeypatch.setattr(sampling,'quick_gates',lambda *a:{'software_fake_gate':True})
    p=FullProjector.__new__(FullProjector);p.task={'condition':{'initial_features':np.zeros(17),'duration':27.}}
    p.robot=object();p.base=np.zeros((19,17));z=np.arange(323).reshape(19,17)*.001
    output,proof=p.prepare(z);assert np.array_equal(output,z) and proof['unchanged']

def test_nonfinite_prepare_refused():
    p=FullProjector.__new__(FullProjector);p.task={'condition':{}}
    with pytest.raises(ValueError,match='nonfinite'):p.prepare(np.full((19,17),np.nan))

def test_unknown_work_not_replayed(tmp_path):
    from ff_prior_pilot_train.storage import atomic_json
    atomic_json(tmp_path/'ledger.json',{'inflight':{'unit':'x'}})
    with pytest.raises(ValueError,match='UNKNOWN'):assert_resume_safe(tmp_path)

def test_full16_autograd_matches_deployment():
    torch.set_num_threads(2);torch.manual_seed(971)
    m=LatentDenoiser(323,65);c=codec()
    task=read_json(ROOT/'ff_prior_full_control/evidence/selection.json')['reconstruction'][0]['task']
    base=np.zeros((19,17));condition=torch.tensor(c.condition(task['condition'],base),dtype=torch.float32)[None]
    x=rollout(m,condition,[7]);assert m.calls==16
    raw,steps=unguided_sample((m,c),base,task,7)
    assert np.array_equal(raw,c.decode(c.coefficients(x.detach().numpy()[0],c.center(task['condition'],base)),base))
    x.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(float(p.grad.norm()) for p in m.parameters())>0 and len(steps)==16

def test_frozen_no_extra_stages():
    assert not P['TEST'] and not P['expand'] and P['TRAIN_added']==0
    assert P['updates']==6000 and P['new_DEV_candidates']==180 and P['candidate_coupled_cap']==113
