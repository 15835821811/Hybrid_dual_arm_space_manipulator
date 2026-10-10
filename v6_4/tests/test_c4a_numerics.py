"""Independent closed-form tests; no optimization or physical integration."""
from unittest.mock import patch
import numpy as np
import pytest
import torch

from v6_4.c4a_frozen import RELEASE, load_frozen_sampler
from v6_4.residual_diffusion import ResidualDDPM, ResidualDiffusionConfig
from v6_4.preference_diffusion_warmstart import inverse_raw
from v6_4.preference_teacher_dataset import ConditionNormalizer
from v6_4.search_effect_teacher import load_search_aware_dataset
from v6_4.route_optimizer_protocol import read, active_intervals, parameter_plan, VERSIONS
from v6_4.route_initializers import raw_seed_plan
from v6_4.task_anchored_reference import build_reference_definition


@pytest.fixture(scope='module')
def ds():
    return load_search_aware_dataset(RELEASE/'snapshot/dataset/manifest.json')


@pytest.mark.parametrize('t', [0, 1, 49, 98, 99])
def test_v_forward_inverse_and_target(t):
    torch.manual_seed(64401)
    model = ResidualDDPM(ResidualDiffusionConfig(3))
    mask = torch.tensor([[False, True, True, False, False, False]])
    m = mask.repeat_interleave(2, -1)
    clean = torch.randn(1, 12)*m
    eps = torch.randn(1, 12)
    ts = torch.tensor([t]); a = model.alpha_bars[t]
    xt = model.q_sample(clean, ts, eps, mask)
    target = (a.sqrt()*eps-(1-a).sqrt()*clean)*m
    with patch.object(model, 'forward', return_value=target):
        prediction = model.prediction(xt, ts, torch.zeros(1, 3), mask)
        torch.testing.assert_close(prediction['x0'], clean, atol=3e-7, rtol=2e-6)
        torch.testing.assert_close(prediction['epsilon'], eps*m, atol=3e-7, rtol=2e-6)
        assert model.loss(clean, ts, eps, torch.zeros(1, 3), mask).item() == 0


def test_cosine100_independent_formula_and_boundary():
    model = ResidualDDPM(ResidualDiffusionConfig(3))
    f = np.cos(((np.arange(101)/100+.008)/1.008)*np.pi/2)**2
    beta = np.minimum(1-f[1:]/f[:-1], .999)
    expected = np.cumprod(1-beta).astype(np.float32)
    np.testing.assert_allclose(model.alpha_bars.numpy(), expected, atol=1e-8, rtol=1e-6)
    assert 0 < model.alpha_bars[-1] < 1e-6
    assert torch.all(model.alpha_bars[1:] < model.alpha_bars[:-1])


def test_ddim20_oracle_full_path_and_terminal_clean():
    model = ResidualDDPM(ResidualDiffusionConfig(3))
    mask = torch.tensor([[False, True, True, False, False, False]])
    m = mask.repeat_interleave(2, -1)
    clean = torch.arange(12)[None].float()*.01*m
    called = []
    def oracle(x, ts, condition, passed_mask):
        assert torch.all(x[~m] == 0)
        called.append(int(ts[0]))
        a = model.alpha_bars[ts, None]
        eps = (x-a.sqrt()*clean)/(1-a).sqrt()
        return (a.sqrt()*eps-(1-a).sqrt()*clean)*m
    with patch.object(model, 'forward', side_effect=oracle):
        out = model.sample_ddim(torch.zeros(1,3), mask, initial_noise=torch.ones(1,12))
    assert called == [99,94,89,83,78,73,68,63,57,52,47,42,36,31,26,21,16,10,5,0]
    torch.testing.assert_close(out, clean, atol=1e-7, rtol=2e-6)
    assert torch.all(out[~m] == 0)


def test_mask_at_loss_forward_noise_and_gradient():
    torch.manual_seed(64402)
    model = ResidualDDPM(ResidualDiffusionConfig(3))
    mask = torch.tensor([[False, True, True, False, False, False]])
    m = mask.repeat_interleave(2, -1)
    x = (torch.randn(1,12)*m).requires_grad_()
    eps = torch.randn(1,12); other = eps.clone(); other[~m] = 1e20
    t = torch.tensor([99]); c = torch.zeros(1,3)
    assert torch.equal(model.q_sample(x,t,eps,mask),model.q_sample(x,t,other,mask))
    assert torch.all(model(eps,t,c,mask)[~m] == 0)
    loss = model.loss(x,t,eps,c,mask); loss.backward()
    assert torch.all(x.grad[~m] == 0)
    bad = x.detach().clone(); bad[~m] = 1e-30
    with pytest.raises(ValueError, match='exactly zero'):
        model.q_sample(bad,t,eps,mask)


def test_scaler_train_only_exact_refit_and_inverse(ds):
    keys = sorted({(s['task_id'],s['preference'],s['family']) for s in ds.samples})
    scaler = ConditionNormalizer.fit([(ds.tasks[t],ds.definitions[t,f],p,f) for t,p,f in keys],
        read(RELEASE/'snapshot/learning_split_manifest.json'))
    assert scaler.to_dict() == ds.condition_scaler.to_dict()
    for z,m in zip(ds.z_m,ds.search_masks):
        latent = ds.residual_scaler.normalize(z,m)
        np.testing.assert_array_equal(inverse_raw(ds.residual_scaler,latent,m),ds.residual_scaler.inverse(latent,m))
        np.testing.assert_allclose(inverse_raw(ds.residual_scaler,latent,m),z,atol=1e-18)
    val = next(t for t in ds.tasks if ds.task_splits[t]=='val')
    with pytest.raises(ValueError,match='TRAIN'):
        ConditionNormalizer.fit([(ds.tasks[val],ds.definitions[val,'v1'],'A','v1')],read(RELEASE/'snapshot/learning_split_manifest.json'))


@pytest.mark.parametrize('bad', ['nan','inf','inactive','norm'])
def test_inverse_illegal_raw_survives_qualification(ds,bad):
    task = next(ds.tasks[s['task_id']] for s in ds.samples)
    m = np.array([False,True,True,False,False,False])
    latent = np.zeros((6,2))
    if bad=='nan': latent[1,0]=np.nan
    elif bad=='inf': latent[1,0]=np.inf
    elif bad=='inactive': latent[0,0]=1e-9
    else: latent[1,:]=100
    raw = inverse_raw(ds.residual_scaler,latent,m)
    before=raw.copy()
    plan,diagnostics=raw_seed_plan(task,dict(source='diffusion',family='v1',preference='A',raw_z_m=raw),1)
    assert plan is None and not diagnostics['raw_legal'] and not diagnostics['raw_repaired'] and not diagnostics['resampled']
    np.testing.assert_array_equal(raw,before)


def test_effective_space_and_interval_disks(ds):
    for t in ds.tasks.values():
        active=active_intervals(build_reference_definition(t)); assert len(active)<=2
        x=np.tile([.012,.016],len(active))
        plan=parameter_plan(t,'v2',x)
        np.testing.assert_array_equal(plan.z_m[active].reshape(-1),x)
        assert np.all(plan.z_m[[i for i in range(6) if i not in active]]==0)
        assert np.linalg.norm(plan.z_m)>.020
        with pytest.raises(ValueError): parameter_plan(t,'v2',np.tile([.016,.016],len(active)))


def test_frozen_checkpoint_determinism_and_authority(ds):
    sampler=load_frozen_sampler()
    task=ds.tasks[next(s['task_id'] for s in ds.samples)]
    noise=np.random.default_rng(64403).standard_normal(12).astype(np.float32)
    a,meta=sampler.sample(task,'A','v1',noise)
    b,_=sampler.sample(task,'A','v1',noise)
    np.testing.assert_array_equal(a,b)
    assert meta['physics_steps']==0 and not meta['raw_repaired']
    assert sampler.sample_units==2 and sum(p.numel() for p in sampler.model.parameters())==152332
    assert not sampler.model.training and all(not p.requires_grad for p in sampler.model.parameters())
