"""Software unit checks only; none substitutes for frozen physical TRAIN tests."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from .world_endpoint import residual,groups
from .world_training import world_loss,accept
from .protocol import P
from ff_prior.coupled_predictor import pose_errors


def test_residual_matches_original_pose_metric():
    goals=np.tile(np.eye(4),(2,1,1));ee=goals.copy()
    ee[0,:3,3]=[.003,.004,0]
    ee[1,:3,:3]=Rotation.from_rotvec([.002,-.01,.007]).as_matrix()
    p,r=pose_errors(ee,goals)
    assert np.allclose(groups(residual(ee,goals)),np.r_[(p/.005)**2,(r/np.deg2rad(1.))**2],atol=1e-14)


def test_analytic_surrogate_gradient_independent_finite_difference():
    rng=np.random.default_rng(4);J=torch.tensor(rng.normal(size=(2,12,323))*.01)
    R=torch.tensor(rng.normal(size=(2,12)));x0=torch.zeros((2,323),dtype=torch.float64)
    x=x0.clone().requires_grad_();loss=world_loss(x,x0,R,J,[True,False]);g=torch.autograd.grad(loss,x)[0]
    direction=torch.tensor(rng.normal(size=(2,323)));direction/=direction.norm();h=1e-5
    slope=(world_loss(x0+h*direction,x0,R,J,[True,False])-world_loss(x0-h*direction,x0,R,J,[True,False]))/(2*h)
    assert torch.allclose(slope,(g*direction).sum(),rtol=1e-6,atol=1e-7)


def test_protection_changes_training_gradient():
    x=torch.zeros((1,323),dtype=torch.float64,requires_grad=True)
    J=torch.zeros((1,12,323),dtype=torch.float64);J[0,0,0]=1
    R=torch.zeros((1,12),dtype=torch.float64);R[0,0]=.95
    g0=torch.autograd.grad(world_loss(x,x.detach(),R,J,[False]),x)[0]
    g1=torch.autograd.grad(world_loss(x,x.detach(),R,J,[True]),x)[0]
    assert g1[0,0]>g0[0,0]>0


def fake_record(q,eligible=True):
    # Tiny pure-logic fixture, never used by workers or physical artifacts.
    return dict(eligible=eligible,Z=np.zeros((19,17)).tolist(),validation=dict(validation_level='MODEL_VALID',
        frozen_goal_position_error_m=[q*.005,0],frozen_goal_rotation_error_rad=[0,0]))


def test_acceptance_cannot_trade_away_qualified_sample():
    before=[fake_record(.8),fake_record(10,False)]
    after=[fake_record(1.01,False),fake_record(.1)]
    accepted,checks=accept(before,after,[True,False],1.)
    assert not accepted and not checks['old_qualified_preserved']


def test_acceptance_detects_actual_improvement():
    ok,checks=accept([fake_record(.8),fake_record(10,False)],[fake_record(.7),fake_record(.5)],[True,False],1.)
    assert ok and all(checks.values())


def test_budget_is_bounded_and_no_new_data():
    # 2 refinements x (4 full-J rounds + worst 3 trials), formal baseline,
    # VAL and DEV/app fixed ceilings. All recon/migration evidence is reused.
    upper=2*4*16*(654+3*7)+16*7+96*3+190*113
    assert upper==108270 and upper<P['coupled_cap']
    assert not P['TEST'] and not P['expand'] and P['TRAIN_added']==0
    assert P['new_DEV_candidates']==180 and P['candidate_coupled_cap']==113


def test_reused_baseline_matches_full_rollout_and_world_vjp():
    from ff_prior_control_transfer.data import load
    from ff_prior_control_transfer.model import TransferDenoiser,rollout
    from ff_prior.storage import read_json
    from .protocol import REUSE
    torch.set_num_threads(2);torch.manual_seed(P['training_seed'])
    codec,arrays,rows,data=load();model=TransferDenoiser(codec)
    saved=torch.load(REUSE/'TRAIN_diagnostic/before.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,saved['weights'][k]) for k,v in model.state_dict().items())
    from .world_training import exact_rollout
    indices=saved['indices'];seeds=saved['seeds'];full=exact_rollout(model,data['c'][indices],seeds)
    for j,i in enumerate(indices):
        raw=codec.decode(codec.coefficients(full[j].detach().numpy(),data['centers'][i]),arrays[1][i])
        with np.load(REUSE/'TRAIN_diagnostic/before'/f'{j:02d}'/'controls.npz') as old:
            assert np.array_equal(raw,old['raw'])
    # Neural path check only: synthetic derivative, explicitly not physical evidence.
    J=torch.ones((16,12,323),dtype=torch.float64)*.001
    r=torch.ones((16,12),dtype=torch.float64)
    loss=world_loss(full,full.detach(),r,J,[True]*16)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.low.net[0].weight.grad.norm()>0 if hasattr(model.low,'net') else sum(p.grad.norm() for p in model.low.parameters())>0
    assert model.detail[-1].bias.grad.norm()>0
