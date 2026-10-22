"""Software tests only; synthetic fixtures are never experimental evidence."""
import numpy as np
import torch
from ff_prior.storage import read_json,digest
from .protocol import P,PACKAGE,PREVIOUS
from .model import TransferDenoiser
from .world_training import independent_gate,exact_rollout,fixed_pool
from .learning import vector_gradient,assign
from ff_prior_rollout_world.test_world import (
    test_residual_matches_original_pose_metric,
    test_analytic_surrogate_gradient_independent_finite_difference,
    test_protection_changes_training_gradient,
    test_acceptance_cannot_trade_away_qualified_sample)


def record(q,eligible=True,stratum='easy'):
    return dict(eligible=eligible,raw_equal_prepared=True,Z=np.zeros((19,17)).tolist(),stratum=stratum,
        validation=dict(validation_level='MODEL_VALID',frozen_goal_position_error_m=[q*.005,0],frozen_goal_rotation_error_rad=[0,0]))


def test_check_pool_cannot_trade_qualification_for_mean():
    old=[record(.8),record(.7),record(10,False)]
    new=[record(1.01,False),record(.5),record(.1)]
    g=independent_gate(old,new,['easy']*3)
    assert g['status']=='BLOCKED' and not g['checks']['original_qualified_preserved']


def test_check_pool_tail_is_separate_from_mean():
    old=[record(.8),record(.7),record(3,False),record(3,False)]
    new=[record(.5),record(.5),record(1.2,False),record(5,False)]
    g=independent_gate(old,new,['easy']*4)
    assert g['status']=='BLOCKED' and not g['checks']['tail_nonregression']


def test_budget_no_expansion_and_no_test():
    upper=2*3*32*(654+3*7)+64*7+32*7+2*32*7+96*3+190*113
    assert upper==152478 and upper<P['coupled_cap']
    assert not P['TEST'] and not P['expand'] and P['TRAIN_added']==0
    assert P['new_DEV_candidates']==180 and P['new_application_candidates']==10


def test_split_exclusion_and_frozen_codec():
    from .data import load
    codec,arrays,rows,data=load();s=read_json(PACKAGE/'evidence/selection.json')
    assert len(fixed_pool(s,'fit'))==len(fixed_pool(s,'check'))==32
    assert set(s['fit_task_hashes']).isdisjoint(s['check_task_hashes'])
    hashes={rows[i]['task']['task_hash'] for i in data['tr']}
    assert not hashes&set(s['check_task_hashes']) and len(data['tr'])==s['adaptation_TRAIN_count']
    assert digest(codec.state)==digest(read_json(PREVIOUS/'codec.json'))
    assert len(data['va'])==48 and len(data['keys'])==len(data['tr'])


def test_detail_update_preserves_old_complete_low_function():
    from .data import load
    from ff_prior_control_transfer.model import TransferDenoiser as Old
    torch.set_num_threads(2);torch.manual_seed(P['training_seed'])
    codec,arrays,rows,data=load();m=TransferDenoiser(codec);old=Old(codec)
    ids=data['tr'][:2];c=data['c'][ids];seeds=[123,456]
    with torch.no_grad():before=exact_rollout(m,c,seeds);reference=exact_rollout(old,c,seeds)
    assert torch.equal(before,reference)
    low=m.low_hash();optim=torch.optim.AdamW(m.trainable(),lr=.001)
    x=exact_rollout(m,c,seeds);loss=(x[:,102:]-torch.ones_like(x[:,102:])*.01).square().mean()
    g=vector_gradient(loss,m);assign(m,g);assert g.norm()>0
    optim.step()
    with torch.no_grad():after=exact_rollout(m,c,seeds)
    assert m.low_hash()==low and all(p.grad is None for p in m.low.parameters())
    assert torch.equal(before[:,:102],after[:,:102]) and not torch.equal(before[:,102:],after[:,102:])
    clone=TransferDenoiser(codec,warm=False);clone.load_state_dict(m.state_dict())
    with torch.no_grad():assert torch.equal(after,exact_rollout(clone,c,seeds))
