"""Software mechanism tests only; random networks are NOT learned-plan evidence."""
import numpy as np
import torch
from ffgd.diffusion import sample,SamplingBudget
from ffgd.model import ConditionalDenoiser


def test_guide_consumed_by_later_network_and_exact_calls():
    torch.set_num_threads(1);torch.manual_seed(13)
    model=ConditionalDenoiser(max_free=30,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.randn(1,3,32))
    calls=[]
    hook=model.register_forward_pre_hook(lambda model,args:calls.append(args[0].detach().numpy().copy()))
    def energy(x):return dict(total=(x-.3).square().mean())
    a=sample(model,condition,23,17,SamplingBudget(steps=7,guide_steps=4),energy)
    hook.remove()
    b=sample(model,condition,23,17,SamplingBudget(steps=7,guide_steps=0))
    assert a['network_calls']==7 and len(calls)==7
    assert a['post_sampling_trajectory_repair_calls']==0
    for i in range(6):np.testing.assert_array_equal(calls[i+1][0],a['trace'][i]['output'])
    assert sum(t['active'] for t in a['trace'][:-1])==3
    assert any(not np.array_equal(a['trace'][i]['input'],b['trace'][i]['input']) for i in (4,5,6))
    assert not np.array_equal(a['free'],b['free'])
    assert max(t['decomposition_error'] for t in a['trace'])<1e-6
    c=sample(model,condition,23,18,SamplingBudget(steps=7,guide_steps=0))
    assert c['initial_noise_hash']!=b['initial_noise_hash']


def test_full_dimension_and_condition_dependency():
    torch.manual_seed(20);model=ConditionalDenoiser(max_free=400,hidden=16,layers=1,heads=2)
    x=torch.randn(1,323);t=torch.tensor([100]);condition=dict(tokens=torch.randn(1,5,32))
    y=model(x,t,condition);assert y.shape==x.shape
    changed=model(x,t,dict(tokens=condition['tokens']+1))
    assert torch.linalg.vector_norm(y-changed)>1e-6
    small=model(x[:,:68],t,dict(tokens=condition['tokens'][:,:2]));assert small.shape==(1,68)


def test_control_change_cap_and_complete_energy_audit():
    torch.set_num_threads(1);torch.manual_seed(17)
    model=ConditionalDenoiser(max_free=20,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32));seen=[]
    def energy(x):return dict(total=(x-3).square().sum())
    result=sample(model,condition,13,2,SamplingBudget(steps=5,guide_steps=3,max_control_change=.005),energy,
                  control_jacobian=10*torch.eye(13,dtype=torch.float64),audit_inactive_energy=True,
                  observer=lambda kind,value:seen.append(kind))
    assert seen.count('initial')==1 and seen.count('before_step')==seen.count('completed_step')==5
    for row in result['trace']:
        assert row['energy_status']=='MEASURED'
        assert np.max(abs(10*(row['guided']-row['clean'])))<=.005+1e-12
        for update in row['updates']:assert update['accepted_maximum_control_change']<=.005+1e-12
