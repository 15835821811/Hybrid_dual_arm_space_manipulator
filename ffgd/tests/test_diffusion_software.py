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


def test_full_rank_metric_polyak_is_bounded_descent():
    torch.set_num_threads(1);torch.manual_seed(18)
    model=ConditionalDenoiser(max_free=20,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32));metric=torch.diag(torch.linspace(1,20,13,dtype=torch.float64))
    def energy(x):return dict(total=((x-.3)@(metric@(x-.3)))/2)
    budget=SamplingBudget(steps=5,guide_steps=3,updates_per_step=3,step_size=1.,step_rule='polyak',max_update_norm=1.)
    result=sample(model,condition,13,2,budget,energy,metric=metric,control_jacobian=torch.eye(13,dtype=torch.float64))
    for row in result['trace']:
        for update in row['updates']:
            assert update['directional_descent']>0 and update['accepted']
            assert update['after']<update['before'] and np.isfinite(update['step_coefficient'])


def test_task_metric_preserves_all_coordinates_and_refreshes_inside_denoising():
    torch.set_num_threads(1);torch.manual_seed(19)
    model=ConditionalDenoiser(max_free=20,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32));A=torch.randn(3,13,dtype=torch.float64)
    def energy(x):
        energy.task_residual=A@x-.2
        return dict(total=energy.task_residual.square().sum()+.01*x.square().sum())
    budget=SamplingBudget(steps=5,guide_steps=3,updates_per_step=3,task_metric=True,step_rule='polyak',step_size=2.,max_update_norm=1.)
    result=sample(model,condition,13,2,budget,energy)
    for row in result['trace']:
        for update in row['updates']:
            info=update['task_metric']
            assert info['refresh_step']==row['index'] and info['full_free_dimension']==13 and info['residual_dimension']==3
            assert update['after']<=update['before'] and update['directional_descent']>0
    assert result['network_calls']==5 and result['post_sampling_trajectory_repair_calls']==0


def test_batched_residual_vjps_match_all_scalar_rows():
    from ffgd.guidance_metric import residual_jacobian
    torch.manual_seed(29);A=torch.randn(11,17,dtype=torch.float64)
    x=torch.randn(17,dtype=torch.float64,requires_grad=True);residual=torch.sin(A@x)+.1*(A@x).square()
    scalar=residual_jacobian(residual,x)
    for batch in (2,4,12):torch.testing.assert_close(residual_jacobian(residual,x,batch_size=batch),scalar,rtol=1e-12,atol=1e-12)


def test_polyak_cap_handles_nonzero_energy_floor_without_extra_line_trials():
    # A nonzero floor makes the zero-bound Polyak coefficient arbitrarily large.
    # This is a software optimization fixture, not physical success evidence.
    torch.set_num_threads(1);torch.manual_seed(45)
    model=ConditionalDenoiser(max_free=12,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32))
    def energy(x):return dict(total=10000.+.5*x.square().sum())
    common=dict(steps=4,guide_steps=3,updates_per_step=3,step_rule='polyak',step_size=2.,max_update_norm=100.)
    original=sample(model,condition,8,23,SamplingBudget(**common),energy)
    capped=sample(model,condition,8,23,SamplingBudget(**common,max_step_coefficient=.25),energy)
    assert any(not update['accepted'] for row in original['trace'] for update in row['updates'])
    for row in capped['trace']:
        for update in row['updates']:
            assert update['accepted'] and update['after']<update['before']
            assert update['step_coefficient']<=.25 and update['trials']<=2
    assert capped['network_calls']==4 and capped['post_sampling_trajectory_repair_calls']==0


def test_rejected_effective_step_backs_off_within_fixed_budget():
    # Nonzero energy floor reproduces the observed Polyak overshoot. A very
    # large uncapped coefficient also checks backoff AFTER the trust caps.
    torch.set_num_threads(1);torch.manual_seed(45)
    model=ConditionalDenoiser(max_free=12,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32))
    calls=[]
    def energy(x):
        calls.append(x.detach().clone())
        return dict(total=10000.+.5*x.square().sum())
    common=dict(steps=4,guide_steps=3,updates_per_step=3,line_trials=2,step_rule='polyak',step_size=2.,max_update_norm=100.)
    result=sample(model,condition,8,23,SamplingBudget(**common,rejection_backoff=True),energy)
    updates=[u for row in result['trace'] for u in row['updates']]
    assert len(updates)==9 and len(calls)<=27
    assert any(not u['accepted'] for u in updates) and any(u['accepted'] for u in updates)
    for row in result['trace']:
        if not row['updates']:continue
        assert row['updates'][0]['feedback_cap_before'] is None
        for u,v in zip(row['updates'],row['updates'][1:]):
            if not u['accepted']:
                assert u['feedback_cap_after']==u['effective_step_coefficient']/4
                assert v['step_coefficient']<=u['feedback_cap_after']
    assert all(u['after']<=u['before'] and u['trials']<=2 for u in updates)
    assert result['network_calls']==4 and result['post_sampling_trajectory_repair_calls']==0
    for a,b in zip(result['trace'],result['trace'][1:]):np.testing.assert_array_equal(a['output'],b['input'])


def test_retained_network_noise_transports_guidance_into_next_actual_network():
    torch.set_num_threads(1);torch.manual_seed(72)
    model=ConditionalDenoiser(max_free=20,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32));inputs=[]
    hook=model.register_forward_pre_hook(lambda m,args:inputs.append(args[0][0].detach().numpy().copy()))
    def energy(x):return dict(total=(x-.3).square().mean())
    result=sample(model,condition,13,11,SamplingBudget(steps=8,guide_steps=5,guidance_transport='retain_network_noise'),energy)
    hook.remove();alpha=np.cumprod(1-np.linspace(.0001,.02,1000))
    assert len(inputs)==8 and result['post_sampling_trajectory_repair_calls']==0
    for i,row in enumerate(result['trace']):
        np.testing.assert_array_equal(inputs[i],row['input'])
        k=row['noise_level'];a=np.sqrt(alpha[k]);s=np.sqrt(1-alpha[k])
        an=1. if i==7 else np.sqrt(alpha[result['trace'][i+1]['noise_level']]);sn=np.sqrt(1-an*an)
        nominal=an*row['clean']+sn*(row['input']-a*row['clean'])/s
        np.testing.assert_allclose(row['output']-nominal,an*(row['guided']-row['clean']),atol=6e-7,rtol=1e-5)
        assert row['decomposition_reference']=='network_clean' and row['decomposition_error']<1e-6
        if i<7:np.testing.assert_array_equal(row['output'],inputs[i+1])
    plain=sample(model,condition,13,11,SamplingBudget(steps=8,guide_steps=0))
    retained=sample(model,condition,13,11,SamplingBudget(steps=8,guide_steps=0,guidance_transport='retain_network_noise'))
    np.testing.assert_array_equal(plain['free'],retained['free'])


def test_quadratic_rejection_uses_two_trials_and_later_network_consumes_result():
    # An exact convex quadratic has a known line minimum. With a tenfold
    # overshoot, both legacy trials fail; interpolation reaches that minimum.
    # This is a numerical/software fixture, not a learned physical plan.
    torch.set_num_threads(1);torch.manual_seed(84)
    model=ConditionalDenoiser(max_free=12,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32));energy_calls=[];network_inputs=[]
    def energy(x):
        energy_calls.append(x.detach().clone())
        return dict(total=.5*x.square().sum())
    common=dict(steps=4,guide_steps=3,step_size=10.,max_update_norm=100.,guidance_transport='retain_network_noise')
    old=sample(model,condition,8,34,SamplingBudget(**common),energy)
    assert all(not u['accepted'] for row in old['trace'] for u in row['updates'])
    energy_calls.clear()
    hook=model.register_forward_pre_hook(lambda m,args:network_inputs.append(args[0][0].detach().numpy().copy()))
    new=sample(model,condition,8,34,SamplingBudget(**common,line_rule='quadratic_rejection'),energy)
    hook.remove()
    assert new['network_calls']==4 and new['post_sampling_trajectory_repair_calls']==0
    assert len(energy_calls)==9
    for row in new['trace']:
        for u in row['updates']:
            assert u['accepted'] and u['trials']==2 and abs(u['line_scale']-.1)<1e-12
            assert u['after']<1e-20 and u['trial_evaluations'][0]['energy']>u['before']
    for i in range(3):np.testing.assert_array_equal(new['trace'][i]['output'],network_inputs[i+1])
    assert not np.array_equal(old['trace'][2]['input'],new['trace'][2]['input'])


def test_quadratic_trial_bounds_and_invalid_observation_fallback():
    import pytest
    from ffgd.diffusion import rejected_trial_scale
    # Values from the preserved physical counterexample, not new evidence.
    scale=rejected_trial_scale(32.52371179203436,180.78006536058697,65.04742358406973,1.,'quadratic_rejection')
    assert abs(scale-.15247602375443928)<1e-14
    assert rejected_trial_scale(1.,1e20,2.,1.,'quadratic_rejection')==.05
    for after in (float('nan'),float('inf')):
        assert rejected_trial_scale(1.,after,2.,.5,'quadratic_rejection')==.25
    assert rejected_trial_scale(1.,2.,0.,1.,'quadratic_rejection')==.5
    with pytest.raises(ValueError):SamplingBudget(line_rule='unbounded_search')


def test_quadratic_failed_trial_backoff_uses_last_actual_scale():
    torch.set_num_threads(1);torch.manual_seed(45)
    model=ConditionalDenoiser(max_free=12,hidden=16,layers=1,heads=2)
    def energy(x):return dict(total=10000.+.5*x.square().sum())
    budget=SamplingBudget(steps=4,guide_steps=3,updates_per_step=3,line_rule='quadratic_rejection',
                          step_rule='polyak',step_size=2.,max_update_norm=100.,rejection_backoff=True)
    result=sample(model,dict(tokens=torch.zeros(1,3,32)),8,23,budget,energy)
    updates=[u for row in result['trace'] for u in row['updates']]
    assert any(not u['accepted'] for u in updates)
    assert any(row['updates'] and row['updates'][-1]['feedback_cap_after'] is not None for row in result['trace'][:-1])
    for row in result['trace']:
        if not row['updates']:continue
        first=row['updates'][0]
        assert first['feedback_cap_before'] is None
        assert first['step_coefficient']==first['uncapped_step_coefficient']
        for u,v in zip(row['updates'],row['updates'][1:]):
            if not u['accepted']:
                assert u['feedback_cap_after']==u['effective_step_coefficient']*u['trial_evaluations'][-1]['scale']*.5
                assert v['step_coefficient']<=u['feedback_cap_after']
    assert all(len(u['trial_evaluations'])<=2 for u in updates)


def test_local_metric_refresh_is_inside_denoising_and_preserves_next_network_input():
    torch.set_num_threads(1);torch.manual_seed(61)
    model=ConditionalDenoiser(max_free=12,hidden=16,layers=1,heads=2)
    condition=dict(tokens=torch.zeros(1,3,32));current=[];refresh=[];observed=[]
    def energy(x):
        if torch.is_grad_enabled():current.append(x.detach().clone())
        return dict(total=.5*(x*torch.arange(1.,9.)).square().sum())
    def metric():
        refresh.append(current[-1].clone())
        return torch.diag(torch.arange(1.,9.,dtype=torch.float64).square()),dict(native_pose_states=0,collision_rows=0)
    hook=model.register_forward_pre_hook(lambda m,args:observed.append(args[0][0].detach().numpy().copy()))
    budget=SamplingBudget(steps=5,guide_steps=3,updates_per_step=3,step_size=.1,max_update_norm=1.,local_metric='instantaneous_multispace',guidance_transport='retain_network_noise')
    result=sample(model,condition,8,56,budget,energy,metric_callback=metric);hook.remove()
    assert len(refresh)==3 and len(current)==9 and result['network_calls']==5
    assert result['post_sampling_trajectory_repair_calls']==0
    for i in range(4):np.testing.assert_array_equal(result['trace'][i]['output'],observed[i+1])
    for j,row in enumerate(result['trace'][2:]):
        np.testing.assert_array_equal(refresh[j].numpy(),row['clean'])
        assert all(u['local_metric']['refresh_step']==j+2 and u['local_metric']['refresh_update']==0 for u in row['updates'])
    import pytest
    with pytest.raises(ValueError,match='one task curvature'):SamplingBudget(task_metric=True,local_metric='instantaneous_multispace')
    with pytest.raises(ValueError,match='missing declared'):sample(model,condition,8,56,budget,energy)
