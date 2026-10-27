"""Software budget and isolation tests; these are not robot success evidence."""
import numpy as np
import pytest
import torch
from ffgd.baselines.compare import optimize, nonlearned_initial
from ffgd.diffusion import SamplingBudget
from ffgd.trajectory import Boundary, BoundaryCodec
from ffgd.architecture_guard import LegacySentinel, ArchitectureViolation


def quadratic(x):
    quadratic.task_residual=x
    return dict(total=(x*x).sum())


def test_baseline_optimizer_enforces_physics_budget_and_keeps_partial():
    sampling=SamplingBudget(task_metric=True,steps=3,guide_steps=3,updates_per_step=1,
                            step_size=.2,max_update_norm=2.,max_control_change=2.)
    result,report=optimize(np.array([1.,-1.]),quadratic,torch.eye(2,dtype=torch.float64),torch.eye(2,dtype=torch.float64),
                          sampling,dict(energy_evaluations=3,gradient_evaluations=3,jacobian_refreshes=3))
    assert report['status']=='INCOMPLETE_PHYSICS_BUDGET'
    assert report['energy_evaluations']==3
    assert report['gradient_evaluations']==2
    assert len(report['trace'])==1
    assert np.linalg.norm(result)<np.sqrt(2.)


def test_nonlearned_initializations_preserve_only_public_boundaries():
    codec=BoundaryCodec(10,2,3.,[Boundary(0.,0,(0,1),(.2,-.3)),Boundary(0.,1,(0,1),(0.,0.)),Boundary(3.,1,(0,1),(0.,0.))])
    values=[nonlearned_initial(codec,np.array([.2,-.3]),np.array([2.,4.]),500+i,i) for i in range(5)]
    assert len({v.tobytes() for v in values})==5
    for z in values:assert np.max(abs(codec.A@codec.decode(z).ravel()-codec.b))<1e-10


def test_main_guard_rejects_already_imported_comparison_entry():
    with pytest.raises(ArchitectureViolation):
        with LegacySentinel([]):
            nonlearned_initial(None,None,None,0,0)


def test_nonlearned_entry_never_reads_checkpoint_or_train_scaler(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from ffgd.baselines import compare
    root=Path(__file__).parents[2]
    config=json.loads((root/'configs/ffgd/comparison_smoke_plan.json').read_text(encoding='utf-8'))
    config['checkpoint']=str(tmp_path/'MUST_NOT_BE_READ.pt')
    config['device']='NO_NEURAL_DEVICE'
    config['profile']=str(root/'configs/ffgd/numerical_profile_v1.json')
    planning=tmp_path/'planning.json';planning.write_text(json.dumps(config),encoding='utf-8')
    def forbidden(*args,**kwargs):raise AssertionError('C loaded learned resources')
    monkeypatch.setattr(compare,'load_model',forbidden)
    # Skip numerical optimization here: this test checks method identity only.
    monkeypatch.setattr(compare,'optimize',lambda initial,*args:(initial,dict(status='FINISHED',trace=[])))
    monkeypatch.setattr(compare,'validate',lambda *args:dict(status='REJECTED',scope='SOFTWARE_IDENTITY_FIXTURE',input_unchanged=True))
    output=tmp_path/'run';output.mkdir()
    result=compare.run(output,dict(method='C_NONLEARNED',planning_config=str(planning),
        task=str(root/'configs/ffgd/dev_v1/free_translation.json'),
        physical_budget=dict(energy_evaluations=9,gradient_evaluations=3,jacobian_refreshes=3)))
    manifest=json.loads((output/'planning_manifest.json').read_text(encoding='utf-8'))
    assert manifest['checkpoint']=={} and not manifest['shared_train_scaler'] and not manifest['learned_network_used']
    assert result['records'][0]['observed_network_calls']==0 and result['requested_candidates']==1
    saved=json.loads((output/'problem.json').read_text(encoding='utf-8'))
    assert saved['initial']=={'snapshot_file':'initial_state.npz'}
