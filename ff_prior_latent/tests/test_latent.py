"""Software fixtures verify algebra; they are not physical experiment evidence."""
import time
import numpy as np
import pytest
import torch
from ff_prior.storage import read_json,atomic_json,Ledger,digest
from ff_prior_latent.learning import Codec,LatentDenoiser,sample


def fixture():
    row=read_json('artifacts/v7_1/qualified_guidance_loop_001/frozen_tasks.json')['training_generation_entries'][0]
    task=row['task']; base=np.tile(task['condition']['initial_features'],(19,1))
    rng=np.random.default_rng(3); s=np.linspace(.01,1,19)
    z=base+np.einsum('t,nj->ntj',s,rng.normal(size=(12,17))*.01)
    codec=Codec.fit(z,np.repeat(base[None],12,0),[task['condition']]*12,[str(i) for i in range(12)],['TRAIN']*12)
    return task,base,z,codec


def test_train_only_codec_reconstructs_known_temporal_subspace():
    task,base,z,codec=fixture()
    assert codec.rank==1 and codec.dim==17
    assert np.max(abs(codec.decode(codec.encode(z[0],base),base)-z[0]))<1e-12
    with pytest.raises(ValueError,match='TRAIN-only'):
        Codec.fit(z,np.repeat(base[None],12,0),[task['condition']]*12,[str(i) for i in range(12)],['TRAIN']*11+['VAL'])


def test_latent_ddim_reproducible_finite_and_keeps_final_projection_exact():
    task,base,z,codec=fixture(); before=digest(task)
    torch.manual_seed(3); net=LatentDenoiser(codec.dim,len(codec.feature_mean)); net.eval()
    sentinel=np.nextafter(base,np.inf); calls=[]
    def projector(raw,step):
        calls.append(step); assert np.isfinite(raw).all()
        return sentinel.copy(),{'software_fixture':True}
    a,info=sample((net,codec),base,task,7,projector)
    b,again=sample((net,codec),base,task,7,projector)
    assert np.array_equal(a,sentinel) and np.array_equal(a,b) and calls==[15,15]
    assert info==again and net.calls==32 and info['actual_network_calls']==16
    assert max(s['decomposition_residual_after'] for s in info['steps'])<1e-12
    assert digest(task)==before


def test_codec_has_no_fit_access_to_validation_or_teacher_fields():
    task,base,z,codec=fixture(); original=codec.center(task['condition'],base)
    edited=dict(task['condition'],teacher_Z=np.full((19,17),1e6),scene_label='private_witness')
    with pytest.raises(ValueError,match='schema violation'):
        codec.center(edited,base)
    assert np.array_equal(original,codec.center(task['condition'],base))
    assert codec.state['fit_split']=='TRAIN' and len(codec.state['fit_ids'])==12


def test_empty_results_cannot_claim_learning_or_diffusion_advantage(tmp_path):
    from ff_prior_latent.reporting import report
    atomic_json(tmp_path/'frozen_tasks.json',{'evaluation_entries':[]})
    result=report(tmp_path,Ledger(tmp_path,{}),'RUNNING')
    assert not result['stability_gate_passed'] and not result['exploratory_advantage_gate_passed'] and not result['full_goal_achieved']
    assert not any(result['advantage_against_each_baseline'].values())


def test_all_methods_share_unchanged_physical_budget():
    from ff_prior_latent.experiment import PROTOCOL
    from ff_prior_feasible.experiment import PROTOCOL as previous
    for name in ('guidance','validation','candidate_coupled_ceiling','candidate_integration_interval_ceiling','candidate_wall_ceiling_s','seed','candidates_per_task'):
        assert PROTOCOL[name]==previous[name]
    assert not any(PROTOCOL[k] for k in ('expansion','TEST','pilot','study','physical_gates_changed','frozen_goal_relabel'))


def test_active_background_worker_prevents_duplicate_launch(tmp_path):
    from ff_prior_latent.job import Monitor,launch
    atomic_json(tmp_path/'run.json',{})
    with Monitor(tmp_path) as monitor:
        with pytest.raises(RuntimeError,match='alive'): launch(tmp_path/'source',tmp_path,True)
        monitor.finish({'status':'FINISHED'})
