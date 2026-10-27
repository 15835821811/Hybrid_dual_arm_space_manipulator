"""Synthetic timing regression only; no physical acceptance is claimed."""
import json
import numpy as np
import torch
from ffgd import api
from ffgd.robot import Robot
from ffgd.contracts import TaskSpec, PoseConstraint
from ffgd.diffusion import SamplingBudget


def test_validation_timeout_preserves_verdict_but_never_selects(tmp_path,monkeypatch):
    torch.set_num_threads(1);r=Robot();snapshot=r.snapshot(r.new_data());clock=[0.]
    monkeypatch.setattr(api.time,'perf_counter',lambda:clock[0])
    model=torch.nn.Linear(323,323)
    checkpoint=dict(update=1,training_config={'scope':'SOFTWARE_TIMING_FIXTURE'},dataset_manifest={'source_kind':'FF_MODEL_PRIOR'},
        robot_physical_hash=r.physical_hash,mean=np.zeros(323),scale=np.ones(323),condition_mean=np.zeros(32),condition_scale=np.ones(32))
    monkeypatch.setattr(api,'load_model',lambda *args:(model,checkpoint,{'scope':'SOFTWARE_TIMING_FIXTURE'}))
    def sample(*args,**kwargs):
        model(torch.zeros((1,323)))
        arrays={name:np.zeros(323) for name in ('input','clean','guided','output')}
        return dict(free=np.zeros(323),trace=[arrays],network_calls=1,initial_noise_hash='software_fixture',post_sampling_trajectory_repair_calls=0)
    monkeypatch.setattr(api,'sample',sample)
    def validation(*args):
        clock[0]=2.
        return dict(status='PLAN_VALID',input_unchanged=True,scope='SOFTWARE_FIXTURE')
    monkeypatch.setattr(api,'validate',validation)
    candidate,result=api.plan(r,snapshot,TaskSpec(4.,(PoseConstraint(4.,0,[1,2,3],np.eye(3)),)),'fixture',
        api.PlanningBudget(seeds=(1,),sampling=SamplingBudget(steps=1,guide_steps=0),max_wall_seconds=1.),{},tmp_path)
    assert candidate is None and result['selected_index'] is None and result['valid']==0
    assert result['records'][0]['status']=='INCOMPLETE_BUDGET'
    assert result['records'][0]['sampled'] and result['records'][0]['validated']
    assert json.loads((tmp_path/'candidate_000/validation.json').read_text())['status']=='PLAN_VALID'
