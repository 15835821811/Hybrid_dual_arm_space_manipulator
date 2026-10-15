from dataclasses import asdict
import numpy as np
import torch
import pytest
from ff_prior.diffusion import NoiseTransformer,ddim,train,load_checkpoint
from ff_prior.dataset import Scaler
from ff_prior.contracts import frozen_task,AcceptanceConfig
from ff_prior.storage import atomic_npz
from ff_prior.spline import Reference
from ff_prior.guidance import FullPathCost,SubspaceGuidance,public_goal_initializer
from ff_prior.proposal_generator import propose
from ff_prior.coupled_predictor import predict_coupled


def test_training_updates_reload_schema_and_exact_ddim_calls(tmp_path):
    # Synthetic software-only fixture, never counted as robot training evidence.
    torch.set_num_threads(2)
    rng=np.random.default_rng(2)
    records=[]
    for i in range(6):
        z=rng.normal(size=(19,17))*.05
        atomic_npz(tmp_path/f"{i}.npz",Z=z)
        c={"initial_features":np.zeros(17),"goals":np.tile(np.eye(4),(2,1,1)),"duration":10.,"pose_mask":np.ones((2,6))}
        records.append({"sample_id":str(i),"arrays":f"{i}.npz","condition":c})
    dataset={"status":"READY","eligible":{"TRAIN":["0","1","2","3"],"VAL":["4","5"],"TEST":[],"DEV":[]}}
    result=train(tmp_path,records,dataset,"software_fixture","initial",updates=2,software_fixture=True)
    assert result["weights_changed"] and result["reload_max_error"]==0
    model,scaler=load_checkpoint(tmp_path/"learning","software_fixture","initial")
    task=frozen_task(records[0]["condition"],{})
    for steps in (1,7,16):
        z,report=ddim(model,scaler,task,steps,123)
        assert report["actual_network_calls"]==steps
        ref=Reference(z,np.zeros(17),10)
        np.testing.assert_array_equal(ref.control[:3],np.zeros((3,17)))
    with pytest.raises(ValueError): load_checkpoint(tmp_path/"learning","wrong","initial")
    resumed=train(tmp_path,records,dataset,"software_fixture","initial",updates=3,software_fixture=True)
    assert len(resumed["history"])==3


def test_guidance_is_inside_reverse_process():
    torch.set_num_threads(2)
    c={"initial_features":np.zeros(17),"goals":np.tile(np.eye(4),(2,1,1)),"duration":10.,"pose_mask":np.ones((2,6))}
    scaler=Scaler.fit(np.zeros((2,19,17)),[c,c],["TRAIN","TRAIN"],"robot","initial")
    model=NoiseTransformer(len(scaler.condition(c)),hidden=32,layers=1,heads=4)
    model.eval()
    seen=[]
    def guide(z,iteration):
        seen.append((iteration,model.calls))
        return z*.9,{"iteration":iteration}
    _,report=ddim(model,scaler,frozen_task(c,{}),7,1,guide)
    assert seen==[(5,6),(6,7)] and report["actual_network_calls"]==7


def test_actual_full_path_subspace_cost_uses_public_task(physical):
    robot,initial,collision=physical
    reference,_=propose(robot,8,0,20261015)
    pred=predict_coupled(initial,reference,robot)
    profile=AcceptanceConfig()
    task=frozen_task({"initial_features":robot.spec.planner_zero,"goals":pred["ee"][-1],
        "duration":10.,"pose_mask":np.ones((2,6))},{"robot_hash":initial.model_hash,"initial_state_id":initial.state_id,
        "pair_policy_hash":collision.policy_hash,"acceptance":asdict(profile)})
    cost=FullPathCost(task,robot,initial,collision,profile)
    initializer=public_goal_initializer(task,robot,initial)
    z,log=SubspaceGuidance(cost)(initializer,0)
    assert log["centered_evaluations"]==16 and log["total_cost_evaluations"]>=20
    assert log["after_cost"]<=log["before_cost"]
    assert np.linalg.norm(SubspaceGuidance(cost).directions[:,-1])>0
    assert not log["full_exact_gradient"] and not log["production_eligible"]
