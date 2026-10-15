import copy
import time
import numpy as np
import pytest
from ff_prior.storage import Ledger,atomic_json,atomic_npz,read_json
from ff_prior.pipeline import freeze_dataset,config_from
from ff_prior.model_gates import validate_model
from ff_prior.coupled_predictor import converge
from ff_prior.proposal_generator import propose
from ff_prior.contracts import AcceptanceConfig
from ff_prior.torque_tracker import track


def test_published_record_mutation_is_rejected(tmp_path):
    ledger=Ledger(tmp_path,{"test":"publication"})
    row={"sample_id":"x","validation_level":"REJECTED","publication":"DRAFT","data_role":"FF_MODEL_PRIOR"}
    atomic_json(tmp_path/"records/x.json",row)
    freeze_dataset(tmp_path,ledger)
    changed=read_json(tmp_path/"records/x.json"); changed["seed"]=123
    atomic_json(tmp_path/"records/x.json",changed)
    with pytest.raises(ValueError,match="immutable"): freeze_dataset(tmp_path,ledger)


def test_forward_goal_tampering_fails_native_label_check(physical):
    r,initial,c=physical
    ref,_=propose(r,2,0,20261015)
    pred,numerical=converge(initial,ref,r)
    pred["ee"][-1,0,0,3]+=.001
    result=validate_model(ref,pred,AcceptanceConfig(),r,initial,c,numerical)
    assert not result["gates"]["label_consistency"] and result["validation_level"]=="REJECTED"


def test_hard_deadline_stops_actual_steps_without_success(physical):
    r,initial,_=physical
    ref,_=propose(r,0,0,20261015)
    before=r.calls.copy()
    arrays,metadata=track(r,initial,ref,.5,15000,time.time()-1)
    assert metadata["status"]=="BUDGET_EXHAUSTED" and metadata["physics_steps"]==0
    assert r.calls["physics_steps"]==before["physics_steps"]
    assert r.calls["forward"]>before["forward"]


def test_required_unimplemented_torque_gate_blocks(tmp_path):
    import yaml
    cfg=config_from("configs/ff_prior/v7_1.proposed.yaml")
    cfg["validation"]["reference_torque_required"]=True
    file=tmp_path/"config.yaml"; file.write_text(yaml.safe_dump(cfg),encoding="utf-8")
    with pytest.raises(ValueError,match="NOT_IMPLEMENTED"): config_from(file)


def test_static_scene_schema_is_frozen_and_dynamic_rejected():
    from ff_prior.scenes import SceneSpec,Obstacle
    scene=SceneSpec((Obstacle("sphere","sphere",(1.,2.,3.),(.1,0.,0.)),Obstacle("obb","box",(0.,0.,1.),(.1,.2,.3))),"held_out_scene")
    assert SceneSpec.from_dict(scene.to_dict()).scene_hash==scene.scene_hash
    with pytest.raises(ValueError): SceneSpec(scene.objects,"")
    bad=scene.to_dict(); bad["motion"]="DYNAMIC"
    with pytest.raises(ValueError): SceneSpec.from_dict(bad)


def test_study_requires_explicit_frozen_inputs_before_physics(tmp_path):
    from ff_prior.study import run_study
    atomic_json(tmp_path/"archive.json",{})
    atomic_json(tmp_path/"budget.json",{})
    with pytest.raises(ValueError,match="explicit frozen"):
        run_study({},tmp_path/"archive.json",tmp_path/"budget.json",tmp_path,tmp_path/"out",dry_run=True)
    assert not (tmp_path/"out").exists()
