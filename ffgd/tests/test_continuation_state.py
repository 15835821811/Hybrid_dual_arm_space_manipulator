"""Software fixtures exercise exact native-state IO, not physical success claims."""
import json
from pathlib import Path
import numpy as np
import pytest
from ffgd.problem_io import load_problem, export_continuation
from ffgd.execution import refine_physics

ROOT=Path(__file__).resolve().parents[2]

def source(tmp_path):
    task=ROOT/'configs/ffgd/dev_v1/short_duration_torque.json'
    robot,snapshot,_,problem=load_problem(task)
    planning=tmp_path/'planning';planning.mkdir()
    (planning/'problem.json').write_text(json.dumps(problem))
    native,_=refine_physics(robot,snapshot,4)
    d=native.new_data();d.time=1.234
    d.qvel[native.base_dofs]=[.01,.02,.03,.04,.05,.06]
    d.qpos[native.qpos_ids[0]]+=1e-6
    d.qvel[native.dof_ids[0]]=.003
    d.ctrl[:]=np.linspace(-.01,.01,native.model.nu)
    actual=native.snapshot(d)
    run=tmp_path/'execution';run.mkdir()
    (run/'provenance.json').write_text(json.dumps(dict(planning_directory=str(planning),physics_substeps=4,execution_model_hash=native.model_hash)))
    (run/'execution.json').write_text(json.dumps(dict(controller='floating_ff_torque_pd_v1')))
    (run/'acceptance.json').write_text(json.dumps(dict(status='EXECUTION_VALID',scope='SYNTHETIC_IO_TEST_ONLY')))
    np.savez(run/'execution.npz',states=np.stack([actual.state]*5),mask=actual.mask,control_substeps=4)
    return task,run,actual

def test_exact_actual_state_and_memory_roundtrip(tmp_path):
    task,run,actual=source(tmp_path)
    result=export_continuation(tmp_path/'next',run,task)
    _,restored,_,_=load_problem(tmp_path/'next/problem.json')
    np.testing.assert_array_equal(actual.state,restored.state)
    assert result['reduced_planner_domain']=='REQUIRES_FULL_STATE_PREDICTION'
    assert result['actual_time']==1.234
    assert result['next_planning_status']=='NOT_RUN'
    assert result['base_velocity']==[.01,.02,.03,.04,.05,.06]

def test_mid_hold_handoff_is_not_silently_reset(tmp_path):
    task,run,_=source(tmp_path)
    with pytest.raises(ValueError,match='CONTROL_UPDATE_BOUNDARY'):
        export_continuation(tmp_path/'next',run,task,1)
    assert not (tmp_path/'next/initial_state.npz').exists()

def test_failed_execution_not_certified_for_handoff(tmp_path):
    task,run,_=source(tmp_path)
    (run/'execution.json').write_text(json.dumps(dict(controller='floating_ff_torque_pd_v1')))
    (run/'acceptance.json').write_text('{"status":"FAILED"}')
    with pytest.raises(ValueError,match='SOURCE_NOT_ACCEPTED'):
        export_continuation(tmp_path/'next',run,task)
