import numpy as np
import pytest
from ffgd.robot import Robot
from ffgd.contracts import Candidate
import ffgd.execution as execution


def test_refinement_preserves_full_state_and_rejects_other_model_changes():
    r=Robot();d=r.new_data();d.qvel[:]=.001;d.ctrl[:]=.002;d.qacc_warmstart[:]=.003;d.time=.7
    snap=r.snapshot(d);rr,ss=execution.refine_physics(r,snap,2)
    assert rr.model.opt.timestep==.001 and rr.physical_hash==r.physical_hash
    np.testing.assert_array_equal(rr.snapshot(rr.restore(ss)).state,snap.state)
    assert execution.refine_physics(r,snap,1)==(r,snap)
    r.model.body_mass[r.base_body]*=1.01;r._hash=None;changed=r.snapshot(d)
    with pytest.raises(ValueError,match='BEYOND_TIMESTEP'):execution.refine_physics(r,changed,2)


def test_native_substeps_hold_control_and_replay_without_state_injection(tmp_path,monkeypatch):
    r=Robot();initial=r.snapshot(r.new_data());r,initial=execution.refine_physics(r,initial,2)
    C=np.tile(r.spec.planner_zero,(24,1));C[3:,0]+=np.linspace(0,.0001,21)
    c=Candidate(C,.2,1);calls=[];original=execution.command
    def counted(*args):
        calls.append(float(args[1].time));return original(*args)
    monkeypatch.setattr(execution,'command',counted)
    bundle,report=execution.track(r,initial,c,tmp_path,dict(physical_geometry_clearance_m=0.),control_substeps=2)
    assert report['status']=='FINISHED' and report['actual_steps']==200
    assert len(calls)==report['controller_updates']==100
    np.testing.assert_allclose(np.diff(calls),.002,atol=1e-15)
    np.testing.assert_array_equal(bundle['ctrl'][::2],bundle['ctrl'][1::2])
    assert np.max(abs(bundle['ctrl']))>0
    assert execution.replay(r,bundle,initial)['max_state_abs']==0
    assert report['state_writes_after_restore']==0 and report['control_period']==.002


def test_recorded_sweep_uses_executed_dt_after_substep_refinement(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import ffgd.problem_io as io
    from ffgd.calibration_momentum import recorded_input_sweep
    planning=Robot();initial=planning.snapshot(planning.new_data())
    r,snapshot=execution.refine_physics(planning,initial,2)
    source=tmp_path/'source';source.mkdir();out=tmp_path/'sweep';out.mkdir()
    candidate=Candidate(np.tile(r.spec.planner_zero,(24,1)),.02,1)
    profile=dict(physical_geometry_clearance_m=0.,replay_state_atol=1e-10)
    bundle,report=execution.track(r,snapshot,candidate,source,profile,control_substeps=2)
    (source/'provenance.json').write_text(json.dumps(dict(planning_directory=str(tmp_path),
        profile=profile,candidate_hash=candidate.identity,physics_substeps=2,physics_dt=.001,
        execution_model_hash=r.model_hash,control_period=.002)))
    monkeypatch.setattr(io,'load_problem',lambda _:(planning,initial,SimpleNamespace(duration=.02),{}))
    result=recorded_input_sweep(out,dict(execution_directory=str(source)))
    assert result['passed']
    assert [row['physics_dt'] for row in result['rows']]==[.001,.0005,.00025]
    assert [row['native_steps'] for row in result['rows']]==[20,40,80]
    assert result['rows'][0]['source_state_error']==0
    assert all(row['held_control_period']==.002 and row['recorded_input_period']==.001 for row in result['rows'])
