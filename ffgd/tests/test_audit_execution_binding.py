"""Artifact identity tests, not a physical success claim."""
import json
import numpy as np
import pytest
from ffgd.contracts import Candidate
from ffgd.architecture_audit import execution_plan_binding


def fixture(tmp_path):
    plan=tmp_path/'plan';folder=plan/'candidate_000';folder.mkdir(parents=True)
    run=tmp_path/'execution';run.mkdir()
    candidate=Candidate(np.zeros((24,17)),5.,13001)
    np.savez_compressed(folder/'candidate.npz',control=candidate.control,duration=candidate.duration,seed=candidate.seed)
    (plan/'planning_result.json').write_text(json.dumps(dict(status='PLAN_VALID',selected_index=0,
        records=[dict(index=0,identity=candidate.identity)])))
    (run/'provenance.json').write_text(json.dumps(dict(planning_directory=str(plan),candidate_hash=candidate.identity)))
    (run/'result.json').write_text(json.dumps(dict(execution=dict(candidate_hash=candidate.identity))))
    np.savez_compressed(run/'execution.npz',control_points=candidate.control,duration=candidate.duration)
    return plan,run,candidate


def test_audit_rejects_execution_from_other_planning_run(tmp_path):
    plan,run,candidate=fixture(tmp_path)
    with pytest.raises(ValueError,match='EXECUTION_OUTSIDE_AUDITED_PLANS'):
        execution_plan_binding(run,[tmp_path/'different_checkpoint_plan'])
    assert execution_plan_binding(run,[plan])['candidate_hash']==candidate.identity


def test_audit_rejects_changed_execution_reference_despite_matching_reports(tmp_path):
    plan,run,candidate=fixture(tmp_path)
    C=candidate.control.copy();C[8,3]=.01
    np.savez_compressed(run/'execution.npz',control_points=C,duration=candidate.duration)
    with pytest.raises(ValueError,match='EXECUTION_REFERENCE_NOT_SELECTED_PLAN'):
        execution_plan_binding(run,[plan])


def test_audit_rejects_mismatched_selected_identity(tmp_path):
    plan,run,candidate=fixture(tmp_path)
    (run/'result.json').write_text(json.dumps(dict(execution=dict(candidate_hash='other'))))
    with pytest.raises(ValueError,match='EXECUTION_SELECTED_IDENTITY_MISMATCH'):
        execution_plan_binding(run,[plan])
