"""Selection follows the task masks and tolerances without modifying evidence."""
import copy
import numpy as np
import pytest
from ffgd.contracts import TaskSpec,PoseConstraint
from ffgd.validation import selection_score


def test_disabled_orientation_cannot_change_candidate_rank():
    task=TaskSpec(1.,(PoseConstraint(1.,0,[0,0,0],np.eye(3),rotation_enabled=False),))
    report=dict(status='PLAN_VALID',task_errors=[dict(time=1.,tip=0,position_m=.001,rotation_rad=3.)],max_nominal_torque_ratio=.1)
    original=copy.deepcopy(report)
    assert selection_score(report,task)==pytest.approx(.3)
    assert report==original
    report['task_errors'][0]['rotation_rad']=0.
    assert selection_score(report,task)==pytest.approx(.3)


def test_declared_tolerances_and_position_mask():
    task=TaskSpec(1.,(PoseConstraint(1.,0,[0,0,0],np.eye(3),rotation_tolerance=.01,position_enabled=False),))
    report=dict(status='PLAN_VALID',task_errors=[dict(time=1.,tip=0,position_m=10.,rotation_rad=.004)],max_nominal_torque_ratio=.1)
    assert selection_score(report,task)==pytest.approx(.5)
    report['status']='REJECTED'
    with pytest.raises(ValueError,match='COMPLETE_VALID'):selection_score(report,task)
    report['status']='PLAN_VALID';report['task_errors'][0]['tip']=1
    with pytest.raises(ValueError,match='MISMATCH'):selection_score(report,task)
