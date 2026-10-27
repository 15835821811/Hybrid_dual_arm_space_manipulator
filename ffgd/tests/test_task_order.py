import numpy as np
from ffgd.contracts import TaskSpec,PoseConstraint,Candidate,Snapshot
from ffgd.validation import input_identity


def test_task_order_has_canonical_identity():
    a=PoseConstraint(1.,0,[1,2,3],np.eye(3));b=PoseConstraint(.4,1,[0,2,1],np.eye(3))
    left=TaskSpec(1.,(a,b));right=TaskSpec(1.,(b,a))
    assert left.poses[0] is b and right.poses[0] is b
    candidate=Candidate(np.zeros((8,17)),1.,7);snapshot=Snapshot('fixture',1,np.zeros(3))
    assert input_identity(candidate,snapshot,left)==input_identity(candidate,snapshot,right)
