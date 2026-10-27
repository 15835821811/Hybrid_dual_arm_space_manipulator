import json
from pathlib import Path
import numpy as np
import torch
from ffgd.robot import Robot
from ffgd.physics import DifferentiableTree
from ffgd.propagation import Propagator
from ffgd.geometry import Geometry
from ffgd.trajectory import Boundary,BoundaryCodec
from ffgd.contracts import Candidate,TaskSpec,PoseConstraint
from ffgd.validation import validate,input_identity


def test_native_validator_readonly_and_rejects_changed_goal():
    torch.set_num_threads(1);r=Robot();d=r.new_data();snapshot=r.snapshot(d);tree=DifferentiableTree(r)
    prop=Propagator(tree,snapshot);geometry=Geometry(r);q0=r.spec.planner_zero;T=.1
    bc=[Boundary(0,j,tuple(range(17)),tuple(q0) if j==0 else (0.,)*17) for j in (0,1,2)]
    codec=BoundaryCodec(8,17,T,bc);candidate=Candidate(np.tile(q0,(8,1)),T,1)
    poses=[]
    for tip,body in enumerate(r.tips):
        R=d.xmat[body].reshape(3,3).copy();pos=d.xpos[body].copy()
        if tip==0:pos+=R@np.array([.0475,0,0])
        poses.append(PoseConstraint(T,tip,pos,R))
    task=TaskSpec(T,tuple(poses));before=input_identity(candidate,snapshot,task)
    profile=json.loads((Path(__file__).parents[2]/'configs/ffgd/numerical_profile_v1.json').read_text())
    report=validate(candidate,snapshot,task,prop,codec,geometry,profile)
    assert report['status']=='PLAN_VALID',report
    assert before==input_identity(candidate,snapshot,task)
    changed=TaskSpec(T,(PoseConstraint(T,0,poses[0].position+[.1,0,0],poses[0].rotation),poses[1]))
    rejected=validate(candidate,snapshot,changed,prop,codec,geometry,profile)
    assert rejected['status']=='REJECTED' and not rejected['gates']['task']
    assert rejected['input_unchanged']
