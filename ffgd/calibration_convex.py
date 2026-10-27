"""Whole-robot convex proxy pressure set against native exact checks."""
import json
import numpy as np
import mujoco
import torch
from .robot import Robot
from .physics import DifferentiableTree,tensor
from .geometry import Geometry
from .convex_guide import ConvexSupportGuide


def run(path,count=64):
    torch.set_num_threads(1);r=Robot();m=r.model;tree=DifferentiableTree(r);geometry=Geometry(r);guide=ConvexSupportGuide(geometry)
    rng=np.random.default_rng(8927);d=r.new_data();target=int(m.joint(r.spec.target_free_joint_name).qposadr[0]);rows=[]
    for case in range(count):
        d.qpos[r.qpos_ids]=r.B@(r.spec.planner_zero+rng.uniform(-.6,.6,17))
        quaternion=rng.normal(size=4);quaternion/=np.linalg.norm(quaternion);d.qpos[r.base_qpos[3:]]=quaternion
        mujoco.mj_kinematics(m,d)
        if case%3:d.qpos[target:target+3]=d.xpos[r.tips[case%2]]+rng.normal(0,.15,3)
        else:d.qpos[target:target+3]=[1.2,0,.05]
        mujoco.mj_kinematics(m,d)
        native=geometry.native_distances(d.qpos)
        values=guide.gaps(tree.kinematics(tensor(d.qpos))).numpy()
        independent=guide.numpy_gaps(d.geom_xpos[geometry.ids],d.geom_xmat[geometry.ids].reshape(-1,3,3))
        false_safe=int(np.sum((values>=1e-7)&(native < -1e-7)))
        false_unsafe=int(np.sum((values<0)&(native>=0)))
        error=float(np.max(abs(values-independent)))
        row=dict(case=case,false_safe=false_safe,false_unsafe=false_unsafe,proxy_min=float(values.min()),native_min=float(native.min()),numpy_abs=error,
                 passed=false_safe==0 and error<1e-10)
        rows.append(row)
    (path/'pair_catalog.json').write_text(json.dumps(geometry.catalog(),indent=2),encoding='utf-8')
    return dict(passed=all(r['passed'] for r in rows),rows=rows,guide=guide.sizes,query_pairs=len(geometry.pairs),
                limitations='finite-axis conservative convex projection; false-unsafe reported, native hard validation retained; sampled audit is not a global proof')
