"""Nonlinear arm FK on a private state; floating-base displacement stays local."""
import mujoco
import numpy as np
from .environment import TIP_OFFSET,rotation_error


def residuals(env,positions,times,g,drift):
    if not hasattr(env,"_preview_data"):
        env._preview_data=mujoco.MjData(env.model)
    data=env._preview_data
    q,_=env.state()
    out=[]
    for qp,t in zip(positions,times):
        data.qpos[:]=env.data.qpos
        displacement=g@(qp-q)+float(t)*drift
        mujoco.mj_integratePos(env.model,data.qpos,displacement,1.)
        mujoco.mj_kinematics(env.model,data)
        reference=env.reference(float(env.data.time)+float(t))
        row=[]
        for arm,index in ((0,0),(1,4)):
            body=env.body_ids[arm]
            rotation=data.xmat[body].reshape(3,3)
            position=data.xpos[body].copy()
            if arm==0: position+=rotation@TIP_OFFSET
            row.extend((position-reference[index],rotation_error(reference[index+1],rotation)))
        out.append(np.concatenate(row))
    return np.asarray(out)
