"""Public initial-state and variable timed-pose tokens, no terminal joint answer."""
import numpy as np


def encode_condition(robot,snapshot,task,momentum):
    d=robot.restore(snapshot);q=robot.Bplus@d.qpos[robot.qpos_ids];v=robot.Bplus@d.qvel[robot.dof_ids]
    tokens=np.zeros((3+len(task.poses)+len(robot.scene.boxes),32),dtype=np.float32)
    tokens[0,0]=1;tokens[0,5:22]=q;tokens[0,22]=task.duration
    tokens[1,1]=1;tokens[1,5:22]=v
    tokens[2,2]=1;tokens[2,5:12]=d.qpos[robot.base_qpos];tokens[2,12:18]=momentum
    tokens[2,18:24]=d.qvel[robot.base_dofs]
    tokens[2,24]=np.linalg.norm(d.qpos[robot.qpos_ids]-robot.B@q)
    tokens[2,25]=np.linalg.norm(d.qvel[robot.dof_ids]-robot.B@v)
    for row,p in enumerate(task.poses,3):
        tokens[row,3]=1;tokens[row,5+p.tip]=1;tokens[row,7:9]=[p.time/task.duration,task.duration]
        tokens[row,9:12]=p.position;tokens[row,12:18]=p.rotation[:,:2].T.ravel()
        tokens[row,18:22]=[p.position_tolerance,p.rotation_tolerance,float(p.position_enabled),float(p.rotation_enabled)]
    for row,box in enumerate(robot.scene.boxes,3+len(task.poses)):
        tokens[row,4]=1;tokens[row,5:8]=box.center;tokens[row,8:11]=box.half_size
        tokens[row,11:17]=box.rotation[:,:2].T.ravel()
    return tokens
