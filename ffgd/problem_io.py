"""Public JSON problem IO. Only state, world tasks and scene configuration."""
import json
from pathlib import Path
import numpy as np
import mujoco
from .contracts import BoxObstacle,SceneSpec,PoseConstraint,TaskSpec,Snapshot
from .robot import Robot
from .api import PlanningBudget
from .diffusion import SamplingBudget


def load_problem(path):
    path=Path(path);value=json.loads(path.read_text(encoding='utf-8-sig'))
    allowed={'schema','id','split','initial','task','scene','description'}
    if set(value)-allowed:raise ValueError('unknown public problem keys')
    scene=SceneSpec(tuple(BoxObstacle(b['center'],b['half_size'],b.get('rotation',np.eye(3))) for b in value.get('scene',{}).get('boxes',[])))
    r=Robot(scene=scene);d=r.new_data();initial=value['initial']
    if set(initial)-{'q17','v17','base_pose','base_velocity','target_pose','target_velocity','snapshot_file'}:raise ValueError('unknown initial-state keys')
    if 'snapshot_file' in initial:
        if len(initial)!=1:raise ValueError('ambiguous snapshot and initial overrides')
        source=path.parent/initial['snapshot_file']
        with np.load(source) as saved:snapshot=Snapshot(str(saved['model_hash']),int(saved['mask']),saved['state'])
        r.restore(snapshot)
    else:
        if 'q17' in initial:d.qpos[r.qpos_ids]=r.B@np.asarray(initial['q17'])
        if 'v17' in initial:d.qvel[r.dof_ids]=r.B@np.asarray(initial['v17'])
        if 'base_pose' in initial:d.qpos[r.base_qpos]=initial['base_pose']
        if 'base_velocity' in initial:d.qvel[r.base_dofs]=initial['base_velocity']
        target=r.model.joint(r.spec.target_free_joint_name);qa=int(target.qposadr[0]);va=int(target.dofadr[0])
        if 'target_pose' in initial:d.qpos[qa:qa+7]=initial['target_pose']
        if 'target_velocity' in initial:d.qvel[va:va+6]=initial['target_velocity']
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise ValueError('nonfinite initial state')
        for adr in (r.base_qpos[3],qa+3):
            if abs(np.linalg.norm(d.qpos[adr:adr+4])-1)>1e-10:raise ValueError('nonunit initial quaternion')
        mujoco.mj_forward(r.model,d);snapshot=r.snapshot(d)
    t=value['task'];task=TaskSpec(float(t['duration']),tuple(PoseConstraint(**p) for p in t['poses']))
    return r,snapshot,task,value


def planning_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    values=dict(config['budget']);values['sampling']=SamplingBudget(**values.get('sampling',{}))
    return config,PlanningBudget(**values)
