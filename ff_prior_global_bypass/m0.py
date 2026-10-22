"""One budgeted physical propagation check before the new tree is used."""
import numpy as np
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .tree import root_node,propagate,Piecewise,minimum_duration


def verify(out,robot,initial,collision,budget):
    root=root_node(robot,initial)
    direction=np.zeros(17);direction[11]=.025;direction[13]=-.02;direction[0]=.005
    duration=max(2.,minimum_duration(direction,robot.spec.planner_velocity_limits,robot.spec.planner_acceleration_limits))
    a,ra,aa=propagate(root,root.q+direction,duration,robot,collision,budget,256,False)
    a.parent=0
    b,rb,ab=propagate(a,root.q+1.4*direction,duration,robot,collision,budget,256,False)
    b.parent=1
    piece=Piecewise([root,a,b],2*duration)
    budget.consume(512);whole=predict_coupled(initial,piece,robot,512,budget.deadline)
    joined=np.concatenate([aa['qpos'],ab['qpos'][1:]],axis=0)
    joined_v=np.concatenate([aa['qvel'],ab['qvel'][1:]],axis=0)
    pos,rot=pose_errors(ab['ee'][-1],whole['ee'][-1])
    checks={'full_joint_base_qpos_continuation':float(np.max(abs(joined-whole['qpos'])))<1e-10,
        'full_qvel_continuation':float(np.max(abs(joined_v-whole['velocity'])))<1e-10,
        'dual_world_poses':max(pos)<1e-10 and max(rot)<1e-10,
        'parent_base_carried':np.array_equal(ab['qpos'][0],aa['qpos'][-1]),
        'time_continues':abs(ab['time'][0]-aa['time'][-1])<1e-12,
        'derived_rest_nodes':np.max(abs(b.qvel))<1e-12 and np.max(abs(a.qvel))<1e-12,
        'base_actually_moves':np.max(abs(a.qpos[robot.base_qpos]-root.qpos[robot.base_qpos]))>1e-8,
        'zero_momentum_solve':ra['numerical_ok'] and rb['numerical_ok']}
    atomic_npz(out/'propagation_check.npz',joined_qpos=joined,whole_qpos=whole['qpos'],
        joined_velocity=joined_v,whole_velocity=whole['velocity'],first_qpos=aa['qpos'],
        second_qpos=ab['qpos'],first_state=a.state,second_state=b.state,root_state=root.state,
        joined_time=np.r_[aa['time'],ab['time'][1:]],whole_ee=whole['ee'],second_ee=ab['ee'])
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':{k:bool(v) for k,v in checks.items()},
        'new_coupled':3,'native_execution':0,'qualification':'PROPAGATION_CHECK_NOT_QUALIFIED_PATH',
        'position_difference_m':pos,'rotation_difference_rad':rot,'first':ra,'second':rb}
    atomic_json(out/'result.json',result)
    return result
