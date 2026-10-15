"""Read-only comparison of prescribed-path and actual-native-path base coupling.

The native-conditioned reconstruction reads all 67 arm coordinates/rates and
integrates the base from its initial snapshot. It never copies per-frame base
state, never projects native geometry through B+, and never certifies execution.
"""
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior.storage import atomic_npz


def base_poses(base):
    base=np.asarray(base)
    result=np.tile(np.eye(4),(len(base),1,1))
    result[:,:3,3]=base[:,:3]
    result[:,:3,:3]=Rotation.from_quat(base[:,[4,5,6,3]]).as_matrix()
    return result


def metrics(actual,predicted):
    p,r=pose_errors(actual,predicted)
    return {'max_position_m':float(np.max(p)),'rms_position_m':float(np.sqrt(np.mean(p*p))),
        'max_rotation_rad':float(np.max(r)),'rms_rotation_rad':float(np.sqrt(np.mean(r*r))),
        'final_position_m':p[-1],'final_rotation_rad':r[-1]}


def reconstruct_native(robot,initial,arrays,indices,deadline):
    d=initial.reset(robot); m=robot.model; a,b=robot.dof_ids,robot.base_dofs
    native_q=arrays['qpos'][:,robot.qpos_ids]; native_v=arrays['qvel'][:,a]
    index_set=set(map(int,indices)); bases=[]; ee=[]
    def velocity(q,v):
        d.qpos[robot.qpos_ids]=q
        robot.kinematics(d,mass=True); M=robot.mass(d)
        value=np.zeros(m.nv); value[a]=v
        value[b]=-np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,a)]@v)
        return value
    for k in range(len(arrays['time'])):
        if time.time()>=deadline: raise RuntimeError('CONSISTENCY_WALL_BUDGET_EXHAUSTED')
        d.qpos[robot.qpos_ids]=native_q[k]
        if k in index_set:
            robot.kinematics(d); bases.append(d.qpos[robot.base_qpos].copy()); ee.append(robot.ee(d))
        if k+1<len(arrays['time']):
            dt=float(arrays['time'][k+1]-arrays['time'][k]); origin=d.qpos.copy()
            v=velocity(native_q[k],native_v[k])
            mujoco.mj_integratePos(m,d.qpos,v,dt/2)
            vm=velocity((native_q[k]+native_q[k+1])/2,(native_v[k]+native_v[k+1])/2)
            d.qpos[:]=origin
            mujoco.mj_integratePos(m,d.qpos,vm,dt)
    return np.asarray(bases),np.asarray(ee)


def analyze(robot,initial,reference,arrays,output,deadline):
    if len(arrays['qpos'])!=len(arrays['time']) or len(arrays['ctrl'])+1!=len(arrays['time']):
        raise ValueError('native state/control alignment invalid')
    dt=float(robot.model.opt.timestep)
    motion_steps=int(round(reference.duration/dt))
    if abs(motion_steps*dt-reference.duration)>1e-9: raise ValueError('motion grid must align with physics')
    intervals=next(n for n in range(min(1000,motion_steps),1,-1) if n%2==0 and motion_steps%n==0)
    coarse=predict_coupled(initial,reference,robot,intervals//2,deadline)
    prediction=predict_coupled(initial,reference,robot,intervals,deadline)
    stride=motion_steps//intervals
    moving=np.arange(0,motion_steps+1,stride)
    if len(arrays['time'])<=motion_steps: raise ValueError('incomplete physical trajectory')
    hold=np.arange(motion_steps+stride,len(arrays['time']),stride)
    if len(hold)==0 or hold[-1]!=len(arrays['time'])-1: hold=np.r_[hold,len(arrays['time'])-1]
    indices=np.r_[moving,hold]
    if not np.allclose(arrays['time'][moving]-arrays['time'][0],prediction['phase']*reference.duration,atol=1e-9,rtol=0):
        raise ValueError('prediction/native timestamps disagree')
    predicted_base=np.concatenate([prediction['base'],np.repeat(prediction['base'][-1:],len(hold),axis=0)])
    predicted_ee=np.concatenate([prediction['ee'],np.repeat(prediction['ee'][-1:],len(hold),axis=0)])
    reconstructed_base,reconstructed_ee=reconstruct_native(robot,initial,arrays,indices,deadline)
    actual_base=arrays['base'][indices]; actual_ee=arrays['ee'][indices]
    nominal_ee=metrics(actual_ee,predicted_ee); nominal_base=metrics(base_poses(actual_base),base_poses(predicted_base))
    conditioned_ee=metrics(actual_ee,reconstructed_ee); conditioned_base=metrics(base_poses(actual_base),base_poses(reconstructed_base))
    convergence_ee=metrics(prediction['ee'][::2],coarse['ee'])
    convergence_base=metrics(base_poses(prediction['base'][::2]),base_poses(coarse['base']))
    native_target=reference.evaluate(arrays['time']-arrays['time'][0])@robot.B.T
    native_tracking=float(abs(arrays['qpos'][:,robot.qpos_ids]-native_target).max())
    agreement=nominal_ee['max_position_m']<=.005 and nominal_ee['max_rotation_rad']<=np.deg2rad(1)
    reconstruction=conditioned_ee['max_position_m']<=.001 and conditioned_ee['max_rotation_rad']<=np.deg2rad(.2)
    numerical=convergence_ee['max_position_m']<=.001 and convergence_ee['max_rotation_rad']<=np.deg2rad(.2)
    atomic_npz(output,indices=indices,time=arrays['time'][indices],predicted_base=predicted_base,
        predicted_ee=predicted_ee,actual_base=actual_base,actual_ee=actual_ee,
        native_conditioned_base=reconstructed_base,native_conditioned_ee=reconstructed_ee)
    return {'status':'MEASURED','reference_hash':reference.reference_hash,'comparison_states':len(indices),
        'prediction_intervals':intervals,'full_motion_and_hold':True,'native_alignment':'state[k+1] follows ctrl[k]',
        'nominal_reference_vs_execution':{'end_effectors':nominal_ee,'base':nominal_base},
        'actual_native_path_reconstruction_vs_execution':{'end_effectors':conditioned_ee,'base':conditioned_base},
        'prediction_grid_convergence':{'end_effectors':convergence_ee,'base':convergence_base},
        'native_tracking_max_rad':native_tracking,'manifold_max_rad':float(abs(arrays['manifold_residual']).max()),
        'generalized_momentum_max':float(abs(arrays['base_generalized_momentum']).max()),
        'diagnostic_flags':{'reference_prediction_within_5mm_1deg':agreement,
            'reference_base_within_1mm_0_2deg':nominal_base['max_position_m']<=.001 and nominal_base['max_rotation_rad']<=np.deg2rad(.2),
            'native_conditioned_within_1mm_0_2deg':reconstruction,
            'native_conditioned_base_within_1mm_0_2deg':conditioned_base['max_position_m']<=.001 and conditioned_base['max_rotation_rad']<=np.deg2rad(.2),
            'prediction_grid_converged':numerical,
            'prediction_base_grid_converged':convergence_base['max_position_m']<=.001 and convergence_base['max_rotation_rad']<=np.deg2rad(.2)},
        'threshold_scope':'diagnostic agreement flags only; all execution acceptance gates unchanged',
        'physical_certification':'NOT_GRANTED_BY_THIS_COMPARISON','all_67_native_joint_rates_used':True,
        'per_frame_base_state_copied':False,'Bplus_used_for_reconstruction':False,
        'hardware_model_accuracy':'NOT_VALIDATED'}
