"""Public initial-state/time/environment conditions; no future executed states."""
import numpy as np
from ff_prior_dynamic_margin.representation import design

def public_features(env,scene):
    knots,_,_,_=design(27.)
    times=np.array([knots[i+1:i+4].mean() for i in range(273)])[2:]
    path=env.path.to_dict();q0=env.robot.Bplus@env.data.qpos[env.robot.qpos_ids]
    v0=env.robot.Bplus@env.data.qvel[env.robot.dof_ids]
    pieces=[];scales=[];schema=[]
    def add(name,value,scale):
        x=np.asarray(value,dtype=np.float64).reshape(-1);s=np.broadcast_to(scale,x.shape).copy()
        schema.append(dict(name=name,start=sum(len(v) for v in pieces),size=len(x)))
        pieces.append(x);scales.append(s)
    add('complete_initial_INTEGRATION',env.initial.values,1.)
    add('planner_q0',q0,1.);add('planner_v0',v0,.1)
    add('world_waypoints',path['waypoint_points_m'],.2)
    add('segment_durations',path['segment_durations_s'],27.)
    add('continuum_world_rotation',path['target_rotation_world'],1.)
    add('grasp_local',env.grasp_local,.2);add('grasp_rotation_local',env.grasp_rotation_local,1.)
    add('approach_offset',env.approach_offset,.1);add('rigid_home',env.rigid_home,2.)
    add('rigid_home_rotation',env.rigid_home_rotation,1.)
    target_p,target_r=env.pose(2)
    add('target_initial_world_position',target_p,2.);add('target_initial_world_rotation',target_r,1.)
    add('target_initial_velocity',env.data.qvel[env.tv],.005)
    add('phase_times',[path['path_start_s'],path['path_end_s'],27.],27.)
    objects=scene.to_dict()['objects'];assert len(objects)<=5
    obs=np.zeros((5,12));obs_scale=np.tile(np.r_[1.,1.,np.full(3,2.),np.full(3,.1),np.ones(4)],(5,1))
    for i,o in enumerate(objects):
        assert o['shape']=='sphere','Only current frozen sphere scenes are supported'
        obs[i]=np.r_[1.,1.,o['position'],o['size'],o['quaternion_wxyz']]
    add('obstacles_mask_shape_position_size_quaternion',obs,obs_scale.reshape(-1))
    local=[]
    for t in times:
        ref=env.reference(float(t))
        local.append(np.concatenate([np.asarray(v).reshape(-1) for v in ref]))
    local=np.asarray(local)
    # Each end: position(3), rotation(9), velocity(3), angular velocity(3).
    local_scale=np.tile(np.r_[np.full(3,.2),np.ones(9),np.full(6,.05)],2)
    assert local.shape==(271,36) and env.initial.values.shape==(836,)
    g=np.concatenate(pieces);scale=np.concatenate(scales)
    assert np.isfinite(g).all() and np.isfinite(local).all()
    return dict(global_condition=g,local_condition=local,global_scale=scale,local_scale=local_scale,
                schema=schema,coefficient_times=times,q0=q0,v0=v0,knots=knots,
                target_prediction='public initial constant velocity and local angular velocity only')

def fit_codec(coefficients,features):
    """Caller provides TRAIN only. No VAL statistics, checkpoint selection or target relabel."""
    assert len(coefficients)==len(features)==8
    q0=np.stack([f['q0'] for f in features]);residual=np.asarray(coefficients)[:,2:]-q0[:,None]
    mean=residual.mean(0);scale=np.maximum(np.sqrt(np.mean((residual-mean)**2,axis=(0,1))),.02)
    g=np.stack([f['global_condition'] for f in features]);local=np.stack([f['local_condition'] for f in features])
    return dict(mean=mean,scale=scale,global_mean=g.mean(0),global_scale=np.maximum(g.std(0),features[0]['global_scale']),
                local_mean=local.mean(0),local_scale=np.maximum(local.std(axis=(0,1)),features[0]['local_scale']),
                schema=features[0]['schema'],fit_split='TRAIN',fit_count=8,free_shape=[271,17])

def encode_conditions(f,codec):
    return ((f['global_condition']-np.asarray(codec['global_mean']))/np.asarray(codec['global_scale']),
            (f['local_condition']-np.asarray(codec['local_mean']))/np.asarray(codec['local_scale']))

def encode_reference(c,f,codec):
    return (c[2:]-f['q0']-np.asarray(codec['mean']))/np.asarray(codec['scale'])

def decode_reference(x,f,codec):
    fixed=np.stack([f['q0'],f['q0']+f['v0']*.1/3])
    return np.concatenate([fixed,np.asarray(x)*np.asarray(codec['scale'])+np.asarray(codec['mean'])+f['q0']])
