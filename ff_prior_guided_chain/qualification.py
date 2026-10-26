"""Original static validator plus full-state dynamic qualification, before execution."""
import numpy as np
from ff_prior.contracts import AcceptanceConfig
from ff_prior.model_gates import validate_model
from ff_prior.coupled_predictor import pose_errors
from ff_prior_pilot_train.storage import atomic_json


def qualify(rep,guide,reference,final,task,path):
    r=guide.r;p=final['pred'];metrics=final['metrics'];initial=guide.pred.initial
    if rep.phase=='A':
        p=dict(p,phase=p['time']/reference.duration)
        p['numerical']=dict(p['numerical'],robot_momentum_residual=p['numerical']['solve_residual'],
            quaternion_norm_error=p['numerical']['quaternion_error'])
        config=AcceptanceConfig(**task['contract']['acceptance'])
        original=validate_model(reference,p,config,r,initial,guide.cc,
            dict(passed=metrics['converged']),guide.budget.deadline)
        gates=original['gates'].copy()
    else:
        original=None
        poly=reference.bounds()
        limits=r.model.jnt_range[r.joints];limited=r.model.jnt_limited[r.joints].astype(bool)
        gates=dict(identity=initial.model_hash==r.identity()['scene_model_hash'],
            finite=all(np.isfinite(v).all() for v in p.values() if isinstance(v,np.ndarray)),
            domain=np.all(poly[0][0]>=r.lower) and np.all(poly[0][1]<=r.upper),
            native_range=np.all(p['qpos'][:,r.qpos_ids][:,limited]>=limits[limited,0]) and np.all(p['qpos'][:,r.qpos_ids][:,limited]<=limits[limited,1]),
            boundary=np.max(abs(reference.evaluate(0)-rep.q0))<1e-10 and np.max(abs(reference.evaluate(0,1)-guide.pred.v0))<1e-10,
            motion=np.all(np.maximum(abs(poly[1][0]),abs(poly[1][1]))<=r.spec.planner_velocity_limits) and np.all(np.maximum(abs(poly[2][0]),abs(poly[2][1]))<=r.spec.planner_acceleration_limits),
            coupled_convergence=metrics['converged'],geometry=metrics['clearance_lower_or_exact_m']>=0,
            geometry_refinement=metrics['geometry_step_m']<=.002 and metrics['geometry_refinement_difference_m']<=.001,
            time_hold=reference.duration==(27. if rep.phase=='B' else 29.))
        scratch=initial.reset(r);scratch.qpos[:]=p['qpos'][-1];r.kinematics(scratch)
        dp,dr=pose_errors(r.ee(scratch),p['ee'][-1]);gates['label_consistency']=dp.max()<1e-10 and dr.max()<1e-10
    peak=np.array([np.linalg.norm(p['momentum'][:,s],axis=1).max() for s in (slice(0,3),slice(3,6))])
    exchange=np.array([np.linalg.norm(p['exchange'][:,s],axis=1).max() for s in (slice(0,3),slice(3,6))])
    gates.update(world_goal_position=metrics['position_max_m']<=.005,
        world_goal_rotation=metrics['rotation_max_rad']<=np.pi/180,
        raw_torque=metrics['maximum_torque_ratio']<=1,
        inherited_momentum=np.all(peak<=1e-5+.002*exchange),
        original_numeric_phase_initial=p['numerical']['initial_qpos_error']<=1e-10 and p['numerical']['initial_qvel_error']<=1e-10)
    gates={k:bool(v) for k,v in gates.items()};passed=all(gates.values())
    report=dict(level='MODEL_VALID' if passed else 'MODEL_REJECTED',gates=gates,original_static_validator=original,
        momentum_peak=peak,momentum_threshold=1e-5+.002*exchange,initial_momentum_subtracted=False,
        metrics=metrics,reference_hash=reference.reference_hash,initial_state_id=initial.state_id,
        actual_execution=False,full_836_replay=False)
    atomic_json(path,report)
    return report
