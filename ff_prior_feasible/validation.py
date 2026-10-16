"""Adaptive independent geometry validation with unchanged numerical tolerances."""
import numpy as np
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from ff_prior.model_gates import validate_model
from ff_prior.proposal_generator import quick_gates
from ff_prior.spline import Reference

CONTRACT={'intervals':[256,512,1024],'maximum_coupled_queries':3,
    'maximum_integration_intervals':1792,'geometry_step_budget_m':.002,
    'aligned_geometry_convergence_m':.001,'endpoint_convergence_m':.001,
    'endpoint_convergence_rad':float(np.deg2rad(.2)),
    'all_geometry_pairs_preserved':True,'preserve_detected_penetration':True}


def compare(coarse,fine,robot):
    pos,rot=pose_errors(fine['ee'][-1],coarse['ee'][-1])
    body=np.linalg.norm(fine['body'][::2]-coarse['body'],axis=-1)
    center=np.linalg.norm(fine['geom'][::2]-coarse['geom'],axis=-1)
    rotation=(fine['geom_rotation'][::2]-coarse['geom_rotation']).reshape(len(coarse['phase']),robot.model.ngeom,3,3)
    extent=float(max(body.max(),(center+np.linalg.norm(rotation,axis=(-1,-2))*robot.model.geom_rbound).max()))
    return {'coarse':len(coarse['phase'])-1,'fine':len(fine['phase'])-1,'terminal_position_m':pos,
        'terminal_rotation_rad':rot,'aligned_body_extent_m':extent,
        'passed':bool(max(pos)<=.001 and max(rot)<=np.deg2rad(.2) and extent<=.001)}


def validate(z,task,robot,initial,collision,profile,deadline):
    c=task['condition']; ref=Reference(z,c['initial_features'],c['duration']); quick=quick_gates(robot,ref)
    result={'frozen_goal_pass':False,'eligible':False,'score':None,'reference_hash':ref.reference_hash,
        'validation_integration_intervals':0,'validation_grids':[]}
    if not all(quick.values()):
        return dict(result,validation={'validation_level':'REJECTED','gates':quick,'reason':'QUICK_GATE_FAILED'})
    coarse=predict_coupled(initial,ref,robot,256,deadline); result['validation_integration_intervals']=256
    comparisons=[]; validations=[]; penetrated=False
    for n in (512,1024):
        fine=predict_coupled(initial,ref,robot,n,deadline); result['validation_integration_intervals']+=n
        comparisons.append(compare(coarse,fine,robot))
        gates=validate_model(ref,fine,profile,robot,initial,collision,
            {'passed':comparisons[-1]['passed'],'comparisons':comparisons.copy()},deadline)
        validations.append({'intervals':n,'validation_level':gates['validation_level'],
            'gates':gates['gates'].copy(),'geometry':gates['geometry']})
        penetrated=penetrated or not gates['gates']['geometry']
        if penetrated:
            gates['gates']['geometry']=False; gates['validation_level']='REJECTED'
        if gates['gates']['geometry_refinement'] and gates['gates']['coupled_convergence']: break
        coarse=fine
    pos,rot=pose_errors(fine['ee'][-1],c['goals'])
    goal=bool(np.all(pos<=profile.position_error_m) and np.all(rot<=profile.rotation_error_rad))
    gates.update(frozen_goal_position_error_m=pos,frozen_goal_rotation_error_rad=rot)
    result.update(validation=gates,validation_grids=validations,frozen_goal_pass=goal,
        eligible=goal and gates['validation_level']=='MODEL_VALID',
        score=float(np.sum((pos/profile.position_error_m)**2)+np.sum((rot/profile.rotation_error_rad)**2)))
    return result
