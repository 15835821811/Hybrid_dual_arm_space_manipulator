"""True nonlinear world-endpoint loss and full latent finite-difference VJP.

Forward uses unchanged 17-to-67, full state and midpoint floating-base dynamics.
This training objective is not a geometry/torque/execution validator.
"""
import time
import numpy as np
import torch
from ff_prior.storage import digest
from ff_prior.contracts import assert_frozen
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import predict_coupled,pose_errors
from .protocol import PROTOCOL as P

def loss_from_ee(ee,goals,position=.005,rotation=np.deg2rad(1)):
    pos,rot=pose_errors(ee,goals)
    value=float((np.sum((pos/position)**2)+np.sum((rot/rotation)**2))/4.)
    if not np.isfinite(value):raise ValueError('NONFINITE_TRUE_ENDPOINT_LOSS')
    return value,pos,rot

def evaluate(x,codec,center,base,task,robot,initial,deadline,intervals=256):
    assert_frozen(task)
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=(codec.dim,) or not np.isfinite(x).all():raise ValueError('invalid endpoint latent')
    z=codec.decode(codec.coefficients(x,center),base)
    ref=Reference(z,task['condition']['initial_features'],task['condition']['duration'])
    pred=predict_coupled(initial,ref,robot,intervals,deadline)
    value,pos,rot=loss_from_ee(pred['ee'][-1],task['condition']['goals'])
    return value,{'ee':pred['ee'][-1],'position_m':pos,'rotation_rad':rot,
                  'numerical':pred['numerical'],'reference_hash':ref.reference_hash,'Z_hash':digest(z)}

def gradient(x,codec,center,base,task,robot,initial,deadline):
    x=np.asarray(x,dtype=np.float64);eps=P['endpoint_difference_normalized']
    if codec.dim!=P['endpoint_gradient_dimensions']:raise ValueError('full102 dimension contract changed')
    before=robot.calls.copy()
    value,central=evaluate(x,codec,center,base,task,robot,initial,deadline)
    losses=np.empty((codec.dim,2));poses=np.empty((codec.dim,2,2,4,4))
    for j in range(codec.dim):
        for k,sign in enumerate((-1.,1.)):
            perturbed=x.copy();perturbed[j]+=sign*eps
            losses[j,k],q=evaluate(perturbed,codec,center,base,task,robot,initial,deadline)
            poses[j,k]=q['ee']
    g=(losses[:,1]-losses[:,0])/(2*eps)
    halves=[]
    for j in (0,codec.dim//2):
        samples=[]
        for sign in (-1.,1.):
            v=x.copy();v[j]+=sign*eps/2
            f,q=evaluate(v,codec,center,base,task,robot,initial,deadline);samples.append({'loss':f,'ee':q['ee']})
        slope=(samples[1]['loss']-samples[0]['loss'])/eps
        error=abs(slope-g[j])/max(abs(slope),abs(g[j]),1e-6)
        halves.append({'dimension':j,'samples':samples,'half_slope':slope,'relative_error':error,
                       'passed':bool(error<=P['endpoint_gradient_half_tolerance'])})
    direction=np.sin(np.arange(codec.dim)+.37);direction/=np.linalg.norm(direction)
    probes=[]
    for sign in (-1.,1.):
        f,q=evaluate(x+sign*eps*direction,codec,center,base,task,robot,initial,deadline)
        probes.append({'loss':f,'ee':q['ee']})
    slope=(probes[1]['loss']-probes[0]['loss'])/(2*eps);expected=float(g@direction)
    mismatch=abs(slope-expected)/max(abs(slope),abs(expected),1e-6)
    fine,fine_record=evaluate(x,codec,center,base,task,robot,initial,deadline,P['endpoint_convergence_intervals'])
    dp,dr=pose_errors(central['ee'],fine_record['ee'])
    converged=bool(max(dp)<=.001 and max(dr)<=np.deg2rad(.2))
    calls={k:robot.calls[k]-before[k] for k in before}
    passed=bool(np.isfinite(g).all() and converged and all(h['passed'] for h in halves)
                and mismatch<=P['endpoint_gradient_direction_tolerance']
                and calls['coupled']==P['queries_per_gradient'])
    proof={'status':'PASS' if passed else 'BLOCKED','task_hash':task['task_hash'],
           'x_hash':digest(x),'base_hash':digest(base),'center_hash':digest(center),'codec_hash':digest(codec.state),
           'true_loss':value,'central':central,'epsilon':eps,'dimensions':codec.dim,
           'half_checks':halves,'direction':direction,'direction_samples':probes,
           'direction_slope':slope,'direction_expected':expected,'direction_relative_error':mismatch,
           'convergence':{'passed':converged,'position_m':dp,'rotation_rad':dr,'fine_loss':fine,'fine':fine_record},
           'calls':calls,'training_only':True,'physical_validation_level':'NOT_A_VALIDATOR',
           'gradient_recomputed_at_current_prediction':True}
    return value,g,proof,{'x':x,'gradient':g,'perturbation_losses':losses,'perturbation_ee':poses}

def attach_gradient(clean,value,g):
    """Exact true forward value, externally computed finite-difference gradient."""
    grad=torch.as_tensor(g,dtype=clean.dtype,device=clean.device)
    return clean.new_tensor(value)+torch.sum((clean-clean.detach())*grad)
