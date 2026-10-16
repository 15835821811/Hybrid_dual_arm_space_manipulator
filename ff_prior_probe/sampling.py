"""Scale only the sampled residual; preserve original DDIM and physical repair."""
import time
import numpy as np
from ff_prior.storage import digest
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior_latent.learning import sample as latent_sample
from ff_prior_feasible.guidance import FeasibleProjector
from ff_prior_feasible.validation import validate

SPECS={
    'structured_feasible':('structured',None),
    'frozen_latent_diffusion':('frozen',1.),
    'augmented_mean_spline':('mean_bow',None),
    'mean_zero':('mean',0.),
    'v025':('v_only',.25),'v050':('v_only',.5),'v100':('v_only',1.),
    'g025':('goal_loss',.25),'g050':('goal_loss',.5),'g100':('goal_loss',1.),
}


def attenuate(raw,center,amplitude):
    if amplitude not in (0.,.25,.5,1.):raise ValueError('amplitude outside frozen grid')
    raw,center=np.asarray(raw,dtype=np.float64),np.asarray(center,dtype=np.float64)
    if raw.shape!=(19,17) or center.shape!=raw.shape or not np.isfinite(raw).all() or not np.isfinite(center).all():
        raise ValueError('finite control matrices required')
    if amplitude==0.:return center.copy()
    if amplitude==1.:return raw.copy()
    return center+amplitude*(raw-center)


def propose(method,k,task,base,seed,bundle,projector):
    kind,amplitude=SPECS[method];condition=task['condition'];saved={}
    center=None if bundle is None else bundle[1].decode(bundle[1].center(condition,base),base)
    def project(raw,iteration):
        value=attenuate(raw,center,amplitude) if kind in ('v_only','goal_loss') else np.asarray(raw).copy()
        saved.update(raw_unscaled=np.asarray(raw),raw_scaled=value,mean=center,
            raw_unscaled_hash=digest(np.asarray(raw)),raw_scaled_hash=digest(value),amplitude=amplitude)
        z,info=projector(value,iteration)
        return z,info
    if kind in ('v_only','goal_loss','frozen'):
        z,info=latent_sample(bundle,base,task,seed,project)
    else:
        raw=np.asarray(base).copy() if kind=='structured' else center.copy()
        if kind=='structured' or (kind=='mean_bow' and k>0):
            s=np.r_[np.linspace(0,1,20)[1:-1],1.]
            raw+=.03*np.sin(np.pi*s)[:,None]*np.random.default_rng(seed).normal(size=(1,17))
        z,g=project(raw,15);info={'actual_network_calls':0,'guidance':[g]}
    info['residual_probe']=saved
    return z,info


def candidate(method,k,task,base,seed,bundle,rr,ii,cc,profile,deadline):
    if method not in SPECS:raise ValueError('unknown frozen method')
    assert_task_compatible(task,rr,ii,cc,profile)
    before=rr.calls.copy();start=time.perf_counter();cpu=time.process_time();deadline=min(deadline,time.time()+15.)
    network_before=bundle[0].calls if bundle else 0
    projector=FeasibleProjector(task,rr,ii,cc,profile,deadline,collision_enabled=True)
    result={'status':'ERROR','eligible':False,'method':method,'candidate_id':k,'seed':seed,'task_hash':task['task_hash'],
        'public_base_hash':digest(base),'private_witness_used':False,'shared_query_ceiling':113,
        'shared_wall_ceiling_s':15.,'shared_integration_interval_ceiling':15872}
    try:
        z,sampling=propose(method,k,task,base,seed,bundle,projector)
        result.update(Z=z,sampling=sampling)
        result.update(validate(z,task,rr,ii,cc,profile,deadline));result['status']='DONE'
        if time.time()>deadline:raise RuntimeError('CANDIDATE_WALL_CEILING_EXCEEDED')
    except Exception as error:result.update(status='ERROR',eligible=False,reason=repr(error))
    calls={n:rr.calls[n]-before[n] for n in before}
    result.update(model_calls=calls,repair_coupled_calls=projector.used,validation_coupled_calls=calls['coupled']-projector.used,
        actual_network_calls=bundle[0].calls-network_before if bundle else 0,
        integration_intervals=128*projector.used+result['validation_integration_intervals'] if result['status']=='DONE' else None,
        wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu)
    if calls['coupled']>113 or projector.used>110 or calls['coupled']-projector.used>3 or (result['integration_intervals'] or 0)>15872:
        result.update(status='ERROR',eligible=False,reason='QUERY_BUDGET_VIOLATION')
    assert_frozen(task);return result
