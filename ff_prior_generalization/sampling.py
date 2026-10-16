"""Same sampler and hard physical guide, with saved pre-guide proposals."""
import time
import numpy as np
from ff_prior.storage import digest
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior_latent.learning import sample
from ff_prior_feasible.guidance import FeasibleProjector
from ff_prior_feasible.validation import validate


class RecordedProjector(FeasibleProjector):
    def __call__(self,z,iteration):
        raw=np.asarray(z,dtype=np.float64).copy()
        result,info=super().__call__(z,iteration)
        if iteration==15: info['raw_Z']=raw
        return result,info


def candidate(method,k,task,base,seed,bundle,rr,ii,cc,profile,deadline):
    assert_task_compatible(task,rr,ii,cc,profile)
    before=rr.calls.copy();start=time.perf_counter();deadline=min(deadline,time.time()+15.)
    network=bundle[0].calls if bundle is not None else 0
    guide=RecordedProjector(task,rr,ii,cc,profile,deadline)
    result={'status':'ERROR','eligible':False,'method':method,'candidate_id':k,'seed':seed,
            'task_hash':task['task_hash'],'public_base_hash':digest(base),'private_witness_used':False}
    try:
        if method in ('structured','learned_mean'):
            z=base.copy() if method=='structured' else bundle[1].decode(bundle[1].center(task['condition'],base),base)
            if method=='structured' or k>0:
                s=np.r_[np.linspace(0,1,20)[1:-1],1.]
                z=z+.03*np.sin(np.pi*s)[:,None]*np.random.default_rng(seed).normal(size=(1,17))
            raw=z.copy();z,info=guide(z,15)
            sampling={'actual_network_calls':0,'guidance':[info],'raw_proposal_hash':digest(raw)}
        elif method in ('expanded_diffusion','frozen_diffusion'):
            z,sampling=sample(bundle,base,task,seed,guide)
        else: raise ValueError('unknown method')
        result.update(Z=z,sampling=sampling,**validate(z,task,rr,ii,cc,profile,deadline),status='DONE')
        if time.time()>deadline: raise RuntimeError('CANDIDATE_WALL_CEILING_EXCEEDED')
    except Exception as error:
        import traceback
        result.update(status='ERROR',eligible=False,reason=repr(error),traceback=traceback.format_exc())
    calls={n:rr.calls[n]-before[n] for n in before}
    result.update(model_calls=calls,repair_coupled_calls=guide.used,
                  validation_coupled_calls=calls['coupled']-guide.used,
                  actual_network_calls=bundle[0].calls-network if bundle is not None else 0,
                  integration_intervals=128*guide.used+result['validation_integration_intervals'] if result['status']=='DONE' else None,
                  wall_s=time.perf_counter()-start)
    if calls['coupled']>113 or guide.used>110 or calls['coupled']-guide.used>3 or (result['integration_intervals'] or 0)>15872:
        result.update(status='ERROR',eligible=False,reason='QUERY_BUDGET_VIOLATION')
    assert_frozen(task)
    return result
