import time,traceback
import numpy as np
from ff_prior.contracts import assert_frozen,assert_task_compatible
from ff_prior.storage import digest
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior_delivery_guidance.sampling import LatentProjector,sample as inherited_sample
from ff_prior_feasible.validation import validate
from ff_prior_generalization.sampling import candidate as legacy_candidate

class FullProjector(LatentProjector):
    def roundtrip(self,z):return np.asarray(z,dtype=np.float64).copy()
    def prepare(self,raw):
        raw=np.asarray(raw,dtype=np.float64);c=self.task['condition']
        if not np.isfinite(raw).all():raise ValueError('nonfinite raw')
        for scale in (1.,.5,.25,.125,.0625,.03125,0.):
            z=raw.copy() if scale==1 else self.base+scale*(raw-self.base)
            if all(quick_gates(self.robot,Reference(z,c['initial_features'],c['duration'])).values()):
                return z,{'coefficient_shrink':scale,'unchanged':bool(np.array_equal(z,raw)),
                          'policy':'preserve_feasible_no_smoothing_or_clip','compression_max_rad':0.}
        return raw.copy(),{'coefficient_shrink':None,'quick_feasible':False,'policy':'rejected_no_gate_relaxation'}

def sample(bundle,base,task,seed,projector):
    decoded,trace=inherited_sample(bundle,base,task,seed,projector)
    final=np.asarray(trace['guidance'][-1]['output_Z']).copy()
    assert np.max(abs(final-decoded))<=1e-12
    trace['final_output_exact_guided_control']=True
    trace['feedback_roundtrip_max_rad']=float(np.max(abs(final-decoded)))
    return final,trace

def candidate(method, k, task, base, seed, bundle, rr, ii, cc, profile, deadline):
    if method != 'late4_diffusion':
        return legacy_candidate(method, k, task, base, seed, bundle, rr, ii, cc, profile, deadline)
    assert_task_compatible(task, rr, ii, cc, profile)
    before = rr.calls.copy()
    network = bundle[0].calls
    start = time.perf_counter()
    deadline = min(deadline, time.time()+15.)
    guide = FullProjector(task, rr, ii, cc, profile, deadline, codec=bundle[1], base=base)
    result = {'status': 'ERROR', 'eligible': False, 'method': method, 'candidate_id': k,
              'seed': seed, 'task_hash': task['task_hash'], 'public_base_hash': digest(base), 'private_witness_used': False}
    try:
        z, sampling = sample(bundle, base, task, seed, guide)
        result.update(Z=z, sampling=sampling, **validate(z, task, rr, ii, cc, profile, deadline), status='DONE')
        if time.time() > deadline:
            raise RuntimeError('CANDIDATE_WALL_CEILING_EXCEEDED')
    except Exception as error:
        result.update(status='ERROR', eligible=False, reason=repr(error), traceback=traceback.format_exc())
    calls = {n: rr.calls[n]-before[n] for n in before}
    intervals = 128*guide.used+result['validation_integration_intervals'] if result['status']=='DONE' else None
    result.update(model_calls=calls, repair_coupled_calls=guide.used, validation_coupled_calls=calls['coupled']-guide.used,
                  actual_network_calls=bundle[0].calls-network, integration_intervals=intervals, wall_s=time.perf_counter()-start)
    if calls['coupled'] > 113 or guide.used > 110 or calls['coupled']-guide.used > 3 or (intervals or 0) > 15872:
        result.update(status='ERROR', eligible=False, reason='QUERY_BUDGET_VIOLATION')
    assert_frozen(task)
    return result
