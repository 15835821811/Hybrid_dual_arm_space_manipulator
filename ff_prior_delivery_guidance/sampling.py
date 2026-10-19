"""Clean-state guidance in the actual codec subspace, with a shared query cap."""
import time
import traceback
import numpy as np
import torch
from ff_prior.contracts import assert_frozen, assert_task_compatible
from ff_prior.diffusion import schedule, prediction_to_clean_noise
from ff_prior.storage import digest
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior_goal.projection import smooth
from ff_prior_feasible.guidance import FeasibleProjector, group_norms, constrained_step, accept_update
from ff_prior_feasible.validation import validate
from ff_prior_generalization.sampling import candidate as legacy_candidate
from .protocol import ALLOCATIONS


def ddim_update(noisy, clean, alpha, next_alpha):
    """Recompose from untouched x_t after changing x_0, including final alpha=1."""
    if not (0. < float(alpha) < 1. and float(alpha) <= float(next_alpha) <= 1.):
        raise ValueError('invalid DDIM alpha')
    epsilon = (noisy - alpha.sqrt() * clean) / (1. - alpha).sqrt()
    residual = float((noisy - alpha.sqrt() * clean - (1. - alpha).sqrt() * epsilon).abs().max())
    value = next_alpha.sqrt() * clean + (1. - next_alpha).sqrt() * epsilon
    if not torch.isfinite(value).all() or not torch.isfinite(epsilon).all():
        raise ValueError('nonfinite guided DDIM state')
    if residual > 1e-10:
        raise ValueError('DDIM decomposition mismatch')
    return value, residual


class LatentProjector(FeasibleProjector):
    def __init__(self, *args, codec, base, **kwargs):
        super().__init__(*args, **kwargs)
        self.codec, self.base = codec, np.asarray(base, dtype=np.float64)
        b = codec.basis
        self.directions = np.einsum('tk,sk,nsj->ntj', b, b, self.directions)
        self.segment_end = 0
        self.seen = set()

    def roundtrip(self, z):
        return self.codec.decode(self.codec.encode(z, self.base), self.base)

    def evaluate(self, z):
        if self.used >= self.segment_end:
            raise RuntimeError('SEGMENT_QUERY_BUDGET_EXHAUSTED')
        return super().evaluate(z)

    def prepare(self, raw):
        """Keep the returned controls representable; never relax strict bounds."""
        c = self.task['condition']
        clipped = np.clip(smooth(raw, c['initial_features']), self.robot.lower, self.robot.upper)
        projected = self.roundtrip(clipped)
        # Codec projection can overshoot a joint bound. Shrinking in coefficient
        # space is representable; post-projection elementwise clipping is not.
        residual = projected - self.base
        for scale in (1., .5, .25, .125, .0625, .03125, 0.):
            z = self.roundtrip(self.base + scale * residual)
            if all(quick_gates(self.robot, Reference(z, c['initial_features'], c['duration'])).values()):
                return z, {'coefficient_shrink': scale,
                           'compression_max_rad': float(abs(projected - clipped).max())}
        # This is an ordinary physical rejection, not a reason to change gates.
        return projected, {'coefficient_shrink': None, 'quick_feasible': False,
                           'compression_max_rad': float(abs(projected - clipped).max())}

    def __call__(self, z, iteration):
        if iteration not in ALLOCATIONS or iteration in self.seen:
            raise ValueError('guidance stage not in frozen once-only schedule')
        self.seen.add(iteration)
        start = self.used
        self.segment_end = min(110, start + ALLOCATIONS[iteration])
        assert_frozen(self.task)
        raw = np.asarray(z, dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite clean estimate')
        proposal, preparation = self.prepare(raw)
        e, d, _ = self.evaluate(proposal)
        original = e.copy()
        history = []
        c = self.task['condition']
        if not (max(group_norms(e)) <= .65 and min(d) >= .0002) and self.segment_end-self.used >= 25:
            values = [self.evaluate(self.roundtrip(proposal + 1e-4 * direction)) for direction in self.directions]
            jg = np.stack([(v[0] - e) / 1e-4 for v in values], axis=1)
            jd = np.stack([(v[1] - d) / 1e-4 for v in values], axis=1)
            step, solver = constrained_step(e, d, jg, jd, True)
            delta = np.einsum('i,ijk->jk', step, self.directions)
            trials = []
            for scale in (1., .5, .25):
                if self.used >= self.segment_end:
                    break
                trial = self.roundtrip(proposal + scale * delta)
                if not all(quick_gates(self.robot, Reference(trial, c['initial_features'], c['duration'])).values()):
                    trials.append({'scale': scale, 'status': 'QUICK_REJECTED'})
                    continue
                ne, nd, _ = self.evaluate(trial)
                accepted = bool(accept_update(e, d, ne, nd, True))
                trials.append({'scale': scale, 'accepted': accepted, 'pose_norms': group_norms(ne),
                               'clearance_m': float(min(nd))})
                if accepted:
                    proposal, e, d = trial, ne, nd
                    break
            history.append({'solver': solver, 'trials': trials})
        return proposal, {'iteration': iteration, 'active': True, 'coupled_calls': self.used-start,
                          'total_repair_queries': self.used, 'allocation': ALLOCATIONS[iteration],
                          'initial_goal_error': original, 'final_goal_error': e,
                          'minimum_guidance_clearance_m': float(min(d)), 'history': history,
                          'raw_Z': raw, 'output_Z': proposal, 'preparation': preparation,
                          'task_hash': self.task['task_hash'], 'private_witness_used': False,
                          'full_path_coupled': True, 'guidance_grid_is_not_acceptance': True,
                          'output_hash': digest(proposal)}


def sample(bundle, base, task, seed, projector):
    model, codec = bundle
    start = model.calls
    center = codec.center(task['condition'], base)
    c = torch.tensor(codec.condition(task['condition'], base), dtype=torch.float32)[None]
    x = torch.randn((1, codec.dim), generator=torch.Generator().manual_seed(seed)).double()
    alpha = schedule().double()
    times = np.linspace(999, 0, 16).round().astype(int)
    steps, guides = [], []
    for i, t in enumerate(times):
        with torch.no_grad():
            v = model(x.float(), torch.tensor([int(t)]), c).double()
            a = alpha[t]
            clean, _ = prediction_to_clean_noise(x, v, a, 'v')
        if not torch.isfinite(clean).all():
            raise ValueError('nonfinite clean latent')
        if i in ALLOCATIONS:
            raw = codec.decode(codec.coefficients(clean[0].numpy(), center), base)
            proposed, guide = projector(raw, i)
            clean = torch.tensor(codec.normalized(codec.encode(proposed, base), center))[None]
            decoded = codec.decode(codec.coefficients(clean[0].numpy(), center), base)
            error = float(abs(decoded-proposed).max())
            if error > 1e-12:
                raise ValueError('guided trajectory lost during latent feedback')
            guide.update(latent_feedback_max_rad=error, feedback_Z_hash=digest(decoded))
            guides.append(guide)
        next_a = alpha[times[i+1]] if i < 15 else torch.tensor(1., dtype=torch.float64)
        x, residual = ddim_update(x, clean, a, next_a)
        steps.append({'iteration': i, 'timestep': int(t), 'alpha_bar': float(a),
                      'decomposition_residual_after': residual, 'state_max_abs': float(x.abs().max())})
    z = codec.decode(codec.coefficients(x[0].numpy(), center), base)
    if model.calls-start != 16:
        raise RuntimeError('network budget violation')
    assert_frozen(task)
    return z, {'actual_network_calls': 16, 'sampling_seed': seed, 'steps': steps, 'guidance': guides,
               'raw_proposal_hash': digest(guides[-1]['raw_Z']), 'latent_dimensions': codec.dim,
               'sampling_precision': 'float64_state_float32_network',
               'guidance_placement': 'late4_clean_with_recomputed_epsilon_and_latent_feedback'}


def candidate(method, k, task, base, seed, bundle, rr, ii, cc, profile, deadline):
    if method != 'late4_diffusion':
        return legacy_candidate(method, k, task, base, seed, bundle, rr, ii, cc, profile, deadline)
    assert_task_compatible(task, rr, ii, cc, profile)
    before = rr.calls.copy()
    network = bundle[0].calls
    start = time.perf_counter()
    deadline = min(deadline, time.time()+15.)
    guide = LatentProjector(task, rr, ii, cc, profile, deadline, codec=bundle[1], base=base)
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
