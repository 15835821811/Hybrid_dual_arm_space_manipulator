"""Instrumented deterministic sampler; algebra matches the frozen DDIM code."""
import numpy as np
import torch
from ff_prior.contracts import assert_frozen
from ff_prior.diffusion import schedule, prediction_to_clean_noise
from ff_prior_next.learning import OffsetScaler, OffsetPrior


def sample(bundle, base, task, seed, projector, observer=None):
    model, scaler, prior = bundle
    scaler, prior = OffsetScaler(scaler, base), OffsetPrior(prior, base)
    assert_frozen(task)
    x = torch.randn((1, 19, 17), generator=torch.Generator().manual_seed(seed)).double()
    c = torch.tensor(scaler.condition(task['condition']), dtype=torch.float32)[None]
    alpha = schedule().double(); times = np.linspace(999, 0, 16).round().astype(int)
    logs, steps, prior_log = [], [], []; start = model.calls
    final = None
    for i, t in enumerate(times):
        with torch.no_grad():
            v = model(x.float(), torch.tensor([int(t)]), c).double()
            a = alpha[t]; x0, epsilon = prediction_to_clean_noise(x, v, a, 'v')
        raw = scaler.inverse_z(x0[0].numpy())
        if observer and i in (0, 7, 14, 15): observer(raw, i, 'network_clean')
        projected = prior.project(raw)
        if observer and i in (0, 7, 14, 15): observer(projected, i, 'after_temporal_prior')
        prior_log.append({'iteration': i, 'rank': prior.metadata['rank'],
                          'correction_norm_rad': float(np.linalg.norm(projected-raw))})
        x0 = torch.tensor(scaler.z(projected), dtype=torch.float64)[None]
        epsilon = (x-a.sqrt()*x0)/(1-a).sqrt()
        if i >= 14:
            z, info = projector(scaler.inverse_z(x0[0].numpy()), i)
            x0 = torch.tensor(scaler.z(z), dtype=torch.float64)[None]
            epsilon = (x-a.sqrt()*x0)/(1-a).sqrt()
            logs.append(info)
            if i == 15: final = np.asarray(z, dtype=np.float64).copy()
        steps.append({'iteration': i, 'timestep': int(t), 'alpha_bar': float(a),
            'decomposition_residual_after': float((x-a.sqrt()*x0-(1-a).sqrt()*epsilon).abs().max())})
        next_a = alpha[times[i+1]] if i+1 < 16 else torch.tensor(1.)
        x = next_a.sqrt()*x0+(1-next_a).sqrt()*epsilon
    if model.calls-start != 16: raise RuntimeError('network budget violation')
    assert_frozen(task)
    return final, {'actual_network_calls': 16, 'sampling_seed': seed, 'prediction_type': 'v',
        'guidance': logs, 'steps': steps, 'clean_prior_projection': prior_log,
        'sampling_precision': 'float64_state_float32_network'}
