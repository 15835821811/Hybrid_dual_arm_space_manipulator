"""Autograd through all sixteen DDIM updates, with the deployment float types."""
import numpy as np
import torch
from ff_prior.diffusion import schedule, prediction_to_clean_noise
from ff_prior_goal.projection import smooth
from ff_prior_delivery_guidance.sampling import LatentProjector
from ff_prior_paired_objectives.objectives import vector_gradient


def rollout(model, condition, seeds):
    x = torch.cat([torch.randn((1, 102), generator=torch.Generator().manual_seed(int(s)))
                   for s in seeds]).double()
    alpha = schedule().double()
    times = np.linspace(999, 0, 16).round().astype(int)
    for j, t in enumerate(times):
        noisy = x
        a = alpha[t]
        v = model(noisy.float(), torch.full((len(seeds),), int(t)), condition).double()
        clean, _ = prediction_to_clean_noise(noisy, v, a, "v")
        # Identical to deployed ddim_update, without detaching its derivative.
        epsilon = (noisy - a.sqrt() * clean) / (1 - a).sqrt()
        next_a = alpha[times[j + 1]] if j < 15 else x.new_tensor(1.)
        x = next_a.sqrt() * clean + (1 - next_a).sqrt() * epsilon
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite full DDIM")
    return x


def teacher_loss(model, data, G, indices, seeds):
    predicted = rollout(model, data["c"][indices], seeds)
    delta = predicted - data["x"][indices].double()
    g = G[[data["lookup"][int(i)] for i in indices]].double()
    local = torch.einsum("bij,bj->bi", g, delta).square().mean()
    value = delta.square().mean() + .05 * local
    return value, predicted


def prepared_output(clean, codec, center, base, task, robot, initial, collision, profile, deadline):
    """Differentiate the exact active prepare branch; no straight-through clipping."""
    base_t = torch.as_tensor(base, dtype=torch.float64)
    basis = torch.as_tensor(codec.basis, dtype=torch.float64)
    scale = torch.as_tensor(codec.latent_scale, dtype=torch.float64)
    offset = torch.as_tensor(codec.latent_mean + center, dtype=torch.float64)
    raw = base_t + basis @ (clean * scale + offset).reshape(6, 17)
    raw_np = raw.detach().numpy()
    projector = LatentProjector(task, robot, initial, collision, profile, deadline,
                               codec=codec, base=base)
    prepared, prep = projector.prepare(raw_np)
    if prep.get("coefficient_shrink") is None:
        raise ValueError("prepare lacks a finite legal branch")
    normalized, final = prepared_fixed(clean, codec, center, base, task,
                                        robot.lower, robot.upper, prep["coefficient_shrink"])
    max_error = float(np.max(np.abs(final.detach().numpy() - prepared)))
    if max_error > 1e-12:
        raise ValueError("differentiable prepare disagrees with deployment")
    return normalized, dict(prep, differentiable_prepare_max_rad=max_error,
                            raw_Z=raw_np, prepared_Z=prepared,
                            lower=robot.lower, upper=robot.upper)


def prepared_fixed(clean, codec, center, base, task, lower, upper, shrink):
    """Read-only neural audit can rebuild the saved active branch without physics."""
    base_t = torch.as_tensor(base, dtype=torch.float64)
    basis = torch.as_tensor(codec.basis, dtype=torch.float64)
    scale = torch.as_tensor(codec.latent_scale, dtype=torch.float64)
    offset = torch.as_tensor(codec.latent_mean + center, dtype=torch.float64)
    raw = base_t + basis @ (clean * scale + offset).reshape(6, 17)
    # smooth is affine in the 19 time controls, independently per coordinate.
    q0 = np.asarray(task["condition"]["initial_features"])
    zero = np.zeros((19, 17))
    affine = smooth(zero, q0)
    matrix = np.stack([smooth(np.eye(19)[:, j, None] * np.ones((1, 17)), np.zeros(17))[:, 0]
                       for j in range(19)], axis=1)
    smoothed = torch.as_tensor(matrix) @ raw + torch.as_tensor(affine)
    clipped = torch.maximum(torch.minimum(smoothed, torch.as_tensor(upper)),
                            torch.as_tensor(lower))
    projected = base_t + basis @ (basis.T @ (clipped - base_t))
    shrunk = base_t + float(shrink) * (projected - base_t)
    final = base_t + basis @ (basis.T @ (shrunk - base_t))
    normalized = ((basis.T @ (final - base_t)).reshape(-1) - offset) / scale
    return normalized, final


def capped_component(base, extra, cap):
    bn, en = float(base.norm()), float(extra.norm())
    coefficient = min(1., cap * bn / max(en, 1e-30))
    return coefficient * extra, coefficient


def assign(model, vector):
    offset = 0
    for p in model.parameters():
        p.grad = vector[offset:offset + p.numel()].view_as(p).clone()
        offset += p.numel()


def combine(base, teacher=None, world=None):
    mixed = base.clone()
    scales = {}
    for name, g in (("teacher", teacher), ("world", world)):
        if g is not None:
            part, s = capped_component(base, g, .5)
            mixed += part
            scales[name] = {"coefficient": s, "norm": float(g.norm()),
                            "weighted_norm": float(part.norm())}
    if not torch.isfinite(mixed).all():
        raise ValueError("nonfinite composed gradient")
    return mixed, scales
