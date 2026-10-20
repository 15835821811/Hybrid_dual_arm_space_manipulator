import copy
import numpy as np
import torch
import pytest
from ff_prior.storage import read_json
from ff_prior_latent.learning import Codec, LatentDenoiser
from ff_prior_endpoint_jacobian.comparison import unguided_sample
from ff_prior_expand256_train.learning import load as warm_load
from ff_prior_paired_objectives.objectives import vector_gradient, get_vector, set_vector
from ff_prior_endpoint_jacobian.state_audit import reconstruct_adamw, state_close
from ff_prior_goal.projection import smooth
from ff_prior_backbone_v2.learning import save_torch
from .objectives import rollout, prepared_fixed, combine, assign
from .protocol import CONTROL, P
from .experiment import assert_resume_safe, Budget

torch.set_num_threads(2)


def inputs():
    model, codec = warm_load(CONTROL / "learning/goal_loss")
    task = read_json(CONTROL / "frozen_tasks.json")["entries"][0]["task"]
    base = np.tile(task["condition"]["initial_features"], (19, 1))
    return model, codec, task, base


def test_full_sampler_exact_and_all_steps_have_gradient():
    model, codec, task, base = inputs()
    condition = torch.tensor(codec.condition(task["condition"], base), dtype=torch.float32)[None]
    calls = model.calls
    raw, _ = unguided_sample((model, codec), base, task, 1234)
    x = rollout(model, condition, [1234])
    assert model.calls-calls == 32
    assert np.array_equal(raw, codec.decode(codec.coefficients(x[0].detach().numpy(), codec.center(task["condition"], base)), base))
    g = vector_gradient(x.square().mean(), model)
    assert torch.isfinite(g).all() and g.norm() > 0


def test_full_sampler_parameter_direction_difference():
    model, codec, task, base = inputs()
    c = torch.tensor(codec.condition(task["condition"], base), dtype=torch.float32)[None]
    v = get_vector(model)
    f = rollout(model, c, [5123]).square().mean()
    g = vector_gradient(f, model).detach()
    direction = g / g.norm()
    eps = 1e-3
    vals = []
    for sign in (-1, 1):
        set_vector(model, v+sign*eps*direction)
        with torch.no_grad():
            vals.append(float(rollout(model, c, [5123]).square().mean()))
    set_vector(model, v)
    observed = (vals[1]-vals[0])/(2*eps)
    expected = float(g@direction)
    assert abs(observed-expected)/max(abs(observed), abs(expected), 1e-6) < .03


@pytest.mark.parametrize("shrink", [1., .5, 0.])
def test_prepare_affine_clip_projection_matches_numpy(shrink):
    _, codec, task, base = inputs()
    center = codec.center(task["condition"], base)
    x = torch.randn(102, dtype=torch.float64, requires_grad=True)
    lower, upper = np.full(17, -.3), np.full(17, .3)
    normalized, z = prepared_fixed(x, codec, center, base, task, lower, upper, shrink)
    raw = codec.decode(codec.coefficients(x.detach().numpy(), center), base)
    clipped = np.clip(smooth(raw, task["condition"]["initial_features"]), lower, upper)
    projected = codec.decode(codec.encode(clipped, base), base)
    wanted = codec.decode(codec.encode(base+shrink*(projected-base), base), base)
    assert np.max(abs(z.detach().numpy()-wanted)) < 1e-12
    grad = torch.autograd.grad(normalized.square().mean(), x)[0]
    assert torch.isfinite(grad).all()
    if shrink == 0:
        assert torch.count_nonzero(grad) == 0


def test_combined_nonempty_adamw_exact_single_update():
    model = LatentDenoiser(102, 65)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    for _ in range(2):
        x = torch.randn(2, 102)
        value = model(x, torch.tensor([333, 999]), torch.randn(2, 65)).square().mean()
        base = vector_gradient(value, model).detach()
        mixed, scales = combine(base, base*10, base*-3)
        assert all(s["weighted_norm"] <= float(base.norm())*.500001 for s in scales.values())
        expected, st = reconstruct_adamw(model, mixed, copy.deepcopy(opt.state_dict()))
        opt.zero_grad()
        assign(model, mixed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        assert torch.allclose(expected, get_vector(model), rtol=2e-6, atol=1e-8)
        state_close(st, opt.state_dict())


def test_nested_state_save(tmp_path):
    path = tmp_path / "new/updates/0001/before.pt"
    save_torch(path, {"x": torch.ones(2)})
    assert torch.equal(torch.load(path, weights_only=False)["x"], torch.ones(2))


def test_resume_unknown_work_and_budget_fail_closed(tmp_path):
    from ff_prior_pilot_train.storage import atomic_json
    atomic_json(tmp_path / "ledger.json", {"inflight": "unknown"})
    with pytest.raises(ValueError):
        assert_resume_safe(tmp_path)
    b = Budget(tmp_path, 1e12)
    with pytest.raises(RuntimeError):
        b.reserve(P["coupled_cap"]+1)


def test_no_superiority_or_test_gate():
    assert not P["TEST"] and not P["expand"] and not P["primary_physics_relaxation"]
    assert "no advantage gate" in P["gate_to_DEV"]
    assert P["new_candidates"] == 360 and P["new_execution_units"] == 72
