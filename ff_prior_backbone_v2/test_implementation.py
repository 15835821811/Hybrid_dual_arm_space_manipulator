import copy
import torch
import numpy as np
import pytest
from ff_prior.storage import read_json, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.diffusion import schedule
from ff_prior_latent.learning import Codec
from ff_prior_paired_objectives.objectives import terms, vector_gradient, get_vector
from ff_prior_endpoint_jacobian.state_audit import reconstruct_adamw, state_close
from ff_prior_delivery_guidance.sampling import sample
from .models import make, parameter_counts
from .protocol import P, CONTROL
from .experiment import assert_resume_safe, Budget
from .learning import load

torch.set_num_threads(2)


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_actual_update_evidence_on_new_nested_path(tmp_path, kind):
    from .learning import save_torch
    from ff_prior.diffusion import weight_hash
    model = make(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    directory = tmp_path / "中文工作区" / "learning" / kind / "updates" / "0001"
    before = weight_hash(model)
    save_torch(directory / "before.pt", {"weights": model.state_dict(), "optimizer": opt.state_dict()})
    value = model(torch.zeros(4, 102), torch.tensor([0, 333, 667, 999]), torch.ones(4, 65)).square().mean()
    value.backward()
    opt.step()
    save_torch(directory / "after.pt", {"weights": model.state_dict(), "optimizer": opt.state_dict()})
    after = weight_hash(model)
    assert before != after and not list(directory.glob("*.tmp"))
    for name, expected in (("before", before), ("after", after)):
        saved = torch.load(directory / (name + ".pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(saved["weights"])
        assert weight_hash(model) == expected
        if name == "after":
            assert all(float(s["step"]) == 1 for s in saved["optimizer"]["state"].values())


def test_parameter_match_and_random_stream_isolation():
    counts = parameter_counts()
    assert counts == {"mlp": 208998, "transformer": 211969}
    assert max(counts.values()) / min(counts.values()) < 1.05
    state = torch.get_rng_state().clone()
    make("mlp")
    make("transformer")
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_condition_and_full_latent_receive_gradients(kind):
    model = make(kind)
    x = torch.randn(4, 102, requires_grad=True)
    c = torch.randn(4, 65, requires_grad=True)
    y = model(x, torch.tensor([0, 333, 667, 999]), c)
    gx, gc = torch.autograd.grad(y.square().sum(), (x, c))
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert torch.all(gx.abs().sum(0) > 0) and torch.all(gc.abs().sum(0) > 0)
    model.eval()
    assert torch.equal(y, model(x, torch.tensor([0, 333, 667, 999]), c))


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_nonempty_adamw_reconstruction(kind):
    model = make(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    clean, c = torch.randn(4, 102), torch.randn(4, 65)
    t, noise, G = torch.tensor([0, 333, 667, 999]), torch.randn(4, 102), torch.randn(4, 12, 102)*.1
    for _ in range(2):
        v, g, _ = terms(model, clean, c, t, noise, schedule(), G)
        value = v + .05*g
        grad = vector_gradient(value, model).detach()
        expected, state = reconstruct_adamw(model, grad, copy.deepcopy(opt.state_dict()))
        opt.zero_grad()
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        assert torch.allclose(get_vector(model), expected, atol=1e-8, rtol=2e-6)
        state_close(state, opt.state_dict())


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_shared_ddim_actual_sixteen_calls_and_four_feedbacks(kind):
    # Real frozen codec/task, but identity projector: software algebra only.
    codec = Codec(read_json(CONTROL / "learning/goal_loss/codec.json"))
    entries = read_json(CONTROL / "frozen_tasks.json")["entries"]
    task = entries[0]["task"]
    base = np.tile(task["condition"]["initial_features"], (19, 1))
    model = make(kind)
    seen = []
    def projector(raw, iteration):
        seen.append(iteration)
        return raw, {"raw_Z": raw}
    z, info = sample((model, codec), base, task, 20263409, projector)
    assert z.shape == (19, 17) and np.isfinite(z).all()
    assert model.calls == 16 and seen == [12, 13, 14, 15]
    assert len(info["steps"]) == 16
    assert max(v["decomposition_residual_after"] for v in info["steps"]) <= 1e-10


def test_resume_blocks_unknown_physics_and_training(tmp_path):
    atomic_json(tmp_path/"ledger.json", {"inflight": {"id": "x"}})
    with pytest.raises(ValueError, match="unknown physics"):
        assert_resume_safe(tmp_path)
    atomic_json(tmp_path/"ledger.json", {"inflight": None})
    atomic_json(tmp_path/"learning/mlp/journal.json", {"status": "PENDING"})
    with pytest.raises(ValueError, match="unknown training"):
        assert_resume_safe(tmp_path)
    atomic_json(tmp_path/"learning/mlp/journal.json", {"status": "COMMITTED"})
    assert_resume_safe(tmp_path)


def test_budget_is_persistent_and_bounded(tmp_path):
    import time
    budget = Budget(tmp_path, time.time()+30)
    budget.reserve(113)
    other = Budget(tmp_path, time.time()+30)
    assert other.total == 113
    with pytest.raises(RuntimeError, match="QUERY_LIMIT"):
        other.reserve(P["coupled_cap"])
    assert 360*113 + 192*3 == 41256 < P["coupled_cap"]


def test_protocol_never_requires_superiority_or_test():
    assert P["nonlinear_world_training_events"] == 0
    assert not any(P[k] for k in ("TEST", "expand", "pilot", "study", "strict_multimodal"))
    assert "no superiority gate" in P["gate_to_DEV"]


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_checkpoint_loader_rejects_mutation(tmp_path, kind):
    from ff_prior_origin_learn_bounded.learning import save_torch
    from ff_prior.storage import file_digest
    from ff_prior.diffusion import weight_hash
    codec = read_json(CONTROL/"learning/goal_loss/codec.json")
    model = make(kind)
    meta = {"codec_hash": digest(codec)}
    atomic_json(tmp_path/"codec.json", codec)
    save_torch(tmp_path/"model.pt", {"weights": model.state_dict(), "metadata": meta})
    atomic_json(tmp_path/"training.json", {"kind": kind, "metadata": meta,
                "checkpoint_sha256": file_digest(tmp_path/"model.pt"), "trained_weight_hash": weight_hash(model)})
    restored, _ = load(tmp_path)
    assert weight_hash(restored) == weight_hash(model)
    with (tmp_path/"model.pt").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError, match="identity"):
        load(tmp_path)
