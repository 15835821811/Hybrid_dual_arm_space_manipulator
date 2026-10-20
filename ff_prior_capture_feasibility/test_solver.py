"""Software-only checks; these do not certify any robot trajectory."""
import numpy as np
import pytest
from types import SimpleNamespace
from ff_prior.storage import read_json
from ff_prior_latent.learning import Codec
from .solver import basis, admissible_difference, accept, solve_step
from .protocol import SOURCE, P
from .experiment import assert_resume_safe
from ff_prior_pilot_train.storage import atomic_json

@pytest.fixture
def codec():
    return Codec(read_json(SOURCE/"learning/aligned/codec.json"))

def test_codec_directions_are_actual_coefficient_map(codec):
    d = basis("codec102", codec)
    y = np.random.default_rng(1234).normal(size=102)
    assert d.shape == (102,19,17)
    assert np.allclose(np.einsum("i,ijk->jk", y, d), codec.decode(y, np.zeros((19,17))), atol=1e-14)
    assert np.linalg.matrix_rank(d.reshape(102,-1)) == 102

def test_legacy_and_full_subspaces(codec):
    full, legacy = basis("full323", codec), basis("legacy24", codec)
    assert np.array_equal(full.reshape(323,323), np.eye(323))
    assert legacy.shape == (24,19,17)
    for value in legacy:
        assert np.allclose(codec.decode(codec.encode(value, np.zeros_like(value)), np.zeros_like(value)), value)

def test_inward_difference_and_blocked_bounds():
    z = np.zeros((19,17)); d = z.copy(); d[0,0] = 1
    z[0,0] = 1
    assert admissible_difference(z,d,-np.ones(17),np.ones(17)) < 0
    z[0,0] = -1
    assert admissible_difference(z,d,-np.ones(17),np.ones(17)) > 0
    z[0,1] = 1; d[0,1] = 1
    assert admissible_difference(z,d,-np.ones(17),np.ones(17)) == 0

def test_acceptance_preserves_pose_and_requires_collision_descent():
    def r(q,d):
        e = np.zeros(12); e[0] = q
        return {"goal":e, "minima":np.array([d])}
    assert accept(r(.5,-.02),r(.6,-.01))
    assert not accept(r(.5,-.02),r(.9,-.01))
    assert not accept(r(.5,-.02),r(.4,-.021))
    assert accept(r(2.,-.02),r(1.,-.01))

def test_solver_resolves_known_linear_collision_without_sacrificing_goal():
    z = np.zeros((19,17)); dirs = np.zeros((2,19,17))
    dirs[0,0,0] = 1; dirs[1,0,1] = 1
    jg = np.zeros((12,2)); jg[0,1] = 1
    step, info = solve_step(z, dirs, np.zeros(12), np.array([-.01]),
                           jg, np.array([[1.,0.]]), -np.ones(17), np.ones(17))
    assert info["linear_pose_feasible"]
    assert -.01+step[0] > -.0001
    assert abs(step[1]) <= P["solver_goal_fraction"]

def test_unknown_work_is_not_repeated(tmp_path):
    atomic_json(tmp_path/"ledger.json", {"inflight":{"id":"uncertain"}})
    with pytest.raises(ValueError, match="UNKNOWN_WORK"):
        assert_resume_safe(tmp_path)
    atomic_json(tmp_path/"ledger.json", {"inflight":None, "unresolved_call_accounting":["uncertain"]})
    with pytest.raises(ValueError, match="UNKNOWN_WORK"):
        assert_resume_safe(tmp_path)

def test_frozen_budget_is_bounded():
    worst = 12*(1+10*(323+4)+4) + (1+10*(102+4)+4) + 8+12+110+4
    assert worst < P["coupled_cap"]
    assert not P["TEST"] and not P["training"] and P["TRAIN_added"] == 0
