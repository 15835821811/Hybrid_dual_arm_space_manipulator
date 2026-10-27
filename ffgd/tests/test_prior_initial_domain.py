import numpy as np
import pytest
from types import SimpleNamespace
from ffgd.data import sample_initial_position


def robot():
    return SimpleNamespace(spec=SimpleNamespace(planner_zero=np.zeros(17)),lower=-np.arange(1,18.),upper=np.arange(1,18.))


def test_factorized_sampling_covers_every_coordinate_and_preserves_other_state():
    r=robot();config=dict(initial_variation=.1,initial_policy='factorized_limits',initial_limit_fraction=.97,initial_strata=8)
    values=[]
    for group in range(17*8):
        a=sample_initial_position(r,config,group,np.random.default_rng(group))
        b=sample_initial_position(r,config,group,np.random.default_rng(group))
        np.testing.assert_array_equal(a,b)
        assert np.all(a>r.lower) and np.all(a<r.upper)
        j=group%17;other=np.arange(17)!=j
        assert np.max(abs(a[other]))<=.1
        stratum=(group//17)%8
        scaled=(a[j]/(r.upper[j]*.97)+1)/2
        assert stratum/8 <= scaled <= (stratum+1)/8
        values.append(a)
    values=np.array(values)
    assert np.all(values.min(0)/r.upper<-.72)
    assert np.all(values.max(0)/r.upper>.72)


def test_original_home_distribution_unchanged_and_unknown_policy_rejected():
    r=robot();config=dict(initial_variation=.1)
    expected=np.random.default_rng(42).uniform(-.1,.1,17)
    np.testing.assert_array_equal(sample_initial_position(r,config,100,np.random.default_rng(42)),expected)
    with pytest.raises(ValueError,match='UNKNOWN_INITIAL_PRIOR_POLICY'):
        sample_initial_position(r,dict(config,initial_policy='unknown'),0,np.random.default_rng(0))
