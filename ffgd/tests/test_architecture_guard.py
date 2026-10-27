import importlib
import pytest
from ffgd.architecture_guard import LegacySentinel
from ffgd.api import PlanningBudget
from ffgd.diffusion import SamplingBudget


def test_legacy_sentinel_actually_raises():
    with LegacySentinel() as guard:
        with pytest.raises(RuntimeError,match='LEGACY_IMPORT_SENTINEL'):
            importlib.import_module('ff_prior_forbidden_sentinel_fixture')
    assert len(guard.report()['forbidden_calls'])==1
    with pytest.raises(ValueError,match='nonfinal'):
        PlanningBudget(sampling=SamplingBudget(steps=60,guide_steps=1))


def test_undeclared_path_data_is_blocked_and_allowed_weights_are_recorded(tmp_path):
    allowed=tmp_path/'weights.pt';allowed.write_bytes(b'test fixture')
    hidden=tmp_path/'historical_success.npy';hidden.write_bytes(b'test fixture')
    with LegacySentinel([allowed]) as guard:
        assert allowed.read_bytes()==b'test fixture'
        with pytest.raises(RuntimeError,match='UNDECLARED_PLANNER_DATA_READ'):hidden.read_bytes()
    report=guard.report()
    assert report['file_audit_enforced'] and len(report['files_read'])==2
    assert len(report['forbidden_calls'])==1
