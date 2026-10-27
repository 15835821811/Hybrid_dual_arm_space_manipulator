import importlib
import pytest
from ffgd.architecture_guard import LegacySentinel
from ffgd.api import PlanningBudget
from ffgd.diffusion import SamplingBudget


@pytest.mark.parametrize('backend',['profile','trace'])
def test_legacy_sentinel_actually_raises(backend):
    with LegacySentinel(call_backend=backend) as guard:
        with pytest.raises(RuntimeError,match='LEGACY_IMPORT_SENTINEL'):
            importlib.import_module('ff_prior_forbidden_sentinel_fixture')
    assert len(guard.report()['forbidden_calls'])==1
    with pytest.raises(ValueError,match='nonfinal'):
        PlanningBudget(sampling=SamplingBudget(steps=60,guide_steps=1))


@pytest.mark.parametrize('backend',['profile','trace'])
def test_already_loaded_legacy_function_is_blocked(backend):
    import types
    module=types.ModuleType('ff_prior_forbidden_fixture')
    exec('def forbidden():\n    raise AssertionError("function body must never execute")',module.__dict__)
    with LegacySentinel(call_backend=backend) as guard:
        with pytest.raises(RuntimeError,match='LEGACY_CALL_SENTINEL'):module.forbidden()
    assert len(guard.report()['forbidden_calls'])==1


def test_main_planner_does_not_retry_after_architecture_violation(tmp_path,monkeypatch):
    # Isolated software fixture: no trained-plan or physical-pass claim.
    import numpy as np
    import torch
    import ffgd.api as api
    from ffgd.robot import Robot
    from ffgd.contracts import TaskSpec,PoseConstraint
    from ffgd.architecture_guard import ArchitectureViolation
    torch.set_num_threads(1);r=Robot();snapshot=r.snapshot(r.new_data())
    checkpoint=dict(update=1,training_config={'scope':'SOFTWARE_SENTINEL_FIXTURE'},dataset_manifest={'source_kind':'FF_MODEL_PRIOR'},
                    robot_physical_hash=r.physical_hash,mean=np.zeros(323),scale=np.ones(323),condition_mean=np.zeros(32),condition_scale=np.ones(32))
    monkeypatch.setattr(api,'load_model',lambda *args:(torch.nn.Linear(1,1),checkpoint,{'scope':'SOFTWARE_SENTINEL_FIXTURE'}))
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(1);raise ArchitectureViolation('LEGACY_CALL_SENTINEL: test fixture')
    monkeypatch.setattr(api,'sample',forbidden)
    with pytest.raises(ArchitectureViolation):
        api.plan(r,snapshot,TaskSpec(4.,(PoseConstraint(4.,0,[1,2,3],np.eye(3)),)),'fixture',PlanningBudget(),{},tmp_path)
    assert len(calls)==1


def test_undeclared_path_data_is_blocked_and_allowed_weights_are_recorded(tmp_path):
    allowed=tmp_path/'weights.pt';allowed.write_bytes(b'test fixture')
    hidden=tmp_path/'historical_success.npy';hidden.write_bytes(b'test fixture')
    with LegacySentinel([allowed]) as guard:
        assert allowed.read_bytes()==b'test fixture'
        with pytest.raises(RuntimeError,match='UNDECLARED_PLANNER_DATA_READ'):hidden.read_bytes()
    report=guard.report()
    assert report['file_audit_enforced'] and len(report['files_read'])==2
    assert len(report['forbidden_calls'])==1


def test_npz_creation_is_output_but_cannot_be_used_as_hidden_input(tmp_path):
    import numpy as np
    output=tmp_path/'new-run';output.mkdir();path=output/'candidate.npz'
    with LegacySentinel([],output_directory=output) as guard:
        np.savez_compressed(path,x=np.ones(3))
        with pytest.raises(RuntimeError,match='UNDECLARED_PLANNER_DATA_READ'):np.load(path)
    assert len(guard.report()['created_outputs'])==1
