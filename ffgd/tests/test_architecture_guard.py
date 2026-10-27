import importlib
import pytest
from ffgd.architecture_guard import LegacySentinel
from ffgd.api import PlanningBudget
from ffgd.diffusion import SamplingBudget


@pytest.mark.parametrize('backend',['profile','trace'])
def test_legacy_sentinel_actually_raises(backend):
    with pytest.raises(RuntimeError,match='LEGACY_IMPORT_SENTINEL'):
        with LegacySentinel(call_backend=backend) as guard:
            importlib.import_module('ff_prior_forbidden_sentinel_fixture')
    assert len(guard.report()['forbidden_calls'])==1
    with pytest.raises(ValueError,match='nonfinal'):
        PlanningBudget(sampling=SamplingBudget(steps=60,guide_steps=1))


@pytest.mark.parametrize('backend',['profile','trace'])
def test_already_loaded_legacy_function_is_blocked(backend):
    import types
    module=types.ModuleType('ff_prior_forbidden_fixture')
    exec('def forbidden():\n    raise AssertionError("function body must never execute")',module.__dict__)
    with pytest.raises(RuntimeError,match='LEGACY_CALL_SENTINEL'):
        with LegacySentinel(call_backend=backend) as guard:module.forbidden()
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
    with pytest.raises(RuntimeError,match='UNDECLARED_PLANNER_DATA_READ'):
        with LegacySentinel([allowed]) as guard:
            assert allowed.read_bytes()==b'test fixture'
            hidden.read_bytes()
    report=guard.report()
    assert report['file_audit_enforced'] and len(report['files_read'])==2
    assert len(report['forbidden_calls'])==1


def test_npz_creation_is_output_but_cannot_be_used_as_hidden_input(tmp_path):
    import numpy as np
    output=tmp_path/'new-run';output.mkdir();path=output/'candidate.npz'
    with pytest.raises(RuntimeError,match='UNDECLARED_PLANNER_DATA_READ'):
        with LegacySentinel([],output_directory=output) as guard:
            np.savez_compressed(path,x=np.ones(3));np.load(path)
    assert len(guard.report()['created_outputs'])==1


def test_main_guard_covers_public_input_loading_before_sampling(tmp_path,monkeypatch):
    import json
    import types
    import ffgd.cli as cli
    import ffgd.problem_io as problem_io
    from ffgd.architecture_guard import ArchitectureViolation
    legacy=types.ModuleType('ff_prior_forbidden_loader_fixture')
    exec('def forbidden():\n    raise AssertionError("loader must not execute forbidden code")',legacy.__dict__)
    monkeypatch.setattr(problem_io,'load_problem',lambda path:legacy.forbidden())
    config=tmp_path/'config.json';task=tmp_path/'task.json';output=tmp_path/'run';output.mkdir()
    config.write_text(json.dumps(dict(checkpoint=str(tmp_path/'weights.pt'),profile=str(tmp_path/'profile.json'),budget={'seeds':[71]})))
    task.write_text(json.dumps({'initial':{}}))
    monkeypatch.setattr(cli,'unit',lambda name,callback,budget:callback(output))
    with pytest.raises(ArchitectureViolation,match='LEGACY_CALL_SENTINEL'):
        cli.main(['plan','--config',str(config),'--task',str(task)])
    report=json.loads((output/'architecture_sentinel.json').read_text())
    result=json.loads((output/'planning_result.json').read_text())
    assert report['forbidden_calls'] and result['sampled']==0 and result['requested_candidates']==1


def test_guard_report_preserves_executed_code_after_module_import_failure():
    import types
    module=types.ModuleType('ffgd.unregistered_software_fixture')
    exec('def function():\n    return 1',module.__dict__)
    with LegacySentinel() as guard:assert module.function()==1
    report=guard.report()
    assert report['actual_methods'][0]['executed_code_sha256']
    identity=next(row for row in report['module_identities'] if row['module']==module.__name__)
    assert identity['source_status']=='UNAVAILABLE_OR_IMPORT_FAILED'


def test_controller_cannot_run_inside_planning_guard():
    from ffgd.execution import command
    from ffgd.architecture_guard import ArchitectureViolation
    with pytest.raises(ArchitectureViolation,match='LEGACY_CALL_SENTINEL'):
        with LegacySentinel() as guard:
            command(None,None,None,None,None)
    assert any('ffgd.execution.command' in name for name in guard.report()['forbidden_calls'])


def test_swallowed_violation_rejects_selected_result(tmp_path,monkeypatch):
    # Software-only stub selection proves a caught exception cannot authorize
    # execution. This does not generate any physical candidate.
    import json
    import types
    import ffgd.cli as cli
    import ffgd.api as api
    import ffgd.problem_io as problem_io
    from ffgd.architecture_guard import ArchitectureViolation
    legacy=types.ModuleType('ff_prior_swallowed_fixture')
    exec('def forbidden():\n    raise AssertionError("must not run")',legacy.__dict__)
    config=tmp_path/'config.json';task=tmp_path/'task.json';profile=tmp_path/'profile.json';output=tmp_path/'run';output.mkdir()
    config.write_text(json.dumps(dict(checkpoint=str(tmp_path/'weights.pt'),profile=str(profile),budget={'seeds':[71]})))
    task.write_text(json.dumps({'initial':{}}));profile.write_text('{}')
    monkeypatch.setattr(problem_io,'load_problem',lambda path:(None,None,None,{'initial':{}}))
    def swallowed(*args,**kwargs):
        try:legacy.forbidden()
        except ArchitectureViolation:pass
        result=dict(status='PLAN_VALID',selected_index=0,records=[],requested_candidates=1,sampled=0,scope='SOFTWARE_FIXTURE')
        (args[6]/'planning_result.json').write_text(json.dumps(result))
        return object(),result
    monkeypatch.setattr(api,'plan',swallowed);monkeypatch.setattr(cli,'unit',lambda name,callback,budget:callback(output))
    with pytest.raises(ArchitectureViolation,match='CAUGHT_INSIDE_PLANNER'):
        cli.main(['plan','--config',str(config),'--task',str(task)])
    result=json.loads((output/'planning_result.json').read_text())
    assert result['status']=='ARCHITECTURE_REJECTED' and result['selected_index'] is None
    assert result['model_selection_before_architecture_rejection']['selected_index']==0


def test_raw_file_descriptors_distinguish_writes_and_reads_on_windows(tmp_path):
    import os
    from ffgd.architecture_guard import ArchitectureViolation
    output=tmp_path/'run';output.mkdir()
    hidden=tmp_path/'hidden.npy';hidden.write_bytes(b'forbidden input')
    with pytest.raises(ArchitectureViolation,match='UNDECLARED_PLANNER_DATA_READ'):
        with LegacySentinel([],output_directory=output):
            descriptor=os.open(str(output/'code.pyc'),os.O_WRONLY|os.O_CREAT|os.O_TRUNC)
            os.write(descriptor,b'code cache');os.close(descriptor)
            descriptor=os.open(str(output/'arrays.npz'),os.O_RDWR|os.O_CREAT|os.O_TRUNC)
            os.close(descriptor)
            os.open(str(hidden),os.O_RDONLY)
