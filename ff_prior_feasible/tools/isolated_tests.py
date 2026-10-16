"""pytest plugin: redirect legacy fixed scratch outputs without changing physics.

Usage: python -m pytest -p ff_prior_feasible.tools.isolated_tests ...
"""
import inspect
from pathlib import Path
from ff_prior.storage import file_digest,atomic_json

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'artifacts/v7_1/feasible_test_isolation_001'
before={}


def pytest_configure(config):
    global before
    paths=list((ROOT/'artifacts/v7_1/test_physics').glob('*'))
    paths.append(ROOT/'artifacts/v7_1/geometry_counterexamples.json')
    before={str(p):file_digest(p) for p in paths if p.is_file()}


def pytest_collection_modifyitems(session,config,items):
    seen=set()
    for item in items:
        module=item.module
        if id(module) in seen or Path(module.__file__).resolve()!=ROOT/'ff_prior/tests/test_physics.py': continue
        seen.add(id(module)); original=module.audit_execution; original_json=module.atomic_json
        def execute(*args, _original=original, **kwargs):
            bound=inspect.signature(_original).bind(*args,**kwargs)
            if Path(bound.arguments['output']).as_posix()=='artifacts/v7_1/test_physics':
                bound.arguments['output']=OUTPUT/'test_physics'
            return _original(*bound.args,**bound.kwargs)
        def write(path,value,_original=original_json):
            if Path(path).as_posix()=='artifacts/v7_1/geometry_counterexamples.json':
                path=OUTPUT/'geometry_counterexamples.json'
            return _original(path,value)
        module.audit_execution=execute; module.atomic_json=write


def pytest_sessionfinish(session,exitstatus):
    changed=[p for p,h in before.items() if file_digest(p)!=h]
    atomic_json(ROOT/'ff_prior_feasible/evidence/test_isolation_verification.json',
        {'pytest_exitstatus':int(exitstatus),'protected_file_hashes':before,'changed_files':changed,
         'redirected_output':str(OUTPUT),'passed':exitstatus==0 and not changed})
    if changed: raise AssertionError('legacy scratch files changed despite isolation')
