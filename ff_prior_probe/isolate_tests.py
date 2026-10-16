"""Reuse redirection logic with new outputs; preserve all previous evidence."""
from pathlib import Path
from ff_prior_feasible.tools import isolated_tests as old
from ff_prior.storage import file_digest,atomic_json

ROOT=Path(__file__).resolve().parents[1]
old.OUTPUT=ROOT/'artifacts/v7_1/probe_test_isolation_001'
pytest_configure=old.pytest_configure
pytest_collection_modifyitems=old.pytest_collection_modifyitems


def pytest_sessionfinish(session,exitstatus):
    changed=[p for p,h in old.before.items() if file_digest(p)!=h]
    atomic_json(ROOT/'ff_prior_probe/evidence/test_isolation_verification.json',
        {'pytest_exitstatus':int(exitstatus),'protected_file_hashes':old.before,'changed_files':changed,
         'redirected_output':str(old.OUTPUT),'passed':exitstatus==0 and not changed})
    if changed: raise AssertionError('legacy evidence changed')
