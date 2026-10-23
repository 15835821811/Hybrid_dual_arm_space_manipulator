"""Run source checks now, or real-terminal interface checks after eligible freeze."""
import argparse
import io
import unittest
from ff_prior.storage import file_digest
from ff_prior_pilot_train.storage import atomic_json
from .inputs import sources
from .protocol import PACKAGE,OUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal',action='store_true')
    args=parser.parse_args()
    if args.terminal:
        assert (PACKAGE/'evidence/contract.json').is_file(),'ACTUAL_TERMINAL_CONTRACT_REQUIRED'
        assert not OUT.exists(),'PREFLIGHT_ONLY_BEFORE_NEW_RUN'
    names=['ff_prior_balanced_brake.test_workflow','ff_prior_balanced_brake.test_braking']
    if args.terminal:names.append('ff_prior_balanced_brake.test_terminal_preflight')
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    text=stream.getvalue();print(text)
    filename='software_tests.json' if args.terminal else 'software_tests_source_only.json'
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),terminal_preflight=bool(args.terminal),
        new_native_steps=0,new_coupled_calls=0,new_neural_samples=0,new_geometry_queries=0,
        source_hashes=sources(),output=text,
        note='Terminal preflight only restores the actual eligible state in private data and reads reference algebra; no geometry scan or integration.')
    if args.terminal:
        record['contract_file_sha256']=file_digest(PACKAGE/'evidence/contract.json')
        record['terminal_file_sha256']=file_digest(PACKAGE/'evidence/terminal.npz')
    atomic_json(PACKAGE/'evidence'/filename,record)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
