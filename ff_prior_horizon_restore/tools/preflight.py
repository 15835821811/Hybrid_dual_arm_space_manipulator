"""No rollout: real model metadata, algebra, budgets, and filesystem checks."""
import io
import json
import platform
import shutil
import sys
import time
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ff_prior_horizon_restore.inputs import identity
from ff_prior_horizon_restore.protocol import P,OUT,PACKAGE
from ff_prior_horizon_restore.job import assert_single_worker
from ff_prior_pilot_train.storage import atomic_json

assert not OUT.exists(),'No prior run may be overwritten'
assert_single_worker()
stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('ff_prior_horizon_restore.test_workflow')
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
sentinel=ROOT/'artifacts/v7_1/horizon_fsprobe_001'
filesystem=[]
for relative in ['units/u00/restore/iter/35/jacobian.npz.4294967295.tmp',
    'units/u00/validated/grid_16384_geometry_scans.json.4294967295.tmp',
    'scenes/capture_5_obstacles/initial_audit.json.4294967295.tmp']:
    target=sentinel/relative;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(b'filesystem sentinel only');assert target.read_bytes()==b'filesystem sentinel only'
    assert len(str(OUT/relative))<260
    filesystem.append(dict(relative=relative,characters=len(str(OUT/relative)),write_read_ok=True))
ident=identity();free=shutil.disk_usage(ROOT).free
assert free>P['output_bytes_cap']+P['future_dynamic_output_bytes']+P['evidence_bytes_reserve']
record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
    errors=len(result.errors),failures=len(result.failures),log=stream.getvalue(),source_hashes=ident['source_hashes'],
    bound_inputs=len(ident['binding']['files']),timestamp=time.time(),python=sys.version,platform=platform.platform(),
    free_bytes=free,filesystem=filesystem,new_coupled_calls=0,new_native_steps=0,new_neural_samples=0,
    note='Actual model metadata compiled, no state propagation, geometry distance queries, neural rebuild or physical steps.',
    development_test_corrections=['Initial negative motion fixture used a legal 0.8rad move; replaced with alternating in-domain extremes.',
        'Frame label assertion assumed an index ignoring path segments; assert labels on all created geometries.'])
atomic_json(PACKAGE/'evidence/software_tests.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','log')},ensure_ascii=True,indent=2))
if not result.wasSuccessful():
    print(stream.getvalue());raise SystemExit(1)
