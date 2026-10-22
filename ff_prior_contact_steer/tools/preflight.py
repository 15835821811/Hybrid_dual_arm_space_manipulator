"""Record meaningful software tests and resource inventory before first physics."""
import contextlib
import io
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ff_prior_contact_steer.inputs import identity
from ff_prior_contact_steer.protocol import PACKAGE,OUT,P
from ff_prior_contact_steer.job import assert_single_worker
from ff_prior_contact_steer.filesystem import probe
from ff_prior_pilot_train.storage import atomic_json

assert not OUT.exists(),'independent output must not exist before first launch'
assert_single_worker()
filesystem=probe()
stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromName('ff_prior_contact_steer.test_workflow')
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
ident=identity()
free=shutil.disk_usage(ROOT).free
assert free>P['output_bytes_cap']+2*1024**3,'insufficient bounded output reserve'
record={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'log':stream.getvalue(),
        'source_hashes':ident['source_hashes'],'timestamp':time.time(),
        'python':sys.version,'platform':platform.platform(),'free_bytes':free,
        'no_heavy_worker_detected':True,'new_physics_calls':0,'new_training_updates':0,
        'scope':'software/resource preflight; not MODEL_VALID or execution proof'}
record['filesystem']=filesystem
atomic_json(PACKAGE/'evidence/software_tests.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','log')},ensure_ascii=True,indent=2))
if not result.wasSuccessful():raise SystemExit(1)
