"""One authorized B.3.1 pure-reference test invocation and external ledger."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path("E:/v64b31work_20261007_01")
DESTINATION = Path("E:/v64b31_staging_20261007_01/reference_tests.json")
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
    def startTest(self, test):
        self.start = time.perf_counter()
        super().startTest(test)
    def stopTest(self, test):
        failed = any(case is test for case, _ in self.failures+self.errors)
        self.records.append({"test_id": test.id(), "status": "FAIL" if failed else "PASS",
            "wall_s": time.perf_counter()-self.start})
        super().stopTest(test)


if DESTINATION.exists():
    raise SystemExit("reference test ledger already exists; do not repeat the authorized invocation")
sources = [ROOT/"v6_4/task_anchored_reference.py", ROOT/"v6_4/tests/test_execution_aware_reference.py"]
before = {str(path.relative_to(ROOT)): sha(path) for path in sources}
stream = io.StringIO()
suite = unittest.defaultTestLoader.loadTestsFromName("v6_4.tests.test_execution_aware_reference")
result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Result).run(suite)
after = {str(path.relative_to(ROOT)): sha(path) for path in sources}
value = {"schema": "v64_b31_pure_reference_tests_v1",
    "status": "PASS" if result.wasSuccessful() and before == after else "FAIL",
    "tests_run": result.testsRun, "records": result.records,
    "errors": [{"test_id": test.id(), "traceback": trace} for test, trace in result.errors],
    "failures": [{"test_id": test.id(), "traceback": trace} for test, trace in result.failures],
    "source_sha256_before": before, "source_sha256_after": after,
    "source_unchanged_during_tests": before == after, "invocations": 1,
    "physics_steps": 0, "geometry_queries": 0, "DDIM_calls": 0, "optimizer_updates": 0,
    "scope": "necessary reference tests only; simulator allocation/forward/step/distance APIs forbidden by mocks; no actual controller run",
    "test_log": stream.getvalue()}
with DESTINATION.open("x", encoding="utf8", newline="\n") as handle:
    json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    handle.write("\n")
print(json.dumps(value, ensure_ascii=False, allow_nan=False))
raise SystemExit(0 if value["status"] == "PASS" else 1)
