"""Record one delivery-only command; never retry or alter scientific evidence."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cwd', type=Path, required=True)
    p.add_argument('--source', type=Path, action='append', default=[])
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('command required')
    a.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    sources = [Path(__file__).resolve(), *a.source]
    record = dict(schema='c3_delivery_command_v1', argv=command, cwd=str(a.cwd.resolve()),
                  role='delivery only; no scientific retry', automatic_retry=False,
                  started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  sources={str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                  environment_overrides={k: env[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                      'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUTF8']})
    (a.output/'started.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    start = time.perf_counter()
    print(f"Delivery command started: {a.output}", flush=True)
    try:
        with (a.output/'stdout.txt').open('wb') as out, (a.output/'stderr.txt').open('wb') as err:
            result = subprocess.run(command, cwd=a.cwd, env=env, stdout=out, stderr=err)
        record['exit_code'] = result.returncode
    except Exception as exc:
        record.update(exit_code=1, launch_error=repr(exc))
    record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  elapsed_wall_s=time.perf_counter()-start)
    record['logs'] = {n: dict(bytes=(a.output/n).stat().st_size,
                     sha256=hashlib.sha256((a.output/n).read_bytes()).hexdigest())
                     for n in ['stdout.txt', 'stderr.txt'] if (a.output/n).exists()}
    (a.output/'receipt.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record['exit_code']


if __name__ == '__main__':
    sys.exit(main())
