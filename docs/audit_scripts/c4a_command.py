"""Retain exact commands, output, exit status and wall time without shell parsing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    a.receipt.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    t = time.perf_counter()
    with (a.receipt/'stdout.txt').open('w', encoding='utf8') as out, (a.receipt/'stderr.txt').open('w', encoding='utf8') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err)
        code = child.wait()
    result = dict(argv=command, cwd=str(Path.cwd()), started_utc=started,
        ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter()-t,
        exit_code=code, pid=child.pid, status='PASS' if code == 0 else 'FAIL',
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        output_sha256={n:hashlib.sha256((a.receipt/n).read_bytes()).hexdigest() for n in ('stdout.txt','stderr.txt')})
    (a.receipt/'receipt.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
