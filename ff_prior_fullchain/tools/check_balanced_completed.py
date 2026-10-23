"""Once-only completed resume observation; no interrupted work may be resumed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_fullchain.job import process_probe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['restore','tracking'])
    args=parser.parse_args()
    package=ROOT/('ff_prior_balanced_'+args.stage)
    run=ROOT/'artifacts/v7_1'/('balanced_'+args.stage+'_001')
    evidence=package/'evidence/completed_001'
    if evidence.exists():raise RuntimeError('COMPLETED_RESUME_ALREADY_RESERVED_NEVER_REPEAT')
    state=read_json(run/'status.json')
    assert state['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    probe=process_probe(state['pid'],state['started_at'])
    assert probe['alive'] is False or (probe['alive'] is True and probe['identity_matches'] is False),'OWNER_EXIT_NOT_CONFIRMED'
    evidence.mkdir(parents=True,exist_ok=False)
    def inventory():
        return {p.relative_to(run).as_posix():dict(bytes=p.stat().st_size,sha256=file_digest(p))
                for p in sorted(run.rglob('*')) if p.is_file()}
    before=inventory()
    atomic_json(evidence/'core_before.json',dict(files=before,bytes=sum(x['bytes'] for x in before.values()),
        written_before_subprocess=True,timestamp=time.time(),scope='all saved run files including status and logs'))
    (evidence/'.gitattributes').write_text('*.bin -text -diff\n',encoding='ascii')
    module='ff_prior_balanced_'+args.stage+'.job'
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-m',module,'resume'],cwd=ROOT,env=env,
                          stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (evidence/'stdout.bin').write_bytes(result.stdout)
    (evidence/'stderr.bin').write_bytes(result.stderr)
    after=inventory()
    atomic_json(evidence/'core_after.json',dict(files=after,bytes=sum(x['bytes'] for x in after.values())))
    import json
    try:response=json.loads(result.stdout.decode('utf-8'))
    except Exception:response=None
    passed=result.returncode==0 and before==after and response is not None and response.get('resume')=='NO_WORK_ALREADY_COMPLETE'
    receipt=dict(status='PASS' if passed else 'FAIL',module=module,returncode=result.returncode,
        response=response,core_files=len(before),core_bytes=sum(x['bytes'] for x in before.values()),
        complete_manifest_hash_size_unchanged=before==after,pre_manifest_persisted_first=True,
        stdout=dict(bytes=len(result.stdout),sha256=file_digest(evidence/'stdout.bin')),
        stderr=dict(bytes=len(result.stderr),sha256=file_digest(evidence/'stderr.bin')),
        exactly_one_invocation=True,new_physical_work_allowed=False)
    atomic_json(evidence/'receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=True,indent=2))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
