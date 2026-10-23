"""One completed no-work resume with pre-persisted full manifests and byte capture."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/v7_1/horizon_restore_001'
EVID=ROOT/'artifacts/review_manifests/horizon_restore_001'
RECEIPT=ROOT/'ff_prior_horizon_restore/evidence/completed_resume.json'
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def manifest():
    rows={}
    for p in sorted(RUN.rglob('*')):
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            rows[p.relative_to(RUN).as_posix()]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    return rows
def main():
    assert not RECEIPT.exists() and not (EVID/'invoked.json').exists(),'RESUME_ALREADY_INVOKED'
    status=json.loads((RUN/'status.json').read_text())
    assert status['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    before=manifest();write(EVID/'before.json',before)
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','ff_prior_horizon_restore.job','resume']
    write(EVID/'invoked.json',{'time':time.time(),'command':command,'before_manifest_persisted':True})
    p=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,
        env=dict(os.environ,PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    for name,value in [('stdout',p.stdout),('stderr',p.stderr)]:
        with (EVID/(name+'.bin')).open('wb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    after=manifest();write(EVID/'after.json',after)
    result=json.loads(p.stdout.decode('utf-8-sig'))
    evidence={'run':'artifacts/v7_1/horizon_restore_001','status':'PASS' if p.returncode==0 and before==after and result.get('resume')=='NO_WORK_ALREADY_COMPLETE' else 'FAIL',
        'command':command,'returncode':p.returncode,'response':result,'core_files':len(before),
        'core_bytes':sum(x['bytes'] for x in before.values()),'identical_before_after':before==after,
        'before_manifest_persisted_before_invocation':True,'new_native_steps':0,
        'before_manifest':'artifacts/review_manifests/horizon_restore_001/before.json',
        'after_manifest':'artifacts/review_manifests/horizon_restore_001/after.json',
        'stdout_bytes':len(p.stdout),'stderr_bytes':len(p.stderr),
        'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr).hexdigest(),
        'stdout_file':'artifacts/review_manifests/horizon_restore_001/stdout.bin','stderr_file':'artifacts/review_manifests/horizon_restore_001/stderr.bin',
        'completed_invocations':1}
    write(RECEIPT,evidence);print(json.dumps(evidence,ensure_ascii=True))
if __name__=='__main__':main()
