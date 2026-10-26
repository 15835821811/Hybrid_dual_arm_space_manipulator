"""One terminal no-work resume; persist before inventory and capture raw bytes."""
from pathlib import Path
import hashlib,json,subprocess,sys,os,time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v7_1/guided_diffusion_chain_001'
OUT=ROOT/'ff_prior_guided_chain_closeout/evidence'
def manifest():
    return {p.relative_to(RUN).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(RUN.rglob('*')) if p.is_file()}
def write(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert json.loads((RUN/'status.json').read_text())['status']=='FINISHED_WITH_BLOCKED_ITEMS'
    before_path=OUT/'resume_before.json'
    assert not before_path.exists(),'ALREADY_ATTEMPTED_NO_RETRY'
    before=manifest();write(before_path,before)
    # Persist claim of invocation before subprocess creation; an interruption is not retried.
    write(OUT/'resume_invocation.json',{'started':time.time(),'argv':[sys.executable,'-m','ff_prior_guided_chain.job','resume'],'before_sha256':hashlib.sha256(before_path.read_bytes()).hexdigest()})
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    p=subprocess.run([sys.executable,'-m','ff_prior_guided_chain.job','resume'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (OUT/'resume_stdout.bin').write_bytes(p.stdout);(OUT/'resume_stderr.bin').write_bytes(p.stderr)
    after=manifest();write(OUT/'resume_after.json',after)
    try:answer=json.loads(p.stdout.decode('utf-8'))
    except Exception:answer={}
    result=dict(status='PASS' if p.returncode==0 and before==after and answer.get('resume')=='NO_WORK_ALREADY_COMPLETE' else 'FAIL_NO_RETRY',returncode=p.returncode,resume=answer.get('resume'),unchanged=before==after,core_files=len(before),core_bytes=sum(v['bytes'] for v in before.values()),stdout_bytes=len(p.stdout),stderr_bytes=len(p.stderr),stdout_sha256=hashlib.sha256(p.stdout).hexdigest(),stderr_sha256=hashlib.sha256(p.stderr).hexdigest(),completed_invocations=1,new_neural=0,new_coupled=0,new_native=0)
    write(OUT/'resume_result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
