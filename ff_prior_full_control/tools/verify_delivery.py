"""Final local delivery checks, without repeated experiment or resume."""
import os
import subprocess
import sys
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_full_control.protocol import ROOT,OUT,PACKAGE


def main():
    trace=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_full_control_run01'
    required=[PACKAGE/'docs/ANALYSIS.md',PACKAGE/'docs/CLAIM_VERDICT.md',PACKAGE/'docs/CLAIM_NOTES.md',
        ROOT/'DIFFUSION_GOAL_CURRENT.md',ROOT/'paper/EXPERIMENT_TRACKER_20261021_162903.md']
    required += [trace/name for name in ('prompt.md','response.md','verdict.json','handoff.md','inputs.json','checks.json','native_math_and_tails.json')]
    assert all(p.is_file() for p in required)
    assert read_json(trace/'checks.json')['status']=='PASS'
    assert read_json(trace/'native_math_and_tails.json')['status']=='PASS'
    v=read_json(trace/'verdict.json');assert v['claim_supported']=='partial' and v['C2_learning_relative_gain']['claim_supported']=='no'
    result={}
    for action in ('status','report'):
        r=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(PACKAGE/'job.ps1'),action],cwd=ROOT,capture_output=True,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
        assert r.returncode==0,(action,r.stderr)
        if action=='status':assert b'FINISHED' in r.stdout
        else:assert b'19' in r.stdout and b'0/96' in r.stdout and b'13/36' in r.stdout
        result[action]=dict(returncode=r.returncode,stdout_bytes=len(r.stdout))
    modified=read_json(PACKAGE/'evidence/preflight.json')['original_modified_files']
    unchanged={p:file_digest(ROOT/p)==h for p,h in modified.items()};assert all(unchanged.values())
    assert read_json(OUT/'status.json')['status']=='FINISHED'
    ledger=read_json(OUT/'ledger.json');assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    resume=read_json(PACKAGE/'evidence/completed/completed_resume.json');assert resume['core_files_unchanged']
    # One cheap post-closeout integrity comparison of the saved raw file manifest;
    # no resume, model inference or physics is called.
    current={p.relative_to(OUT).as_posix():file_digest(p) for p in OUT.rglob('*') if p.is_file()}
    assert current==resume['after']
    proof=dict(status='PASS',terminal_commands=result,raw_files=len(current),raw_files_unchanged=True,
        user_files_unchanged=unchanged,required_artifacts={p.relative_to(ROOT).as_posix():file_digest(p) for p in required},
        scope='Bounded execution/report delivery; not scientific gain or long-term goal achievement')
    atomic_json(PACKAGE/'evidence/completed/delivery_check.json',proof)
    print('PASS: status/report, raw manifest, user files, independent verdict and deliverables')


if __name__=='__main__':main()
