"""Check delivery and preserved evidence without experiments or repeat resume."""
import os
import subprocess
from ff_prior.storage import read_json, file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_control_transfer.protocol import ROOT, OUT, PACKAGE

trace=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01'
required=[PACKAGE/'docs'/n for n in ('ANALYSIS.md','CLAIM_VERDICT.md','CLAIM_NOTES.md')]
required += [ROOT/'DIFFUSION_GOAL_CURRENT.md', ROOT/'paper/EXPERIMENT_TRACKER_20261021_165617.md']
required += [trace/n for n in ('prompt.md','response.md','verdict.json','handoff.md','inputs.json','checks.json','native_math.json','regression_provenance.json')]
assert all(p.is_file() for p in required)
assert read_json(trace/'checks.json')['status']=='PASS'
assert read_json(trace/'native_math.json')['status']=='PASS'
v=read_json(trace/'verdict.json')
assert v['claim_supported']=='partial' and v['claims']['C2']['evaluation_status']=='not_evaluated'
commands={}
for action in ('status','report'):
    r=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(PACKAGE/'job.ps1'),action],cwd=ROOT,capture_output=True,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    assert r.returncode==0,(action,r.stderr)
    assert (b'FINISHED_WITH_BLOCKED_ITEMS' in r.stdout) if action=='status' else (b'6/16' in r.stdout and b'BLOCKED' in r.stdout)
    commands[action]={'returncode':r.returncode,'stdout_bytes':len(r.stdout)}
modified=read_json(PACKAGE/'evidence/preflight.json')['original_modified_files']
unchanged={p:file_digest(ROOT/p)==h for p,h in modified.items()}
assert all(unchanged.values())
assert read_json(OUT/'status.json')['status']=='FINISHED_WITH_BLOCKED_ITEMS'
ledger=read_json(OUT/'ledger.json')
assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
resume=read_json(PACKAGE/'evidence/completed/completed_resume.json')
current={p.relative_to(OUT).as_posix():file_digest(p) for p in OUT.rglob('*') if p.is_file()}
assert resume['core_files_unchanged'] and current==resume['after']
atomic_json(PACKAGE/'evidence/completed/delivery_check.json',dict(status='PASS',terminal_commands=commands,
    raw_files=len(current),raw_files_unchanged=True,user_files_unchanged=unchanged,
    required_artifacts={p.relative_to(ROOT).as_posix():file_digest(p) for p in required},
    scope='Bounded conditional experiment delivery, not long-term scientific completion'))
print('PASS: terminal commands, independent verdict, raw manifest, user preservation and reports')
