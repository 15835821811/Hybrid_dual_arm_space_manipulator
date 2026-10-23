"""Close the already reviewed failed prefix; never resume it again."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json,file_digest
from ff_prior_pilot_train.storage import atomic_json
out=ROOT/'ff_prior_fullchain/evidence/balanced_tracking_closeout.json'
assert not out.exists(),'CLOSE_ONLY_ONCE'
review='paper/review-traces/experiment-result-to-claim/20261023_balanced_dynamic/verdict.json'
v=read_json(ROOT/review)
assert v['claims']['C1']['claim_supported']=='no' and v['checks']['failed']==0
resume='ff_prior_balanced_tracking/evidence/completed_001/receipt.json';r=read_json(ROOT/resume)
assert r['status']=='PASS' and r['complete_manifest_hash_size_unchanged'] and r['exactly_one_invocation']
assert r['response']['resume']=='NO_WORK_ALREADY_COMPLETE'
analysis='ff_prior_balanced_tracking/docs/ANALYSIS.md'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert subprocess.check_output(['git','show',commit+':'+analysis],cwd=ROOT).decode('utf-8').replace('\r\n','\n')==(ROOT/analysis).read_text(encoding='utf-8').replace('\r\n','\n')
for name in ('stdout','stderr'):
    path=ROOT/'ff_prior_balanced_tracking/evidence/completed_001'/f'{name}.bin'
    assert file_digest(path)==r[name]['sha256']
    import hashlib
    assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT)).hexdigest()==r[name]['sha256']
atomic_json(out,dict(closed=True,run='artifacts/v7_1/balanced_tracking_001',commit=commit,
    analysis=analysis,independent_review=review,execution_status='EXECUTION_FAILED',
    completed_resume=resume,completed_resume_calls=1,core_files=r['core_files'],core_bytes=r['core_bytes'],
    native_steps=9830,replay_steps=9830,coupled_calls=0,elapsed_s=83.609,
    independent_checks=v['checks']['total'],unresolved_checks=0,
    actual_handoff=True,executed_prefix_replay_exact=True,full_dynamic_pass=False,
    waypoint_arrivals=5,waypoint_denominator=7,final_hold_reached=False,
    source_files=187,bound_inputs=256,failure_mechanism=v['failure_mechanism'],
    actual_prefix_media='ff_prior_balanced_delivery/final_001/index.html',
    paper_writing_provisional=True,new_physics_during_closeout=False,long_term_goal_complete=False))
print(out)
