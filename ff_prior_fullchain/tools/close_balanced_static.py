"""Close the reviewed new static run and route its authorized dynamic binding."""
import subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_pilot_train.storage import atomic_json

run=ROOT/'artifacts/v7_1/balanced_restore_001'
out=ROOT/'ff_prior_fullchain/evidence/balanced_restore_closeout.json'
assert not out.exists(),'CLOSE_ONLY_ONCE'
review='paper/review-traces/experiment-result-to-claim/20261023_balanced_static/verdict.json'
v=read_json(ROOT/review)
assert v['claims']['C1']['claim_supported']=='yes'
resume='ff_prior_balanced_restore/evidence/completed_001/receipt.json'
r=read_json(ROOT/resume)
assert r['status']=='PASS' and r['complete_manifest_hash_size_unchanged']
assert r['response']['resume']=='NO_WORK_ALREADY_COMPLETE'
report=read_json(run/'units/u00/execution/report.json')
assert report['level']=='EXECUTION_VERIFIED' and all(report['gates'].values())
with np.load(run/'units/u00/execution/native_trace.npz') as a:
    terminal=digest(a['integration'][-1]);assert a['integration'].shape==(13751,836)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
analysis='ff_prior_balanced_restore/docs/ANALYSIS.md'
committed=subprocess.check_output(['git','show',commit+':'+analysis],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
assert committed==(ROOT/analysis).read_text(encoding='utf-8').replace('\r\n','\n')
atomic_json(out,dict(closed=True,run=run.relative_to(ROOT).as_posix(),commit=commit,analysis=analysis,
    producer_git_head='5a21e08868cbff6a90d93e8225a43b77e8302bcd',source_files=214,bound_inputs=246,
    source_identity='Git base plus214 actual source hashes/executed_source.zip; pre-existing working-tree changes retained',
    independent_review=review,completed_resume=resume,completed_resume_calls=1,
    candidate_denominator=1,model_valid=1,executed=1,replay_verified=1,
    terminal_state_digest=terminal,coupled_queries=673,reserved_model_intervals=500016,
    native_steps_including_replay=27500,elapsed_s=773.7029999997467,
    core_files_unchanged=r['core_files'],core_bytes=r['core_bytes'],
    independent_checks=619,unresolved_checks=0,paper_writing_provisional=True,
    body_diagnostic_original_status='PARTIAL_9912_OF_13751',
    body_diagnostic_completion='ff_prior_balanced_delivery/static_body_completion/body_obstacle_metadata.json',
    readonly_delivery_receipt='ff_prior_balanced_delivery/readonly_completion_receipt.json',
    new_physics_during_closeout=False,long_term_goal_complete=False,
    next_stage='authorized same-scene actual-terminal dynamic; source already frozen',
    checksums={'review':file_digest(ROOT/review),'analysis':file_digest(ROOT/analysis)}))
p=ROOT/'ff_prior_fullchain/control.json';c=read_json(p)
c.update(active_worker=False,active_run_role='most_recent_closed_run',
    current_phase='D3_balanced_2R5C_static_closed',next_phase='D3_AUTHORIZED_BALANCED_TERMINAL_TRACKING_FREEZE',
    next_phase_action='Bind only verified same-scene actual terminal, run terminal software preflight, launch one already-frozen dynamic worker.')
c['phases'][-1]['status']='STATIC_CLOSED_EXECUTION_VERIFIED_DYNAMIC_AUTHORIZED'
atomic_json(p,c)
print(out)
