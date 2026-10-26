from pathlib import Path
import json,hashlib,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'ff_prior_guided_chain_closeout/evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
budget=read(E/'final_budget.json');auth=read(ROOT/'ff_prior_guided_chain/evidence/authorization.json')
assert time.time()-auth['started_at']<=budget['charged_conservative']['wall_s'],'FINALIZATION_TIME_RESERVE_EXCEEDED'
folders=list(budget['measured_owned_folders_bytes'])
owned=sum(p.stat().st_size for f in folders for p in (ROOT/f).rglob('*') if p.is_file())
assert owned+1024**2<=budget['charged_conservative']['bytes'],'FINALIZATION_BYTE_RESERVE_EXCEEDED'
resume=read(E/'resume_result.json');assert resume['status']=='PASS'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
for n in ('resume_stdout.bin','resume_stderr.bin'):
    raw=subprocess.check_output(['git','show',head+':ff_prior_guided_chain_closeout/evidence/'+n],cwd=ROOT)
    assert raw==(E/n).read_bytes()
review=ROOT/'paper/review-traces/experiment-result-to-claim/20261026_guided_chain/verdict.json'
verdict=read(review)
value=dict(unit='guided_diffusion_chain_001',status='CLOSED_NEGATIVE_GUIDANCE_DERIVATIVE',report_commit=head,
 source_commit='4474adeb7225ff86412e1b1ea33bff70ab22d63f',freeze_commit='4e4fde56',
 identity_basis='218 actual source hashes plus executed zip, 89 bound input hashes and dependency versions; not a pure HEAD assertion',
 report='ff_prior_guided_chain/docs/ANALYSIS.md',report_sha256=sha(ROOT/'ff_prior_guided_chain/docs/ANALYSIS.md'),
 worker_terminal='FINISHED_WITH_BLOCKED_ITEMS',worker_s=88.093,neural_forwards=13,saved_ddim_steps=12,
 accepted_guidance=0,completed_samples=0,model_certifications=0,coupled=342,model_intervals=124528,
 native=0,replay=0,model_intervals_are_native=False,
 saved_check_report='ff_prior_guided_chain_closeout/evidence/saved_checks.json',saved_checks=1094,saved_mismatches=0,
 review=review.relative_to(ROOT).as_posix(),review_sha256=sha(review),claims=verdict['claims'],writing_status='provisional',
 resume=resume,resume_before_sha256=sha(E/'resume_before.json'),resume_after_sha256=sha(E/'resume_after.json'),
 committed_raw_resume_bytes_verified=True,final_budget='ff_prior_guided_chain_closeout/evidence/final_budget.json',
 final_budget_sha256=sha(E/'final_budget.json'),finalization_within_reserve=True,
 forbid_repeat_resume_review_or_run=True,automatic_new_unit=False,
 long_term_goal_complete=False,training=False,expansion=False,advantage_comparison=False,TEST=False,
 visualization='artifacts/v7_1/guided_diffusion_chain_001/delivery/index.html',visualization_kind='MODEL_ONLY_NO_NATIVE',
 next_phase='READ_ONLY_DERIVATIVE_REPAIR_DECISION_NO_AUTOMATIC_RUN')
path=ROOT/'ff_prior_fullchain/evidence/guided_chain_closeout.json'
assert not path.exists(),'CLOSEOUT_ALREADY_SEALED'
path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=value['status'],report_commit=head,within_finalization_reserve=True)))
