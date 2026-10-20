"""Independent bookkeeping closure; no training, neural calls, or physics."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[3]
OUT=ROOT/'artifacts/v7_1/endpoint_jacobian_bounded_001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e=read(P/'independent_evidence.json')
assert e['status']=='PASS' and e['unguided_raw_full16_rebuilt']==192
ledger=read(OUT/'ledger.json')
calls={
 'diagnostic':sum(x['calls'] for x in e['events'][:24]),
 'training_segment':sum(x['calls'] for x in e['events'][24:]),
 'candidate':sum(read(p)['model_calls']['coupled'] for p in OUT.glob('candidates/*/*/*.json')),
 'unguided':sum(read(p)['calls']['coupled'] for p in OUT.glob('unguided/*/*/*.json'))
}
for k,n in calls.items():assert n==ledger['calls'][k+'.coupled']
steps=sum(read(p)['metadata']['physics_steps']+read(p)['replay']['physics_steps'] for p in OUT.glob('executions/*/*/report.json'))
assert steps==ledger['calls']['execution.physics_steps']==378000
assert sum(calls.values())==50779
assert read(OUT/'query_budget.json')['reserved_coupled_upper']==57396
sel=read(ROOT/'ff_prior_endpoint_jacobian/evidence/selection.json')
per={str(t):sum(v['accepted'] for v,c in zip(e['events'][:24],sel['cases']) if c['t']==t) for t in [333,667,999]}
assert per=={'333':8,'667':8,'999':8}
resume_path=ROOT/'ff_prior_endpoint_jacobian/evidence/completed_resume.json'
resume=read(resume_path)
assert resume['calls']==1 and resume['old_run_resume_calls']==0 and resume['result']['resume']=='NO_WORK_ALREADY_COMPLETE'
assert resume['unchanged_file_count']==len(resume['before'])==1249 and resume['after_equals_before']
for p,h in resume['before'].items():assert sha(OUT/p)==h
ev=e['events'][24:];accepted=[v for v in ev if v['accepted']]
summary={'status':'PASS','coupled_by_stage':calls,'actual_coupled_total':sum(calls.values()),
 'reserved':57396,'native_plus_replay_steps':steps,'M1_acceptance_per_noise':per,
 'M1_matched_base_acceptable':sum(x['selected_base_admissible'] for x in e['events'][:24]),
 'M2_accepted':len(accepted),'M2_matched_base_acceptable':sum(x['selected_base_admissible'] for x in accepted),
 'M2_combined_only_acceptable':sum(not x['selected_base_admissible'] for x in accepted),
 'resume_performed_by_executor_only':True,'reviewer_resume_calls':0,'resume_snapshot_files_current_hash_match':1249,
 'completed_resume_sha256':sha(resume_path),'main_script_sha256':sha(P/'independent_review.py'),
 'main_result_sha256':sha(P/'independent_evidence.json'),'script_sha256':sha(Path(__file__))}
(P/'bookkeeping_evidence.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
i=read(P/'inputs.json')
i['reviewed_artifacts'][str(resume_path.relative_to(ROOT))]={'sha256':sha(resume_path),'bytes':resume_path.stat().st_size,'mtime_ns':resume_path.stat().st_mtime_ns}
(P/'inputs.json').write_text(json.dumps(i,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,indent=2))
