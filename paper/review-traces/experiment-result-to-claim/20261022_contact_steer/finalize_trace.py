"""Compose final verdict from already-completed review outputs only."""
import datetime,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
raw=read(HERE/'checker_result.json');supp=read(HERE/'supplement_result.json');diag=read(HERE/'nominal_arithmetic_diagnostic.json');inputs=read(HERE/'inputs.json');s=raw['stats']
coreprefix=str((ROOT/'artifacts/v7_1/contact_steer_001').resolve())+'\\'
core=[x for x in inputs['manifest'] if x['path'].startswith(coreprefix)]
corebytes=sum(x['bytes'] for x in core)
assert len(core)==2682 and corebytes==180418308,(len(core),corebytes)
assert len(raw['failed_checks'])==2 and diag['all_match_original_two_threads_exactly'] and supp['status']=='PASS'
evidence={'status':'PASS_WITH_DOCUMENTED_LIMITATIONS','raw_checks':raw['checks_total'],'raw_passed_directly':raw['checks_passed'],'targeted_checks_reconciled_exactly_under_producer_blas_threads':2,'unresolved_arithmetic_checks':0,'supplement_checks':len(supp['checks']),'supplement_passed':sum(x['passed'] for x in supp['checks']),'core_files':len(core),'core_bytes':corebytes,'original_raw_fail_output_preserved':True,'new_physics_queries':0,'new_geometric_queries':0,'optimizer_calls_by_reviewer':0,'resume_calls_by_reviewer':0,'old_reviews_content_read':False}
write(HERE/'evidence_verdict.json',evidence)
verdict={
 'experiment':'contact_steer_001','producer_commit':'73be52493b09c4ff27a34340efe36bc8a653b97d','review_role':'fresh independent paper architect/reviewer','claim_supported':'no','confidence':'high','confidence_scope':'high for this bounded negative primary claim and descriptive evidence; medium for component causality',
 'intended_claim':'Original five-obstacle full initial state to original dual world goals in 27 seconds, full323 MODEL_VALID, native67 execution, independent complete836 replay.',
 'claims':{
  'C1':{'claim_supported':'no','confidence':'high','reason':'0/3 full qualified; no node reaches Q<=1; native execution/replay never entered; three frozen execution opportunities blocked.'},
  'Caux':{'claim_supported':'partial','confidence':'medium','supported':'249 additional sampled/model-screened nodes, exact context deduplication without repeated executed action keys, and within-prefix best Q reductions 61.3683-76.4158%.','not_supported':'Continuous-time safety, component-level causal benefit of Jacobian/deduplication, Diffusion advantage, generalization, full-trajectory feasibility, actual execution.'}},
 'what_results_support':'In one development task, the bounded combined contact-aware continuation produced 249 retained sample-screened model nodes from three frozen historical prefixes and reduced their best world-pose residuals; all primary identities and saved arithmetic agree with documented limitations.',
 'what_results_dont_support':'No complete original-goal MODEL_VALID trajectory, native67 execution, full836 replay, dynamic handoff, certified continuous-time collision freedom, efficacy of the unrun full323 stage, component causality, neural advantage, or impossibility of the original task.',
 'missing_evidence':['Complete original-task full-route model certification under all unchanged gates.','Native67 execution from original initial state and independent entire836 replay including required hold.','All omitted near phase/pair identities and distances; 4900 omitted records have counts only.','A paper-to-claim audit.','Causal/component/generalization evidence if those broader claims are later intended; not required to close this negative engineering run.'],
 'suggested_claim_revision':'On one five-obstacle development task, extending three frozen historical model prefixes with the bounded contact-aware planner produced 249 additional sample-screened model nodes and reduced best normalized world-pose residual by 61.37-76.42%. None reached the original final goal tolerance or yielded a complete certified trajectory, so no native execution or replay was attempted.',
 'next_experiments_needed':[],
 'next_experiments_scope':'No further experiment is authorized/recommended to launch now. Close the bounded repair and make an architecture-level decision; any future full-horizon constrained restoration test needs separate frozen scope/costs and unchanged final physical gates.',
 'paper_claim_audit':'unavailable','provisional':True,'provisional_reason':'provisional - no paper claim audit available','pending_external_review':False,
 'evidence_check':evidence,'metrics':s,
 'limitations':['All140 active sets hit96-cap; omitted pair identities/distances not saved.','Half-step is one selected coordinate per linearization with no preregistered pass threshold.','Only saved dynamics residuals can be checked without new physics; not independently recomputed.','u00/u02 hit50-parent cap with11/83 eligible parents remaining; u01 exhausts40 available parents, max new time17s.','Exact dedup leaves near-identical branches; no component ablation.','Three selected prefixes are not independent tasks or an unbiased success-rate sample.','SciPy warning scope is proposal evaluation; returned deltas in bounds; internal overshoot magnitude/frequency not logged.','Generic sqp=0 counter is uninstrumented for420 recorded SLSQP solves.','Two nominal arithmetic discrepancies at24 BLAS threads reproduce exactly at producer2 threads.'],
 'routing_action':'pivot_architecture_and_narrow_claim','handoff_blocked':True,'full_engineering_goal_achieved':False,
 'supplement':'Single final supplement PASS; executor acknowledged narrow editorial fixes in message. No second full raw check or resume call.'}
write(HERE/'verdict.json',verdict)
routing={'experiment':'contact_steer_001','action':'pivot','route':'architecture_level_decision_and_negative_result_closeout','confirmed_main_claim':False,'claim_to_keep':'bounded descriptive sampled-model continuation only','paper_writing':'provisional_only_no_claim_audit','ablation_planner_triggered':False,'new_physics_authorized_by_review':False,'automatic_local_retry':False,'automatic_training':False,'automatic_TEST':False,'resume_or_old_review':False,'static_execution_gate':'BLOCKED_NO_COMPLETE_MODEL_VALID_ROUTE','dynamic_handoff':'BLOCKED_NO_ACTUAL_QUALIFIED_TERMINAL_STATE','required_next_action':'Executor integrates reviewer verdict and limitations into closeout/status. Any future global27s/323 constrained restoration route remains an untested architecture hypothesis needing separate scope and frozen budget.','do_not_infer':['Original task infeasible','More search necessarily fails','Unrun full323/native/replay stages failed','Diffusion or dedup caused improvement','Review or software audit PASS means physical engineering PASS'],'long_term_goal':'not achieved','pending_external_review':False,'paper_claim_audit':'unavailable'}
write(HERE/'routing.json',routing)
existing={x['path'] for x in inputs['manifest']}
for x in supp['inputs']:
    if x['path'] not in existing:inputs['manifest'].append(dict(x,scope='single final supplement snapshot'));existing.add(x['path'])
inputs['review_completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();inputs['evidence_summary']=evidence
inputs['executor_editorial_followup']='Executor reports applying wording qualifications and SQP-counter disclosure after the reviewed report snapshot; no re-read/re-audit was requested or performed.'
write(HERE/'inputs.json',inputs)
tracefiles=[]
for p in sorted(HERE.iterdir()):
    if p.is_file() and p.name!='trace_manifest.json':
        b=p.read_bytes();tracefiles.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
write(HERE/'trace_manifest.json',{'files':tracefiles,'all_writes_confined_to':str(HERE),'no_commit':True})
print(json.dumps({'primary':'no','auxiliary':'partial','confidence':'high bounded/medium attribution','evidence':evidence,'trace_files':len(tracefiles),'routing':'pivot_architecture_and_narrow_claim'},ensure_ascii=False,indent=2))
