"""Finalize the same review from existing results and sealed scalar metadata only."""
from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent;ROOT=O.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
R=read(O/'audit_results.json');I=read(O/'inputs.json')
p=ROOT/'artifacts/v7_1/balanced_brake_001/execution/result.json';old=read(p)
meta={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'read_scope':'sealed scalar report metadata only; no old raw-array reanalysis'}
I['files'][str(p.relative_to(ROOT))]=meta
ctx={'source':str(p.relative_to(ROOT)),'source_identity':meta,'predecessor_report_status':old['status'],'predecessor_report_momentum_peak':old['momentum_peak'],
 'predecessor_report_threshold':old['momentum_threshold'],'new_run_threshold':R['momentum']['threshold'],
 'old_over_new_threshold_ratio':[a/b for a,b in zip(old['momentum_threshold'],R['momentum']['threshold'])],
 'inferred_predecessor_exchange_peak':[(x-1e-5)/.002 for x in old['momentum_threshold']],
 'interpretation':'Initial inherited momentum violates both new-run thresholds before any burn input. Same formula, smaller current-run exchange scale; small new drift does not repair or excuse formal gate failure. Old scalar report is provenance, not independently reaudited old raw evidence.'}
write(O/'momentum_context.json',ctx);write(O/'inputs.json',I)
V={'audit_skill':'experiment-result-to-claim','review_role':'fresh independent paper architect/reviewer','review_trace':'20261023_satellite_origin',
 'run':'moving_satellite_001','claim_supported':'partial','confidence':'high','C1':{'claim_supported':'no','confidence':'high','claim':'Complete all-gate moving-satellite engineering continuation','reason':'Unchanged original robot momentum gate fails at the initial state and every subsequent saved state.'},
 'Caux':{'claim_supported':'yes','confidence':'high','claim':'Bounded disclosed physical-force satellite origin, saved tracking/replay/body evidence, with explicit failed robot momentum qualification'},
 'what_results_support':['One 29 s APP_DEV continuation, 14500 execution and 14500 producer replay steps, full 14501x836 saved replay equality.','Satellite COM world+Z pulse: 1000 burn inputs, 0.04 N s impulse; final 0.002 m/s and 0.056002 m displacement; 27 s engine-off velocity invariant.','All other recorded original and satellite-ledger gates reproduce as passing; whole-body/body and active/omitted pair records reconciled.','Momentum failure already inherited at state zero under this low-motion run exchange-scaled threshold; tiny new drift is not the dominant residual.'],
 'what_results_dont_support':['All-gate engineering success or long-term goal completion.','Relaxing the gate, subtracting initial momentum, canceling robot against satellite momentum, or reclassifying this run as passing.','Hardware propulsion/fuel exhaust, continuous-time collision proof, Diffusion advantage or learned dynamic control, generalization or TEST.','Causal obstacle correction: no binding/violated saved row, geometry projection, or SLSQP solve.'],
 'missing_evidence':['A continuation satisfying the unchanged robot momentum requirement.','Ultimate historical mechanism of inherited residual not diagnosed; old raw runs remain sealed.','Whole-paper claim audit unavailable.'],
 'suggested_claim_revision':'A single disclosed-force origin and engine-off tracking continuation has consistent saved physical accounting and exact saved full-state replay, but fails the original robot momentum gate from its inherited initial state.',
 'next_experiments_needed':[],'future_work_constraint':'No further experiment authorized by this review; any later repair needs a separately frozen design and cannot alter this run or its gates.',
 'routing_action':'narrow_claim_and_close_failed_C1_with_supported_Caux','pending_external_review':False,
 'paper_claim_audit':'unavailable','writing_status':'provisional — no paper claim audit available',
 'failed_gates':R['failed_gates'],'producer_gate_disagreements':R['gate_disagreements'],'unresolved_arithmetic_discrepancies':[],
 'grouped_checks_passed':len(R['checks']),'grouped_checks_total':len(R['checks']),
 'limits':['Nearest-distance engine not re-queried; saved distances, indices, connectors, policy, static sphere bounds/body positions and displacement reconstructed.','No new physical integration/replay; saved producer replay compared arithmetically.','No old raw-array analysis; predecessor scalar report thresholds only.'],
 'source_identity':R['identity'],'new_review_physics_steps':0,'new_review_replay_steps':0,'completed_resume_calls_by_reviewer':0,
 'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write(O/'verdict.json',V)
f=ROOT/'findings.md';entry='\n\n## 2026-10-23 moving_satellite_001 — fresh result-to-claim review\n\n'+(O/'findings.md').read_text(encoding='utf-8').split('\n',2)[2]+'\nFull trace: paper/review-traces/experiment-result-to-claim/20261023_satellite_origin.\n'
with f.open('a',encoding='utf-8') as h:h.write(entry)
now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime(2026,10,23,12,9,14,tzinfo=datetime.timezone.utc)
write(O/'review_budget.json',{'review_start_utc':start.isoformat(),'review_finished_utc':now.isoformat(),'elapsed_review_wall_s':(now-start).total_seconds(),'wall_cap_s':600,'raw_audit_execution_s':R['elapsed_s'],'complete_raw_audit_invocations':1,'new_physics_steps':0,'new_replay_steps':0,'new_resume_calls':0,'output_bytes_before_this_receipt':sum(p.stat().st_size for p in O.iterdir() if p.is_file()),'output_cap_bytes':16*1024**2,'scope':'trace plus one findings.md append; outputs remain far below cap'})
print(json.dumps({'verdict':V['claim_supported'],'C1':V['C1']['claim_supported'],'Caux':V['Caux']['claim_supported'],'threshold_ratio':ctx['old_over_new_threshold_ratio'],'elapsed_review_wall_s':(now-start).total_seconds(),'trace':str(O)},ensure_ascii=False))
