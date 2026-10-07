from pathlib import Path
from datetime import datetime, timezone
import json,sys,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from v6_4.route_optimizer_protocol import ROOT,read,write,verify_frozen,TaskSpec,PreferenceSpec
from v6_4.route_candidate_evaluator import NominalCandidateEvaluator
from v6_4.continuous_route_optimizer import optimize
run=ROOT/'v6_4/output/continuous_route_optimizer_20261007_01'
tid=sys.argv[1]
p=verify_frozen(run)
frozen=next(t for t in p['tasks'] if t['task_id']==tid)
task=TaskSpec.from_dict(read(run/'frozen_tasks'/tid/'task.json'))
prefs=[PreferenceSpec(**v) for v in read(run/'frozen_tasks'/tid/'preferences.json')]
evaluator=NominalCandidateEvaluator(run,task,frozen)
receipt={'argv':sys.argv,'cwd':str(Path.cwd()),'started_utc':datetime.now(timezone.utc).isoformat(),'role':'independent task scheduling only; frozen API and shared per-task budget unchanged'}
try:
 result=optimize(task,prefs,evaluator.execution_identity,evaluator,run/'planning'/tid)
 receipt['stop_reason']=result['budget']['stop_reason']
 receipt['exit_code']=1 if receipt['stop_reason']=='TOOL_ERROR' else 0
except Exception:
 receipt['exit_code']=1; receipt['error']=traceback.format_exc()
finally:
 receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
 write(run/'scheduling_receipts'/(tid+'.json'),receipt)
sys.exit(receipt['exit_code'])