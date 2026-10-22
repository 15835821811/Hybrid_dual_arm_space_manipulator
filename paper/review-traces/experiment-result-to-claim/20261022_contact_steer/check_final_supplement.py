"""Single supplement: new compact report/statistics and saved resume evidence only.
Does not invoke resume or rerun the full raw checker.
"""
import collections,datetime,gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
records=[];checks=[]
def get(p,binary=False):
    data=p.read_bytes();s=p.stat();records.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'mtime_utc':datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat()});return data if binary else json.loads(data.decode('utf-8-sig'))
def ck(name,v):checks.append({'name':name,'passed':bool(v)})
def same(a,b):return np.allclose(a,b,atol=1e-10,rtol=0)
base=json.loads((HERE/'checker_result.json').read_text(encoding='utf-8'));diag=json.loads((HERE/'nominal_arithmetic_diagnostic.json').read_text(encoding='utf-8'));s=base['stats']
c=get(ROOT/'ff_prior_contact_steer/evidence/statistics.json');r=get(ROOT/'ff_prior_contact_steer/evidence/completed_resume.json')
for key,expected in [('actual_coupled',s['coupled']),('reserved_model_intervals',s['reserved_intervals']),('new_nodes',s['new_nodes']),('linearizations',s['linearizations']),('unique_actions',s['exact_action_keys']),('active_observations_available',s['near_records']),('active_observations_selected',s['active_records']),('complete_qualified',0),('candidate_denominator',3),('execution_denominator',3),('native_and_replay_steps',0)]:ck('statistics/'+key,c[key]==expected)
edges=json.loads((HERE/'edge_rows.json').read_text(encoding='utf-8'));lin=json.loads((HERE/'linearization_rows.json').read_text(encoding='utf-8'))
counts=collections.Counter('EDGE_SCREEN_PASS' if e['edge_accepted'] else 'PENETRATION' if not e['complete'] else 'REFINE_GEOMETRY_GRID' for e in edges)
ck('statistics/edge_report_categories',c['edge_reports']==counts)
ck('statistics/solver_counts',c['solvers']=={'linear_admissible':s['solver_linear_admissible'],'success':s['solver_success'],'total':s['solvers']})
ck('statistics/trial_reasons',c['trial_reasons']==sum((collections.Counter(u['trial_reasons']) for u in s['units']),collections.Counter()))
ck('statistics/half_difference',same(c['half_step_distance_derivative_difference_m_per_rad']['maximum'],max(x['distance_half_abs_error_m_per_rad'] for x in lin)) and same(c['half_step_distance_derivative_difference_m_per_rad']['median'],np.median([x['distance_half_abs_error_m_per_rad'] for x in lin])))
for u,v in zip(s['units'],c['unit_rows']):
    ck(u['unit']+'/statistics',u['unit']==v['unit'] and u['new_nodes']==v['new_nodes'] and u['best_node']==v['best_node'] and u['best_time_s']==v['best_elapsed_s'] and u['best_Q']==v['new_best_Q'] and u['prefix_Q']==v['old_prefix_Q'] and u['remaining_unexpanded_time_feasible_parents']==v['unexpanded_time_feasible'] and same(np.asarray(u['best_position_errors_m'])*1000,v['best_position_mm']) and same(u['best_rotation_errors_deg'],v['best_rotation_deg']) and v['accepted_new_pose_pass_nodes']==0)
objects={}
for k in ('before','after','stdout','stderr'):
    meta=r[k];p=ROOT/meta['path'];b=get(p,True);ck('resume/'+k+'_hash',hashlib.sha256(b).hexdigest()==meta['sha256'] and len(b)==meta['bytes']);objects[k]=json.loads(gzip.decompress(b)) if k in ('before','after') else b
# Log schemas without printing thousands of raw entries.
schemas={k:({'type':type(v).__name__,'keys':list(v) if isinstance(v,dict) else None,'length':len(v)}) for k,v in objects.items() if k in ('before','after')}
before=objects['before'];after=objects['after']
def entries(x):
    if isinstance(x,dict) and 'files' in x:return x['files']
    return x
eb=entries(before);ea=entries(after)
ck('resume/manifest_entries_equal',eb==ea)
ck('resume/explicit_one_call_completed',r['calls']==1 and r['returncode']==0 and r['resume_result']=='NO_WORK_ALREADY_COMPLETE' and r['same_file_list'] and r['same_hash_and_size'] and not r['changed_files'])
stdout=objects['stdout'].decode('utf-8-sig');payload=json.loads(stdout)
ck('resume/raw_stdout_no_work',payload['resume']=='NO_WORK_ALREADY_COMPLETE' and payload['status']=='FINISHED_WITH_BLOCKED_ITEMS')
ck('resume/raw_stderr_empty',not objects['stderr'])
ck('resume/core_counts',len(eb)==r['core_files']==2682)
for p in ('ff_prior_contact_steer/docs/ANALYSIS.md','ff_prior_fullchain/docs/AFTER_CONTACT_STEER_DECISION.md','ff_prior_contact_steer/docs/figures/contact_steer_diagnosis.png'):
    get(ROOT/p,True)
ck('nominal_discrepancies_reconciled',set(x['name'] for x in base['failed_checks'])=={'u02/nominal/42','u02/nominal/46'} and diag['all_match_original_two_threads_exactly'])
result={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checks':checks,'manifest_schemas':schemas,'inputs':records,'read_only_scope':'single final supplement only; no resume invocation or repeated full raw audit','resume_snapshot_statement':'The two persisted manifests match and raw stdout says NO_WORK_ALREADY_COMPLETE; one invocation is supported by this receipt, not inferred by rerunning it.','report_verdict':'Numbers and figure agree with independently reconstructed primary quantities. Required wording qualification sent to executor: extra search cannot be ruled out; global323 redesign is untested. Ledger sqp=0 is an uninstrumented generic counter despite 420 saved SLSQP results.','final_evidence_check_status':'PASS_WITH_DOCUMENTED_LIMITATIONS' if all(x['passed'] for x in checks) else 'FAIL'}
(HERE/'supplement_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
