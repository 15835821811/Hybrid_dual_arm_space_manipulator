"""Verify planning arithmetic, bound identities and current routing; zero simulation."""
from pathlib import Path
from datetime import datetime
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'ff_prior_guided_chain_plan'
STAMP='20261026_113732'
def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
b=read('ff_prior_guided_chain_plan/evidence/budget_derivation.json')
m=read('ff_prior_guided_chain_plan/evidence/model_reuse.json')
c=read('ff_prior_fullchain/control.json')
prev=read(f'ff_prior_guided_chain_plan/history/control_before_{STAMP}.json')
s=read('ff_prior_guided_chain_plan/evidence/implementation_spec.json')
checks={}
checks['eleven_metadata_checks']=len(m['checks'])==11 and all(m['checks'].values())
checks['checkpoint_and_bound_input_hashes_unchanged']=all(sha(ROOT/p)==v['sha256'] and (ROOT/p).stat().st_size==v['bytes'] for p,v in m['files'].items())
checks['coupled_derivation']=sum(b['coupled_derivation'].values())==b['coupled_enumerated']==39072
checks['coupled_cap']=b['coupled_enumerated']+b['coupled_headroom_for_same_schedules_only']==40000
checks['model_intervals']=sum(b['model_interval_derivation'].values())==b['model_interval_enumerated']<=b['reserved_model_interval_cap']
checks['interval_derivation']=b['model_interval_derivation']['columns']==38012*2048 and b['model_interval_derivation']['dense_centers_and_trials']==12*5*sum([512,1024,2048,4096,8192,16384])
checks['native_steps']=sum(p['execution']+p['replay'] for p in b['phases'])==83500
checks['timeline']=all(math.isclose((p['to_s']-p['from_s'])/.002,p['execution']) and p['execution']==p['replay'] for p in b['phases']) and all(a['to_s']==z['from_s'] for a,z in zip(b['phases'],b['phases'][1:]))
checks['network_calls']=sum(p['neural_calls'] for p in b['phases'])==b['neural_calls_cap']==48
checks['fd_dimensions']=4*sum(s['fd_dimensions'])==b['coupled_derivation']['fd_columns']
checks['wall_storage_allocations']=sum(b['wall_allocations_s'].values())==b['proposed_total_unit_caps']['wall_s'] and sum(b['storage_allocations_bytes'].values())==b['proposed_total_unit_caps']['bytes']
for k in ['coupled','native','wall_s','bytes']:
    checks[f'available_{k}']=math.isclose(b['conservative_before_this_plan'][k]-b['this_plan_reservation'][k],b['available_after_reservation'][k])
    checks[f'extra_{k}']=math.isclose(max(0,b['proposed_total_unit_caps'][k]-b['available_after_reservation'][k]),b['exact_additional_required'][k])
    checks[f'after_extension_{k}']=math.isclose(b['available_after_reservation'][k]+b['rounded_extension_proposal'][k]-b['proposed_total_unit_caps'][k],b['after_extension_and_full_reservation'][k]) and b['after_extension_and_full_reservation'][k]>=0
    for scope in ['global','D3']:
        checks[f'proposed_{scope}_{k}']=b[f'original_{scope}_caps'][k]+b['rounded_extension_proposal'][k]==b[f'proposed_{scope}_caps_if_approved'][k]
checks['old_closed_identity_preserved']=all(c[k]==prev[k] for k in ['active_run','active_run_role','active_module','active_source_commit','active_binding_commit','active_report','active_closeout_receipt','authorization_evidence','goal_status'])
checks['physics_and_automation_unchanged']=all(c[k] is False for k in ['allow_physics_relaxation','allow_final_TEST','allow_dataset_expansion','allow_parallel_physics_workers','active_worker']) and c['automation_id'] is None and c['automation_status']=='DELETED_BY_USER_REQUEST'
checks['not_implemented_or_frozen']=not s['production_implemented'] and not s['frozen'] and not s['worker_started'] and not (ROOT/'ff_prior_guided_chain').exists()
checks['correct_next_phase']=c['next_phase']=='D3_DIFFUSION_GUIDED_CHAIN_PLAN_READY_BUDGET_EXTENSION_REQUIRED'
checks['original_initial_identity']=sha(ROOT/s['original_initial'])==s['original_initial_sha256']
for name in ['EXPERIMENT_PLAN','EXPERIMENT_TRACKER']:
    paths=[ROOT/f'paper/{name}.md',ROOT/f'paper/{name}_{STAMP}.md',P/f'docs/{name}.md',P/f'docs/{name}_{STAMP}.md']
    checks[name+'_version_copies_equal']=len({sha(p) for p in paths})==1
    checks[name+'_prior_latest_saved']=(P/f'history/{name}_before_{STAMP}.md').exists()
checks['new_counts_zero']=all(m[k]==0 for k in ['new_model_forward_calls','new_samples','optimizer_updates','new_data','native_steps','coupled_calls','geometry_queries'])
report={'created_at':datetime.now().astimezone().isoformat(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'pass_count':sum(checks.values()),'total':len(checks),'scope':'Arithmetic, identity and planning-state verification only. No old raw reanalysis, network forward, dynamics, geometry, native or replay calls.'}
(P/'evidence/plan_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest={}
for p in P.rglob('*'):
    if p.is_file() and p.name!='file_manifest.json':
        manifest[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
for rel in ['paper/EXPERIMENT_PLAN.md','paper/EXPERIMENT_TRACKER.md','DIFFUSION_GOAL_CURRENT.md','ff_prior_fullchain/control.json','ff_prior_fullchain/docs/CONTINUOUS_DELIVERY_GOAL.md']:
    p=ROOT/rel;manifest[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
(P/'evidence/file_manifest.json').write_text(json.dumps({'created_at':report['created_at'],'files':manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
existing_manifest=(ROOT/'MANIFEST.md').read_text(encoding='utf-8')
with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
    for rel in ['ff_prior_guided_chain_plan/tools/check_plan.py','ff_prior_guided_chain_plan/evidence/assembly_receipt.json','ff_prior_guided_chain_plan/evidence/plan_checks.json','ff_prior_guided_chain_plan/evidence/file_manifest.json']:
        if f'| {STAMP} | /experiment-plan | {rel} |' not in existing_manifest:
            f.write(f'| {STAMP} | /experiment-plan | {rel} | engineering | 方案算术、文件身份和状态同步核对；0新物理 |\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'passed':sum(checks.values()),'failed':[k for k,v in checks.items() if not v]},ensure_ascii=False))
raise SystemExit(0 if all(checks.values()) else 1)

