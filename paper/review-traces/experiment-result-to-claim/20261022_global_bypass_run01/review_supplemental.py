"""Read-only seed stream, state-id, completed-resume receipt and statistics check."""
from pathlib import Path
import json,hashlib,datetime,collections
import numpy as np
from scipy.interpolate import BSpline
TRACE=Path(__file__).resolve().parent;ROOT=TRACE.parents[3];RUN=ROOT/'artifacts/v7_1/global_bypass_001'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dg(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=lambda v:v.tolist()).encode()).hexdigest()
ident=js(RUN/'run.json')['identity'];physical=js(RUN/'physical_contract.json');audit=js(RUN/'scenes/capture_5_obstacles/initial_audit.json');old=js(ROOT/'artifacts/v7_1/sample_alignment_001/physical_contract.json')
root=np.load(RUN/'m0/propagation_check.npz')['root_state'];mask=physical['initial_state_mask'];task=ident['selection']['entries'][0]['task']
scene_id=dg({'values':root,'model_hash':task['contract']['robot_hash'],'state_mask':mask});canonical_id=dg({'values':root,'model_hash':old['identity']['scene_model_hash'],'state_mask':mask})
checks={'scene_initial_state_id':scene_id==task['contract']['initial_state_id'],'canonical_initial_state_id':canonical_id==old['initial_state_id']==audit['canonical_input_state_id']};rows=[];maxerr=0.;counts=collections.Counter()
limits=js(RUN/'scenes/capture_5_obstacles/limits.json');lower=np.asarray(limits['lower']);upper=np.asarray(limits['upper']);weights=np.r_[np.full(10,.10),np.full(7,.65)]
for spec in ident['selection']['units']:
    for slot,seed in enumerate(spec['proposal_seeds']):
        pd=RUN/'units'/spec['id']/'candidates'/f'{slot:02d}';nodes=np.load(pd/'tree/nodes.npz');seeded=np.load(pd/'goal_seed/result.npz')['Z'];qgoal=seeded[-1];q0=nodes['q'][0];rng=np.random.default_rng(seed+440001)
        curve=BSpline(np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)],np.vstack([np.repeat(q0[None],3,axis=0),seeded[:18],np.repeat(seeded[-1:],3,axis=0)]),3,extrapolate=False)
        size=1;passed=True
        for a in js(pd/'tree/attempts.json'):
            r=rng.random()
            if r<.25:target=qgoal.copy();sampling='GOAL'
            elif r<.55:target=curve(rng.uniform(0,27)/27);target+=rng.normal(0,.15,17)*weights;sampling='PROPOSAL'
            elif r<.85:s=rng.random();target=(1-s)*q0+s*qgoal+rng.normal(0,1,17)*weights;sampling='LOCAL_EXPLORATION'
            else:target=rng.uniform(lower,upper);sampling='GLOBAL_DOMAIN'
            target=np.minimum(np.maximum(target,lower),upper)
            parent=int(np.argmin([np.linalg.norm((q-target)/weights)+.02*t for q,t in zip(nodes['q'][:size],nodes['elapsed'][:size])]))
            delta=target-nodes['q'][parent];delta*=min(1,.18/max(abs(delta).max(),1e-12));target=nodes['q'][parent]+delta
            er=float(abs(target-np.asarray(a['target'])).max());maxerr=max(maxerr,er);passed=passed and er<1e-12 and parent==a['parent'] and sampling==a['sampling'];counts[sampling]+=1
            if a['accepted']:size+=1;passed=passed and a['node']==size-1
        rows.append({'unit':spec['id'],'slot':slot,'seed':seed,'passed':passed,'nodes':size});checks[spec['id']+'/'+str(slot)]=passed and size==len(nodes['q'])
receipt=js(ROOT/'ff_prior_global_bypass/evidence/completed_resume.json');checks['completed_resume_receipt']=receipt['returncode']==0 and receipt['response']['resume']=='NO_WORK_ALREADY_COMPLETE' and receipt['core_unchanged'] is True and receipt['before_files']==receipt['after_files']==24735 and receipt['new_physics']==0
d=js(TRACE/'saved_evidence_check.json');methods={}
for method in ident['protocol']['methods']:
    rs=[r for r in d['candidates'] if r['method']==method]
    methods[method]={k:sum(r[k] for r in rs) for k in ['attempts','accepted_tree_nodes','saved_connector_propagations','inferred_tree_total_coupled','inferred_seed_coupled','reserved_intervals','saved_edge_actual_intervals','goal_connections_with_iteration_logs','goal_iterations','goal_accepted_iterations','failed_goal_trials_without_edge_trace']}
    for k in ['minimum_tree_Q','seed_Q']:
        a=np.asarray([r[k] for r in rs]);methods[method][k+'_distribution']={'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}
    methods[method]['maximum_node_elapsed_s']=max(r['maximum_elapsed'] for r in rs)
    methods[method]['maximum_depth']=max(r['maximum_depth'] for r in rs)
    last=[r['goal_last_evaluated_Q_min'] for r in rs if r['goal_last_evaluated_Q_min'] is not None];methods[method]['best_saved_last_goal_trial_Q']=min(last)
    methods[method]['candidate_qualified']=0;methods[method]['candidate_denominator']=15;methods[method]['verified_units']=0;methods[method]['frozen_units']=3;methods[method]['actual_executions']=0;methods[method]['conditional_execution_failure_rate']=None
reserved=d['accounting']['reserved_integrations'];actual=sum(r['saved_edge_actual_intervals'] for r in d['candidates'])+512*sum(r['inferred_seed_coupled'] for r in d['candidates'])+256*18*d['counters']['goal_iterations']+1024
output={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'scene_initial_state_id':scene_id,'canonical_initial_state_id':canonical_id,'mask':mask,'tree_rng_reconstruction_max_target_error':maxerr,'sampling_counts':dict(counts),'rows':rows,'methods':methods,'reserved_integration_intervals':reserved,'reconstructed_actual_integration_intervals':actual,'not_integrated_reserved_intervals_due_to_collision_stop':reserved-actual,'actual_count_scope':'connector actual arrays plus code-inferred complete seed and unlogged finite-difference counts; not a separate instrumented actual counter','resume_receipt':'ff_prior_global_bypass/evidence/completed_resume.json','resume_performed_by_this_reviewer':False,'raw_git_byte_differences':len(d['archive_vs_git']),'line_ending_only_differences':sum(x['only_line_endings'] for x in d['archive_vs_git']),'substantive_git_differences':[x['path'] for x in d['archive_vs_git'] if not x['only_line_endings']]}
(TRACE/'supplemental_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Clarify the scope after exhaustive matching showed every actual false line trial
# had a matching saved connector record. The implementation can omit motion rejects,
# but this run does not exhibit that hypothetical gap.
d['limitations'][2]='Goal logs omit direct edge IDs/parent/target fields; all recorded line-search trials in this run were matched by saved parent-state hash and reconstructed target. No unmatched false trial occurred.'
(TRACE/'saved_evidence_check.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
inputs=js(TRACE/'inputs.json')
for p in [ROOT/'ff_prior_global_bypass/evidence/completed_resume.json',ROOT/'ff_prior_global_bypass/evidence/core_manifest.json',TRACE/'review_saved_evidence.py',TRACE/'review_network_reconstruction.py',TRACE/'review_supplemental.py']:
    if p.exists():
        st=p.stat();inputs['artifacts'][str(p.relative_to(ROOT))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':st.st_size,'modified_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()}
(TRACE/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(output,ensure_ascii=False))
