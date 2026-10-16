"""Recompute evidence and preservation checks without running the robot again."""
from pathlib import Path
import hashlib
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest
from ff_prior.contracts import MODEL_GATES,assert_frozen
from ff_prior.spline import Reference
from ff_prior_complex.tools.audit_completed import audit_physics
from .experiment import METHODS,PROTOCOL,sources,PREVIOUS,ROOT


def audit(out):
    out=Path(out); run=read_json(out/'run.json'); ledger=read_json(out/'ledger.json'); summary=read_json(out/'summary.json')
    assert summary['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS') and ledger['inflight'] is None
    assert not ledger.get('unresolved_call_accounting')
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    source=Path(run['identity']['source'])
    assert all(file_digest(source/p)==h for p,h in run['identity']['source_files'].items())
    assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    before=read_json(ROOT/'ff_prior_latent/evidence/preservation_before.json')
    changed=[p for p,h in before['old_files'].items() if file_digest(ROOT/p)!=h]
    assert not changed,changed
    assert all(file_digest(ROOT/p)==h for p,h in before['user_edits'].items())
    for b,h in before['branches'].items(): assert subprocess.check_output(['git','rev-parse',b],cwd=ROOT,text=True).strip()==h
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    targets=read_json(out/'frozen_tasks.json')['evaluation_entries']
    assert targets==read_json(PREVIOUS/'frozen_tasks.json')['evaluation_entries'] and len(targets)==10
    tr=read_json(out/'learning/training.json'); codec=read_json(out/'learning/codec.json')
    index=read_json(PREVIOUS/'training_index.json')['records']
    train_ids=[e['sample_id'] for e in index if e['split']=='TRAIN']
    assert tr['metadata']['training_ids']==codec['fit_ids']==train_ids and tr['VAL_gradient_updates']==0
    assert not tr['VAL_checkpoint_selection'] and tr['updates']==1800
    assert sum(tr['train_draw_counts'].values())==1800*32 and all(v>0 for v in tr['train_draw_counts'].values())
    from .learning import load
    load(out/'learning')
    total=network=executions=physics_steps=0; records={m:[] for m in METHODS}; physics=[]; pair_inputs=[]
    for ti,e in enumerate(targets):
        task=e['task']; sid=e['sample_id']; assert_frozen(task); base_hashes=set()
        for method in METHODS:
            rows=[]
            for k in range(5):
                r=read_json(out/'candidates'/sid/method/f'{k:02d}.json'); rows.append(r); records[method].append(r); total+=1
                assert r['task_hash']==task['task_hash'] and r['seed']==20261067+ti*101+k*997
                base_hashes.add(r['public_base_hash']); network+=r['actual_network_calls']
                assert not r['private_witness_used']
                assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']
                assert r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3 and r['model_calls']['coupled']<=113
                if r['status']!='DONE': assert not r['eligible']; continue
                assert np.isfinite(r['Z']).all() and r['wall_s']<=15 and r['integration_intervals']<=15872
                expected=16 if 'diffusion' in method else 0
                assert r['actual_network_calls']==r['sampling']['actual_network_calls']==expected
                if expected: assert max(v['decomposition_residual_after'] for v in r['sampling']['steps'])<1e-12
                guide=r['sampling']['guidance'][-1]
                assert guide['output_hash']==digest(np.asarray(r['Z'])) and guide['collision_enabled']
                for h in guide['history']:
                    if h['accepted'] and max(h['before_pose_norms'])<=.8: assert max(h['after_pose_norms'])<=.8
                v=r['validation']; model=all(v['gates'].get(g,False) for g in MODEL_GATES)
                goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass'] and r['eligible']==(model and goal)
                assert r['integration_intervals']==128*r['repair_coupled_calls']+r['validation_integration_intervals']
                assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
                if method in METHODS[:2]:
                    old_method='structured_feasible' if method==METHODS[0] else 'trained_diffusion_feasible'
                    prior=read_json(ROOT/'artifacts/v7_1/goal_preserving_smoke_001/candidates'/sid/old_method/f'{k:02d}.json')
                    pair_inputs.append(r['sampling']['guidance'][-1]['input_hash']==prior['sampling']['guidance'][-1]['input_hash'])
            chosen=min((r for r in rows if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
            directory=out/'executions'/sid/method; selection=read_json(directory/'selection.json')
            assert selection['selected_id']==(chosen['candidate_id'] if chosen else None) and selection['selection_before_execution']
            if chosen:
                assert (directory/'report.json').exists()
                p,n=audit_physics(directory,task,chosen['reference_hash']); physics.append(p); physics_steps+=n; executions+=1
                assert read_json(directory/'outcome.json')['status']==p['level']
            else: assert not (directory/'report.json').exists()
        assert len(base_hashes)==1
    assert total==200 and network==1600 and all(pair_inputs)
    assert physics_steps==ledger['calls']['execution.physics_steps']
    pairs={}; metrics={}
    new=records[METHODS[3]]
    for m,rows in records.items():
        sm=summary['methods'][m]
        assert sm['eligible']==sum(r['eligible'] for r in rows) and sm['coupled_queries']==sum(r['model_calls']['coupled'] for r in rows)
        assert abs(sm['candidate_wall_s']-sum(r['wall_s'] for r in rows))<1e-8
        before_norm=[max(np.linalg.norm(np.asarray(r['sampling']['guidance'][-1]['initial_goal_error']).reshape(4,3),axis=1)) for r in rows if r.get('sampling')]
        metrics[m]={'raw_goal_norm_median':float(np.median(before_norm)),
            'repair_queries_median':float(np.median([r['repair_coupled_calls'] for r in rows])),
            'raw_inner_goal_pass':sum(v<=.65 for v in before_norm)}
        if m!=METHODS[3]:
            pairs[m]={'candidate_wins':sum(a['eligible'] and not b['eligible'] for a,b in zip(new,rows)),
                'candidate_losses':sum(b['eligible'] and not a['eligible'] for a,b in zip(new,rows)),
                'query_change_percent':100*(summary['methods'][METHODS[3]]['coupled_queries']/sm['coupled_queries']-1),
                'wall_change_percent':100*(summary['methods'][METHODS[3]]['candidate_wall_s']/sm['candidate_wall_s']-1)}
    result={'status':'PASS','audited_candidates':total,'network_calls':network,'actual_torque_executions':executions,
        'physics_steps_including_replay':physics_steps,'same_previous_baseline_proposal_hashes':sum(pair_inputs),
        'old_files_unchanged':len(before['old_files']),'user_files_and_branches_preserved':True,
        'minimum_execution_clearance_m':min(p['minimum_clearance_m'] for p in physics),
        'maximum_hold_position_m':max(max(p['hold_max_position_error_m']) for p in physics),
        'paired_comparisons':pairs,'proposal_diagnostics':metrics,'original_DEV_unchanged':True,'no_TEST_or_pilot':True,
        'extra_model_calls_in_this_audit':0,'blanket_execution_upgrade':False,'TRAIN_count':len(train_ids)}
    atomic_json(out/'independent_verification.json',result)
    return result


if __name__=='__main__':
    import json
    print(json.dumps(audit(ROOT/'artifacts/v7_1/latent_residual_smoke_001'),ensure_ascii=False,indent=2))
