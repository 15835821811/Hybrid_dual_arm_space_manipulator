"""Read-only raw evidence audit; adds zero simulator or network evaluations."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json, atomic_json, digest, file_digest
from ff_prior.contracts import MODEL_GATES, EXECUTION_GATES, assert_frozen
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import pose_errors
from ff_prior_cost.experiment import inputs, sources, METHODS, PROTOCOL, train_targets
from ff_prior_goal.experiment import validation_targets
from ff_prior_cost.reporting import metrics, choose_on_val


def main():
    source=Path('artifacts/v7_1/smoke_002'); out=Path('artifacts/v7_1/equal_budget_residual_001')
    run=read_json(out/'run.json'); summary=read_json(out/'summary.json'); ledger=read_json(out/'ledger.json')
    assert summary['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    assert ledger['attempts']=={'diagnostic':12,'teacher':36,'training':1,'candidates':200,'execution':40}
    assert len(ledger['completed'])==289
    assert all(file_digest(source/p)==h for p,h in run['identity']['source_files'].items())
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    _,_,dev,rows,_=inputs(source); groups=read_json(out/'frozen_tasks.json')
    train=train_targets(rows)
    assert groups['DEV']==dev and groups['VAL']==validation_targets(rows)
    assert read_json(out/'train_targets.json')==train
    assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    before=read_json('ff_prior_cost/evidence/preflight.json')
    assert all(file_digest(p)==h for p,h in before['user_dirty_file_sha256'].items())
    assert all(file_digest(p)==h for p,h in before['previous_run_files'].items())
    for b in ('main','v6.4-a-diffusion-planning'):
        assert subprocess.check_output(['git','rev-parse',b],text=True).strip()==before['branches'][b]
    for directory,filename in [('ff_prior/evidence/smoke_002','evidence_manifest.json'),
            ('ff_prior/evidence/diffusion_repair','manifest.json'),
            ('ff_prior_next/evidence/public_seed_residual_001','evidence_manifest.json'),
            ('ff_prior_fixed/evidence/fixed_train_consistency_001','evidence_manifest.json'),
            ('ff_prior_goal/evidence/goal_projection_001','evidence_manifest.json')]:
        p=Path(directory); assert all(file_digest(p/k)==h for k,h in read_json(p/filename)['files'].items())
    for split,entries in (('TRAIN',train),('VAL',groups['VAL'])):
        for ti,e in enumerate(entries):
            r=read_json(out/'teachers'/split/f'{ti:03d}.json')
            if r['status']!='DONE': continue
            assert r['task']==e['task'] and r['parent_sample_id']==e['sample_id']
            assert r['family_id']==e['family_id'] and r['execution_evidence']=='NOT_EXECUTED'
            assert r['training_eligible']==(split=='TRAIN' and r['eligible'])
            assert all(r['validation']['gates'].get(k,False) for k in MODEL_GATES)==(r['validation']['validation_level']=='MODEL_VALID')
    trained=read_json(out/'learning/training.json')
    if trained['status']=='PASS':
        assert trained['checkpoint_sha256']==file_digest(out/'learning/model.pt')
        assert len(trained['training_ids'])==30 and len(trained['validation_ids'])==6
        assert set(trained['training_task_hashes'])=={e['task']['task_hash'] for e in train}
        assert not set(trained['training_task_hashes']) & {e['task']['task_hash'] for e in dev+groups['VAL']}
        assert trained['reload_error']==0 and trained['sanity']['status']=='PASS'
    freeze=read_json(out/'pre_DEV_freeze.json')
    assert not freeze['DEV_results_used'] and freeze['selected_method']==choose_on_val(out)
    import json
    events=[json.loads(line) for line in (out/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    frozen_time=next(e['timestamp'] for e in events if e['stage']=='protocol_frozen_before_DEV')
    assert all(e['timestamp']>=frozen_time for e in events if e.get('split')=='DEV')
    network=0; executions=0; steps=0; clearance=[]; hold_pos=[]; hold_rot=[]; caps=[]
    for split,entries in groups.items():
        for method in METHODS:
            for ti,e in enumerate(entries):
                task=e['task']; assert_frozen(task); candidates=[]
                for k in range(5):
                    r=read_json(out/'candidates'/split/method/f'{ti:03d}_{k:02d}.json'); candidates.append(r)
                    assert r['task_hash']==task['task_hash'] and r['candidate_id']==k and r['method']==method
                    if r['status']!='DONE':
                        assert not r['eligible']; continue
                    assert r['seed']==(20261017 if split=='DEV' else 20261023)+ti*101+k*997
                    assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
                    count=r['model_calls']['coupled']; caps.append(count)
                    assert count==r['repair_coupled_calls']+r['validation_coupled_calls']
                    assert count<=113 and r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3
                    sampling=r['sampling']; network+=sampling['actual_network_calls']
                    assert sampling['actual_network_calls']==(0 if method==METHODS[0] else 16)
                    active=[g for g in sampling['guidance'] if g.get('active')]
                    assert [g['iteration'] for g in active]==([14,15] if method=='frozen_diffusion_two' else [15])
                    assert sum(g['coupled_calls'] for g in active)==r['repair_coupled_calls']
                    if method!=METHODS[0]:
                        assert all(v['decomposition_residual_after']<1e-12 for v in sampling['steps'])
                    v=r['validation']; model=all(v['gates'].get(g,False) for g in MODEL_GATES)
                    goal=('frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005
                        and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1))
                    assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass']
                    assert r['eligible']==(model and goal) and not r['private_teacher_fields_accessed']
                eligible=[r for r in candidates if r['eligible']]
                chosen=min(eligible,key=lambda r:(r['score'],r['candidate_id'])) if eligible else None
                directory=out/'executions'/split/method/f'{ti:03d}'
                selection=read_json(directory/'selection.json')
                assert selection['selected_id']==(chosen['candidate_id'] if chosen else None)
                assert selection['selection_before_execution'] and selection['task_hash']==task['task_hash']
                if not chosen:
                    assert not (directory/'report.json').exists(); continue
                if not (directory/'report.json').exists():
                    assert ledger['completed'][f'execution_{split}_{ti}_{method}']['status']=='EXECUTION_FAILED'; continue
                report=read_json(directory/'report.json'); executions+=1
                assert read_json(directory/'frozen_task.json')==task
                assert report['metadata']['reference_hash']==chosen['reference_hash']
                assert report['scene_model_hash']==run['robot_hash'] and report['initial_state_id']==run['initial_state_id']
                assert report['metadata']['robot_state_writes_during_step_loop']==0
                assert not report['metadata']['external_wrench_applied']
                with np.load(directory/'native_trace.npz') as arrays:
                    assert arrays['qpos'].shape[1]==81 and arrays['ctrl'].shape[1]==67
                    assert len(arrays['qpos'])==len(arrays['ctrl'])+1
                    assert digest(arrays['ctrl'])==report['ctrl_hash']
                    pos,rot=pose_errors(arrays['ee'],task['condition']['goals'])
                    hold=arrays['time']-arrays['time'][0]>=task['condition']['duration']-1e-9
                    assert np.max(abs(pos[hold].max(axis=0)-report['hold_max_position_error_m']))<1e-12
                    assert np.max(abs(rot[hold].max(axis=0)-report['hold_max_rotation_error_rad']))<1e-12
                    assert report['replay']['physics_steps']==len(arrays['ctrl'])
                    steps+=2*len(arrays['ctrl'])
                if report['level']=='EXECUTION_VERIFIED':
                    assert all(report['gates'][g] for g in EXECUTION_GATES) and report['replay']['passed']
                    assert max(report['replay']['maximum_errors'].values())<=1e-10
                clearance.append(report['minimum_clearance_m'])
                hold_pos.append(max(report['hold_max_position_error_m'])); hold_rot.append(max(report['hold_max_rotation_error_rad']))
            assert summary['methods'][split][method]==metrics(out,split,method,len(entries))
    full=read_json(out/'verification.json')['complete_model_calls']
    assert full['physics_steps']==steps
    for key in ('coupled','physics_steps','torque_rollout'):
        assert sum(v for k,v in ledger['calls'].items() if k.endswith('.'+key))==full[key]
    result={'status':'PASS','run_status':summary['status'],'formal_network_calls':network,
        'real_torque_executions':executions,'physics_steps':steps,'maximum_candidate_queries':max(caps),
        'minimum_execution_clearance_m':min(clearance) if clearance else None,
        'maximum_hold_position_m':max(hold_pos) if hold_pos else None,
        'maximum_hold_rotation_rad':max(hold_rot) if hold_rot else None,
        'all_old_runs_and_user_files_preserved':True,'same_frozen_DEV':True,'physical_gates_unchanged':True,
        'checkpoint_selection_VAL_only':True,'diagnostic_split_TRAIN_VAL_only':True,
        'extra_model_physics_network_calls_in_audit':0,'pilot':'NOT_RUN','TEST':'NOT_RUN'}
    atomic_json(out/'independent_verification.json',result)
    dest=Path('ff_prior_cost/evidence/equal_budget_residual_001'); dest.mkdir(parents=True,exist_ok=True)
    files=[p for p in out.rglob('*.json') if p.name!='launcher.json']+[out/'REPORT.md',out/'executed_source.zip',out/'events.jsonl']
    for p in files:
        target=dest/p.relative_to(out); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    atomic_json(dest/'evidence_manifest.json',{'weights_included':False,'native_traces_included':False,
        'files':{p.relative_to(dest).as_posix():file_digest(p) for p in sorted(dest.rglob('*'))
                 if p.is_file() and p.name!='evidence_manifest.json'}})
    print(result)


if __name__=='__main__': main()
