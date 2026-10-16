"""Read-only verification of frozen tasks, raw controls, gates and paired budgets."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,digest,file_digest
from ff_prior.contracts import MODEL_GATES,EXECUTION_GATES,assert_frozen
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import pose_errors
from ff_prior_complex.experiment import PROTOCOL,METHODS,sources,inputs
from ff_prior_complex.reporting import analyze


def audit_physics(directory,task,reference_hash):
    p=read_json(directory/'report.json')
    assert read_json(directory/'frozen_task.json')==task
    assert p['task_hash']==task['task_hash']
    assert p['metadata']['reference_hash']==reference_hash
    assert p['scene_model_hash']==task['contract']['robot_hash']
    assert p['initial_state_id']==task['contract']['initial_state_id']
    assert p['metadata']['robot_state_writes_during_step_loop']==0
    assert not p['metadata']['external_wrench_applied']
    with np.load(directory/'native_trace.npz') as a:
        assert a['qpos'].shape[1]==81 and a['ctrl'].shape[1]==67
        assert len(a['qpos'])==len(a['ctrl'])+1
        assert digest(a['ctrl'])==p['ctrl_hash']
        pos,rot=pose_errors(a['ee'],task['condition']['goals'])
        hold=a['time']-a['time'][0]>=task['condition']['duration']-1e-9
        assert np.max(abs(pos[hold].max(axis=0)-p['hold_max_position_error_m']))<1e-12
        assert np.max(abs(rot[hold].max(axis=0)-p['hold_max_rotation_error_rad']))<1e-12
        assert p['replay']['physics_steps']==len(a['ctrl'])
        steps=2*len(a['ctrl'])
        assert a['time'][-1]-a['time'][0]<=30.+1e-9
    if p['level']=='EXECUTION_VERIFIED':
        assert all(p['gates'][g] for g in EXECUTION_GATES) and p['replay']['passed']
        assert max(p['replay']['maximum_errors'].values())<=1e-10
        assert p['minimum_clearance_m']>=0 and max(p['hold_max_position_error_m'])<=.005
        assert max(p['hold_max_rotation_error_rad'])<=np.deg2rad(1)
    return p,steps


def main():
    out=Path('artifacts/v7_1/complex_train_val_001'); source=Path('artifacts/v7_1/smoke_002')
    run=read_json(out/'run.json'); ledger=read_json(out/'ledger.json'); summary=read_json(out/'summary.json')
    assert summary['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    assert all(file_digest(source/p)==h for p,h in run['identity']['source_files'].items())
    assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    before=read_json('ff_prior_complex/evidence/preflight.json')
    assert all(file_digest(p)==h for p,h in before['user_dirty_file_sha256'].items())
    assert all(file_digest(p)==h for p,h in before['previous_run_files'].items())
    for branch,value in before['branches'].items():
        assert subprocess.check_output(['git','rev-parse',branch],text=True).strip()==value
    for directory,name in [('ff_prior/evidence/smoke_002','evidence_manifest.json'),
        ('ff_prior/evidence/diffusion_repair','manifest.json'),
        ('ff_prior_next/evidence/public_seed_residual_001','evidence_manifest.json'),
        ('ff_prior_fixed/evidence/fixed_train_consistency_001','evidence_manifest.json'),
        ('ff_prior_goal/evidence/goal_projection_001','evidence_manifest.json'),
        ('ff_prior_cost/evidence/equal_budget_residual_001','evidence_manifest.json')]:
        folder=Path(directory)
        assert all(file_digest(folder/p)==h for p,h in read_json(folder/name)['files'].items())
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    _,_,dev,rows,_=inputs(source)
    assert digest(dev)==run['identity']['tasks_hash']
    by_id={r['sample_id']:(r,s) for r,s in rows}
    frozen=read_json(out/'frozen_complex_tasks.json'); entries=frozen['entries']
    assert len(entries)==12 and digest(entries)==frozen['digest'] and frozen['before_any_comparison']
    events=[json.loads(v) for v in (out/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    freeze_time=next(e['timestamp'] for e in events if e['stage']=='all_tasks_frozen')
    assert all(e['timestamp']>=freeze_time for e in events if e['stage'] in ('paired_candidates','selected_torque_and_replay'))
    count=network=steps=executions=0; caps=[]; times=[]; physics=[]; witness_verified=0
    dataset=[]
    for ti,e in enumerate(entries):
        if e['status']!='FROZEN': continue
        sid=e['sample_id']; task=e['task']; assert_frozen(task)
        parent,split=by_id[e['parent_sample_id']]
        assert e['split']==split and e['family_id']==parent['family_id']
        assert e['transfer']['status']=='PASS' and all(e['transfer']['checks'].values())
        assert not e['method_results_used'] and not e['goal_relabelled']
        assert task['contract']['acceptance']==read_json(source/'benchmark_tasks.json')[0]['task']['contract']['acceptance']
        assert read_json(out/'tasks'/sid/'task.json')==task
        assert e['witness_model']['eligible']
        with np.load(out/'tasks'/sid/'witness.npz') as a:
            assert np.array_equal(a['Z'],e['Z'])
            assert Reference(a['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash==e['reference_hash']
        if e['kind']=='static_detour':
            screen=e['screen']; assert len(screen['attempts'])<=24 and screen['attempts'][-1]['passed']
            assert screen['attempts'][-1]['direct_clearance_m']<-.0001
            assert not screen['MODEL_VALID_claimed'] and not screen['method_results_used']
        directory=out/'witness_executions'/sid
        if (directory/'report.json').exists():
            p,n=audit_physics(directory,task,e['reference_hash']); physics.append(p); steps+=n; executions+=1
        if read_json(directory/'outcome.json')['status']!='EXECUTION_VERIFIED': continue
        witness_verified+=1
        augmented=read_json(out/'records'/f'{sid}.json'); dataset.append(augmented)
        assert augmented['training_eligible']==(split=='TRAIN') and not augmented['checkpoint_used_this_new_row']
        assert augmented['task']==task and augmented['validation_level']=='EXECUTION_VERIFIED'
        assert augmented['family_id']==parent['family_id']
        for method in METHODS:
            candidates=[]
            for k in range(5):
                r=read_json(out/'candidates'/sid/method/f'{k:02d}.json'); candidates.append(r); count+=1
                assert r['task_hash']==task['task_hash'] and r['candidate_id']==k and r['method']==method
                assert r['seed']==20261047+ti*101+k*997
                assert not r['private_witness_fields_accessed']
                assert r['shared_wall_ceiling_s']==8. and r['shared_coupled_ceiling']==113
                caps.append(r['model_calls']['coupled']); times.append(r['wall_s'])
                assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']
                assert r['model_calls']['coupled']<=113 and r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3
                if r['status']!='DONE': assert not r['eligible']; continue
                assert r['wall_s']<=8.
                sampling=r['sampling']; network+=sampling['actual_network_calls']
                assert sampling['actual_network_calls']==(16 if method=='frozen_diffusion' else 0)
                if method=='frozen_diffusion': assert all(s['decomposition_residual_after']<1e-12 for s in sampling['steps'])
                assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
                v=r['validation']; model=all(v['gates'].get(g,False) for g in MODEL_GATES)
                goal=('frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005
                    and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1))
                assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass']
                assert r['eligible']==(model and goal)
            eligible=[r for r in candidates if r['eligible']]
            chosen=min(eligible,key=lambda r:(r['score'],r['candidate_id'])) if eligible else None
            directory=out/'executions'/sid/method; selection=read_json(directory/'selection.json')
            assert selection['task_hash']==task['task_hash'] and selection['selection_before_execution']
            assert selection['selected_id']==(chosen['candidate_id'] if chosen else None)
            if chosen and (directory/'report.json').exists():
                p,n=audit_physics(directory,task,chosen['reference_hash']); physics.append(p); steps+=n; executions+=1
            if not chosen: assert not (directory/'report.json').exists()
    assert count<=180 and executions<=48
    assert sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==steps
    expected=analyze(out)
    for k,v in expected.items(): assert summary[k]==v,k
    trainfamilies={r['family_id'] for r in dataset if r['split']=='TRAIN'}
    valfamilies={r['family_id'] for r in dataset if r['split']=='VAL'}
    assert not trainfamilies & valfamilies
    atomic_json(out/'dataset.json',{'schema':'complex_augmentation_1','data_root':str(out.resolve()),
        'records':dataset,'TRAIN':[r['sample_id'] for r in dataset if r['split']=='TRAIN'],
        'VAL':[r['sample_id'] for r in dataset if r['split']=='VAL'],'training_performed':False,
        'old_dataset_modified':False,'TEST':[]})
    # A concrete combined index references old files, without copying or upgrading their levels.
    fixed=Path('artifacts/v7_1/fixed_train_consistency_001')
    prior_ids=read_json(fixed/'training_dataset.json')
    combined=[]
    for split in ('TRAIN','VAL'):
        for sid in prior_ids[split]:
            root=fixed if sid.startswith('trainfix_') else source
            record=root/('corrections/records' if sid.startswith('trainfix_') else 'records')/f'{sid}.json'
            row=read_json(record); arrays=root/row['arrays']
            combined.append({'sample_id':sid,'split':split,'family_id':row['family_id'],
                'record':str(record),'arrays':str(arrays),'record_sha256':file_digest(record),
                'arrays_sha256':file_digest(arrays),'validation_level':row['validation_level'],
                'execution_status':row['execution_status'],'new_in_this_run':False})
    for row in dataset:
        record=out/'records'/f"{row['sample_id']}.json"; arrays=out/row['arrays']
        combined.append({'sample_id':row['sample_id'],'split':row['split'],'family_id':row['family_id'],
            'record':str(record),'arrays':str(arrays),'record_sha256':file_digest(record),
            'arrays_sha256':file_digest(arrays),'validation_level':row['validation_level'],
            'execution_status':row['execution_status'],'new_in_this_run':True})
    assert not {r['family_id'] for r in combined if r['split']=='TRAIN'} & {r['family_id'] for r in combined if r['split']=='VAL'}
    atomic_json(out/'combined_train_val_index.json',{'schema':'referenced_train_val_augmentation_1',
        'path_base':'repository_root','records':combined,
        'counts':{s:sum(r['split']==s for r in combined) for s in ('TRAIN','VAL')},
        'new_counts':{s:sum(r['split']==s and r['new_in_this_run'] for r in combined) for s in ('TRAIN','VAL')},
        'old_labels_not_upgraded':True,'network_retrained':False,
        'scene_conditioning_note':'new static scenes have distinct model/state identities; honor each record scene and transfer audit'} )
    result={'status':'PASS','run_status':summary['status'],'audited_candidates':count,
        'formal_network_calls':network,'real_torque_executions_including_witnesses':executions,
        'witnesses_execution_verified':witness_verified,'physics_steps':steps,
        'maximum_candidate_queries':max(caps) if caps else 0,'maximum_candidate_wall_s':max(times) if times else 0,
        'minimum_execution_clearance_m':min(p['minimum_clearance_m'] for p in physics) if physics else None,
        'maximum_hold_position_m':max(max(p['hold_max_position_error_m']) for p in physics) if physics else None,
        'maximum_hold_rotation_rad':max(max(p['hold_max_rotation_error_rad']) for p in physics) if physics else None,
        'all_old_runs_and_user_files_preserved':True,'original_DEV_preserved_and_not_rerun':True,
        'all_complex_tasks_frozen_before_comparison':True,'physical_gates_unchanged':True,
        'TRAIN_VAL_family_lineage_disjoint':True,'no_private_witness_in_candidate_API':True,
        'extra_model_physics_network_calls_in_audit':0,'pilot':'NOT_RUN','TEST':'NOT_RUN'}
    atomic_json(out/'independent_verification.json',result)
    dest=Path('ff_prior_complex/evidence/complex_train_val_001'); dest.mkdir(parents=True,exist_ok=True)
    files=[p for p in out.rglob('*.json') if p.name!='launcher.json']+[out/'REPORT.md',out/'executed_source.zip',out/'events.jsonl']
    for p in files:
        target=dest/p.relative_to(out); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    atomic_json(dest/'evidence_manifest.json',{'weights_included':False,'native_traces_included':False,
        'files':{p.relative_to(dest).as_posix():file_digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json'}})
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
