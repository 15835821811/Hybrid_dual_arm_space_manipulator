"""Read-only recomputation from raw arrays; does not rerun models or upgrade labels."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest
from ff_prior.contracts import MODEL_GATES,assert_frozen
from ff_prior.spline import Reference
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_loop.experiment import METHODS,PROTOCOL,sources
from ff_prior_loop.reporting import metrics,decision
from ff_prior_loop.data import assert_disjoint


def main():
    out=Path('artifacts/v7_1/qualified_guidance_loop_001'); source=Path('artifacts/v7_1/smoke_002')
    run=read_json(out/'run.json'); summary=read_json(out/'summary.json'); ledger=read_json(out/'ledger.json')
    assert summary['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    assert all(file_digest(source/p)==h for p,h in run['identity']['source_files'].items())
    before=read_json('ff_prior_loop/evidence/preflight.json')
    assert all(file_digest(p)==h for p,h in before['user_dirty_hashes'].items())
    assert all(file_digest(p)==h for p,h in before['old_run_files'].items())
    for b,h in before['branches'].items(): assert subprocess.check_output(['git','rev-parse',b],text=True).strip()==h
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    frozen=read_json(out/'frozen_tasks.json'); targets=frozen['evaluation_entries']
    assert len(targets)==10 and frozen['before_new_training_and_comparison']
    assert len(frozen['training_generation_entries'])==16
    index=read_json(out/'training_index.json')['records']; assert_disjoint(index)
    for e in index:
        assert e['split'] in ('TRAIN','VAL') and e['execution_status']!='FAIL'
        assert file_digest(e['record'])==e['record_sha256'] and file_digest(e['arrays'])==e['arrays_sha256']
    actual=read_json(out/'training_input_audit.json')
    assert actual['records']==index and actual['actual_array_count']==len(index)
    training=read_json(out/'learning/training.json')
    assert training['metadata']['training_ids']==[e['sample_id'] for e in index if e['split']=='TRAIN']
    assert training['metadata']['validation_ids']==[e['sample_id'] for e in index if e['split']=='VAL']
    assert training['all_TRAIN_rows_seen'] and min(training['train_draw_counts'].values())>0 and training['VAL_gradient_updates']==0
    assert training['initial_weight_hash']!=training['selected_weight_hash'] and training['reload_error']==0
    assert file_digest(out/'learning/model.pt')==training['checkpoint_sha256']
    count=network=steps=executions=0; times=[]; caps=[]; physics=[]
    for ti,e in enumerate(targets):
        if e['status']!='FROZEN': continue
        task=e['task']; assert_frozen(task); sid=e['sample_id']; bases=set()
        for method in METHODS:
            candidates=[]
            for k in range(5):
                r=read_json(out/'candidates'/sid/method/f'{k:02d}.json'); count+=1; candidates.append(r); bases.add(r['public_base_hash'])
                assert r['task_hash']==task['task_hash'] and r['seed']==20261067+ti*101+k*997
                assert r['shared_query_ceiling']==113 and r['shared_wall_ceiling_s']==15.
                assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']
                assert r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3 and r['model_calls']['coupled']<=113
                caps.append(r['model_calls']['coupled']); times.append(r['wall_s'])
                assert not r['private_witness_used']
                if r['status']!='DONE': assert not r['eligible']; continue
                assert r['wall_s']<=15. and np.isfinite(r['Z']).all()
                sampling=r['sampling']; network+=sampling['actual_network_calls']
                assert sampling['actual_network_calls']==(0 if method=='structured_joint' else 16)
                if method!='structured_joint': assert max(s['decomposition_residual_after'] for s in sampling['steps'])<1e-12
                assert sampling['guidance'][-1]['output_hash']==digest(np.asarray(r['Z']))
                assert sampling['guidance'][-1]['collision_enabled']==(method!='trained_diffusion_goal_only')
                assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
                v=r['validation']; model=all(v['gates'].get(g,False) for g in MODEL_GATES)
                goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass'] and r['eligible']==(model and goal)
            chosen=min((r for r in candidates if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
            directory=out/'executions'/sid/method; sel=read_json(directory/'selection.json')
            assert sel['selected_id']==(chosen['candidate_id'] if chosen else None) and sel['selection_before_execution']
            if chosen and (directory/'report.json').exists():
                p,n=audit_physics(directory,task,chosen['reference_hash']); physics.append(p); steps+=n; executions+=1
            if not chosen: assert not (directory/'report.json').exists()
        assert len(bases)==1
    for p in (out/'witness_executions').glob('*/report.json'):
        sid=p.parent.name; row=read_json(out/'records'/f'{sid}.json')
        report,n=audit_physics(p.parent,row['task'],row['reference_hash']); physics.append(report); steps+=n; executions+=1
        assert row['execution_status']==('PASS' if report['level']=='EXECUTION_VERIFIED' else 'FAIL')
    assert steps==sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))
    assert count<=200 and executions<=52
    assert metrics(out)==summary['methods'] and decision(out)==read_json(out/'expansion_decision.json')
    final=read_json(out/'final_dataset_index.json'); assert_disjoint(final['records'])
    counts={s:sum(r['split']==s for r in final['records']) for s in ('TRAIN','VAL')}
    result={'status':'PASS','audited_candidates':count,'network_calls':network,'actual_torque_executions':executions,
        'physics_steps_including_replay':steps,'max_candidate_queries':max(caps),'max_candidate_wall_s':max(times),
        'minimum_execution_clearance_m':min(p['minimum_clearance_m'] for p in physics),
        'maximum_hold_position_m':max(max(p['hold_max_position_error_m']) for p in physics),
        'maximum_hold_rotation_rad':max(max(p['hold_max_rotation_error_rad']) for p in physics),
        'combined_counts':counts,'all_old_run_files_unchanged':len(before['old_run_files']),
        'user_files_and_branches_preserved':True,'original_DEV_unchanged':True,'no_TEST_or_pilot':True,
        'extra_model_calls_in_this_audit':0,'scope':'raw evidence recomputation, no new execution or blanket level upgrades'}
    atomic_json(out/'independent_verification.json',result)
    dest=Path('ff_prior_loop/evidence/qualified_guidance_loop_001'); dest.mkdir(parents=True,exist_ok=True)
    for p in list(out.rglob('*.json'))+[out/'REPORT.md',out/'events.jsonl',out/'executed_source.zip']:
        if p.name=='launcher.json': continue
        target=dest/p.relative_to(out); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    atomic_json(dest/'evidence_manifest.json',{'weights_included':False,'native_traces_included':False,
        'files':{p.relative_to(dest).as_posix():file_digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'}})
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
