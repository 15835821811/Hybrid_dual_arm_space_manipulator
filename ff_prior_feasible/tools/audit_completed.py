"""Independent evidence recomputation without executing or upgrading trajectories."""
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
from ff_prior_feasible.experiment import METHODS,PROTOCOL,sources,PREVIOUS


def main():
    out=Path('artifacts/v7_1/goal_preserving_smoke_001'); source=Path('artifacts/v7_1/smoke_002')
    run=read_json(out/'run.json'); summary=read_json(out/'summary.json'); ledger=read_json(out/'ledger.json')
    assert summary['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS') and ledger['inflight'] is None
    assert not ledger.get('unresolved_call_accounting')
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    assert all(file_digest(source/p)==h for p,h in run['identity']['source_files'].items())
    before=read_json('ff_prior_feasible/evidence/preservation_before.json')
    assert all(file_digest(p)==h for p,h in before['old_files'].items())
    assert all(file_digest(p)==h for p,h in before['user_edits'].items())
    for b,h in before['branches'].items(): assert subprocess.check_output(['git','rev-parse',b],text=True).strip()==h
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    targets=read_json(out/'frozen_tasks.json')['evaluation_entries']
    assert targets==read_json(PREVIOUS/'frozen_tasks.json')['evaluation_entries'] and len(targets)==10
    count=network=steps=executions=0; caps=[]; times=[]; intervals=[]; physics=[]; refinement=0; paired_inputs=0
    for ti,e in enumerate(targets):
        task=e['task']; assert_frozen(task); sid=e['sample_id']; bases=set()
        for method in METHODS:
            candidates=[]
            for k in range(5):
                r=read_json(out/'candidates'/sid/method/f'{k:02d}.json'); candidates.append(r); count+=1
                bases.add(r['public_base_hash']); assert r['task_hash']==task['task_hash'] and r['seed']==20261067+ti*101+k*997
                assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']
                assert r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3 and r['model_calls']['coupled']<=113
                assert not r['private_witness_used']; caps.append(r['model_calls']['coupled']); times.append(r['wall_s'])
                network+=r['actual_network_calls']
                if r['status']!='DONE': assert not r['eligible']; continue
                assert r['wall_s']<=15 and r['integration_intervals']<=15872 and np.isfinite(r['Z']).all()
                intervals.append(r['integration_intervals']); sampling=r['sampling']
                assert r['actual_network_calls']==sampling['actual_network_calls']==(0 if method=='structured_feasible' else 16)
                if method!='structured_feasible': assert max(s['decomposition_residual_after'] for s in sampling['steps'])<1e-12
                guide=sampling['guidance'][-1]
                assert guide['output_hash']==digest(np.asarray(r['Z']))
                assert guide['collision_enabled']==(method!='trained_diffusion_goal_only')
                for h in guide['history']:
                    if h['accepted'] and max(h['before_pose_norms'])<=.8: assert max(h['after_pose_norms'])<=.8
                assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
                v=r['validation']; model=all(v['gates'].get(g,False) for g in MODEL_GATES)
                goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass'] and r['eligible']==(model and goal)
                if r['validation_grids'][-1]['intervals']==1024: refinement+=1
                assert r['validation_integration_intervals']==256+sum(g['intervals'] for g in r['validation_grids'])
                assert r['integration_intervals']==128*r['repair_coupled_calls']+r['validation_integration_intervals']
                if any(not g['gates']['geometry'] for g in r['validation_grids']): assert not r['eligible']
            chosen=min((r for r in candidates if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
            directory=out/'executions'/sid/method; selection=read_json(directory/'selection.json')
            assert selection['selected_id']==(chosen['candidate_id'] if chosen else None) and selection['selection_before_execution']
            if chosen and (directory/'report.json').exists():
                p,n=audit_physics(directory,task,chosen['reference_hash']); physics.append(p); steps+=n; executions+=1
                assert read_json(directory/'outcome.json')['status']==p['level']
            if not chosen: assert not (directory/'report.json').exists()
        assert len(bases)==1
        for k in range(5):
            a=read_json(out/'candidates'/sid/METHODS[2]/f'{k:02d}.json')
            b=read_json(out/'candidates'/sid/METHODS[3]/f'{k:02d}.json')
            if a.get('sampling') and b.get('sampling'):
                assert a['sampling']['guidance'][-1]['input_hash']==b['sampling']['guidance'][-1]['input_hash']; paired_inputs+=1
    assert steps==ledger['calls'].get('execution.physics_steps',0) and count==200 and executions<=40
    for method in METHODS:
        rows=[read_json(p) for p in (out/'candidates').glob(f'*/{method}/*.json')]
        m=summary['methods'][method]
        assert sum(r['eligible'] for r in rows)==m['eligible']
        assert sum(r['model_calls']['coupled'] for r in rows)==m['coupled_queries']
        assert sum(r.get('integration_intervals') or 0 for r in rows)==m['integration_intervals']
    result={'status':'PASS','audited_candidates':count,'network_calls':network,'actual_torque_executions':executions,
        'physics_steps_including_replay':steps,'max_candidate_queries':max(caps),'max_candidate_wall_s':max(times),
        'max_candidate_integration_intervals':max(intervals),'adaptive_1024_candidates':refinement,
        'paired_guidance_input_hashes_verified':paired_inputs,
        'minimum_execution_clearance_m':min(p['minimum_clearance_m'] for p in physics),
        'maximum_hold_position_m':max(max(p['hold_max_position_error_m']) for p in physics),
        'maximum_hold_rotation_rad':max(max(p['hold_max_rotation_error_rad']) for p in physics),
        'old_files_unchanged':len(before['old_files']),'user_files_and_branches_preserved':True,
        'same_frozen_VAL_tasks_and_seeds':True,'original_DEV_unchanged':True,'no_TEST_or_pilot':True,
        'extra_model_calls_in_this_audit':0,'scope':'raw evidence audit; no new execution or blanket level upgrades'}
    atomic_json(out/'independent_verification.json',result)
    dest=Path('ff_prior_feasible/evidence/goal_preserving_smoke_001'); dest.mkdir(parents=True,exist_ok=True)
    for p in list(out.rglob('*.json'))+[out/'REPORT.md',out/'events.jsonl',out/'executed_source.zip']:
        if p.name=='launcher.json': continue
        target=dest/p.relative_to(out); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    atomic_json(dest/'evidence_manifest.json',{'weights_included':False,'native_traces_included':False,
        'files':{p.relative_to(dest).as_posix():file_digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'}})
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
