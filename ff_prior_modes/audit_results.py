"""Read-only recomputation of labels, gradients, paired budgets and real traces."""
from pathlib import Path
import hashlib
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest,plain
from ff_prior.contracts import MODEL_GATES,assert_frozen
from ff_prior.spline import Reference
from ff_prior.initial_state import STATE
from ff_prior_complex.tools.audit_completed import audit_physics
from .experiment import ROOT,PREVIOUS,METHODS,PROTOCOL,sources
from . import data,learning


def audit(out):
    out=Path(out);run=read_json(out/'run.json');ledger=read_json(out/'ledger.json');summary=read_json(out/'summary.json')
    assert summary['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS') and ledger['inflight'] is None
    assert not ledger.get('unresolved_call_accounting')
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    source=Path(run['identity']['source']);assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    assert all(file_digest(source/p)==h for p,h in run['identity']['source_files'].items())
    before=read_json(ROOT/'ff_prior_modes/evidence/preservation_before.json')
    assert all(file_digest(ROOT/p)==h for p,h in before['old_files'].items())
    assert all(file_digest(ROOT/p)==h for p,h in before['user_edits'].items())
    for b,h in before['branches'].items():assert subprocess.check_output(['git','rev-parse',b],cwd=ROOT,text=True).strip()==h
    with zipfile.ZipFile(out/'executed_source.zip') as z:assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    frozen=read_json(out/'frozen_tasks.json');entries=frozen['generation_entries'];targets=frozen['evaluation_entries']
    assert digest(entries)==digest(data.select(read_json(PREVIOUS/'training_index.json')['records']))
    assert targets==read_json(PREVIOUS/'frozen_tasks.json')['evaluation_entries'] and len(targets)==10
    generated=admitted=witnesses=steps=0;physics=[];generation_queries=0
    for entry in entries:
        assert entry['split']=='TRAIN';assert_frozen(entry['task']);references=[np.asarray(entry['Z'])]
        for k in range(6):
            sid=entry['sample_id']+f'_v{k}';r=read_json(out/'generation'/f'{sid}.json');generated+=1
            assert r['task_hash']==entry['task']['task_hash'] and r['calls']['coupled']<=113
            generation_queries+=r['calls']['coupled']
            if not r['admitted']:continue
            assert r['physical_eligible'] and r['validation_result']['eligible'] and r['wall_s']<=15
            v=r['validation_result']['validation'];assert all(v['gates'].get(g,False) for g in MODEL_GATES)
            assert max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
            z=np.asarray(r['Z']);dist=min(float(np.sqrt(np.mean((z-q)**2))) for q in references)
            assert dist>=.005 and abs(dist-r['minimum_reference_rms_rad'])<1e-12;references.append(z);admitted+=1
            row=read_json(out/'records'/f'{sid}.json');assert row['task']==entry['task'] and not row['goal_relabelled']
            assert row['family_id']==entry['family_id'] and row['split']=='TRAIN'
            with np.load(out/row['arrays']) as a:assert np.array_equal(a['Z'],z)
            with np.load(out/row['complete_initial_state']) as a:
                assert digest({'values':a['values'],'model_hash':entry['task']['contract']['robot_hash'],'state_mask':int(STATE)})==entry['task']['contract']['initial_state_id']
            assert row['reference_hash']==Reference(z,entry['task']['condition']['initial_features'],entry['task']['condition']['duration']).reference_hash
            directory=out/'witness_executions'/sid
            if (directory/'report.json').exists():
                p,n=audit_physics(directory,entry['task'],row['reference_hash']);physics.append(p);steps+=n;witnesses+=1
                assert row['execution_status']==('PASS' if p['level']=='EXECUTION_VERIFIED' else 'FAIL')
            else:assert row['validation_level']=='MODEL_VALID' and row['execution_status']=='NOT_RUN'
    assert generated==48 and generation_queries==ledger['calls']['generation.coupled']
    index=read_json(out/'training_index.json')['records'];expected=data.combined(out);assert index==expected
    train_ids=[e['sample_id'] for e in index if e['split']=='TRAIN'];val_ids=[e['sample_id'] for e in index if e['split']=='VAL']
    assert len(val_ids)==48 and len(train_ids)==208+summary['dataset']['new_training_eligible']
    sens=read_json(out/'sensitivity_summary.json');sq=0;sr=[]
    for p in (out/'sensitivity').glob('*.json'):
        r=read_json(p);sq+=r['calls']['coupled'];sr.append(r)
        assert r['anchor_sample_id'] in train_ids
        if r['status']=='PASS':
            assert r['calls']['coupled']==39 and all(c['passed'] for c in r['half_step_checks'])
            assert r['mixed_direction_relative_error']<=.05
            assert file_digest(p.with_suffix('.npz'))==sens['jacobian_files'][r['task_hash']]
    assert sq==ledger['calls']['sensitivity.coupled'] and sq<=7020
    training={}
    for mode in ('v_only','goal_loss'):
        path=out/'learning'/mode/'training.json'
        if not path.exists():continue
        r=read_json(path);learning.load(path.parent);training[mode]=r
        assert r['metadata']['training_ids']==train_ids and r['metadata']['validation_ids']==val_ids
        assert r['updates']==1800 and r['VAL_gradient_updates']==0 and not r['VAL_checkpoint_selection']
        assert sum(r['train_draw_counts'].values())==57600 and all(v>0 for v in r['train_draw_counts'].values())
    matched=False
    if len(training)==2:
        a,b=training.values();matched=a['initial_weight_hash']==b['initial_weight_hash'] and a['rng_proofs']==b['rng_proofs'] and a['train_draw_counts']==b['train_draw_counts']
        assert matched and a['metadata']['data_hash']==b['metadata']['data_hash']
    total=network=executions=0;baselines=[];rows_by_method={m:[] for m in METHODS}
    for ti,entry in enumerate(targets):
        task=entry['task'];sid=entry['sample_id'];assert_frozen(task);bases=set()
        for method in METHODS:
            rows=[]
            for k in range(5):
                r=read_json(out/'candidates'/sid/method/f'{k:02d}.json');rows.append(r);rows_by_method[method].append(r);total+=1
                assert r['task_hash']==task['task_hash'] and r['seed']==20261067+ti*101+k*997
                bases.add(r['public_base_hash']);network+=r['actual_network_calls'];assert not r['private_witness_used']
                assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']<=113
                if r['status']!='DONE':assert not r['eligible'];continue
                assert r['wall_s']<=15 and r['integration_intervals']<=15872 and np.isfinite(r['Z']).all()
                assert r['actual_network_calls']==(0 if method in (METHODS[0],METHODS[2]) else 16)
                if r['actual_network_calls']:assert max(v['decomposition_residual_after'] for v in r['sampling']['steps'])<1e-12
                guide=r['sampling']['guidance'][-1];assert guide['output_hash']==digest(np.asarray(r['Z'])) and guide['collision_enabled']
                for h in guide['history']:
                    if h['accepted'] and max(h['before_pose_norms'])<=.8:assert max(h['after_pose_norms'])<=.8
                v=r['validation'];model=all(v['gates'].get(g,False) for g in MODEL_GATES)
                goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass'] and r['eligible']==(model and goal)
                assert r['integration_intervals']==128*r['repair_coupled_calls']+r['validation_integration_intervals']
                if method in METHODS[:2]:
                    old_method='structured_feasible' if method==METHODS[0] else 'latent_diffusion_feasible'
                    old=read_json(ROOT/'artifacts/v7_1/latent_residual_smoke_001/candidates'/sid/old_method/f'{k:02d}.json')
                    baselines.append(guide['input_hash']==old['sampling']['guidance'][-1]['input_hash'])
            chosen=min((r for r in rows if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
            directory=out/'executions'/sid/method;selection=read_json(directory/'selection.json')
            assert selection['selected_id']==(chosen['candidate_id'] if chosen else None) and selection['selection_before_execution']
            if chosen:
                p,n=audit_physics(directory,task,chosen['reference_hash']);physics.append(p);steps+=n;executions+=1
                assert read_json(directory/'outcome.json')['status']==p['level']
            else:assert not (directory/'report.json').exists()
        assert len(bases)==1
    assert total==250 and all(baselines)
    assert steps==ledger['calls'].get('witness.physics_steps',0)+ledger['calls'].get('execution.physics_steps',0)
    diagnostics={}
    for m,rows in rows_by_method.items():
        sm=summary['methods'][m];assert sm['eligible']==sum(r['eligible'] for r in rows)
        assert sm['coupled_queries']==sum(r['model_calls']['coupled'] for r in rows)
        norm=np.array([np.linalg.norm(np.asarray(r['sampling']['guidance'][-1]['initial_goal_error']).reshape(4,3),axis=1) for r in rows if r.get('sampling')])
        if len(norm):diagnostics[m]={'pre_repair_pose_pass':int(sum(norm.max(1)<=1.)),
            'pre_repair_position_median_mm':float(np.median(norm[:,:2].max(1))*5.),
            'pre_repair_rotation_median_deg':float(np.median(norm[:,2:].max(1))),
            'repair_queries_median':float(np.median([r['repair_coupled_calls'] for r in rows]))}
    result={'status':'PASS','generated_attempts':generated,'admitted_before_execution':admitted,'new_training_rows':len(train_ids)-208,
        'TRAIN_rows':len(train_ids),'VAL_rows':len(val_ids),'sensitivity_anchors':len(sr),'sensitivity_coupled_queries':sq,
        'matched_training_initial_weights_and_rng':matched,'audited_candidates':total,'actual_network_calls':network,
        'preselected_TRAIN_executions':witnesses,'actual_VAL_executions':executions,'physics_steps_including_replay':steps,
        'same_previous_baseline_inputs':sum(baselines),'old_files_unchanged':len(before['old_files']),
        'user_edits_and_branches_preserved':True,'original_DEV_unchanged':True,'TEST_executed':False,
        'minimum_execution_clearance_m':min(p['minimum_clearance_m'] for p in physics),
        'maximum_hold_position_m':max(max(p['hold_max_position_error_m']) for p in physics),
        'proposal_diagnostics':diagnostics,'additional_physics_in_this_audit':0,'blanket_execution_upgrade':False}
    atomic_json(out/'independent_verification.json',result);return result


if __name__=='__main__':
    import json
    print(json.dumps(audit(ROOT/'artifacts/v7_1/qualified_modes_smoke_001'),default=plain,ensure_ascii=False,indent=2))
