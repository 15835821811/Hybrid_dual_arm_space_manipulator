"""Audit saved raw controls, cost accounting, frozen tasks and real traces."""
from pathlib import Path
import hashlib
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,atomic_json,file_digest,digest,plain
from ff_prior.contracts import MODEL_GATES,assert_frozen
from ff_prior.initial_state import STATE
from ff_prior.spline import Reference
from ff_prior_goal.projection import smooth
from ff_prior_complex.tools.audit_completed import audit_physics
from .experiment import ROOT,METHODS,PROTOCOL,targets,sources
from .sampling import SPECS,attenuate
from .reporting import diversity


def audit(out):
    out=Path(out);run=read_json(out/'run.json');ledger=read_json(out/'ledger.json');summary=read_json(out/'summary.json')
    assert summary['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS') and ledger['inflight'] is None
    assert not ledger.get('unresolved_call_accounting')
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    source=Path(run['identity']['source']);assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    assert all(file_digest(source/p)==h for p,h in run['identity']['source_files'].items())
    before=read_json(ROOT/'ff_prior_probe/evidence/preservation_before.json')
    assert all(file_digest(ROOT/p)==h for p,h in before['old_files'].items())
    assert all(file_digest(ROOT/p)==h for p,h in before['user_edits'].items())
    for branch,h in before['branches'].items():assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==h
    with zipfile.ZipFile(out/'executed_source.zip') as z:assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    entries=read_json(out/'frozen_tasks.json')['evaluation_entries'];assert entries==targets() and len(entries)==8
    total=network=physics_steps=executions=0;physical=[];costs={m:[] for m in METHODS};raw_pairs=0
    for ti,e in enumerate(entries):
        task=e['task'];sid=e['sample_id'];assert e['split']=='TRAIN' and 'Z' not in e;assert_frozen(task)
        with np.load(out/'tasks'/sid/'initial_state.npz') as a:
            assert digest({'values':a['values'],'model_hash':task['contract']['robot_hash'],'state_mask':int(STATE)})==task['contract']['initial_state_id']
        base=read_json(out/'transfer_audits'/f'{sid}.json')['public_base'];paired={};base_hashes=set()
        for m in METHODS:
            rows=[read_json(out/'candidates'/sid/m/f'{k:02d}.json') for k in range(5)];costs[m].extend(rows);paired[m]=rows
            for k,r in enumerate(rows):
                total+=1;network+=r['actual_network_calls'];base_hashes.add(r['public_base_hash'])
                assert r['split']=='TRAIN' and r['task_hash']==task['task_hash'] and r['seed']==20261113+ti*101+k*997
                assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']<=113
                assert not r['private_witness_used']
                if r['status']!='DONE':assert not r['eligible'];continue
                assert r['wall_s']<=15 and r['integration_intervals']<=15872 and np.isfinite(r['Z']).all()
                assert r['integration_intervals']==128*r['repair_coupled_calls']+r['validation_integration_intervals']
                kind,amp=SPECS[m];expected=16 if kind in ('frozen','v_only','goal_loss') else 0
                assert r['actual_network_calls']==expected
                sampling=r['sampling'];probe=sampling['residual_probe'];guide=sampling['guidance'][-1]
                original=np.asarray(probe['raw_unscaled']);scaled=np.asarray(probe['raw_scaled'])
                assert digest(original)==probe['raw_unscaled_hash'] and digest(scaled)==probe['raw_scaled_hash']
                assert np.array_equal(scaled,attenuate(original,probe['mean'],amp) if kind in ('v_only','goal_loss') else original)
                if m=='mean_zero':assert np.array_equal(scaled,probe['mean'])
                if expected:assert max(s['decomposition_residual_after'] for s in sampling['steps'])<1e-12
                assert guide['input_hash']==digest(scaled) and guide['output_hash']==digest(np.asarray(r['Z'])) and guide['collision_enabled']
                for h in guide['history']:
                    if h['accepted'] and max(h['before_pose_norms'])<=.8:assert max(h['after_pose_norms'])<=.8
                v=r['validation'];model=all(v['gates'].get(g,False) for g in MODEL_GATES)
                goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass'] and r['eligible']==(model and goal)
                assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
            if ti in PROTOCOL['witness_task_slots']:
                chosen=min((r for r in rows if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                d=out/'executions'/sid/m;selection=read_json(d/'selection.json')
                assert selection['selection_before_execution'] and selection['selected_id']==(chosen['candidate_id'] if chosen else None)
                if chosen:
                    p,n=audit_physics(d,task,chosen['reference_hash']);physical.append(p);executions+=1;physics_steps+=n
                    assert read_json(d/'outcome.json')['status']==p['level']
                else:assert not (d/'report.json').exists()
            else:assert not (out/'executions'/sid/m).exists()
            d=diversity([r['Z'] for r in rows if r['eligible']]);s=summary['methods'][m]['by_task'][sid]
            assert d['distinct']==s['distinct'] and abs(d['pairwise_rms_rad']-s['pairwise_rms_rad'])<1e-12
        for prefix in ('v','g'):
            for k in range(5):
                triplet=[paired[prefix+s][k] for s in ('025','050','100')]
                if all(r['status']=='DONE' for r in triplet):
                    assert len({r['sampling']['residual_probe']['raw_unscaled_hash'] for r in triplet})==1;raw_pairs+=1
        assert len(base_hashes)==1 and digest(base) in base_hashes
    assert total==400 and physics_steps==ledger['calls'].get('execution.physics_steps',0)
    for m,rows in costs.items():
        sm=summary['methods'][m]
        assert sm['eligible']==sum(r['eligible'] for r in rows) and sm['coupled_queries']==sum(r['model_calls']['coupled'] for r in rows)
        assert abs(sm['candidate_wall_s']-sum(r['wall_s'] for r in rows))<1e-8
    assert sum(r['model_calls']['coupled'] for rows in costs.values() for r in rows)==ledger['calls']['candidate.coupled']
    result={'status':'PASS','audited_candidates':total,'actual_network_calls':network,'paired_three_amplitude_unscaled_proposals':raw_pairs,
        'actual_preselected_TRAIN_executions':executions,'physics_steps_including_replay':physics_steps,
        'old_files_unchanged':len(before['old_files']),'user_edits_and_branches_preserved':True,'original_DEV_unchanged':True,
        'minimum_execution_clearance_m':min(p['minimum_clearance_m'] for p in physical) if physical else None,
        'maximum_hold_position_m':max(max(p['hold_max_position_error_m']) for p in physical) if physical else None,
        'new_training_or_expansion':False,'VAL_DEV_TEST_executed':False,'additional_physics_in_audit':0,'full_goal_achieved':False}
    atomic_json(out/'independent_verification.json',result);return result


if __name__=='__main__':
    import json
    print(json.dumps(audit(ROOT/'artifacts/v7_1/train_residual_probe_001'),default=plain,ensure_ascii=False,indent=2))
