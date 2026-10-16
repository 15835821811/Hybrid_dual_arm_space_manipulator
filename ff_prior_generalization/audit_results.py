"""Verify frozen task isolation, unchanged checkpoints, candidates and native traces."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior.contracts import assert_frozen, all_pass, MODEL_GATES
from ff_prior.initial_state import STATE
from ff_prior.spline import Reference
from ff_prior_complex.tools.audit_completed import audit_physics
from .protocol import ROOT, METHODS, PROTOCOL
from .data import assert_isolated
from .reporting import pre_norm


def check_candidate(r,task,seed,method,k):
    assert r['task_hash']==task['task_hash'] and r['seed']==seed and r['method']==method and r['candidate_id']==k
    assert not r['private_witness_used'] and r['model_calls']['coupled']<=113
    if r['status']!='DONE':assert not r['eligible'];return
    assert r['wall_s']<=15 and r['integration_intervals']<=15872 and np.isfinite(r['Z']).all()
    assert r['repair_coupled_calls']+r['validation_coupled_calls']==r['model_calls']['coupled']
    assert r['repair_coupled_calls']<=110 and r['validation_coupled_calls']<=3
    assert r['actual_network_calls']==(16 if method in ('frozen_diffusion','expanded_diffusion') else 0)
    if r['actual_network_calls']:assert max(s['decomposition_residual_after'] for s in r['sampling']['steps'])<1e-12
    guide=r['sampling']['guidance'][-1]
    assert digest(np.asarray(guide['raw_Z']))==r['sampling']['raw_proposal_hash']
    assert guide['output_hash']==digest(np.asarray(r['Z'])) and np.isfinite(pre_norm(r))
    for h in guide['history']:
        if h['accepted'] and max(h['before_pose_norms'])<=.8:assert max(h['after_pose_norms'])<=.8
    v=r['validation'];goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
    assert goal==r['frozen_goal_pass'] and r['eligible']==(all_pass(v['gates'],MODEL_GATES) and goal)
    assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash


def verify(out,identity,guard):
    from .experiment import sources
    out=Path(out);ledger=read_json(out/'ledger.json');problems=[]
    assert ledger['inflight'] is None and ledger['identity']=={'identity_hash':digest(identity)}
    assert identity['protocol']==PROTOCOL and identity['source_hashes']==sources()
    assert all(file_digest(ROOT/p)==h for p,h in identity['source_files'].items())
    assert read_json(out/'protected_goal_metadata.json')==guard and digest(guard)==identity['guard_hash']
    assert digest(read_json(out/'original_DEV.json'))==identity['original_DEV_hash']
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    binding=read_json(out/'checkpoint_binding.json')
    assert binding['training_updates']==0 and not binding['validation_used_for_checkpoint_selection']
    from .protocol import OLD
    source=Path(identity['source'])
    assert binding['old']==file_digest(OLD/'learning/goal_loss/model.pt')
    assert binding['new']==file_digest(source/'learning/goal_loss/model.pt')
    assert binding['codec']==digest(read_json(source/'learning/goal_loss/codec.json'))
    before=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    for group in ('old_files','user_edits'):assert all(file_digest(ROOT/p)==h for p,h in before[group].items())
    for branch,h in before['branches'].items():assert subprocess.check_output(['git','rev-parse',branch],text=True).strip()==h
    frozen=read_json(out/'frozen_tasks.json');entries=frozen['entries']
    assert len(entries)==12 and digest(entries)==frozen['digest'] and frozen['before_any_comparison']
    events=[json.loads(s) for s in (out/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    ft=next(e['timestamp'] for e in events if e['stage']=='all_new_tasks_frozen')
    assert all(e['timestamp']>=ft for e in events if e['stage'] in ('new_validation_teacher_torque_replay','new_validation_candidates','new_validation_torque_replay'))
    traces=[];steps=network=count=0;by_stage_calls={}
    selected=[]
    for entry in entries:
        if entry['status']!='FROZEN':problems.append(entry['sample_id']+':TASK_GENERATION_BLOCKED');continue
        sid=entry['sample_id'];slot=entry['slot'];task=entry['task'];assert_frozen(task)
        key=f'freeze_{slot:02d}_{entry["attempt"]}'
        assert read_json(out/'generation'/f'{key}.json')==entry and ledger['completed'][key]['status']=='FROZEN'
        assert task['contract']['acceptance']==read_json(out/'run.json')['acceptance']
        assert all(ledger['completed'].get(f'freeze_{slot:02d}_{a}',{}).get('status')!='FROZEN' for a in range(entry['attempt']))
        assert_isolated(entry,guard+selected)
        selected.append({'sample_id':sid,'split':'NEW_VAL','family_id':entry['family_id'],'goals':task['condition']['goals']})
        assert read_json(out/'tasks'/sid/'task.json')==task and entry['witness_model']['eligible']
        assert file_digest(out/'tasks'/sid/'witness.npz')==entry['witness_sha256']
        assert file_digest(out/'tasks'/sid/'initial_state.npz')==entry['initial_state_sha256']
        with np.load(out/'tasks'/sid/'initial_state.npz') as a:
            assert digest({'values':a['values'],'model_hash':task['contract']['robot_hash'],'state_mask':int(STATE)})==task['contract']['initial_state_id']
        with np.load(out/'tasks'/sid/'witness.npz') as a:
            assert Reference(a['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash==entry['reference_hash']
        directory=out/'witness_executions'/sid
        if (directory/'report.json').exists():
            p,n=audit_physics(directory,task,entry['reference_hash']);traces.append(p);steps+=n
            assert p['level']==read_json(directory/'outcome.json')['status']
        else:problems.append('MISSING_TEACHER_TRACE:'+sid)
        for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
            base_hashes=set()
            for method in METHODS:
                records=[]
                for k in range(5):
                    r=read_json(out/'candidates'/str(sj)/sid/method/f'{k:02d}.json');records.append(r);count+=1
                    check_candidate(r,task,seed+slot*101+k*997,method,k)
                    assert r['sample_id']==sid and r['seed_group']==sj and r['split']=='NEW_VAL'
                    network+=r['actual_network_calls'];base_hashes.add(r['public_base_hash'])
                    for key,value in r['model_calls'].items():by_stage_calls['candidate.'+key]=by_stage_calls.get('candidate.'+key,0)+value
                chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                directory=out/'executions'/str(sj)/sid/method;selection=read_json(directory/'selection.json')
                assert selection['selected_id']==(chosen['candidate_id'] if chosen else None) and selection['selection_before_execution'] and selection['task_hash']==task['task_hash']
                outcome=read_json(directory/'outcome.json')
                if chosen and (directory/'report.json').exists():
                    p,n=audit_physics(directory,task,chosen['reference_hash']);traces.append(p);steps+=n
                    assert p['level']==outcome['status']
                elif chosen:problems.append('MISSING_SELECTED_TRACE:'+str(directory))
                else:assert outcome['status']=='NO_CANDIDATE'
            assert len(base_hashes)==1
        for method in METHODS:
            source=out/'candidates'/'0'/sid/method/'00.json'
            for stage in ('raw','smoothed_clipped'):
                p=out/'proposal_diagnostics'/sid/method/f'{stage}.json';r=read_json(p)
                if r['status']!='DONE':problems.append('PROPOSAL_DIAGNOSTIC_ERROR:'+str(p));continue
                assert r['source_candidate_sha256']==file_digest(source) and r['task_hash']==task['task_hash'] and not r['affects_selection']
                assert r['calls']['coupled']<=3 and r['validation_integration_intervals']<=1792
                v=r['validation'];goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                assert r['eligible']==(all_pass(v['gates'],MODEL_GATES) and goal)
                for key,value in r['calls'].items():by_stage_calls['diagnostic.'+key]=by_stage_calls.get('diagnostic.'+key,0)+value
    assert count==len(selected)*60 and count<=720
    for stage in ('candidate','diagnostic'):
        assert all(ledger['calls'].get(k,0)==v for k,v in by_stage_calls.items() if k.startswith(stage+'.'))
    assert steps==ledger['calls'].get('witness.physics_steps',0)+ledger['calls'].get('execution.physics_steps',0)
    return {'status':'PASS' if not problems and not ledger.get('unresolved_call_accounting') else 'BLOCKED',
            'problems':problems,'frozen_tasks':len(selected),'candidate_records':count,'network_calls':network,
            'audited_execution_traces':len(traces),'physics_steps_including_replay':steps,
            'minimum_execution_clearance_m':min((p['minimum_clearance_m'] for p in traces),default=None),
            'maximum_hold_position_m':max((max(p['hold_max_position_error_m']) for p in traces),default=None),
            'source_files_models_user_edits_branches_unchanged':True,'training_updates':0,
            'TEST_executed':False,'additional_physics_calls':0,'full_goal_achieved':False}
