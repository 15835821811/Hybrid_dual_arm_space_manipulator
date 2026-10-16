"""Independent arithmetic/provenance/native-trace audit; no new physics calls."""
from pathlib import Path
import hashlib
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.initial_state import STATE
from ff_prior.contracts import assert_frozen,MODEL_GATES,all_pass
from ff_prior.spline import Reference,from_arrays
from ff_prior_complex.tools.audit_completed import audit_physics


def verify(out):
    from .experiment import ROOT,PROTOCOL,METHODS,sources,inputs
    from . import data
    from .learning import load
    out=Path(out);run=read_json(out/'run.json');ledger=read_json(out/'ledger.json')
    assert ledger['inflight'] is None
    assert run['identity']['source_hashes']==sources() and run['identity']['protocol']==PROTOCOL
    source=Path(run['identity']['source']);_,_,dev,_,files=inputs(source)
    assert files==run['identity']['source_files'] and digest(dev)==run['identity']['tasks_hash']
    assert run['acceptance']==read_json(source/'baseline_audit.json')['acceptance']
    before=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    for group in ('old_files','user_edits'):
        assert all(file_digest(ROOT/p)==h for p,h in before[group].items()),group
    for branch,h in before['branches'].items():assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==h
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    frozen=read_json(out/'frozen_tasks.json');assert frozen['DEV']==dev
    index=read_json(out/'training_index.json')['records'];old=data.old_index()
    assert index[:len(old)]==old and sum(e['split']=='VAL' for e in index)==48
    heldout=data.heldout_goals(old,dev,source)
    accepted=[];hashes={read_json(ROOT/e['record'])['reference_hash'] for e in old}
    for path in sorted((out/'records').glob('*.json')):
        r=data.check_new_record(out,path)
        assert ledger['completed'][r['sample_id']]['status']=='ADMITTED'
        assert not data.goal_neighbours(r['task'],heldout)
        assert r['reference_hash'] not in hashes;hashes.add(r['reference_hash'])
        with np.load(out/r['arrays']) as a:assert from_arrays(a).reference_hash==r['reference_hash']
        with np.load(out/r['complete_initial_state']) as a:
            assert digest({'values':a['values'],'model_hash':r['task']['contract']['robot_hash'],'state_mask':int(STATE)})==r['task']['contract']['initial_state_id']
        if r['execution_status']!='FAIL':accepted.append(r)
        assert (r['validation_level']=='EXECUTION_VERIFIED')==(r['execution_status']=='PASS')
    new={e['sample_id']:e for e in index if e['sample_id'].startswith('expand_train_')}
    assert set(new)=={r['sample_id'] for r in accepted}
    for r in accepted:
        e=new[r['sample_id']]
        for k in ('record','arrays'):assert file_digest(ROOT/e[k])==e[k+'_sha256']
    generated=[read_json(p) for p in (out/'generation').glob('*.json')]
    assert len(generated)<=1200 and len(generated)==ledger['attempts']['generation']
    assert sum(r['calls']['coupled'] for r in generated)==ledger['calls']['generation.coupled']
    assert all(not any(r['calls'][k] for k in ('rrt','sqp','torque_rollout','physics_steps')) for r in generated)
    trainpath=out/'learning/goal_loss/training.json';train=read_json(trainpath) if trainpath.exists() else None
    if train:
        bundle=load(trainpath.parent);assert train['status']=='PASS'
        expected={e['sample_id'] for e in index if e['split']=='TRAIN'}
        assert set(train['train_draw_counts'])==expected and all(v>0 for v in train['train_draw_counts'].values())
        assert sum(train['train_draw_counts'].values())==3600*32
        assert set(bundle[1].state['fit_ids'])==expected and bundle[1].state['fit_split']=='TRAIN'
        assert train['VAL_gradient_updates']==0 and train['initial_weight_hash']!=train['trained_weight_hash']
    calls=network=0;problems=[]
    def check_candidate(r,task,method,seed):
        nonlocal calls,network
        assert r['task_hash']==task['task_hash'] and r['seed']==seed
        assert not r['private_witness_used']
        calls+=r['model_calls']['coupled'];network+=r['actual_network_calls']
        assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']<=113
        if r['status']!='DONE':
            assert not r['eligible'];problems.append(r.get('reason',r['status']));return
        assert r['wall_s']<=15 and r['integration_intervals']<=15872 and np.isfinite(r['Z']).all()
        expected=0 if method in ('structured','learned_mean') else 16
        assert r['actual_network_calls']==expected
        if expected:assert max(s['decomposition_residual_after'] for s in r['sampling']['steps'])<1e-12
        guide=r['sampling']['guidance'][-1]
        assert guide['output_hash']==digest(np.asarray(r['Z']))
        for h in guide['history']:
            if h['accepted'] and max(h['before_pose_norms'])<=.8:assert max(h['after_pose_norms'])<=.8
        v=r['validation'];model=all_pass(v['gates'],MODEL_GATES)
        goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
        assert model==(v['validation_level']=='MODEL_VALID') and goal==r['frozen_goal_pass'] and r['eligible']==(model and goal)
        assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
    for ti,e in enumerate(frozen['mechanism_entries']):
        for sj,seed in enumerate(PROTOCOL['mechanism_seeds']):
            pair=[]
            for method in ('frozen_diffusion','diffusion_goal_only'):
                r=read_json(out/'mechanism'/f'mechanism_{ti}_{sj}_{method}.json')
                check_candidate(r,e['task'],method,seed+ti*101);pair.append(r)
            if all(r['status']=='DONE' for r in pair):assert pair[0]['sampling']['raw_proposal_hash']==pair[1]['sampling']['raw_proposal_hash']
    physics=[];steps=0
    for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
        for ti,e in enumerate(dev):
            task=e['task'];sid=f'DEV_{ti:02d}';assert_frozen(task);base_hashes=set()
            with np.load(out/'DEV_tasks'/sid/'initial_state.npz') as a:
                assert digest({'values':a['values'],'model_hash':task['contract']['robot_hash'],'state_mask':int(STATE)})==task['contract']['initial_state_id']
            for method in METHODS:
                candidates=[]
                for k in range(5):
                    r=read_json(out/'candidates'/str(sj)/sid/method/f'{k:02d}.json');candidates.append(r)
                    if r['status']=='BLOCKED':problems.append(r['reason']);continue
                    check_candidate(r,task,method,seed+ti*101+k*997);base_hashes.add(r['public_base_hash'])
                chosen=min((r for r in candidates if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                directory=out/'executions'/str(sj)/sid/method;selection=read_json(directory/'selection.json')
                assert selection['selection_before_execution'] and selection['selected_id']==(chosen['candidate_id'] if chosen else None)
                if chosen and (directory/'report.json').exists():
                    p,n=audit_physics(directory,task,chosen['reference_hash']);physics.append(p);steps+=n
                    assert p['level']==read_json(directory/'outcome.json')['status']
                elif chosen:problems.append('MISSING_EXECUTION_TRACE')
                else:assert read_json(directory/'outcome.json')['status']=='NO_CANDIDATE'
            assert len(base_hashes)==1
    selection=read_json(out/'audit_selection.json')
    for sid in selection['random_stratified']+selection['risk']:
        r=read_json(out/'records'/f'{sid}.json');directory=out/'witness_executions'/sid
        if (directory/'report.json').exists():
            p,n=audit_physics(directory,r['task'],r['reference_hash']);physics.append(p);steps+=n
            assert (p['level']=='EXECUTION_VERIFIED')==(r['execution_status']=='PASS')
        else:problems.append('MISSING_WITNESS_TRACE')
    assert steps==sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))
    assert calls==ledger['calls'].get('candidate.coupled',0)+ledger['calls'].get('mechanism.coupled',0)
    return {'status':'PASS' if not problems and not ledger.get('unresolved_call_accounting') else 'BLOCKED',
        'problems':problems,'actual_new_TRAIN_arrays':len(accepted),'training_rows_actually_consumed':len(train['train_draw_counts']) if train else 0,
        'new_rows_all_consumed':bool(train and all(train['train_draw_counts'].get(r['sample_id'],0)>0 for r in accepted)),
        'candidate_network_calls':network,'audited_execution_traces':len(physics),'physics_steps_including_replay':steps,
        'old_files_unchanged':len(before['old_files']),'user_edits_and_branches_unchanged':True,
        'original_DEV_unchanged':True,'TEST_arrays_read':False,'TEST_goal_metadata_split_guard_only':True,
        'minimum_execution_clearance_m':min((p['minimum_clearance_m'] for p in physics),default=None),
        'maximum_hold_position_m':max((max(p['hold_max_position_error_m']) for p in physics),default=None),
        'additional_audit_physics_calls':0,'full_goal_achieved':False}
