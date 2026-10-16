"""Recompute counts, checkpoint identity and native trace evidence without simulation."""
from pathlib import Path
import hashlib
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.contracts import all_pass,MODEL_GATES,assert_frozen
from ff_prior.initial_state import STATE
from ff_prior.spline import Reference
from ff_prior.diffusion import weight_hash
from ff_prior_complex.tools.audit_completed import audit_physics


def verify(out,identity,index,rows,codec):
    from .experiment import ROOT,PROTOCOL,METHODS,sources
    from .learning import load,CONTRACT,FROZEN
    out=Path(out);source=Path(identity['source']);ledger=read_json(out/'ledger.json');problems=[]
    assert ledger['inflight'] is None and ledger['identity']=={'run_identity_hash':digest(identity)}
    assert sources()==identity['source_hashes'] and identity['protocol']==PROTOCOL
    assert all(file_digest(source/p)==h for p,h in identity['source_files'].items())
    assert read_json(out/'training_index.json')['records']==index==read_json(source/'training_index.json')['records']
    assert read_json(out/'run.json')['acceptance']==read_json(source/'run.json')['acceptance']
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    before=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    for group in ('old_files','user_edits'):assert all(file_digest(ROOT/p)==h for p,h in before[group].items())
    for branch,h in before['branches'].items():assert subprocess.check_output(['git','rev-parse',branch],text=True).strip()==h
    audit=read_json(out/'training_input_audit.json');assert audit['actual_array_count']==len(index) and audit['records']==index
    with np.load(out/'loaded_training_arrays.npz') as a:z,bases=a['Z'].copy(),a['bases'].copy()
    assert len(z)==len(index)
    for i,e in enumerate(index):
        for key in ('record','arrays'):assert file_digest(ROOT/e[key])==e[key+'_sha256']
        with np.load(ROOT/e['arrays']) as a:assert np.array_equal(z[i],a['Z'])
        assert rows[i]==read_json(ROOT/e['record']) and e['execution_status']!='FAIL'
    train_ids=[e['sample_id'] for e in index if e['split']=='TRAIN'];val_ids=[e['sample_id'] for e in index if e['split']=='VAL']
    assert 12096<=len(train_ids)<=12144 and len(val_ids)==48 and not set(train_ids)&set(val_ids)
    trained=(out/'learning/goal_loss/training.json').exists()
    actual_draws=0;anchors=0
    if trained:
        tr=read_json(out/'learning/goal_loss/training.json');model,loaded=load(out/'learning/goal_loss')
        assert tr['status']=='PASS' and tr['updates']==CONTRACT['updates']
        assert tr['trained_weight_hash']==weight_hash(model)!=tr['initial_weight_hash']
        assert set(tr['train_draw_counts'])==set(train_ids) and all(v>0 for v in tr['train_draw_counts'].values())
        actual_draws=sum(tr['train_draw_counts'].values());assert actual_draws==CONTRACT['updates']*CONTRACT['batch']
        assert tr['VAL_gradient_updates']==0 and not tr['VAL_checkpoint_selection']
        assert loaded.state['fit_split']=='TRAIN' and digest(loaded.state)==digest(codec.state)==digest(read_json(FROZEN/'codec.json'))
        from ff_prior_pilot_train.learning import load as parent_load
        parent_model,_=parent_load(FROZEN);assert tr['initial_weight_hash']==weight_hash(parent_model)
        assert tr['metadata']['warm_start_checkpoint_sha256']==file_digest(FROZEN/'model.pt')
        fit_ids=set(loaded.state['fit_ids'])
        mask=np.array([e['sample_id'] in fit_ids for e in index]);conditions=[r['condition'] for r in rows]
        assert [e['sample_id'] for e,m in zip(index,mask) if m]==loaded.state['fit_ids']
        assert loaded.state['fit_hash']==digest({'z':z[mask],'bases':bases[mask],'conditions':[c for c,m in zip(conditions,mask) if m],'ids':loaded.state['fit_ids']})
        encoded=np.stack([loaded.encode(v,b) for v,b in zip(z,bases)])
        reconstructed=bases+np.einsum('tk,nkj->ntj',loaded.basis,encoded.reshape(len(z),loaded.rank,17))
        errors=np.max(abs(reconstructed-z),axis=(1,2))
        trmask=np.array([e['split']=='TRAIN' for e in index]);newmask=trmask & ~mask
        representation=tr['metadata']['representation']
        assert representation['frozen_codec_fit_rows']==10000 and representation['new_TRAIN_rows']==sum(newmask)
        assert representation['TRAIN_max_abs_projection_error_rad']==float(max(errors[trmask]))
        assert representation['new_TRAIN_max_abs_projection_error_rad']==float(max(errors[newmask]))
        assert tr['metadata']['data_hash']==digest({'z':z,'bases':bases,'conditions':conditions,'ids':[e['sample_id'] for e in index],
            'splits':[e['split'] for e in index],'states':[r['task']['contract']['initial_state_id'] for r in rows]})
        summary=read_json(out/'sensitivity_summary.json');assert summary['status']=='PASS' and summary['TRAIN_only']
        keys={r['task']['task_hash'] for e,r in zip(index,rows) if e['split']=='TRAIN'}
        id_slots={e['sample_id']:i for i,e in enumerate(index)}
        assert set(summary['jacobian_files'])==keys;anchors=len(keys);Gs={};sensitivity_calls=0
        for key in keys:
            p=out/'sensitivity'/f'{key}.npz';proof=read_json(p.with_suffix('.json'))
            assert file_digest(p)==summary['jacobian_files'][key] and proof['status']=='PASS'
            assert proof['task_hash']==key and proof['anchor_sample_id'] in train_ids
            anchor=id_slots[proof['anchor_sample_id']]
            assert rows[anchor]['task']['task_hash']==key and proof['reference_hash']==digest(z[anchor])
            assert proof['mixed_direction_relative_error']<=.05 and all(c['passed'] for c in proof['half_step_checks'])
            sensitivity_calls+=proof['calls']['coupled'];assert proof['calls']['coupled']==39
            with np.load(p) as a:Gs[key]=a['G'].copy()
            assert Gs[key].shape==(12,loaded.dim) and np.isfinite(Gs[key]).all()
        matrices=np.stack([Gs[r['task']['task_hash']] for e,r in zip(index,rows) if e['split']=='TRAIN'])
        assert tr['metadata']['sensitivity_matrix_hash']==digest(matrices)
        assert tr['metadata']['sensitivity_hash']==digest(summary)
        assert sensitivity_calls==ledger['calls'].get('sensitivity.coupled',0)+ledger['calls'].get('cached_sensitivity.coupled',0)
        assert ledger['calls'].get('cached_sensitivity.new_coupled',0)==0
        parent_source=FROZEN.parent.parent
        for key in keys:
            entry=ledger['completed']['sensitivity_'+key]
            if 'reused_from' in entry:
                assert entry['reused_from']==str(parent_source)
                for suffix in ('.json','.npz'):
                    rel='sensitivity/'+key+suffix
                    assert file_digest(out/rel)==file_digest(parent_source/rel)
                assert entry['source_proof_sha256']==file_digest(parent_source/'sensitivity'/f'{key}.json')
    else:problems.append('TRAINING_NOT_COMPLETED')
    dev=read_json(out/'frozen_tasks.json')['DEV'];assert digest(dev)==identity['tasks_hash']
    network=calls=steps=0;traces=[];candidate_count=0
    for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
        for ti,e in enumerate(dev):
            task=e['task'];sid=f'DEV_{ti:02d}';assert_frozen(task);base_hashes=set()
            with np.load(out/'DEV_tasks'/sid/'initial_state.npz') as a:
                assert digest({'values':a['values'],'model_hash':task['contract']['robot_hash'],'state_mask':int(STATE)})==task['contract']['initial_state_id']
            for method in METHODS:
                candidates=[]
                for k in range(5):
                    r=read_json(out/'candidates'/str(sj)/sid/method/f'{k:02d}.json');candidates.append(r);candidate_count+=1
                    if r['status']=='BLOCKED':problems.append(r['reason']);continue
                    assert r['task_hash']==task['task_hash'] and r['seed']==seed+ti*101+k*997 and not r['private_witness_used']
                    calls+=r['model_calls']['coupled'];network+=r['actual_network_calls'];base_hashes.add(r['public_base_hash'])
                    if r['status']!='DONE':problems.append(r.get('reason',r['status']));assert not r['eligible'];continue
                    assert r['actual_network_calls']==(0 if method in ('structured','learned_mean') else 16)
                    assert r['wall_s']<=15 and r['integration_intervals']<=15872 and np.isfinite(r['Z']).all()
                    assert r['repair_coupled_calls']+r['validation_coupled_calls']==r['model_calls']['coupled']<=113
                    if r['actual_network_calls']:assert max(s['decomposition_residual_after'] for s in r['sampling']['steps'])<1e-12
                    guide=r['sampling']['guidance'][-1];assert guide['output_hash']==digest(np.asarray(r['Z']))
                    for h in guide['history']:
                        if h['accepted'] and max(h['before_pose_norms'])<=.8:assert max(h['after_pose_norms'])<=.8
                    v=r['validation'];valid=all_pass(v['gates'],MODEL_GATES)
                    goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                    assert goal==r['frozen_goal_pass'] and r['eligible']==(valid and goal)
                    assert r['reference_hash']==Reference(r['Z'],task['condition']['initial_features'],task['condition']['duration']).reference_hash
                chosen=min((r for r in candidates if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                directory=out/'executions'/str(sj)/sid/method;selection=read_json(directory/'selection.json')
                assert selection['selection_before_execution'] and selection['selected_id']==(chosen['candidate_id'] if chosen else None)
                if chosen:
                    p,n=audit_physics(directory,task,chosen['reference_hash']);steps+=n;traces.append(p)
                    assert p['level']==read_json(directory/'outcome.json')['status']
                else:assert read_json(directory/'outcome.json')['status']=='NO_CANDIDATE'
            assert len(base_hashes)==1
    assert candidate_count==240 and calls==ledger['calls'].get('candidate.coupled',0)
    assert steps==ledger['calls'].get('execution.physics_steps',0)
    return {'status':'PASS' if not problems and not ledger.get('unresolved_call_accounting') else 'BLOCKED',
        'problems':problems,'actual_arrays':len(index),'TRAIN_rows':len(train_ids),'actual_training_draws':actual_draws,
        'sensitivity_anchors_verified':anchors,'candidate_records':candidate_count,'network_calls':network,
        'audited_execution_traces':len(traces),'physics_steps_including_replay':steps,
        'minimum_execution_clearance_m':min((p['minimum_clearance_m'] for p in traces),default=None),
        'maximum_hold_position_m':max((max(p['hold_max_position_error_m']) for p in traces),default=None),
        'old_data_sources_user_edits_branches_unchanged':True,'TEST_executed':False,'additional_physics_calls':0,'full_goal_achieved':False}
