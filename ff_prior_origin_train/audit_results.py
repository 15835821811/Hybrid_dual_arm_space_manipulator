"""Offline independent recomputation of arrays, real updates and raw traces."""
from pathlib import Path
import hashlib
import subprocess
import zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.contracts import assert_frozen,all_pass,MODEL_GATES
from ff_prior.initial_state import STATE
from ff_prior.diffusion import weight_hash
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_generalization.audit_results import check_candidate
from ff_prior_goal.projection import smooth
from .protocol import ROOT,SOURCE,OLD,METHODS,PRODUCERS,PROTOCOL as P


def verify(out,identity,index,rows,codec,arrays):
    from .experiment import sources
    from .learning import load,CONTRACT
    from ff_prior_coverage_train.learning import load as old_load
    from .mean import fit as mean_fit
    out=Path(out);ledger=read_json(out/'ledger.json');problems=[]
    assert ledger['inflight'] is None and ledger['identity']=={'identity_hash':digest(identity)}
    assert identity['protocol']==P and sources()==identity['source_hashes']
    assert all(file_digest(ROOT/p)==h for p,h in identity['source_files'].items())
    with zipfile.ZipFile(out/'executed_source.zip') as archive:
        assert all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in sources().items())
    preserved=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    for group in ('old_files','user_edits'):assert all(file_digest(ROOT/p)==h for p,h in preserved[group].items())
    for branch,h in preserved['branches'].items():
        assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==h
    assert read_json(out/'training_index.json')['records']==index==read_json(SOURCE/'training_index.json')['records']
    assert read_json(out/'frozen_validation.json')==read_json(SOURCE/'frozen_validation.json')
    assert read_json(out/'run.json')['acceptance']==read_json(SOURCE/'run.json')['acceptance']
    input_audit=read_json(out/'training_input_audit.json')
    assert input_audit['actual_array_count']==12736 and input_audit['records']==index and not input_audit['TEST_arrays_read']
    z,bases,conditions,ids,splits,states=arrays
    with np.load(out/'loaded_training_arrays.npz') as a:
        assert np.array_equal(a['Z'],z) and np.array_equal(a['bases'],bases)
    for i,e in enumerate(index):
        for key in ('record','arrays'):assert file_digest(ROOT/e[key])==e[key+'_sha256']
        with np.load(ROOT/e['arrays']) as a:assert np.array_equal(z[i],a['Z'])
        assert rows[i]==read_json(ROOT/e['record']) and e['execution_status']!='FAIL'
        assert states[i]==rows[i]['task']['contract']['initial_state_id']
    trmask=np.array([s=='TRAIN' for s in splits]);tids=np.asarray(ids)[trmask].tolist()
    vids=np.asarray(ids)[~trmask].tolist();assert len(tids)==12688 and len(vids)==48
    coverage=[e['sample_id'] for e in index if e['split']=='TRAIN' and e.get('producer_run_id') in PRODUCERS]
    assert len(coverage)==544 and sorted(coverage)==identity['coverage_ids']
    validation=read_json(out/'frozen_validation.json');newval_ids={e['sample_id'] for e in validation['entries']}
    assert len(newval_ids)==8 and not newval_ids&set(ids)
    mean=mean_fit(out/'learning',arrays,codec,coverage)
    assert mean.state['fit_ids']==tids and mean.state['mean_contract']['TRAIN_only']
    assert not newval_ids&set(mean.state['fit_ids'])
    binding=read_json(out/'checkpoint_binding.json')
    assert binding['old']==file_digest(OLD/'learning/goal_loss/model.pt')
    assert binding['fixed_codec']==digest(codec.state) and not binding['validation_checkpoint_selection']
    assert binding['mean_codec_sha256']==file_digest(out/'learning/weighted_mean.json')
    draws=anchors=0
    if (out/'learning/goal_loss/training.json').exists():
        tr=read_json(out/'learning/goal_loss/training.json');model,loaded=load(out/'learning/goal_loss')
        assert tr['status']=='PASS' and tr['updates']==6000==CONTRACT['updates']
        assert tr['trained_weight_hash']==weight_hash(model)!=tr['initial_weight_hash']
        parent,pc=old_load(OLD/'learning/goal_loss')
        assert tr['initial_weight_hash']==weight_hash(parent)
        assert digest(loaded.state)==digest(codec.state)==digest(pc.state)
        assert tr['metadata']['warm_start_checkpoint_sha256']==file_digest(OLD/'learning/goal_loss/model.pt')
        assert binding['new']==file_digest(out/'learning/goal_loss/model.pt')
        counts=tr['train_draw_counts'];assert set(counts)==set(tids) and all(v>0 for v in counts.values())
        draws=sum(counts.values());assert draws==384000
        assert tr['group_draw_counts']=={'old':192000,'coverage':192000}
        assert tr['group_rows']=={'old':12144,'coverage':544}
        assert sum(counts[sid] for sid in coverage)==192000
        assert all(counts[sid] in (352,353) for sid in coverage)
        assert all(counts[sid] in (15,16) for sid in set(tids)-set(coverage))
        assert tr['metadata']['new_coverage_ids']==sorted(coverage)
        assert tr['VAL_gradient_updates']==0 and not tr['VAL_checkpoint_selection']
        assert tr['metadata']['validation_ids']==vids and tr['metadata']['training_ids']==tids
        assert tr['metadata']['data_hash']==digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states})
        fit_ids=codec.state['fit_ids'];slots={sid:i for i,sid in enumerate(ids)}
        fi=[slots[sid] for sid in fit_ids];assert len(fi)==10000 and all(splits[i]=='TRAIN' for i in fi)
        assert codec.state['fit_hash']==digest({'z':z[fi],'bases':bases[fi],
            'conditions':[conditions[i] for i in fi],'ids':fit_ids})
        summary=read_json(out/'sensitivity_summary.json')
        keys={r['task']['task_hash'] for r,s in zip(rows,splits) if s=='TRAIN'}
        assert summary['status']=='PASS' and summary['TRAIN_only'] and len(keys)==12472
        assert set(summary['jacobian_files'])==keys;anchors=len(keys);Gs={};cached=0;new=0
        for key in keys:
            path=out/'sensitivity'/f'{key}.npz';proof=read_json(path.with_suffix('.json'))
            assert file_digest(path)==summary['jacobian_files'][key] and proof['status']=='PASS'
            assert proof['task_hash']==key and proof['anchor_sample_id'] in tids
            i=slots[proof['anchor_sample_id']]
            assert rows[i]['task']['task_hash']==key and proof['reference_hash']==digest(z[i])
            assert proof['calls']['coupled']==39 and proof['mixed_direction_relative_error']<=.05
            assert all(c['passed'] for c in proof['half_step_checks'])
            with np.load(path) as a:Gs[key]=a['G'].copy()
            assert Gs[key].shape==(12,loaded.dim) and np.isfinite(Gs[key]).all()
            entry=ledger['completed']['sensitivity_'+key]
            if 'reused_from' in entry:
                cached+=1;assert Path(entry['reused_from']).resolve()==OLD.resolve()
                for suffix in ('.json','.npz'):
                    assert file_digest(out/'sensitivity'/f'{key}{suffix}')==file_digest(OLD/'sensitivity'/f'{key}{suffix}')
                assert entry['source_proof_sha256']==file_digest(OLD/'sensitivity'/f'{key}.json')
            else:new+=1
        assert cached==11960 and new==512
        assert ledger['calls'].get('sensitivity.coupled',0)==512*39
        assert ledger['calls'].get('cached_sensitivity.coupled',0)==11960*39
        assert ledger['calls'].get('cached_sensitivity.new_coupled',0)==0
        matrices=np.stack([Gs[r['task']['task_hash']] for r,s in zip(rows,splits) if s=='TRAIN'])
        assert tr['metadata']['sensitivity_matrix_hash']==digest(matrices) and tr['metadata']['sensitivity_hash']==digest(summary)
    else:problems.append('ACTUAL_TRAINING_NOT_COMPLETED')
    frozen=read_json(out/'frozen_tasks.json');entries=frozen['entries']
    assert len(entries)==12 and frozen['before_training'] and frozen['before_comparison']
    assert digest(entries)==identity['comparison_entries_hash']
    assert {e['sample_id'] for e in entries if e['split']=='NEW_VAL'}==newval_ids
    olddev=read_json(OLD/'frozen_tasks.json')['DEV']
    for i,e in enumerate(entries[:4]):assert e['task']==olddev[i]['task'] and e['sample_id']==f'DEV_{i:02d}'
    traces=[];steps=network=count=0;calls={};diagnostics=0
    run=read_json(out/'run.json');lower,upper=np.asarray(run['joint_lower']),np.asarray(run['joint_upper'])
    for entry in entries:
        sid=entry['sample_id'];ti=entry['comparison_slot'];task=entry['task'];assert_frozen(task)
        with np.load(out/'comparison_tasks'/sid/'initial_state.npz') as a:
            assert digest({'values':a['values'],'model_hash':task['contract']['robot_hash'],'state_mask':int(STATE)})==task['contract']['initial_state_id']
        for sj,seed in enumerate(P['sampling_seeds']):
            base_hashes=set()
            for method in METHODS:
                records=[]
                for k in range(5):
                    r=read_json(out/'candidates'/str(sj)/sid/method/f'{k:02d}.json');records.append(r);count+=1
                    assert r['sample_id']==sid and r['seed_group']==sj and r['split']==entry['split']
                    if r['status']=='BLOCKED':problems.append('BLOCKED_CANDIDATE:'+str((sid,method,k)));continue
                    check_candidate(r,task,seed+ti*101+k*997,method,k)
                    if r['status']!='DONE':problems.append('CANDIDATE_ERROR:'+str((sid,method,k)))
                    assert ledger['completed'][f'candidate_{sj}_{sid}_{method}_{k}']['status']==r['status']
                    network+=r['actual_network_calls'];base_hashes.add(r['public_base_hash'])
                    for key,value in r['model_calls'].items():calls['candidate.'+key]=calls.get('candidate.'+key,0)+value
                chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                directory=out/'executions'/str(sj)/sid/method;selection=read_json(directory/'selection.json')
                assert selection['selection_before_execution'] and selection['task_hash']==task['task_hash']
                assert selection['selected_id']==(chosen['candidate_id'] if chosen else None)
                outcome=read_json(directory/'outcome.json')
                assert outcome['status']==ledger['completed'][f'execution_{sj}_{sid}_{method}']['status']
                if chosen and (directory/'report.json').exists():
                    p,n=audit_physics(directory,task,chosen['reference_hash']);traces.append(p);steps+=n
                    assert p['level']==outcome['status']
                elif chosen:problems.append('MISSING_NATIVE_TRACE:'+str(directory))
                else:assert outcome['status']=='NO_CANDIDATE'
            assert len(base_hashes)==1
        for method in METHODS:
            source=out/'candidates'/'0'/sid/method/'00.json';original=read_json(source)
            for stage in ('raw','smoothed_clipped'):
                p=out/'proposal_diagnostics'/sid/method/f'{stage}.json';r=read_json(p);diagnostics+=1
                if r['status']!='DONE':problems.append('DIAGNOSTIC_BLOCKED:'+str(p));continue
                assert r['source_candidate_sha256']==file_digest(source) and not r['affects_selection']
                assert r['task_hash']==task['task_hash'] and r['calls']['coupled']<=3
                proposed=np.asarray(original['sampling']['guidance'][-1]['raw_Z'])
                if stage=='smoothed_clipped':proposed=np.clip(smooth(proposed,task['condition']['initial_features']),lower,upper)
                assert digest(proposed)==r['proposal_Z_hash']
                v=r['validation'];goal='frozen_goal_position_error_m' in v and max(v['frozen_goal_position_error_m'])<=.005 and max(v['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
                assert r['eligible']==(all_pass(v['gates'],MODEL_GATES) and goal)
                for key,value in r['calls'].items():calls['diagnostic.'+key]=calls.get('diagnostic.'+key,0)+value
    assert count==720 and diagnostics==96
    assert all(ledger['calls'].get(k,0)==v for k,v in calls.items())
    assert steps==ledger['calls'].get('execution.physics_steps',0)
    return {'status':'PASS' if not problems and not ledger.get('unresolved_call_accounting') else 'BLOCKED',
        'problems':problems,'actual_arrays':12736,'TRAIN_rows':12688,'coverage_TRAIN_rows':544,
        'actual_training_draws':draws,'sensitivity_anchors_verified':anchors,
        'cached_anchors':11960 if anchors else 0,'new_anchors':512 if anchors else 0,
        'candidate_records':count,'proposal_diagnostics':diagnostics,'network_calls':network,
        'audited_execution_traces':len(traces),'physics_steps_including_replay':steps,
        'minimum_execution_clearance_m':min((r['minimum_clearance_m'] for r in traces),default=None),
        'maximum_hold_position_m':max((max(r['hold_max_position_error_m']) for r in traces),default=None),
        'sources_models_user_edits_branches_unchanged':True,'TEST_executed':False,
        'additional_audit_physics_network_training_calls':0,'full_goal_achieved':False}
