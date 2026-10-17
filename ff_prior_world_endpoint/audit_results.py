"""Recompute admission, true-world losses/gradients, training draws and saved traces."""
from pathlib import Path
import hashlib,subprocess,zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.contracts import assert_frozen,all_pass,MODEL_GATES
from ff_prior.initial_state import STATE
from ff_prior.diffusion import weight_hash
from ff_prior.spline import Reference
from ff_prior.coupled_predictor import pose_errors
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_generalization.audit_results import check_candidate
from ff_prior_goal.projection import smooth
from ff_prior_origin_learn_bounded.distribution import population,CoverageSampler
from .protocol import ROOT,SOURCE,METHODS,PROTOCOL as P
from .endpoint import loss_from_ee
from .identity import sources
from . import data

def audit_data(out,targets,ledger):
    out=Path(out);problems=[];steps=0
    try:
        frozen=read_json(out/'fixed_TRAIN_targets.json')
        assert digest(frozen['entries'])==digest(targets) and frozen['before_generation'] and frozen['TRAIN_only']
        rs=[data.check_new_record(out,p) for p in sorted((out/'records').glob('*.json'))]
        for target in targets:
            references=[np.asarray(target['Z'])]
            for r in rs:
                if r['parent_sample_id']!=target['parent_sample_id']:continue
                assert r['task']==target['task'] and r['family_id']==target['family_id'] and not r['goal_relabelled']
                assert ledger['completed'][r['sample_id']]['status']=='ADMITTED'
                with np.load(out/r['arrays']) as a:z=a['Z'].copy()
                assert min(np.sqrt(np.mean((z-v)**2)) for v in references)>=.005
                references.append(z)
        selection=read_json(out/'data_audit_selection.json')
        assert selection['before_execution'] and not selection['replacement_after_failure']
        expected=[]
        for target in targets:
            group=[r for r in rs if r['parent_sample_id']==target['parent_sample_id']]
            if group:expected.append(min(group,key=lambda r:digest({'endpoint_witness':r['sample_id']}))['sample_id'])
        assert selection['ids']==expected
        for sid in expected:
            row=next(r for r in rs if r['sample_id']==sid)
            proof,n=audit_physics(out/'TRAIN_executions'/sid,row['task'],row['reference_hash'])
            steps+=n
            assert proof['level']=='EXECUTION_VERIFIED' and row['execution_status']=='PASS'
        assert steps==ledger['calls'].get('TRAIN_execution.physics_steps',0)
        assert len(expected)==16 and len(rs)>=32
    except Exception as error:
        import traceback
        problems.append({'reason':repr(error),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not problems else 'BLOCKED','problems':problems,
            'TRAIN_native_traces':len(expected) if 'expected' in locals() else 0,
            'physics_steps_including_replay':steps,'additional_physics_calls':0}

def verify_gradient(path,codec,base,center,row):
    proof=read_json(path);arrpath=Path(path).with_suffix('.npz')
    assert proof['status']=='PASS' and proof['arrays_sha256']==file_digest(arrpath)
    assert proof['task_hash']==row['task']['task_hash'] and proof['training_only']
    assert proof['dimensions']==102 and proof['gradient_recomputed_at_current_prediction']
    assert proof['codec_hash']==digest(codec.state) and proof['base_hash']==digest(base) and proof['center_hash']==digest(center)
    with np.load(arrpath) as a:
        x=a['x'].copy();g=a['gradient'].copy();losses=a['perturbation_losses'].copy();poses=a['perturbation_ee'].copy()
    assert x.shape==g.shape==(102,) and proof['x_hash']==digest(x)
    assert poses.shape==(102,2,2,4,4)
    goals=row['task']['condition']['goals'];value,_,_=loss_from_ee(proof['central']['ee'],goals)
    assert np.isclose(value,proof['true_loss'],rtol=1e-12,atol=1e-12)
    rebuilt=np.array([[loss_from_ee(v,goals)[0] for v in pair] for pair in poses])
    assert np.allclose(rebuilt,losses,rtol=1e-12,atol=1e-12)
    eps=P['endpoint_difference_normalized'];assert proof['epsilon']==eps
    assert np.array_equal(g,(losses[:,1]-losses[:,0])/(2*eps))
    ref=Reference(codec.decode(codec.coefficients(x,center),base),row['condition']['initial_features'],row['condition']['duration'])
    assert ref.reference_hash==proof['central']['reference_hash']
    for h in proof['half_checks']:
        v=[loss_from_ee(s['ee'],goals)[0] for s in h['samples']];slope=(v[1]-v[0])/eps
        err=abs(slope-g[h['dimension']])/max(abs(slope),abs(g[h['dimension']]),1e-6)
        assert h['passed'] and err<=P['endpoint_gradient_half_tolerance']
    probes=[loss_from_ee(s['ee'],goals)[0] for s in proof['direction_samples']]
    measured=(probes[1]-probes[0])/(2*eps);expected=g@np.asarray(proof['direction'])
    assert abs(measured-expected)/max(abs(measured),abs(expected),1e-6)<=P['endpoint_gradient_direction_tolerance']
    dp,dr=pose_errors(proof['central']['ee'],proof['convergence']['fine']['ee'])
    assert max(dp)<=.001 and max(dr)<=np.deg2rad(.2) and proof['convergence']['passed']
    assert proof['calls']['coupled']==212 and proof['calls']['physics_steps']==0
    return proof

def verify(out,identity,index,rows,codec,arrays):
    from .learning import load,CONTRACT
    from ff_prior_origin_learn_bounded.mean import fit as mean_fit
    out=Path(out);ledger=read_json(out/'ledger.json');problems=[]
    assert ledger['inflight'] is None and ledger['identity']=={'identity_hash':digest(identity)}
    assert sources()==identity['source_hashes']
    assert all(file_digest(ROOT/p)==h for p,h in identity['source_files'].items())
    with zipfile.ZipFile(out/'executed_source.zip') as z:
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in sources().items())
    protected=read_json(ROOT/'ff_prior_expand/evidence/preservation_before.json')
    for group in ('old_files','user_edits'):assert all(file_digest(ROOT/p)==h for p,h in protected[group].items())
    for branch,h in protected['branches'].items():
        assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==h
    targets=read_json(out/'fixed_TRAIN_targets.json')['entries'];assert digest(targets)==identity['targets_hash']
    dproof=audit_data(out,targets,ledger)
    if dproof['status']!='PASS':problems.append('FIXED_TRAIN_AUDIT_BLOCKED')
    assert read_json(out/'data_verification.json')==dproof
    assert read_json(out/'run.json')['acceptance']==read_json(SOURCE/'run.json')['acceptance']
    cache=read_json(out/'runtime_cache_audit.json')
    assert cache['peak_entries']<=64 and cache['peak_scene_models']<=4 and cache['peak_private_bytes']<=16*1024**3
    z,bases,conditions,ids,splits,states=arrays;old_index=read_json(SOURCE/'training_index.json')['records']
    assert index[:len(old_index)]==old_index and index==read_json(out/'training_index.json')['records']
    proof=read_json(out/'training_input_audit.json')
    assert proof['actual_array_count']==len(index) and proof['reused_verified_initializers']==len(old_index)
    assert proof['parent_loaded_arrays_sha256']==file_digest(SOURCE/'loaded_training_arrays.npz')
    assert proof['parent_input_audit_sha256']==file_digest(SOURCE/'training_input_audit.json') and not proof['TEST_arrays_read']
    with np.load(SOURCE/'loaded_training_arrays.npz') as a:
        assert np.array_equal(a['Z'],z[:len(old_index)]) and np.array_equal(a['bases'],bases[:len(old_index)])
    for i,e in enumerate(index):
        assert all(file_digest(ROOT/e[k])==e[k+'_sha256'] for k in ('record','arrays'))
        with np.load(ROOT/e['arrays']) as a:assert np.array_equal(a['Z'],z[i])
        assert rows[i]==read_json(ROOT/e['record']) and states[i]==rows[i]['task']['contract']['initial_state_id']
    tids=[sid for sid,s in zip(ids,splits) if s=='TRAIN'];newids=[e['sample_id'] for e in index[len(old_index):]]
    assert len(tids)==13774+len(newids) and len(ids)-len(tids)==48
    binding=read_json(out/'checkpoint_binding.json')
    assert binding['old']==file_digest(SOURCE/'learning/goal_loss/model.pt') and binding['fixed_codec']==digest(codec.state)
    mean=mean_fit(out/'learning',arrays,codec,newids)
    assert binding['mean_codec_sha256']==file_digest(out/'learning/weighted_mean.json')
    trained=False;world_events=0;draws=0
    if (out/'learning/goal_loss/training.json').exists():
        tr=read_json(out/'learning/goal_loss/training.json');model,loaded=load(out/'learning/goal_loss')
        assert tr['status']=='PASS' and tr['updates']==6000 and tr['metadata']['contract']==CONTRACT
        assert digest(loaded.state)==digest(codec.state)
        assert binding['new']==file_digest(out/'learning/goal_loss/model.pt')
        assert tr['trained_weight_hash']==weight_hash(model)!=tr['initial_weight_hash']
        assert tr['metadata']['training_ids']==tids and set(tr['train_draw_counts'])==set(tids)
        assert tr['metadata']['data_hash']==digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states})
        slots,keys,weights,pop=population(conditions,ids,splits);sampler=CoverageSampler(keys,P['training_seed']+2)
        for _ in range(6000):sampler.take(64)
        assert tr['sampling_population']==pop and all(tr[k]==v for k,v in sampler.report(tids).items())
        assert np.array_equal(sampler.row_counts+sampler.origin_row_counts,[tr['train_draw_counts'][s] for s in tids])
        assert all(v>0 for v in tr['train_draw_counts'].values());draws=sum(tr['train_draw_counts'].values())
        assert draws==384000 and tr['VAL_gradient_updates']==0 and not tr['VAL_checkpoint_selection']
        mapping={sid:i for i,sid in enumerate(ids)}
        for e in tr['world_records']:
            path=out/'world_gradients'/f'{e["event"]:03d}.json';i=mapping[e['sample_id']]
            gproof=verify_gradient(path,codec,bases[i],codec.center(conditions[i],bases[i]),rows[i])
            assert splits[i]=='TRAIN' and file_digest(path)==e['proof_sha256']
            assert gproof['input_weight_hash']==e['input_weight_hash']!=e['output_weight_hash']
            assert gproof['update']==e['update']==50*(e['event']+1)
            assert ledger['completed'][f'world_gradient_{e["event"]:03d}']['status']=='PASS'
            world_events+=1
        assert world_events==120==tr['world_gradient_updates']
        assert ledger['calls']['world_gradient.coupled']==120*212
        a=read_json(out/'learning/goal_loss/world_diagnostic_before.json')
        b=read_json(out/'learning/goal_loss/world_diagnostic_after.json');ratios=[]
        assert len(a['cases'])==len(b['cases'])==32
        assert a['model_hash']==tr['initial_weight_hash'] and b['model_hash']==tr['trained_weight_hash']
        for x,y in zip(a['cases'],b['cases']):
            assert (x['sample_id'],x['t'],x['seed'])==(y['sample_id'],y['t'],y['seed'])
            row=rows[mapping[x['sample_id']]]
            for v in (x,y):assert np.isclose(v['loss'],loss_from_ee(v['ee'],row['condition']['goals'])[0])
            ratios.append(y['loss']/max(x['loss'],1e-12))
        assert float(np.median(ratios))==tr['world_alignment']['median_paired_loss_ratio']
        trained=True
    else:problems.append('ACTUAL_WORLD_TRAINING_NOT_COMPLETED')
    smoke=verify_smoke(out,identity,ledger,problems)
    return dict(smoke,status='PASS' if not problems and not ledger.get('unresolved_call_accounting') else 'BLOCKED',
                problems=problems,TRAIN_rows=len(tids),actual_arrays=len(ids),actual_training_draws=draws,
                actual_training_verified=trained,actual_full102_endpoint_gradients=world_events,
                TRAIN_native_traces=dproof['TRAIN_native_traces'],
                TRAIN_physics_steps=dproof['physics_steps_including_replay'],
                sources_models_user_edits_branches_unchanged=True,TEST_executed=False,
                additional_audit_physics_training_calls=0,full_goal_achieved=False)


def verify_smoke(out,identity,ledger,problems):
    frozen=read_json(out/'frozen_tasks.json');entries=frozen['entries']
    assert len(entries)==12 and frozen['before_training'] and frozen['before_comparison']
    assert digest(entries)==identity['comparison_entries_hash']
    assert entries==read_json(SOURCE/'frozen_tasks.json')['entries']
    olddev=read_json(SOURCE/'frozen_tasks.json')['entries'][:4]
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
    return {'candidate_records':count,'proposal_diagnostics':diagnostics,'network_calls':network,
            'audited_execution_traces':len(traces),'physics_steps_including_replay':steps,
            'minimum_execution_clearance_m':min((r['minimum_clearance_m'] for r in traces),default=None),
            'maximum_hold_position_m':max((max(r['hold_max_position_error_m']) for r in traces),default=None)}
