"""Audit actual ingestion/updates, raw candidates, and native execution traces."""
import hashlib
import zipfile
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.contracts import assert_frozen
from ff_prior_origin_learn_bounded.distribution import population,CoverageSampler
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from .protocol import ROOT,SOURCE,OLD,CACHE,CACHE_HASHES,METHODS,PROTOCOL
from .inputs import identity_for
from . import learning


def verify(out,identity,arrays,rows,G,ledger):
    errors=[];candidate_count=executions=physics_steps=0;training_status='BLOCKED'
    try:
        expected,(_,index,entries,new_ids)=identity_for()
        assert expected==identity and read_json(out/'run.json')['identity']==identity
        assert read_json(out/'frozen_tasks.json')['entries']==entries==read_json(OLD/'frozen_tasks.json')['entries']
        with zipfile.ZipFile(out/'executed_source.zip') as archive:
            assert all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in identity['source_hashes'].items())
        assert ledger.data['inflight'] is None and not ledger.data.get('unresolved_call_accounting')
        z,bases,conditions,ids,splits,states=arrays
        assert len(ids)==14014+len(new_ids) and splits.count('TRAIN')==13966+len(new_ids) and splits.count('VAL')==48
        assert read_json(out/'training_input_audit.json')['records']==index
        for e in index:
            assert file_digest(ROOT/e['record'])==e['record_sha256']
        for name,h in CACHE_HASHES.items():assert file_digest(CACHE/name)==h
        with np.load(out/'loaded_training_arrays.npz') as a:
            assert np.array_equal(a['Z'],z) and np.array_equal(a['bases'],bases)
        sreport=read_json(out/'sensitivity_summary.json')
        assert sreport['status']=='PASS' and sreport['TRAIN_only']
        for b in sreport['bindings'].values():
            assert file_digest(ROOT/b['proof'])==b['proof_sha256'] and file_digest(ROOT/b['array'])==b['array_sha256']
        trained=out/'learning/goal_loss/training.json'
        training_proof={'status':'BLOCKED','reason':'no complete training checkpoint'}
        if trained.exists() and read_json(trained)['status']=='PASS':
            t=read_json(trained);model,codec=learning.load(out/'learning/goal_loss')
            meta=t['metadata'];train_ids=[sid for sid,s in zip(ids,splits) if s=='TRAIN']
            assert meta['training_ids']==train_ids and set(meta['new_E256_ids'])==set(new_ids)
            assert meta['validation_ids']==[sid for sid,s in zip(ids,splits) if s=='VAL']
            assert meta['data_hash']==digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states})
            assert meta['sensitivity_summary_hash']==digest(sreport) and meta['sensitivity_matrix_hash']==digest(G)
            assert meta['warm_checkpoint_sha256']==file_digest(OLD/'learning/goal_loss/model.pt')
            assert t['initial_weight_hash']==read_json(OLD/'learning/goal_loss/training.json')['trained_weight_hash']
            assert t['trained_weight_hash']!=t['initial_weight_hash']
            assert t['updates']==6000 and t['actual_TRAIN_draws']==384000
            assert t['VAL_gradient_updates']==0 and not t['VAL_checkpoint_selection']
            assert set(t['train_draw_counts'])==set(train_ids) and all(v>0 for v in t['train_draw_counts'].values())
            assert sum(t['train_draw_counts'].values())==384000
            assert [(s['start_update'],s['end_update'],s['draws']) for s in t['segments']]==[(i+1,i+20,1280) for i in range(0,6000,20)]
            assert read_json(out/'learning/goal_loss/training_journal.json')=={'status':'COMMITTED','committed_updates':6000}
            _,keys,_,_=population(conditions,ids,splits)
            sampler=CoverageSampler(keys,learning.CONTRACT['seed']+2)
            for _ in range(6000):sampler.take(64)
            sampled=sampler.report(train_ids)
            assert sampled['sampler_state_hash']==t['sampler_state_hash']
            reconstructed=sampler.row_counts+sampler.origin_row_counts
            assert dict(zip(train_ids,reconstructed.tolist()))==t['train_draw_counts']
            assert digest(codec.state)==digest(read_json(OLD/'learning/goal_loss/codec.json'))
            training_status='PASS'
            training_proof={'status':'PASS','actual_updates':6000,'TRAIN_rows_seen':len(train_ids),'draws':384000,
                            'weight_reload':True,'warm_weights_exact':True,'row_consumption_reconstructed':True,
                            'gradient_training_rerun':False,'G_reused':sreport['reused'],'G_new':sreport['new'],
                            'G_new_coupled_queries':sreport['new_coupled_queries'],'VAL_updates':0}
        atomic_json(out/'training_verification.json',training_proof)
        for sj,seed in enumerate(PROTOCOL['sampling_seeds']):
            for ti,e in enumerate(entries):
                task=e['task'];sid=e['sample_id'];assert_frozen(task)
                for method in METHODS:
                    records=[]
                    for k in range(5):
                        r=read_json(out/'candidates'/str(sj)/sid/method/f'{k:02d}.json');records.append(r);candidate_count+=1
                        assert r['method']==method and r['candidate_id']==k and r['seed']==seed+ti*101+k*997
                        assert r['task_hash']==task['task_hash'] and not r['private_witness_used']
                        assert r['sampling_method']==('late4_diffusion' if method.endswith('diffusion') else method)
                        if r['status']=='BLOCKED':
                            assert method=='expanded_diffusion' and training_status!='PASS' and not r['eligible'];continue
                        assert r['model_calls']['coupled']==r['repair_coupled_calls']+r['validation_coupled_calls']<=113
                        assert 0<=r['repair_coupled_calls']<=110 and 0<=r['validation_coupled_calls']<=3
                        if r['status']!='DONE':assert not r['eligible'];continue
                        assert r['wall_s']<=15. and r['integration_intervals']<=15872
                        check_candidate_validation(r,task)
                        n=16 if method.endswith('diffusion') else 0
                        assert r['actual_network_calls']==r['sampling']['actual_network_calls']==n
                        if n:
                            assert len(r['sampling']['steps'])==16
                            assert all(s['decomposition_residual_after']<=1e-10 for s in r['sampling']['steps'])
                            guides=r['sampling']['guidance']
                            assert [g['iteration'] for g in guides]==[12,13,14,15]
                            assert all(g['coupled_calls']<=cap and g['latent_feedback_max_rad']<=1e-12 for g,cap in zip(guides,[27,27,27,29]))
                            assert sum(g['coupled_calls'] for g in guides)==r['repair_coupled_calls']
                    chosen=min((r for r in records if r['eligible']),key=lambda r:(r['score'],r['candidate_id']),default=None)
                    directory=out/'executions'/str(sj)/sid/method
                    selection=read_json(directory/'selection.json');outcome=read_json(directory/'outcome.json');executions+=1
                    assert selection['task_hash']==task['task_hash'] and selection['selection_before_execution']
                    assert selection['selected_id']==(chosen['candidate_id'] if chosen else None)
                    if chosen:
                        if (directory/'report.json').exists():
                            p,steps=audit_physics(directory,task,chosen['reference_hash']);physics_steps+=steps
                            assert outcome['status']==p['level']
                        else:assert outcome['status']=='ERROR'
                    else:assert outcome['status']=='NO_CANDIDATE'
        assert candidate_count==720 and executions==144
    except Exception as error:
        import traceback
        errors.append({'reason':repr(error),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'training_status':training_status,
            'candidate_records_checked':candidate_count,'execution_units_checked':executions,
            'native_plus_replay_steps':physics_steps,'TEST_read':False,
            'scope':'actual input/update/checkpoint accounting and raw native trace audit; no gradient or physics rerun'}
