import zipfile,hashlib
import numpy as np
import torch
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from .inputs import identity
from .protocol import CONTROL,WARM,PROTOCOL as P
from . import endpoint,learning

def verify(out,ident,ledger):
    errors=[];gradients=0;new_candidates=0;new_execution=0;physics_steps=0
    try:
        assert identity()==ident and read_json(out/'run.json')['identity']==ident
        with zipfile.ZipFile(out/'executed_source.zip') as z:
            assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in ident['source_hashes'].items())
        assert ledger.data['inflight'] is None and not ledger.data.get('unresolved_call_accounting')
        diag=read_json(out/'single_step_summary.json');assert diag['cases']==24 and diag['updates_retained']==0
        from ff_prior_delivery_train.learning import load as warm_load
        from ff_prior_paired_objectives.objectives import predict
        from ff_prior.diffusion import weight_hash
        from .data import load as load_data
        model,codec=warm_load(WARM/'learning/goal_loss')
        arrays,rows,G,data=load_data(codec)
        records=[]
        for path in sorted(out.glob('diagnostic/*/result.json'))+sorted(out.glob('world_events/*/result.json')):
            event=read_json(path);records.append(event)
            before=torch.load(path.parent/'before.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(before['weights']);model.eval()
            assert weight_hash(model)==event['input_weight_hash']
            assert event['weighted_world_norm']<=.1*event['base_norm']+1e-8
            assert np.isfinite(event['base_norm']) and np.isfinite(event['world_norm'])
            if not event['updates_retained']:assert event['input_weight_hash']==event['output_weight_hash']
            for trial in event['trials']:
                if trial['accepted']:
                    from .optimization import admissible
                    assert all(a['domain'] and admissible(b,a) for b,a in zip(event['before'],trial['after']))
            for q in path.parent.glob('gradient_*.json'):
                pr=read_json(q);assert file_digest(q.with_suffix('.npz'))==pr['arrays_sha256']
                assert pr['dimensions']==102 and pr['calls']['coupled']==212
                with np.load(q.with_suffix('.npz')) as a:
                    x=a['x'];poses=a['perturbation_ee'];values=a['perturbation_losses'];g=a['gradient']
                    assert digest(x)==pr['x_hash']
                    i=data['ids'][pr['sample_id']]
                    with torch.no_grad():current=predict(model,data,i,pr['t'],pr['noise_seed']).numpy().astype(float)
                    assert np.array_equal(x,current) and pr['input_weight_hash']==weight_hash(model)
                    goals=rows[i]['task']['condition']['goals']
                    recalculated=np.array([[endpoint.loss_from_ee(v,goals)[0] for v in pair] for pair in poses])
                    assert np.allclose(recalculated,values,rtol=1e-12,atol=1e-12)
                    assert abs(endpoint.loss_from_ee(pr['central']['ee'],goals)[0]-pr['true_loss'])<1e-9
                    assert pr['base_hash']==digest(arrays[1][i]) and pr['codec_hash']==digest(codec.state)
                    # Goals are frozen by identity-selected TRAIN task and source record.
                    target=next(e for e in ident['selection']['diagnostic']+ident['selection']['events'] if e['sample_id']==pr['sample_id'])
                    assert target['task_hash']==pr['task_hash']
                    assert np.allclose((values[:,1]-values[:,0])/(2*pr['epsilon']),g,rtol=1e-12,atol=1e-12)
                    assert np.isfinite(poses).all() and np.isfinite(g).all()
                gradients+=1
        assert len(list(out.glob('diagnostic/*/result.json')))==24
        if diag['status']=='PASS':
            t=read_json(out/'learning/training.json');old=read_json(CONTROL/'learning/goal_loss/training.json')
            assert t['updates']==6000 and t['segments']==old['segments'] and t['actual_TRAIN_draws']==384000
            assert t['sampling']['train_draw_counts']==old['train_draw_counts'] and t['world_events']==60
            assert t['initial_weight_hash']==old['initial_weight_hash'] and t['VAL_updates']==0
            assert gradients==144 and len(list(out.glob('world_events/*/result.json')))==60
            learning.load(out/'learning')
            progress=torch.load(out/'learning/progress.pt',map_location='cpu',weights_only=False)
            assert progress['updates']==6000 and all(float(v['step'])==6000 for v in progress['optimizer']['state'].values())
            bare=list(out.glob('unguided/*/*/*.json'));assert len(bare)==192
            for q in bare:
                v=read_json(q);assert len(v['steps'])==16 and v['physics_guidance_calls']==0
                assert max(s['residual'] for s in v['steps'])<=1e-10 and v['calls']['coupled']<=3
            entries=read_json(CONTROL/'frozen_tasks.json')['entries']
            for sj,seed in enumerate(P['sampling_seeds']):
                for ti,e in enumerate(entries):
                    task=e['task'];sid=e['sample_id'];rs=[]
                    for k in range(5):
                        c=read_json(out/'candidates'/str(sj)/sid/f'{k:02d}.json');rs.append(c);new_candidates+=1
                        assert c['seed']==seed+ti*101+k*997 and c['task_hash']==task['task_hash']
                        assert c['model_calls']['coupled']<=113
                        if c['status']=='DONE':
                            check_candidate_validation(c,task)
                            assert c['actual_network_calls']==16 and len(c['sampling']['guidance'])==4
                            assert max(s['decomposition_residual_after'] for s in c['sampling']['steps'])<=1e-10
                            assert all(g['latent_feedback_max_rad']<=1e-12 and g['coupled_calls']<=cap for g,cap in zip(c['sampling']['guidance'],[27,27,27,29]))
                    chosen=min((c for c in rs if c['eligible']),key=lambda c:(c['score'],c['candidate_id']),default=None)
                    d=out/'executions'/str(sj)/sid;o=read_json(d/'outcome.json');sel=read_json(d/'selection.json');new_execution+=1
                    assert sel['selected_id']==(chosen['candidate_id'] if chosen else None)
                    if chosen and (d/'report.json').exists():
                        report,n=audit_physics(d,task,chosen['reference_hash']);physics_steps+=n;assert o['status']==report['level']
                    elif chosen:assert o['status']=='ERROR'
                    else:assert o['status']=='NO_CANDIDATE'
        else:
            assert gradients==24 and not (out/'learning/model.pt').exists() and not (out/'candidates').exists()
        assert read_json(out/'query_budget.json')['reserved_coupled_upper']<=54000
    except Exception as exc:
        import traceback
        errors.append({'reason':repr(exc),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'full102_gradients_checked':gradients,'new_candidates':new_candidates,
       'new_execution_units':new_execution,'new_native_and_replay_steps':physics_steps,'reused_candidates':540,'reused_execution_units':108,
       'reuse_is_evidence_not_new_success':True,'no_training_or_physics_repeated_by_audit':True,'TEST':False}
