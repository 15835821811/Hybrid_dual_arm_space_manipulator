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
            components=[];record_gradient_status=[]
            for q in sorted(path.parent.glob('gradient_*.json')):
                pr=read_json(q);assert file_digest(q.with_suffix('.npz'))==pr['arrays_sha256']
                assert pr['dimensions']==102 and pr['calls']['coupled']==212
                with np.load(q.with_suffix('.npz')) as a:
                    x=a['x'];poses=a['perturbation_ee'];values=a['perturbation_losses'];g=a['gradient']
                    assert digest(x)==pr['x_hash']
                    i=data['ids'][pr['sample_id']]
                    with torch.no_grad():current=predict(model,data,i,pr['t'],pr['noise_seed']).numpy().astype(float)
                    assert np.array_equal(x,current) and pr['input_weight_hash']==weight_hash(model)
                    from ff_prior_paired_objectives.objectives import vector_gradient
                    prediction=predict(model,data,i,pr['t'],pr['noise_seed'])
                    components.append(vector_gradient(endpoint.attach_gradient(prediction,pr['true_loss'],g),model).detach())
                    goals=rows[i]['task']['condition']['goals']
                    recalculated=np.array([[endpoint.loss_from_ee(v,goals)[0] for v in pair] for pair in poses])
                    assert np.allclose(recalculated,values,rtol=1e-12,atol=1e-12)
                    assert abs(endpoint.loss_from_ee(pr['central']['ee'],goals)[0]-pr['true_loss'])<1e-9
                    assert pr['base_hash']==digest(arrays[1][i]) and pr['codec_hash']==digest(codec.state)
                    # Goals are frozen by identity-selected TRAIN task and source record.
                    target=next(e for e in ident['selection']['diagnostic']+ident['selection']['events'] if e['sample_id']==pr['sample_id'])
                    assert target['task_hash']==pr['task_hash']
                    # Independently reconstruct the group Jacobian and analytic chain,
                    # rather than asserting the old scalar finite-difference formula.
                    from ff_prior.coupled_predictor import pose_errors
                    def group_q(ee):
                        pos,rot=pose_errors(ee,goals)
                        return np.r_[(pos/.005)**2,(rot/np.deg2rad(1))**2]
                    cq=group_q(pr['central']['ee'])
                    pq=np.array([[group_q(ee) for ee in pair] for pair in poses])
                    jac=(pq[:,1]-pq[:,0])/(2*pr['epsilon'])
                    ex=np.exp((cq-cq.max())/P['smooth_temperature']);weights=ex/ex.sum()+.0125
                    assert np.array_equal(a['central_groups'],cq) and np.array_equal(a['perturbation_groups'],pq)
                    assert np.array_equal(a['group_jacobian'],jac) and np.array_equal(a['analytic_smooth_weights'],weights)
                    assert np.allclose(jac@weights,g,rtol=1e-12,atol=1e-12)
                    assert np.allclose((values[:,1]-values[:,0])/(2*pr['epsilon']),a['legacy_scalar_gradient'],rtol=1e-12,atol=1e-12)
                    half_pass=[]
                    for h in pr['half_checks']:
                        hh=[endpoint.loss_from_ee(v['ee'],goals)[0] for v in h['samples']]
                        slope=(hh[1]-hh[0])/pr['epsilon'];expected=g[h['dimension']]
                        err=abs(slope-expected)/max(abs(slope),abs(expected),1e-6)
                        assert np.isclose(err,h['relative_error'],rtol=1e-10,atol=1e-12)
                        half_pass.append(err<=P['endpoint_gradient_half_tolerance'])
                        assert bool(h['passed'])==bool(half_pass[-1])
                    dd=[endpoint.loss_from_ee(v['ee'],goals)[0] for v in pr['direction_samples']]
                    direction=np.sin(np.arange(102)+.37);direction/=np.linalg.norm(direction)
                    assert np.array_equal(np.asarray(pr['direction']),direction)
                    slope=(dd[1]-dd[0])/(2*pr['epsilon']);expected=float(g@direction)
                    err=abs(slope-expected)/max(abs(slope),abs(expected),1e-6)
                    assert np.isclose(err,pr['direction_relative_error'],rtol=1e-10,atol=1e-12)
                    dp,dr=pose_errors(pr['central']['ee'],pr['convergence']['fine']['ee'])
                    converged=bool(max(dp)<=.001 and max(dr)<=np.deg2rad(.2))
                    passed=bool(np.isfinite(g).all() and converged and all(half_pass) and err<=P['endpoint_gradient_direction_tolerance'])
                    assert pr['status']==('PASS' if passed else 'BLOCKED')
                    record_gradient_status.append(pr['status'])
                    assert np.isfinite(poses).all() and np.isfinite(g).all()
                gradients+=1
            assert event['gradient_status']==('PASS' if all(s=='PASS' for s in record_gradient_status) else 'BLOCKED')
            from .state_audit import reconstruct_adamw,state_close
            from .optimization import optimizer_hash
            from ff_prior_paired_objectives.objectives import get_vector,set_vector
            original=get_vector(model)
            assert event['world_gradient_roles']==['event','control'] and len(components)==2
            with np.load(path.parent/'parameter_proposals.npz') as proposal:
                world=torch.stack(components).mean(0)
                assert np.array_equal(world.numpy(),proposal['world_gradient'])
                assert np.array_equal(torch.stack(components).numpy(),proposal['world_component_gradients'])
                batch=before['base_batch']
                if batch is None:
                    losses=[]
                    for i,t,seed in before['specs']+before['controls']:
                        noise=torch.randn((1,102),generator=torch.Generator().manual_seed(seed))
                        losses.append(learning.base_loss(model,data,G,[i],torch.tensor([t]),noise))
                    loss=torch.stack(losses).mean()
                else:loss=learning.base_loss(model,data,G,batch['indices'],batch['t'],batch['noise'])
                base=vector_gradient(loss,model).detach()
                assert np.array_equal(base.numpy(),proposal['base_gradient'])
                bv,bs=reconstruct_adamw(model,base,before['optimizer'])
                mv,ms=reconstruct_adamw(model,base+event['world_coefficient']*world,before['optimizer'])
                assert np.allclose((bv-original).numpy(),proposal['base_delta'],rtol=0,atol=6e-8)
                assert np.allclose((mv-original).numpy(),proposal['combined_delta'],rtol=0,atol=6e-8)
                assert event['optimizer_before_hash']==optimizer_hash(before['optimizer'])
                after=torch.load(path.parent/'after.pt',map_location='cpu',weights_only=False)
                assert file_digest(path.parent/'after.pt')==event['after_sha256']
                assert event['optimizer_after_hash']==optimizer_hash(after['optimizer'])
                expected_state=before['optimizer'] if not event['updates_retained'] else ms if event['accepted'] else bs
                state_close(after['optimizer'],expected_state)
                if not event['updates_retained']:assert optimizer_hash(after['optimizer'])==optimizer_hash(before['optimizer'])
                expected_vector=original if not event['updates_retained'] else original+event['fraction']*torch.tensor(proposal['combined_delta']) if event['accepted'] else original+torch.tensor(proposal['base_delta'])
                model.load_state_dict(after['weights']);assert torch.allclose(get_vector(model),expected_vector,rtol=0,atol=6e-8)
                assert weight_hash(model)==event['output_weight_hash']
                model.load_state_dict(before['weights'])
                assert [t['fraction'] for t in event['base_trials']]==[t['fraction'] for t in event['trials']]
                # Check saved actual clean estimates and world error arithmetic for both arms, without propagation.
                for name,delta in [('trials',proposal['combined_delta']),('base_trials',proposal['base_delta'])]:
                    for trial in event[name]:
                        set_vector(model,original+trial['fraction']*torch.tensor(delta))
                        for spec,record in zip(before['specs']+before['controls'],trial['after']):
                            i,t,seed=spec
                            with torch.no_grad():x=predict(model,data,i,t,seed).numpy().astype(float)
                            assert digest(x)==record['x_hash']
                            value,pos,rot=endpoint.loss_from_ee(record['ee'],rows[i]['task']['condition']['goals'])
                            assert abs(value-record['loss'])<1e-9 and np.allclose(pos,record['position_m']) and np.allclose(rot,record['rotation_rad'])
                set_vector(model,original)
                assert event['actual_coupled_calls']==424+len(event['before'])+sum(len(t['after']) for t in event['trials']+event['base_trials'])
        assert len(list(out.glob('diagnostic/*/result.json')))==24
        diagnostic_records=records[:24]
        assert diag['accepted']==sum(v['accepted'] for v in diagnostic_records)
        for t in P['noise_times']:
            assert diag['accepted_per_noise'][str(t)]==sum(v['accepted'] for v,c in zip(diagnostic_records,ident['selection']['cases']) if c['t']==t)
        expected_gate=(diag['accepted']>=P['diagnostic_minimum'] and min(diag['accepted_per_noise'].values())>=P['diagnostic_minimum_per_noise'] and all(v['gradient_status']=='PASS' for v in diagnostic_records))
        assert diag['status']==('PASS' if expected_gate else 'BLOCKED')
        if diag['status']=='PASS':
            t=read_json(out/'learning/training.json');old=read_json(CONTROL/'learning/goal_loss/training.json')
            assert t['updates']==6000 and t['segments']==old['segments'] and t['actual_TRAIN_draws']==384000
            assert t['sampling']['train_draw_counts']==old['train_draw_counts'] and t['world_events']==60
            assert t['initial_weight_hash']==old['initial_weight_hash'] and t['VAL_updates']==0
            assert gradients==168 and len(list(out.glob('world_events/*/result.json')))==60
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
            assert gradients==48 and not (out/'learning/model.pt').exists() and not (out/'candidates').exists()
        assert read_json(out/'query_budget.json')['reserved_coupled_upper']<=P['total_queries']
    except Exception as exc:
        import traceback
        errors.append({'reason':repr(exc),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'full102_gradients_checked':gradients,'new_candidates':new_candidates,
       'new_execution_units':new_execution,'new_native_and_replay_steps':physics_steps,'reused_candidates':540,'reused_execution_units':108,
       'reuse_is_evidence_not_new_success':True,'no_training_or_physics_repeated_by_audit':True,'TEST':False}
