from pathlib import Path
import time
import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.diffusion import weight_hash,save_torch
from ff_prior_latent.learning import Codec,LatentDenoiser
from ff_prior_origin_learn_bounded.learning import load as old_load
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_coverage_train.storage import atomic_json,atomic_npz
from ff_prior_world_endpoint import endpoint
from .protocol import ARMS,WARM,PROTOCOL as P
from .objectives import random_batch,terms,predict,vector_gradient,gradient_statistics

def load(directory):
    directory=Path(directory);r=read_json(directory/'training.json')
    if file_digest(directory/'model.pt')!=r['checkpoint_sha256']:raise ValueError('checkpoint changed')
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    if saved['metadata']!=r['metadata']:raise ValueError('checkpoint metadata mismatch')
    codec=Codec(read_json(directory/'codec.json'))
    if digest(codec.state)!=r['metadata']['codec_hash']:raise ValueError('codec changed')
    model=LatentDenoiser(codec.dim,len(codec.feature_mean));model.load_state_dict(saved['weights']);model.eval()
    if weight_hash(model)!=r['trained_weight_hash']:raise ValueError('weights changed')
    return model,codec

def probes(out,name,model,codec,data,arrays,rows,cache,ledger,deadline):
    path=out/'probes'/f'{name}.json';key='TRAIN_probe_'+name
    if key in ledger.data['completed']:
        r=read_json(path)
        if r['weight_hash']!=weight_hash(model):raise ValueError('probe checkpoint changed')
        return r
    ledger.begin(key,'TRAIN_probes',P['wall_budget_s'],5);results=[];calls={}
    for j,group in enumerate(data['groups']):
        i=group[0];rr,ii,_,_=cache.get(rows[i]);before=rr.calls.copy()
        for k,t in enumerate(P['world_noise_times']):
            seed=P['diagnostic_noise_seed']+j*2+k
            with torch.no_grad():xx=predict(model,data,i,t,seed).numpy().astype(float)
            f,r=endpoint.evaluate(xx,codec,data['centers'][i],arrays[1][i],rows[i]['task'],rr,ii,deadline,512)
            results.append(dict(r,loss=f,sample_id=arrays[3][i],t=t,seed=seed,x_hash=digest(xx)))
        for k in before:calls[k]=calls.get(k,0)+rr.calls[k]-before[k]
    result={'weight_hash':weight_hash(model),'cases':results,'TRAIN_only':True,'calls':calls}
    atomic_json(path,result);ledger.finish({'status':'DONE'},calls);return result

def select(probe_results):
    before=np.asarray([r['loss'] for r in probe_results['warm']['cases']]);metrics={}
    local=np.asarray([r['loss'] for r in probe_results['B_local']['cases']])
    identities=[(r['sample_id'],r['t'],r['seed']) for r in probe_results['warm']['cases']]
    for arm in ARMS:
        if [(r['sample_id'],r['t'],r['seed']) for r in probe_results[arm]['cases']]!=identities:raise ValueError('unpaired probes')
        after=np.asarray([r['loss'] for r in probe_results[arm]['cases']])
        a=float(np.median(after/np.maximum(before,1e-12)));b=float(np.median(after/np.maximum(local,1e-12)))
        metrics[arm]={'median_ratio_vs_warm':a,'median_ratio_vs_local_control':b,'improved_cases':int(sum(after<before)),
                      'eligible':bool(arm in ('C_world','D_both') and a<=.8 and b<=1.)}
    eligible=[a for a in ARMS if metrics[a]['eligible']]
    chosen=min(eligible,key=lambda a:(metrics[a]['median_ratio_vs_warm'],a)) if eligible else None
    return {'status':'PASS' if chosen else 'BLOCKED','selected_arm':chosen,'metrics':metrics,
        'selection_data':'TRAIN only; fixed final6000 checkpoints','before_DEV':True,'TEST_allowed':False}

def train(out,codec,data,arrays,rows,G,cache,ledger,deadline,progress):
    (out/'learning').mkdir(parents=True,exist_ok=True)
    decision=out/'TRAIN_selection.json'
    if decision.exists():
        for arm in ARMS:load(out/'learning'/arm)
        return read_json(decision)
    models={a:old_load(WARM/'learning/goal_loss')[0] for a in ARMS}
    initial=weight_hash(models['A_v'])
    if any(weight_hash(m)!=initial for m in models.values()):raise ValueError('unpaired weights')
    optimizers={a:torch.optim.AdamW(m.parameters(),lr=P['lr']) for a,m in models.items()}
    sampler=CoverageSampler(data['keys'],P['training_seed']+2)
    ids=[arrays[3][int(i)] for i in data['slots']]
    metadata={'protocol':P,'initial_weight_hash':initial,'codec_hash':digest(codec.state),
        'warm_checkpoint_sha256':file_digest(WARM/'learning/goal_loss/model.pt'),
        'input_proof_sha256':file_digest(out/'input_verification.json'),'training_ids':ids}
    checkpoint=out/'learning/progress.pt';start=0;records={a:[] for a in ARMS};sequence=[];history={a:[] for a in ARMS}
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['metadata']!=metadata:raise ValueError('resume metadata changed')
        for a in ARMS:models[a].load_state_dict(saved['weights'][a]);optimizers[a].load_state_dict(saved['optimizers'][a])
        sampler=CoverageSampler(data['keys'],P['training_seed']+2,saved['sampler'])
        start=saved['updates'];records=saved['records'];sequence=saved['sequence'];history=saved['history']
    if start==0:probes(out,'warm',models['A_v'],codec,data,arrays,rows,cache,ledger,deadline)
    for step in range(start,P['training_updates']):
        if time.time()>=deadline:raise RuntimeError('PAIRED_TRAINING_WALL_EXHAUSTED')
        if step%50==0:
            cache.guard()
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=P['max_output_bytes']:
                raise RuntimeError('PAIRED_STORAGE_BUDGET_EXHAUSTED')
            progress('paired_four_arm_training',update=step+1,total=6000)
        batch=sampler.take(P['batch']);slots=data['slots'][batch]
        t,noise=random_batch(step,P['batch'],102)
        seq={'update':step+1,'batch':batch,'t':t.numpy(),'noise_hash':digest(noise.numpy())}
        batch_hash=digest(seq);sequence.append(dict(seq,arms={a:batch_hash for a in ARMS}))
        for arm,(lw,ww) in ARMS.items():
            model=models[arm];optimizer=optimizers[arm];model.train()
            v,l,_=terms(model,data['x'][slots],data['c'][slots],t,noise,data['alpha'],G[batch]);loss=v+lw*l
            event_record=None
            if ww and (step+1)%P['world_every_updates']==0:
                event=(step+1)//P['world_every_updates']-1;j=event%16
                i=data['groups'][j][(event//16)%4];tt=P['world_noise_times'][(event//16)%2];seed=P['world_noise_seed']+event
                pred=predict(model,data,i,tt,seed);xx=pred.detach().numpy().astype(float);input_hash=weight_hash(model)
                key=f'{arm}_gradient_{event:03d}';path=out/'world_gradients'/arm/f'{event:03d}.json'
                if key not in ledger.data['completed']:
                    ledger.begin(key,'world_gradient',P['wall_budget_s'],240)
                    progress('paired_true_world_gradient',arm=arm,update=step+1,event=event+1,total=120)
                    rr,ii,_,_=cache.get(rows[i])
                    path.parent.mkdir(parents=True,exist_ok=True)
                    save_torch(path.with_suffix('.pt'),{'weights':model.state_dict(),'weight_hash':input_hash,
                        'sample_id':arrays[3][i],'t':tt,'seed':seed,'update':step+1,'codec_hash':digest(codec.state)})
                    value,g,proof,arr=endpoint.gradient(xx,codec,data['centers'][i],arrays[1][i],rows[i]['task'],rr,ii,deadline)
                    atomic_npz(path.with_suffix('.npz'),**arr)
                    proof.update(sample_id=arrays[3][i],update=step+1,event=event,noise_t=tt,noise_seed=seed,
                        input_weight_hash=input_hash,arrays_sha256=file_digest(path.with_suffix('.npz')),
                        input_checkpoint_sha256=file_digest(path.with_suffix('.pt')))
                    atomic_json(path,proof);ledger.finish({'status':proof['status']},proof['calls'])
                else:
                    proof=read_json(path);value=proof['true_loss']
                    with np.load(path.with_suffix('.npz')) as a:g=a['gradient'].copy()
                if proof['status']!='PASS' or proof['x_hash']!=digest(xx) or proof['input_weight_hash']!=input_hash:raise ValueError('stale/invalid world gradient')
                world=endpoint.attach_gradient(pred,value,g);loss=loss+ww*world
                vectors={name:vector_gradient(term,model) for name,term in [('v',v),('local',lw*l),('world',ww*world)]}
                event_record={'event':event,'update':step+1,'sample_id':arrays[3][i],'noise_t':tt,'noise_seed':seed,
                    'input_weight_hash':input_hash,'true_world_loss':value,'components':gradient_statistics(vectors),
                    'proof_sha256':file_digest(path)}
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite parameter gradient')
            optimizer.step()
            if event_record:
                event_record.update(parameter_gradient_norm=float(norm),output_weight_hash=weight_hash(model))
                records[arm].append(event_record)
            if (step+1)%100==0:history[arm].append({'update':step+1,'v':float(v.detach()),'local':float(l.detach()),'total':float(loss.detach()),'gradient_norm':float(norm)})
        if (step+1)%50==0:
            save_torch(checkpoint,{'metadata':metadata,'weights':{a:m.state_dict() for a,m in models.items()},
                'optimizers':{a:o.state_dict() for a,o in optimizers.items()},'sampler':sampler.state(),
                'updates':step+1,'records':records,'sequence':sequence,'history':history})
    atomic_json(out/'paired_random_sequence.json',sequence)
    sampling=sampler.report(ids);counts=sampler.row_counts+sampler.origin_row_counts
    probe_results={'warm':read_json(out/'probes/warm.json')}
    for arm,model in models.items():
        path=out/'learning'/arm;atomic_json(path/'codec.json',codec.state)
        meta=dict(metadata,arm=arm,loss_weights=ARMS[arm]);save_torch(path/'model.pt',{'metadata':meta,'weights':model.state_dict()})
        after=probes(out,arm,model,codec,data,arrays,rows,cache,ledger,deadline);probe_results[arm]=after
        before=probe_results['warm']['cases'];ratios=[b['loss']/max(a['loss'],1e-12) for a,b in zip(before,after['cases'])]
        result={'status':'PASS' if all(counts>0) and weight_hash(model)!=initial else 'FAIL',
            'metadata':meta,'updates':6000,'initial_weight_hash':initial,'trained_weight_hash':weight_hash(model),
            'checkpoint_sha256':file_digest(path/'model.pt'),'world_gradient_updates':len(records[arm]),
            'world_records':records[arm],'history':history[arm],'train_draw_counts':dict(zip(ids,counts.tolist())),
            'all_TRAIN_rows_seen':bool(all(counts>0)),'random_sequence_sha256':file_digest(out/'paired_random_sequence.json'),
            'actual_consumed_sequence_hashes':[s['arms'][arm] for s in sequence],
            'world_alignment':{'median_paired_loss_ratio':float(np.median(ratios)),'pre_registered_improvement':bool(np.median(ratios)<=.8)},
            'VAL_gradient_updates':0,'VAL_checkpoint_selection':False,**sampling}
        atomic_json(path/'training.json',result);load(path)
    result=select(probe_results)
    if any(read_json(out/'learning'/a/'training.json')['status']!='PASS' for a in ARMS):
        result.update(status='BLOCKED',selected_arm=None,reason='actual training incomplete')
    atomic_json(decision,result);return result
