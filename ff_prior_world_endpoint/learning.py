"""Actual v training with current-prediction nonlinear world endpoint gradients."""
from pathlib import Path
import time
import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.diffusion import schedule,prediction_to_clean_noise,weight_hash,save_torch
from ff_prior_latent.learning import Codec,LatentDenoiser,sample
from ff_prior_origin_learn_bounded.learning import load as parent_load
from ff_prior_origin_learn_bounded.distribution import population,CoverageSampler
from ff_prior_coverage_train.storage import atomic_json,atomic_npz
from . import endpoint
from .protocol import ROOT,SOURCE,PROTOCOL as P
CONTRACT={'version':'full102_nonlinear_endpoint_v1','updates':6000,'batch':64,'seed':20262301,
          'lr':2e-4,'world_weight':.1,'world_every':50,'world_updates':120,
          'world_forward':'unmodified full floating midpoint propagation256; fine512 convergence',
          'world_gradient':'two-sided full102 at current prediction; no teacher-local G',
          'TRAIN_only':True,'VAL_checkpoint_selection':False,'obstacle_conditioned':False,
          'codec':'exact frozen13774 codec; public conditions only'}
def load(directory):
    directory=Path(directory);r=read_json(directory/'training.json')
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    if saved['metadata']!=r['metadata'] or saved['metadata']['contract']!=CONTRACT:
        raise ValueError('endpoint checkpoint metadata mismatch')
    if file_digest(directory/'model.pt')!=r['checkpoint_sha256']:raise ValueError('checkpoint bytes changed')
    codec=Codec(read_json(directory/'codec.json'))
    if digest(codec.state)!=r['metadata']['codec_hash']:raise ValueError('codec changed')
    model=LatentDenoiser(codec.dim,len(codec.feature_mean));model.load_state_dict(saved['weights']);model.eval()
    return model,codec

def train(directory,arrays,rows,targets,new_ids,cache,ledger,progress,deadline):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);out=directory.parents[1]
    model,codec=parent_load(SOURCE/'learning/goal_loss');torch.set_num_threads(2)
    z,bases,conditions,ids,splits,states=arrays
    mask=np.asarray(splits)=='TRAIN';slots,keys,weights,pop=population(conditions,ids,splits)
    tids=np.asarray(ids)[mask].tolist();mapping={sid:i for i,sid in enumerate(ids)}
    if set(new_ids)-set(tids):raise ValueError('correction rows outside TRAIN')
    atomic_json(directory/'codec.json',codec.state)
    encoded=np.stack([codec.encode(v,b) for v,b in zip(z,bases)])
    centers=np.stack([codec.center(c,b) for c,b in zip(conditions,bases)])
    xall=codec.normalized(encoded,centers)
    call=np.stack([codec.condition(c,b) for c,b in zip(conditions,bases)])
    groups=[]
    for target in targets:
        pool=[mapping[sid] for sid in sorted(new_ids) if rows[mapping[sid]]['parent_sample_id']==target['parent_sample_id']]
        if not pool:raise ValueError('fixed TRAIN target lacks correction')
        groups.append(pool)
    meta={'contract':CONTRACT,'codec_hash':digest(codec.state),'training_ids':tids,
          'validation_ids':np.asarray(ids)[~mask].tolist(),'new_ids':sorted(new_ids),
          'data_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states}),
          'warm_start_checkpoint_sha256':file_digest(SOURCE/'learning/goal_loss/model.pt'),
          'world_groups':[[ids[i] for i in g] for g in groups],'world_targets_hash':digest(targets)}
    if (directory/'training.json').exists():
        result=read_json(directory/'training.json')
        if result['metadata']!=meta:raise ValueError('completed training identity changed')
        load(directory);return result
    torch.manual_seed(P['training_seed']);optimizer=torch.optim.AdamW(model.parameters(),lr=P['lr'])
    sampler=CoverageSampler(keys,P['training_seed']+2)
    x=torch.tensor(xall[mask],dtype=torch.float32);c=torch.tensor(call[mask],dtype=torch.float32)
    alpha=schedule();initial_hash=weight_hash(model);counts=np.zeros(len(tids),dtype=np.int64)
    start=0;history=[];world_records=[];wall_before=0.
    if (directory/'progress.pt').exists():
        saved=torch.load(directory/'progress.pt',map_location='cpu',weights_only=False)
        if saved['metadata']!=meta:raise ValueError('resume metadata changed')
        model.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);sampler=CoverageSampler(keys,P['training_seed']+2,saved['sampler'])
        start=saved['updates'];counts=saved['counts'];history=saved['history'];world_records=saved['world_records']
        initial_hash=saved['initial_hash'];wall_before=saved['wall_s']
    def prediction(i,t,seed):
        clean=torch.tensor(xall[i:i+1],dtype=torch.float32)
        cond=torch.tensor(call[i:i+1],dtype=torch.float32)
        noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
        a=alpha[t];noisy=a.sqrt()*clean+(1-a).sqrt()*noise
        output,_=prediction_to_clean_noise(noisy,model(noisy,torch.tensor([t]),cond),a,'v')
        return output[0]
    def diagnose(stage):
        path=directory/f'world_diagnostic_{stage}.json';key='TRAIN_endpoint_diagnostic_'+stage
        if key in ledger.data['completed']:
            r=read_json(path)
            if r['model_hash']!=weight_hash(model):raise ValueError('diagnostic checkpoint changed')
            return r
        ledger.begin(key,'TRAIN_endpoint_diagnostic',P['wall_budget_s'],2);calls={};results=[]
        model.eval()
        for j,group in enumerate(groups):
            i=group[0];rr,ii,_,_=cache.get(rows[i]);before=rr.calls.copy()
            for k,t in enumerate(P['world_noise_times']):
                with torch.no_grad():xx=prediction(i,t,P['diagnostic_noise_seed']+j*2+k).numpy().astype(float)
                value,r=endpoint.evaluate(xx,codec,centers[i],bases[i],rows[i]['task'],rr,ii,deadline,512)
                results.append({'sample_id':ids[i],'task_hash':rows[i]['task']['task_hash'],'t':t,
                                'seed':P['diagnostic_noise_seed']+j*2+k,'loss':value,
                                'position_m':r['position_m'],'rotation_rad':r['rotation_rad'],'ee':r['ee'],
                                'x_hash':digest(xx),'training_only':True})
            for name in before:calls[name]=calls.get(name,0)+rr.calls[name]-before[name]
        r={'status':'DONE','stage':stage,'model_hash':weight_hash(model),'cases':results,
           'calls':calls,'selection':'fixed TRAIN targets and independent noise; no validation selection'}
        atomic_json(path,r);ledger.finish({'status':'DONE'},calls);return r
    if start==0:diagnose('before')
    started=time.perf_counter()
    for step in range(start,P['training_updates']):
        if time.time()>=deadline:raise RuntimeError('ENDPOINT_TRAINING_WALL_EXHAUSTED')
        model.train();batch=sampler.take(P['batch']);np.add.at(counts,batch,1)
        t=torch.randint(0,1000,(P['batch'],));noise=torch.randn_like(x[batch]);a=alpha[t,None]
        noisy=a.sqrt()*x[batch]+(1-a).sqrt()*noise;target=a.sqrt()*noise-(1-a).sqrt()*x[batch]
        vloss=(model(noisy,t,c[batch])-target).square().mean();loss=vloss;event_record=None
        if (step+1)%P['world_every_updates']==0:
            event=(step+1)//P['world_every_updates']-1;j=event%len(groups)
            i=groups[j][(event//len(groups))%len(groups[j])]
            tt=P['world_noise_times'][(event//len(groups))%2];seed=P['world_noise_seed']+event
            predicted=prediction(i,tt,seed);xx=predicted.detach().numpy().astype(float)
            key=f'world_gradient_{event:03d}';proofpath=out/'world_gradients'/f'{event:03d}.json'
            arrpath=proofpath.with_suffix('.npz');input_weight=weight_hash(model)
            if key not in ledger.data['completed']:
                ledger.begin(key,'world_gradient',P['wall_budget_s'],P['world_updates'])
                progress('true_world_endpoint_gradient',update=step+1,event=event+1,total=P['world_updates'],
                         TRAIN_sample=ids[i],noise_t=tt,dimensions=codec.dim)
                rr,ii,_,_=cache.get(rows[i])
                value,g,proof,arr=endpoint.gradient(xx,codec,centers[i],bases[i],rows[i]['task'],rr,ii,deadline)
                proof.update(sample_id=ids[i],update=step+1,event=event,noise_t=tt,noise_seed=seed,
                             input_weight_hash=input_weight,training_only=True)
                atomic_npz(arrpath,**arr);proof['arrays_sha256']=file_digest(arrpath)
                atomic_json(proofpath,proof);ledger.finish({'status':proof['status']},proof['calls'])
            else:
                proof=read_json(proofpath)
                with np.load(arrpath) as arr:g=arr['gradient'].copy()
                value=proof['true_loss']
            if proof['status']!='PASS' or proof['x_hash']!=digest(xx) or proof['input_weight_hash']!=input_weight:
                raise ValueError('actual endpoint gradient missing, failed, or stale')
            endpoint_term=endpoint.attach_gradient(predicted,value,g)
            loss=vloss+P['world_weight']*endpoint_term
            event_record={'event':event,'update':step+1,'sample_id':ids[i],'noise_t':tt,
                          'true_world_loss':value,'gradient_l2':float(np.linalg.norm(g)),
                          'proof_sha256':file_digest(proofpath),'input_weight_hash':input_weight}
        if not torch.isfinite(loss):raise ValueError('NONFINITE_TRAINING_LOSS')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('NONFINITE_TRAINING_GRADIENT')
        optimizer.step()
        if event_record is not None:
            event_record.update(parameter_gradient_norm=float(norm),output_weight_hash=weight_hash(model))
            world_records.append(event_record)
        if (step+1)%100==0:
            history.append({'update':step+1,'v_mse':float(vloss.detach()),'total_loss':float(loss.detach())})
        if (step+1)%20==0 or event_record is not None:
            save_torch(directory/'progress.pt',{'metadata':meta,'weights':model.state_dict(),
                'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'sampler':sampler.state(),
                'updates':step+1,'counts':counts,'history':history,'world_records':world_records,
                'initial_hash':initial_hash,'wall_s':wall_before+time.perf_counter()-started})
    after=diagnose('after');before=read_json(directory/'world_diagnostic_before.json')
    ratios=[b['loss']/max(a['loss'],1e-12) for a,b in zip(before['cases'],after['cases'])]
    assert all((a['sample_id'],a['t'],a['seed'])==(b['sample_id'],b['t'],b['seed']) for a,b in zip(before['cases'],after['cases']))
    alignment={'cases':len(ratios),'median_paired_loss_ratio':float(np.median(ratios)),
               'improved_cases':sum(r<1 for r in ratios),'pre_registered_improvement':bool(np.median(ratios)<=.8),
               'TRAIN_diagnostic_only':True}
    save_torch(directory/'model.pt',{'metadata':meta,'weights':model.state_dict()})
    result={'status':'PASS' if all(counts>0) and len(world_records)==120 and weight_hash(model)!=initial_hash else 'FAIL',
            'metadata':meta,'updates':6000,'world_gradient_updates':len(world_records),
            'checkpoint_sha256':file_digest(directory/'model.pt'),'initial_weight_hash':initial_hash,
            'trained_weight_hash':weight_hash(model),'train_draw_counts':dict(zip(tids,counts.tolist())),
            'all_TRAIN_rows_seen':bool(all(counts>0)),**sampler.report(tids),'sampling_population':pop,
            'world_records':world_records,'world_alignment':alignment,'history':history,
            'VAL_gradient_updates':0,'VAL_checkpoint_selection':False,'wall_s':wall_before+time.perf_counter()-started,
            'benefit':'REQUIRES_FROZEN_PHYSICAL_COMPARISON'}
    atomic_json(directory/'training.json',result);load(directory);return result
