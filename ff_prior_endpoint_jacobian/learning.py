import time
import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_origin_learn_bounded.learning import save_torch
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_paired_objectives.objectives import terms,vector_gradient
from ff_prior_delivery_train.learning import load as warm_load
from ff_prior_latent.learning import Codec,LatentDenoiser
from .optimization import controlled_step
from .protocol import CONTROL,WARM,PROTOCOL as P

def base_loss(model,data,G,indices,t,noise):
    gi=[data['lookup'][int(i)] for i in indices]
    v,l,_=terms(model,data['x'][indices],data['c'][indices],t,noise,data['alpha'],G[gi])
    return v+.05*l

def diagnostic(out,selection,codec,arrays,rows,G,data,cache,ledger,budget,progress):
    model,_=warm_load(WARM/'learning/goal_loss');torch.set_num_threads(2)
    targets=[data['ids'][e['sample_id']] for e in selection['diagnostic']];cases=[]
    for j,c in enumerate(selection['cases']):
        key=f'M1_{j:02d}';ledger.begin(key,'diagnostic',21600.,24)
        progress('M1_TRAIN_controlled_single_step',case=j+1,total=24,t=c['t'])
        i=targets[c['target']];t=c['t'];seed=c['noise_seed'];noise=torch.randn((1,102),generator=torch.Generator().manual_seed(seed))
        control_i=targets[c['control_target']]
        control_noise=torch.randn((1,102),generator=torch.Generator().manual_seed(seed+100000))
        loss=(base_loss(model,data,G,[i],torch.tensor([t]),noise)+base_loss(model,data,G,[control_i],torch.tensor([t]),control_noise))/2
        g=vector_gradient(loss,model).detach()
        optimizer=torch.optim.AdamW(model.parameters(),lr=P['lr'])
        rr,_,_,_=cache.get(rows[i]);before=rr.calls.copy()
        result=controlled_step(out/'diagnostic'/f'{j:02d}',model,optimizer,g,codec,data,arrays,rows,cache,
          [(i,t,seed)],[(targets[c['control_target']],t,seed+100000)],budget.deadline,budget,False)
        result.update(case=j,t=t,sample_id=arrays[3][i]);cases.append(result)
        ledger.finish({'status':'GRADIENT_BLOCKED' if result['gradient_status']!='PASS' else 'PASS' if result['accepted'] else 'NO_DESCENT'}, {'coupled':result['actual_coupled_calls']})
    per={str(t):sum(c['accepted'] for c in cases if c['t']==t) for t in P['noise_times']}
    result={'status':'PASS' if sum(c['accepted'] for c in cases)>=16 and min(per.values())>=4 and all(c['gradient_status']=='PASS' for c in cases) else 'BLOCKED',
        'cases':24,'accepted':sum(c['accepted'] for c in cases),'accepted_per_noise':per,'rows':cases,'physical_certification':False,'updates_retained':0}
    atomic_json(out/'single_step_summary.json',result);return result

def train(out,selection,codec,arrays,rows,G,data,cache,ledger,budget,progress):
    directory=out/'learning';directory.mkdir(exist_ok=True);model,_=warm_load(WARM/'learning/goal_loss')
    initial_hash=weight_hash(model);torch.manual_seed(P['training_seed']);torch.set_num_threads(2)
    opt=torch.optim.AdamW(model.parameters(),lr=P['lr']);sampler=CoverageSampler(data['keys'],P['training_seed']+2)
    baseline=read_json(CONTROL/'learning/goal_loss/training.json');segments=[];events=[];proofs=[];started=time.time()
    targets=[data['ids'][e['sample_id']] for e in selection['events']]
    event_map={e['update']:e for e in selection['world_sequence']}
    for step in range(6000):
        if time.time()>=budget.deadline:raise RuntimeError('M2_TIME_LIMIT')
        if step%20==0:ledger.begin(f'train_segment_{step//20:03d}','training_segment',21600.,300);proofs=[];segment_coupled=0
        local=sampler.take(64);indices=data['tr'][local];t=torch.randint(0,1000,(64,));noise=torch.randn((64,102))
        loss=base_loss(model,data,G,indices,t,noise)
        proofs.append(digest({'update':step+1,'batch':local,'t':t.numpy(),'noise':noise.numpy()}))
        if step+1 in event_map:
            e=event_map[step+1];g=vector_gradient(loss,model).detach()
            specs=[(targets[k],e['t'],e['noise_seeds'][j]) for j,k in enumerate(e['targets'])]
            controls=[(targets[k],e['t'],e['noise_seeds'][j+1]) for j,k in enumerate(e['control_targets'])]
            progress('M2_actual_world_event',update=step+1,event=(step+1)//100,total=60)
            event=controlled_step(out/'world_events'/f'{step+1:04d}',model,opt,g,codec,data,arrays,rows,cache,specs,controls,budget.deadline,budget,True,base_batch={'indices':indices.tolist(),'t':t.clone(),'noise':noise.clone()})
            events.append(event);segment_coupled+=event['actual_coupled_calls']
            if event['gradient_status']!='PASS':raise RuntimeError('WORLD_GRADIENT_INVALID; no silent local-only completion')
        else:
            opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite training gradient')
            opt.step()
        if (step+1)%20==0:
            segment={'start_update':step-18,'end_update':step+1,'draws':1280,'batch_rng_digest':digest(proofs)}
            if segment!=baseline['segments'][step//20]:raise ValueError('ordinary RNG/row sequence mismatch with A')
            segments.append(segment)
            save_torch(directory/'progress.pt',{'weights':model.state_dict(),'optimizer':opt.state_dict(),'updates':step+1,'rng':torch.get_rng_state(),'sampler':sampler.state()})
            atomic_json(directory/'journal.json',{'status':'COMMITTED','updates':step+1});ledger.finish({'status':'PASS'}, {'coupled':segment_coupled})
    ids=[arrays[3][i] for i in data['tr']];sampling=sampler.report(ids)
    sampling['train_draw_counts']=dict(zip(ids,(sampler.row_counts+sampler.origin_row_counts).tolist()))
    if sampling['train_draw_counts']!=baseline['train_draw_counts']:raise ValueError('draw counts mismatch')
    atomic_json(directory/'codec.json',codec.state)
    metadata={'protocol':P,'control_checkpoint':file_digest(CONTROL/'learning/goal_loss/model.pt'),'warm_checkpoint':file_digest(WARM/'learning/goal_loss/model.pt'),
       'data_hash':baseline['metadata']['data_hash'],'codec_hash':digest(codec.state)}
    save_torch(directory/'model.pt',{'weights':model.state_dict(),'metadata':metadata})
    result={'status':'PASS','updates':6000,'actual_TRAIN_draws':384000,'TRAIN_rows_seen':len(ids),'VAL_updates':0,'world_events':60,
      'world_gradients':120,'world_accepted':sum(e['accepted'] for e in events),'segments':segments,'initial_weight_hash':initial_hash,
      'trained_weight_hash':weight_hash(model),'checkpoint_sha256':file_digest(directory/'model.pt'),'metadata':metadata,
      'ordinary_rng_matches_A':True,'sampling':sampling,'wall_s':time.time()-started}
    atomic_json(directory/'training.json',result);load(directory);return result

def load(directory):
    t=read_json(directory/'training.json');saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    if t['checkpoint_sha256']!=file_digest(directory/'model.pt') or saved['metadata']!=t['metadata']:raise ValueError('B checkpoint changed')
    codec=Codec(read_json(directory/'codec.json'));model=LatentDenoiser(codec.dim,len(codec.feature_mean));model.load_state_dict(saved['weights']);model.eval()
    if weight_hash(model)!=t['trained_weight_hash']:raise ValueError('B reload failed')
    return model,codec
