import numpy as np
import torch
from ff_prior_latent.learning import LatentDenoiser
from ff_prior.diffusion import schedule,prediction_to_clean_noise,weight_hash
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_backbone_v2.learning import save_torch
from ff_prior_origin_learn_bounded.distribution import CoverageSampler
from ff_prior_paired_objectives.objectives import vector_gradient
from ff_prior_sample_alignment.objectives import assign,capped_component
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .codec import FullCodec
from .protocol import P

def rollout(model,c,seeds):
    x=torch.cat([torch.randn((1,323),generator=torch.Generator().manual_seed(int(s))) for s in seeds]).double()
    alpha=schedule().double();times=np.linspace(999,0,16).round().astype(int)
    for j,t in enumerate(times):
        a=alpha[t];v=model(x.float(),torch.full((len(seeds),),int(t)),c).double()
        clean,_=prediction_to_clean_noise(x,v,a,'v')
        epsilon=(x-a.sqrt()*clean)/(1-a).sqrt()
        next_a=alpha[times[j+1]] if j<15 else x.new_tensor(1.)
        x=next_a.sqrt()*clean+(1-next_a).sqrt()*epsilon
    if not torch.isfinite(x).all():raise ValueError('nonfinite full DDIM')
    return x

def load(directory):
    r=read_json(directory/'training.json');s=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    assert file_digest(directory/'model.pt')==r['checkpoint_sha256'] and s['metadata']==r['metadata']
    codec=FullCodec(read_json(directory/'codec.json'));assert codec.dim==323 and digest(codec.state)==r['metadata']['codec_hash']
    model=LatentDenoiser(323,65);model.load_state_dict(s['weights']);model.eval()
    assert weight_hash(model)==r['trained_weight_hash']
    return model,codec

def train(out,codec,arrays,data,ledger,budget,progress):
    directory=out/'learning/full';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'training.json').exists():load(directory);return
    torch.manual_seed(P['training_seed']);model=LatentDenoiser(323,65);original=weight_hash(model)
    meta={'protocol':P,'codec_hash':digest(codec.state),'TRAIN_ids':[arrays[3][i] for i in data['tr']],
        'data_hash':digest({'Z':arrays[0],'bases':arrays[1],'ids':arrays[3],'splits':arrays[4]})}
    optimizer=torch.optim.AdamW(model.parameters(),lr=P['lr'],weight_decay=P['weight_decay'])
    sampler=CoverageSampler(data['keys'],P['training_seed']+2);rng=torch.Generator().manual_seed(P['training_seed'])
    counts=np.zeros(len(data['tr']),dtype=np.int64);segments=[];history=[];start=0;rollouts=0
    if (directory/'progress.pt').exists():
        s=torch.load(directory/'progress.pt',map_location='cpu',weights_only=False)
        assert read_json(directory/'journal.json')['status']=='COMMITTED' and s['metadata']==meta
        model.load_state_dict(s['weights']);optimizer.load_state_dict(s['optimizer']);rng.set_state(s['rng'])
        sampler=CoverageSampler(data['keys'],P['training_seed']+2,s['sampler'])
        counts,segments,history,start,rollouts=[s[k] for k in ('counts','segments','history','updates','rollouts')]
    else:
        save_torch(directory/'initial.pt',{'weights':model.state_dict(),'metadata':meta})
        atomic_json(directory/'codec.json',codec.state)
    scale=torch.tensor(codec.latent_scale,dtype=torch.float64)
    for begin in range(start,P['updates'],20):
        ledger.begin(f'train_{begin//20}','training',P['wall_s'],300)
        atomic_json(directory/'journal.json',{'status':'PENDING','start':begin+1})
        proofs=[]
        for step in range(begin,begin+20):
            budget.check();batch=sampler.take(64);indices=data['tr'][batch];np.add.at(counts,batch,1)
            t=torch.randint(0,1000,(64,),generator=rng);noise=torch.randn((64,323),generator=rng)
            a=data['alpha'][t,None];x=data['x'][indices];c=data['c'][indices]
            v=model(a.sqrt()*x+(1-a).sqrt()*noise,t,c)
            loss=(v-(a.sqrt()*noise-(1-a).sqrt()*x)).square().mean()
            base=vector_gradient(loss,model).detach();mixed=base.clone();extra=None;coef=0.;tv=None
            event=(step+1)%P['rollout_every']==0
            seeds=[P['rollout_seed']+step*101+k for k in range(8)]
            save=step+1 in P['audit_updates'];dest=directory/'updates'/f'{step+1:04d}'
            if save:save_torch(dest/'before.pt',{'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'indices':indices,'t':t,'noise':noise,'full_seeds':seeds})
            if event:
                full=rollout(model,c[:8],seeds)
                # All 323 controls affect this loss; there is no low-rank projection.
                tv=((full-data['x'][indices[:8]].double())*scale).square().mean()/scale.square().mean()
                extra=vector_gradient(tv,model).detach();part,coef=capped_component(base,extra,P['rollout_norm_cap'])
                mixed+=part;rollouts+=1
            optimizer.zero_grad();assign(model,mixed)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite weights')
            if save:
                save_torch(dest/'after.pt',{'weights':model.state_dict(),'optimizer':optimizer.state_dict()})
                atomic_npz(dest/'gradients.npz',base=base.numpy(),teacher=np.zeros(0) if extra is None else extra.numpy(),mixed=mixed.numpy())
                atomic_json(dest/'update.json',{'update':step+1,'clip_norm':float(norm),'teacher_scale':coef,'v_loss':float(loss.detach()),'full16_loss':None if tv is None else float(tv.detach())})
            proofs.append(digest({'step':step+1,'batch':batch,'t':t.numpy(),'noise':noise.numpy()}))
            if (step+1)%100==0:
                history.append({'step':step+1,'v_loss':float(loss.detach()),'full16_loss':None if tv is None else float(tv.detach())})
                progress('M2_training_full323',updates=step+1,total=P['updates'])
        segments.append({'start':begin+1,'end':begin+20,'draws':1280,'digest':digest(proofs)})
        save_torch(directory/'progress.pt',{'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.get_state(),
            'sampler':sampler.state(),'metadata':meta,'counts':counts,'segments':segments,'history':history,'updates':begin+20,'rollouts':rollouts})
        atomic_json(directory/'journal.json',{'status':'COMMITTED','updates':begin+20});ledger.finish({'status':'PASS'}, {})
    assert np.all(counts>0) and rollouts==1200 and weight_hash(model)!=original
    save_torch(directory/'model.pt',{'metadata':meta,'weights':model.state_dict()})
    atomic_json(directory/'training.json',{'status':'PASS','metadata':meta,'updates':P['updates'],'full16_teacher_batches':rollouts,
        'train_draw_counts':dict(zip(meta['TRAIN_ids'],counts.tolist())),'TRAIN_rows_seen':int((counts>0).sum()),
        'segments':segments,'history':history,'actual_TRAIN_draws':int(counts.sum()),'initial_weight_hash':original,
        'trained_weight_hash':weight_hash(model),'checkpoint_sha256':file_digest(directory/'model.pt'),
        'VAL_gradient_updates':0,'VAL_checkpoint_selection':False,'TEST':False})
    load(directory)
