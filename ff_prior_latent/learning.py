"""TRAIN-only temporal coefficients, conditional ridge center, and v diffusion."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from ff_prior.contracts import public_condition, assert_frozen
from ff_prior.diffusion import schedule, prediction_to_clean_noise, save_torch, weight_hash
from ff_prior.storage import atomic_json, read_json, file_digest, digest

CONTRACT = {'version':'centered_temporal_latent_v1','fit':'TRAIN_only',
    'retained_temporal_energy':1-1e-7,'ridge_lambda':10.,'updates':1800,'batch':32,
    'lr':2e-4,'seed':20261081,'prediction':'v','hidden':256,'hidden_layers':3,
    'selection':'fixed_final_update; VAL diagnostic only', 'ddim_steps':16,
    'features':'public_condition_48 + public_initializer_terminal_delta_17',
    'scene_conditioned':False,'data_added':False}


def feature(condition, base):
    return np.r_[public_condition(condition),np.asarray(base)[-1]-condition['initial_features']]


class Codec:
    def __init__(self, state):
        self.state=state
        for k in ('basis','feature_mean','feature_scale','ridge','latent_mean','latent_scale'):
            setattr(self,k,np.asarray(state[k],dtype=np.float64))
        self.rank=self.basis.shape[1]; self.dim=self.rank*17
        if not np.allclose(self.basis.T@self.basis,np.eye(self.rank),atol=1e-12):
            raise ValueError('non-orthonormal temporal basis')
    def condition(self,c,base):
        return (feature(c,base)-self.feature_mean)/self.feature_scale
    def center(self,c,base):
        return np.r_[1.,self.condition(c,base)]@self.ridge
    def encode(self,z,base):
        return (self.basis.T@(np.asarray(z)-base)).reshape(-1)
    def decode(self,y,base):
        return np.asarray(base)+self.basis@np.asarray(y).reshape(self.rank,17)
    def normalized(self,y,center):
        return (y-center-self.latent_mean)/self.latent_scale
    def coefficients(self,x,center):
        return x*self.latent_scale+self.latent_mean+center
    @classmethod
    def fit(cls,z,bases,conditions,ids,splits):
        if not splits or any(s!='TRAIN' for s in splits): raise ValueError('codec fit must be TRAIN-only')
        z,bases=np.asarray(z),np.asarray(bases)
        if z.shape!=bases.shape or z.shape!=(len(ids),19,17) or len(set(ids))!=len(ids):
            raise ValueError('invalid fitting population')
        residual=z-bases
        matrix=residual.transpose(1,0,2).reshape(19,-1)
        u,s,_=np.linalg.svd(matrix,full_matrices=False)
        energy=np.cumsum(s*s)
        rank=int(np.searchsorted(energy/energy[-1],CONTRACT['retained_temporal_energy'])+1) if energy[-1]>0 else 1
        basis=u[:,:rank]
        y=np.einsum('tk,ntj->nkj',basis,residual).reshape(len(z),-1)
        c=np.stack([feature(v,b) for v,b in zip(conditions,bases)])
        cm,cs=c.mean(0),c.std(0); cs=np.where(cs<1e-8,1.,cs)
        design=np.c_[np.ones(len(c)),(c-cm)/cs]
        penalty=np.eye(design.shape[1])*CONTRACT['ridge_lambda']; penalty[0,0]=0.
        ridge=np.linalg.solve(design.T@design+penalty,design.T@y)
        rest=y-design@ridge; mean,scale=rest.mean(0),rest.std(0)
        scale=np.maximum(scale,1e-4)
        state={'basis':basis,'feature_mean':cm,'feature_scale':cs,'ridge':ridge,
            'latent_mean':mean,'latent_scale':scale,'fit_ids':ids,'fit_split':'TRAIN',
            'fit_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids}),
            'rank':rank,'retained_energy':float(energy[rank-1]/energy[-1]) if energy[-1]>0 else 1.,
            'reconstruction_max_rad':float(abs(basis@basis.T@matrix-matrix).max())}
        if not np.isfinite(rest).all(): raise ValueError('nonfinite codec fit')
        return cls(state)


class LatentDenoiser(nn.Module):
    def __init__(self,dim,condition_dim):
        super().__init__(); self.calls=0
        self.net=nn.Sequential(nn.Linear(dim+condition_dim+32,256),nn.SiLU(),
            nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,dim))
    def forward(self,x,t,c):
        self.calls+=1
        freq=torch.exp(torch.arange(16,device=x.device)*(-np.log(10000)/15))
        phase=t[:,None]*freq[None]
        return self.net(torch.cat([x,c,phase.sin(),phase.cos()],1))


def load(directory):
    directory=Path(directory); report=read_json(directory/'training.json')
    state=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    if state['metadata']!=report['metadata'] or state['metadata']['contract']!=CONTRACT:
        raise ValueError('checkpoint identity mismatch')
    if file_digest(directory/'model.pt')!=report['checkpoint_sha256'] or file_digest(directory/'codec.json')!=state['metadata']['codec_sha256']:
        raise ValueError('checkpoint or codec changed')
    codec=Codec(read_json(directory/'codec.json'))
    model=LatentDenoiser(codec.dim,len(codec.feature_mean)); model.load_state_dict(state['weights']); model.eval()
    return model,codec


def train(directory,arrays,progress,deadline):
    z,bases,conditions,ids,splits,states=arrays; directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    if set(splits)!={'TRAIN','VAL'}: raise ValueError('only TRAIN/VAL accepted')
    tr=np.array([s=='TRAIN' for s in splits]); va=~tr
    tc=[c for c,s in zip(conditions,splits) if s=='TRAIN']; tids=np.asarray(ids)[tr].tolist()
    codec=Codec.fit(z[tr],bases[tr],tc,tids,['TRAIN']*sum(tr))
    cp=directory/'codec.json'
    if cp.exists():
        if digest(read_json(cp))!=digest(codec.state): raise ValueError('codec changed on resume')
    else: atomic_json(cp,codec.state)
    meta={'contract':CONTRACT,'codec_sha256':file_digest(cp),'training_ids':tids,
        'validation_ids':np.asarray(ids)[va].tolist(),'data_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states})}
    if (directory/'training.json').exists():
        r=read_json(directory/'training.json')
        if r['metadata']!=meta: raise ValueError('completed training input changed')
        load(directory); return r
    c=np.stack([codec.condition(v,b) for v,b in zip(conditions,bases)])
    centers=np.stack([codec.center(v,b) for v,b in zip(conditions,bases)])
    y=np.stack([codec.encode(v,b) for v,b in zip(z,bases)])
    xall=codec.normalized(y,centers)
    torch.set_num_threads(2); torch.manual_seed(CONTRACT['seed'])
    model=LatentDenoiser(codec.dim,c.shape[1]); original=weight_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONTRACT['lr']); alpha=schedule()
    x=torch.tensor(xall[tr],dtype=torch.float32); ct=torch.tensor(c[tr],dtype=torch.float32)
    xv=torch.tensor(xall[va],dtype=torch.float32); cv=torch.tensor(c[va],dtype=torch.float32)
    vg=torch.Generator().manual_seed(CONTRACT['seed']+1); vn=torch.randn(xv.shape,generator=vg)
    counts=np.zeros(len(x),dtype=int); order=[]; history=[]; start=0; wall_before=0.
    if (directory/'progress.pt').exists():
        saved=torch.load(directory/'progress.pt',map_location='cpu',weights_only=False)
        if saved['metadata']!=meta: raise ValueError('training resume input changed')
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['rng'])
        start,counts,order,history,original,wall_before=(saved[k] for k in ('updates','counts','order','history','original','wall_s'))
    def diagnostics(shuffle=False):
        result=[]; model.eval()
        for t in (0,500,999):
            tt=torch.full((len(xv),),t); a=alpha[t]; noisy=a.sqrt()*xv+(1-a).sqrt()*vn
            with torch.no_grad(): clean,_=prediction_to_clean_noise(noisy,model(noisy,tt,cv.roll(1,0) if shuffle else cv),a,'v')
            err=(clean.numpy()-xall[va])*codec.latent_scale
            trajectory=np.einsum('tk,nkj->ntj',codec.basis,err.reshape(len(err),codec.rank,17))
            result.append({'t':t,'trajectory_rmse_rad':float(np.sqrt(np.mean(trajectory**2)))})
        return result
    wall=time.perf_counter()
    for step in range(start,CONTRACT['updates']):
        if time.time()>=deadline: raise RuntimeError('TRAINING_WALL_BUDGET_EXHAUSTED')
        if len(order)<32: order.extend(torch.randperm(len(x)).tolist())
        batch=order[:32]; order=order[32:]; np.add.at(counts,batch,1)
        model.train(); t=torch.randint(0,1000,(32,)); noise=torch.randn_like(x[batch]); a=alpha[t,None]
        prediction=model(a.sqrt()*x[batch]+(1-a).sqrt()*noise,t,ct[batch]); target=a.sqrt()*noise-(1-a).sqrt()*x[batch]
        loss=(prediction-target).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient')
        optimizer.step()
        if (step+1)%100==0:
            history.append({'update':step+1,'train_v_mse':float(loss.detach()),'val_noise_bands':diagnostics()})
            progress('train_latent_diffusion',updates=step+1,total=CONTRACT['updates'])
        if (step+1)%20==0 or step+1==CONTRACT['updates']:
            save_torch(directory/'progress.pt',{'metadata':meta,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                'updates':step+1,'counts':counts,'order':order,'history':history,'original':original,'wall_s':wall_before+time.perf_counter()-wall})
    save_torch(directory/'model.pt',{'metadata':meta,'weights':model.state_dict()})
    mean_prediction=np.stack([codec.decode(v,b) for v,b in zip(centers,bases)])
    report={'status':'PASS' if all(counts>0) and weight_hash(model)!=original else 'FAIL','metadata':meta,
        'updates':CONTRACT['updates'],'checkpoint_sha256':file_digest(directory/'model.pt'),
        'initial_weight_hash':original,'trained_weight_hash':weight_hash(model),'history':history,
        'train_draw_counts':dict(zip(tids,counts.tolist())),'all_TRAIN_rows_seen':bool(all(counts>0)),
        'VAL_gradient_updates':0,'VAL_checkpoint_selection':False,'noise_bands':diagnostics(),'shuffled_condition_noise_bands':diagnostics(True),
        'temporal_rank':codec.rank,'latent_dimensions':codec.dim,'control_dimensions':323,
        'ridge_trajectory_rmse_rad':{s:float(np.sqrt(np.mean((mean_prediction[m]-z[m])**2))) for s,m in [('TRAIN',tr),('VAL',va)]},
        'zero_residual_trajectory_rmse_rad':{s:float(np.sqrt(np.mean((bases[m]-z[m])**2))) for s,m in [('TRAIN',tr),('VAL',va)]},
        'wall_s':wall_before+time.perf_counter()-wall,'benefit':'REQUIRES_PAIRED_PHYSICAL_EVIDENCE'}
    atomic_json(directory/'training.json',report); load(directory); return report


def sample(bundle,base,task,seed,projector):
    model,codec=bundle; assert_frozen(task); start=model.calls
    center=codec.center(task['condition'],base)
    c=torch.tensor(codec.condition(task['condition'],base),dtype=torch.float32)[None]
    x=torch.randn((1,codec.dim),generator=torch.Generator().manual_seed(seed)).double()
    alpha=schedule().double(); times=np.linspace(999,0,16).round().astype(int); steps=[]; raw=None
    for i,t in enumerate(times):
        with torch.no_grad():
            v=model(x.float(),torch.tensor([int(t)]),c).double()
            a=alpha[t]; x0,epsilon=prediction_to_clean_noise(x,v,a,'v')
        residual=float((x-a.sqrt()*x0-(1-a).sqrt()*epsilon).abs().max())
        if not torch.isfinite(x0).all(): raise ValueError('nonfinite latent estimate')
        if i==15:
            raw=codec.decode(codec.coefficients(x0[0].numpy(),center),base)
            # Last denoising step has next alpha=1: the physical projection is
            # the returned clean trajectory, with no subsequent noisy update.
            z,guide=projector(raw,15)
        next_a=alpha[times[i+1]] if i+1<16 else torch.tensor(1.,dtype=torch.float64)
        x=next_a.sqrt()*x0+(1-next_a).sqrt()*epsilon
        steps.append({'iteration':i,'timestep':int(t),'alpha_bar':float(a),'decomposition_residual_after':residual})
    if model.calls-start!=16: raise RuntimeError('network budget violation')
    assert_frozen(task)
    return z,{'actual_network_calls':16,'sampling_seed':seed,'prediction_type':'v','steps':steps,'guidance':[guide],
        'latent_dimensions':codec.dim,'low_rank_noise_and_clean_share_basis':True,'condition_center_hash':digest(center),
        'raw_proposal_hash':digest(raw),'sampling_precision':'float64_state_float32_network',
        'guidance_placement':'final_clean_denoising_step; no subsequent noisy update'}
