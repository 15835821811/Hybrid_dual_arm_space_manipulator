import numpy as np
import torch
from torch import nn
from ff_prior_latent.learning import LatentDenoiser
from ff_prior_sample_alignment.learning import load as old_load
from ff_prior.diffusion import schedule,prediction_to_clean_noise
from .protocol import SOURCE,P

class TransferDenoiser(nn.Module):
    def __init__(self,codec,warm=True):
        super().__init__();self.calls=0
        self.low=LatentDenoiser(102,65)
        if warm:self.low.load_state_dict(old_load(SOURCE/'learning/aligned')[0].state_dict())
        self.detail=nn.Sequential(nn.Linear(323+65+32,128),nn.SiLU(),nn.Linear(128,221))
        nn.init.zeros_(self.detail[-1].weight);nn.init.zeros_(self.detail[-1].bias)
        self.register_buffer('detail_limit',torch.tensor(codec.state['detail_limit'],dtype=torch.float64))
        self.register_buffer('alpha',schedule().double())

    def components(self,x,t,c):
        self.calls+=1
        low_v=self.low(x[:,:102].float(),t,c)
        freq=torch.exp(torch.arange(16,device=x.device)*(-np.log(10000)/15));phase=t[:,None]*freq[None]
        h=self.detail(torch.cat((x.float(),c,phase.sin(),phase.cos()),1)).double()
        detail=self.detail_limit*torch.tanh(h/self.detail_limit)
        return low_v,detail

    def clean(self,x,t,c):
        v,detail=self.components(x,t,c)
        low,_=prediction_to_clean_noise(x[:,:102],v.double(),self.alpha[t,None],'v')
        return torch.cat((low,detail),1)


def initial_noise(seeds):
    # Drawing 323 at once changes the first 102 samples in torch.randn. Preserve
    # the exact old low-dimensional RNG stream, and isolate extra coordinates.
    return torch.cat([torch.cat((torch.randn((1,102),generator=torch.Generator().manual_seed(int(s))),
        torch.randn((1,221),generator=torch.Generator().manual_seed(int(s)+P['detail_noise_offset']))),1) for s in seeds]).double()


def rollout(model,c,seeds):
    x=initial_noise(seeds);alpha=schedule().double();times=np.linspace(999,0,16).round().astype(int)
    for j,t in enumerate(times):
        a=alpha[t];clean=model.clean(x,torch.full((len(seeds),),int(t)),c)
        epsilon=(x-a.sqrt()*clean)/(1-a).sqrt()
        na=alpha[times[j+1]] if j<15 else x.new_tensor(1.)
        x=na.sqrt()*clean+(1-na).sqrt()*epsilon
    if not torch.isfinite(x).all():raise ValueError('nonfinite full rollout')
    return x


def teacher_loss(model,codec,c,target,seeds):
    full=rollout(model,c,seeds)
    scale=torch.tensor(codec.latent_scale,dtype=torch.float64)
    basis=torch.tensor(codec.basis,dtype=torch.float64)
    error=(full-target.double())*scale
    physical=torch.einsum('tk,nkj->ntj',basis,error.reshape(-1,19,17))
    mse=physical.square().mean()/scale.square().mean()
    temporal=torch.diff(physical,n=2,dim=1).square().mean()/scale.square().mean()
    return mse+P['temporal_error_weight']*temporal
