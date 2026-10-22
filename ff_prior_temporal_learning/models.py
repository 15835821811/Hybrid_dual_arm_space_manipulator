import math
import torch
from torch import nn
from .protocol import P

class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.norm=nn.GroupNorm(6,width)
        self.conv=nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation)
        self.film=nn.Linear(width,width*2)
    def forward(self,x,c):
        scale,bias=self.film(c).chunk(2,dim=1)
        y=self.norm(x)*(1+.1*scale[:,:,None])+.1*bias[:,:,None]
        return x+.25*self.conv(torch.nn.functional.silu(y))

class SequenceNetwork(nn.Module):
    def __init__(self,condition_dim):
        super().__init__();w=P['channels'];self.condition_dim=condition_dim;self.calls=0
        self.condition=nn.Sequential(nn.Linear(condition_dim,w*2),nn.SiLU(),nn.Linear(w*2,w))
        self.noise_time=nn.Sequential(nn.Linear(32,w),nn.SiLU(),nn.Linear(w,w))
        times=torch.linspace(0,1,271);features=[times]
        for frequency in (1,2,4,8):features.extend([torch.sin(2*math.pi*frequency*times),torch.cos(2*math.pi*frequency*times)])
        self.register_buffer('position',torch.stack(features)[None])
        self.input=nn.Conv1d(17+36+9,w,1)
        self.blocks=nn.ModuleList([Block(w,d) for d in P['dilations']]);self.output=nn.Conv1d(w,17,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,t,global_condition,local_condition):
        self.calls+=1
        freq=torch.exp(torch.arange(16,device=x.device)*(-math.log(10000)/15))
        phase=t[:,None]*freq[None]
        c=self.condition(global_condition)+self.noise_time(torch.cat([phase.sin(),phase.cos()],1))
        z=self.input(torch.cat([x.transpose(1,2),local_condition.transpose(1,2),self.position.expand(len(x),-1,-1)],1))
        for block in self.blocks:z=block(z,c)
        return self.output(torch.nn.functional.silu(z)).transpose(1,2)

def sample(model,global_condition,local_condition,seeds,method):
    from ff_prior.diffusion import schedule
    batch=len(global_condition)
    if method=='deterministic_sequence':
        x=torch.zeros((batch,271,17),dtype=torch.float32)
        return model(x,torch.zeros(batch),global_condition,local_condition).double(),[]
    x=torch.stack([torch.randn((271,17),generator=torch.Generator().manual_seed(int(seed)),dtype=torch.float64) for seed in seeds])
    alpha=schedule().double();times=torch.linspace(999,0,16).round().long();records=[]
    for j,t in enumerate(times):
        a=alpha[t];v=model(x.float(),t.repeat(batch).float(),global_condition,local_condition).double()
        clean=a.sqrt()*x-(1-a).sqrt()*v
        noise=(1-a).sqrt()*x+a.sqrt()*v
        next_a=alpha[times[j+1]] if j<15 else torch.tensor(1.,dtype=torch.float64)
        residual=(a.sqrt()*clean+(1-a).sqrt()*noise-x).detach().abs().max().item()
        if not torch.isfinite(clean).all() or residual>1e-10:raise RuntimeError('DDIM_NUMERIC_CHECK')
        x=next_a.sqrt()*clean+(1-next_a).sqrt()*noise
        records.append(dict(t=int(t),decomposition_residual=residual))
    return x,records
