"""Conditional Transformer over the complete free-variable vector.

Scalar tokens support heterogeneous boundary ranks without learned truncation.
Task masks permit variable waypoint counts; no scenario-name switches.
"""
import math
import torch
from torch import nn


class ConditionalDenoiser(nn.Module):
    def __init__(self,max_free=816,condition_dim=32,hidden=256,layers=4,heads=8):
        super().__init__()
        self.max_free=max_free;self.hidden=hidden;self.condition_dim=condition_dim
        self.value=nn.Linear(1,hidden)
        self.coordinate=nn.Embedding(max_free,hidden)
        self.condition=nn.Linear(condition_dim,hidden)
        self.time=nn.Sequential(nn.Linear(hidden,hidden*4),nn.SiLU(),nn.Linear(hidden*4,hidden))
        layer=nn.TransformerDecoderLayer(hidden,heads,hidden*4,dropout=0.,batch_first=True,norm_first=True,activation='gelu')
        self.decoder=nn.TransformerDecoder(layer,layers,norm=nn.LayerNorm(hidden))
        self.output=nn.Linear(hidden,1)

    def forward(self,x,t,condition):
        if x.ndim!=2 or x.shape[1]>self.max_free:raise ValueError('unsupported full free dimension')
        tokens=condition['tokens'];mask=condition.get('padding_mask')
        if tokens.shape[0]!=x.shape[0] or tokens.shape[2]!=self.condition_dim:raise ValueError('condition shape mismatch')
        freq=torch.exp(-math.log(10000)*torch.arange(self.hidden//2,device=x.device)/(self.hidden//2))
        phase=t[:,None]*freq[None,:];time=self.time(torch.cat((phase.sin(),phase.cos()),-1))
        h=self.value(x[:,:,None])+self.coordinate(torch.arange(x.shape[1],device=x.device))[None,:,:]+time[:,None,:]
        context=self.condition(tokens)
        return self.output(self.decoder(h,context,memory_key_padding_mask=mask)).squeeze(-1)
