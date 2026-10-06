"""Paired B.1 denoisers; common 30-free / 32-full reference representation."""
from __future__ import annotations
import torch
from torch import nn
from v6_4.diffusion_model import (ConditionalDDPM, ConditionalDenoiser,
    DiffusionConfig, build_noise_schedule, timestep_embedding)


class FlatReferenceDenoiser(nn.Module):
    """Original M0 body, retrained with the same typed information flattened."""
    def __init__(self, config):
        super().__init__()
        self.body=ConditionalDenoiser(config)

    def forward(self, noisy, timesteps, condition):
        return self.body(noisy,timesteps,condition['flat'])


class ModulatedCrossBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.norms=nn.ModuleList([nn.LayerNorm(width,elementwise_affine=False) for _ in range(3)])
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(width,6*width))
        self.self_attention=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))

    def forward(self,x,context,global_time,valid):
        shifts_scales=self.modulation(global_time).chunk(6,dim=-1)
        def mod(i):
            shift,scale=shifts_scales[2*i:2*i+2]
            return self.norms[i](x)*(1+scale[:,None,:])+shift[:,None,:]
        y=mod(0)
        x=x+self.self_attention(y,y,y,need_weights=False)[0]
        y=mod(1)
        x=x+self.cross_attention(y,context,context,key_padding_mask=~valid,need_weights=False)[0]
        return x+self.ffn(mod(2))


class ConditionedControlpointDenoiser(nn.Module):
    """32 ordered control tokens, typed unordered geometry cross-attention."""
    def __init__(self,config,*,global_dim=93,token_dim=40,type_count=8):
        super().__init__()
        self.config=config
        w=config.hidden_dim
        self.control_input=nn.Linear(17,w)
        self.boundary_input=nn.Linear(17,w)
        self.position=nn.Parameter(torch.randn(1,32,w)*.02)
        self.token_input=nn.Linear(token_dim,w)
        self.type_embedding=nn.Embedding(type_count,w,padding_idx=0)
        self.global_input=nn.Sequential(nn.Linear(global_dim,w),nn.SiLU(),nn.Linear(w,w))
        self.time_input=nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w))
        self.blocks=nn.ModuleList([ModulatedCrossBlock(w,config.heads) for _ in range(config.layers)])
        self.final_norm=nn.LayerNorm(w)
        self.output=nn.Linear(w,17)

    def forward(self,noisy,timesteps,condition):
        b=noisy.shape[0]
        if noisy.shape!=(b,30,17) or timesteps.shape!=(b,):
            raise ValueError('common free-control/timestep dimensions differ')
        if torch.any((timesteps<0)|(timesteps>=self.config.diffusion_steps)):
            raise ValueError('timestep outside frozen schedule')
        valid=condition['token_mask'].bool()
        if valid.shape!=condition['token_types'].shape or not torch.all(valid.any(dim=1)):
            raise ValueError('condition needs finite valid typed context')
        # Boundary inputs duplicate q0/dq0 already supplied to both models.
        # They are fixed task controls scaled by physical work ranges, not latent outputs.
        fixed=condition['fixed_controls_scaled']
        if fixed.shape!=(b,2,17):raise ValueError('two algebraically fixed controls required')
        x=torch.cat((self.boundary_input(fixed),self.control_input(noisy)),dim=1)+self.position
        features=torch.where(valid[:,:,None],condition['token_features'],0.)
        types=torch.where(valid,condition['token_types'].long(),0)
        context=self.token_input(features)+self.type_embedding(types)
        g=self.global_input(condition['global'])+self.time_input(timestep_embedding(timesteps,self.config.hidden_dim))
        for block in self.blocks:x=block(x,context,g,valid)
        return self.output(self.final_norm(x[:,2:]))


class PilotDDPM(ConditionalDDPM):
    """Shared exact V5 cosine100 / v-MSE / deterministic DDIM20 algebra."""
    def __init__(self,model,flat_dim,*,global_dim=93,token_dim=40,type_count=8):
        nn.Module.__init__(self)
        self.model_name=model
        self.config=DiffusionConfig(condition_dim=flat_dim,objective='v_mse',parameterization='epsilon_residual')
        self.denoiser=(FlatReferenceDenoiser(self.config) if model=='M0' else
            ConditionedControlpointDenoiser(self.config,global_dim=global_dim,token_dim=token_dim,type_count=type_count))
        if model not in ('M0','M1'):raise ValueError('only the two declared pilot models are permitted')
        betas,alpha=build_noise_schedule(self.config)
        self.register_buffer('betas',betas);self.register_buffer('alpha_bars',alpha)

    @torch.no_grad()
    def sample_ddim(self,condition,*,steps=20,generator=None,initial_noise=None):
        if steps!=self.config.ddim_steps:raise ValueError('pilot freezes DDIM20')
        anchor=condition['global'];shape=(len(anchor),30,17)
        sample=(torch.randn(shape,device=anchor.device,dtype=anchor.dtype,generator=generator)
                if initial_noise is None else initial_noise.clone())
        if sample.shape!=shape:raise ValueError('paired latent shape mismatch')
        times=torch.linspace(99,0,steps,device=anchor.device).round().long()
        was_training=self.training;self.eval()
        try:
            for i,time in enumerate(times):
                t=torch.full((len(anchor),),int(time),device=anchor.device,dtype=torch.long)
                previous=self.alpha_bars[times[i+1]] if i+1<len(times) else sample.new_tensor(1.)
                epsilon,clean=self.predict_epsilon_and_x0(sample,t,condition)
                sample=previous.sqrt()*clean+(1-previous).sqrt()*epsilon
            return sample
        finally:self.train(was_training)

    def architecture_identity(self):
        return {'model':self.model_name,'full_controls':[32,17],'free_latent':[30,17],
            'boundary_elimination':'same original codec, C0 and C1 fixed by task; no clipping',
            'body':'original ConditionalDenoiser + same-information flat input' if self.model_name=='M0' else
                   '32 control tokens, typed cross-attention, global/timestep modulation in each block',
            'layers':4,'hidden_dim':128,'heads':4,'ffn_ratio':4,
            'parameters':sum(p.numel() for p in self.parameters()),
            'schedule':self.schedule_identity(),'prediction':self.parameterization_identity(),
            'modules':{name:type(module).__name__ for name,module in self.named_modules()},
            'new_controller_or_second_diffusion':False}
