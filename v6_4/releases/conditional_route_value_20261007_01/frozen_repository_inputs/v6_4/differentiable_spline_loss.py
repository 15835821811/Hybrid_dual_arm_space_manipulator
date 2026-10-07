"""Shared differentiable reference loss, before the disclosed execution repair."""
import numpy as np
import torch
from torch import nn
from v6_4.trajectory_codec import CubicBSplineCodec


class DecodedReferenceLoss(nn.Module):
    def __init__(self,mean,scale,q_ranges,dq_scales,*,sample_count=136):
        super().__init__()
        codec=CubicBSplineCodec(np.zeros(17),np.zeros(17))
        for key,value in [('control_mean',mean),('control_scale',scale),('q_ranges',q_ranges),('dq_scales',dq_scales),
                          ('q_basis',codec.basis(np.linspace(0,27,sample_count))),('dq_basis',codec.basis(np.linspace(0,27,sample_count),1))]:
            self.register_buffer(key,torch.as_tensor(np.asarray(value),dtype=torch.float32))
        if torch.any(self.control_scale<=0) or torch.any(self.q_ranges<=0) or torch.any(self.dq_scales<=0):
            raise ValueError('original positive scales required')

    def decode(self,standardized_free,fixed_controls):
        free=standardized_free*self.control_scale+self.control_mean
        controls=torch.cat((fixed_controls,free),dim=1)
        return torch.einsum('tc,bcd->btd',self.q_basis,controls),torch.einsum('tc,bcd->btd',self.dq_basis,controls)

    def forward(self,predicted_clean,label_clean,fixed_controls,alpha_bars):
        eligible=alpha_bars>=.1
        if not torch.any(eligible):
            zero=predicted_clean.sum()*0.
            return zero,zero
        q,dq=self.decode(predicted_clean[eligible],fixed_controls[eligible])
        label_q,label_dq=self.decode(label_clean[eligible],fixed_controls[eligible])
        return ((q-label_q)/self.q_ranges).square().mean(),((dq-label_dq)/self.dq_scales).square().mean()


def paired_training_loss(model,decoder,clean,condition,timesteps,epsilon):
    noisy,_=model.q_sample(clean,timesteps,epsilon)
    v=model.denoiser(noisy,timesteps,condition)
    signal,noise=model._coefficients(timesteps)
    target=signal*epsilon-noise*clean
    loss_noise=(v-target).square().mean()
    predicted_clean=signal*noisy-noise*v
    loss_q,loss_dq=decoder(predicted_clean,clean,condition['fixed_controls'],model.alpha_bars[timesteps])
    return loss_noise+.1*loss_q+.01*loss_dq,{'noise':loss_noise,'decoded_q':loss_q,'decoded_dq':loss_dq}
