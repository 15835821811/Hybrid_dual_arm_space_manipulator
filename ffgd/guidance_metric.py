"""Full-rank Sobolev metric for a denoising gradient, never trajectory smoothing.

I + weight * integral(A''^T A'')/T is SPD on every normalized free coordinate.
It changes a descent direction inside a sampling step; no control point is fixed
and no path optimization or post-sampling correction is performed here.
"""
import numpy as np
import torch
from .trajectory import basis


def acceleration_metric(codec,scale,limits,grid,weight):
    if not np.isfinite(weight) or weight<0:raise ValueError('invalid metric weight')
    times=np.asarray(grid);dt=np.diff(times)
    quadrature=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]/codec.duration
    controls=codec.N.reshape(codec.n,codec.dof,codec.free_dim)*np.asarray(scale)[None,None,:]
    operator=np.einsum('tn,njf->tjf',basis(codec.n,times,codec.duration,2),controls)/np.asarray(limits)[None,:,None]
    operator=operator*np.sqrt(quadrature[:,None,None]/codec.dof)
    A=operator.reshape(-1,codec.free_dim)
    return torch.tensor(np.eye(codec.free_dim)+weight*(A.T@A),dtype=torch.float64)
