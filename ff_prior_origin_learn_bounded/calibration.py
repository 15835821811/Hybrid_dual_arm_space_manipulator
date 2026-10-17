"""Recalibrate public features using TRAIN only, preserving the warm-start map."""
import copy
import numpy as np
import torch
from ff_prior.storage import digest
from ff_prior.diffusion import weight_hash
from ff_prior_latent.learning import Codec,feature
from .distribution import population


def calibrate(old,arrays):
    z,bases,conditions,ids,splits,_=arrays
    slots,keys,w,p=population(conditions,ids,splits)
    f=np.stack([feature(conditions[i],bases[i]) for i in slots])
    mu=np.average(f,axis=0,weights=w)
    scale=np.sqrt(np.average((f-mu)**2,axis=0,weights=w));scale=np.where(scale<1e-8,1.,scale)
    a=scale/old.feature_scale;b=(mu-old.feature_mean)/old.feature_scale
    ridge=old.ridge.copy();ridge[0]=old.ridge[0]+b@old.ridge[1:]
    ridge[1:]=a[:,None]*old.ridge[1:]
    state=copy.deepcopy(old.state)
    state.update(feature_mean=mu,feature_scale=scale,ridge=ridge,
        condition_calibration={'TRAIN_only':True,'population':p,'training_ids':[ids[i] for i in slots],
            'public_features_hash':digest(f),'old_codec_hash':digest(old.state),
            'affine_a':a,'affine_b':b,'physical_center_preserved':True,
            'basis_and_latent_normalization_unchanged':True})
    return Codec(state)


def rebase(model,old,new):
    if not all(np.array_equal(getattr(old,k),getattr(new,k)) for k in ('basis','latent_mean','latent_scale')):
        raise ValueError('warm-start basis/latent normalization changed')
    a=new.feature_scale/old.feature_scale;b=(new.feature_mean-old.feature_mean)/old.feature_scale
    result=copy.deepcopy(model); layer=result.net[0];n=old.dim;c=len(old.feature_mean)
    with torch.no_grad():
        W=layer.weight[:,n:n+c].double().clone()
        layer.weight[:,n:n+c].copy_((W*torch.as_tensor(a)).to(layer.weight.dtype))
        layer.bias.copy_((layer.bias.double()+W@torch.as_tensor(b)).to(layer.bias.dtype))
    result.eval();return result


def equivalence(model,rebased,old,new,arrays):
    _,bases,conditions,ids,splits,_=arrays
    slots=np.flatnonzero(np.asarray(splits)=='TRAIN')
    co=np.stack([old.condition(conditions[i],bases[i]) for i in slots])
    cn=np.stack([new.condition(conditions[i],bases[i]) for i in slots])
    oldcent=np.c_[np.ones(len(co)),co]@old.ridge
    newcent=np.c_[np.ones(len(cn)),cn]@new.ridge
    center_error=float(np.max(abs(oldcent-newcent)))
    if center_error>1e-9:raise ValueError('affine calibration changed physical center')
    maximum=0.;relative=0.;model.eval();rebased.eval()
    # Every actual TRAIN condition, deterministic latent inputs; not a physical or denoising-success test.
    with torch.no_grad():
        for start in range(0,len(slots),256):
            stop=min(start+256,len(slots));n=stop-start
            x=torch.sin(torch.arange(n*old.dim,dtype=torch.float32).reshape(n,old.dim)*.013)
            t=torch.tensor([0,500,999]*((n+2)//3))[:n]
            u=model(x,t,torch.tensor(co[start:stop],dtype=torch.float32))
            v=rebased(x,t,torch.tensor(cn[start:stop],dtype=torch.float32))
            error=abs(u-v);tol=1e-5+5e-5*abs(u)
            if not bool(torch.all(error<=tol)):raise ValueError('affine warm-start prediction changed beyond float tolerance')
            maximum=max(maximum,float(error.max()));relative=max(relative,float((error/tol).max()))
    return {'status':'PASS','TRAIN_conditions_tested':len(slots),'old_weight_hash':weight_hash(model),
        'rebased_weight_hash':weight_hash(rebased),'old_codec_hash':digest(old.state),'new_codec_hash':digest(new.state),
        'max_center_difference':center_error,'max_prediction_difference':maximum,
        'maximum_fraction_of_float_tolerance':relative,'float_atol':1e-5,'float_rtol':5e-5,
        'old_max_abs_normalized_condition':float(abs(co).max()),'new_max_abs_normalized_condition':float(abs(cn).max()),
        'scope':'algebraic affine map plus actual TRAIN numeric checks; no physical-success assertion'}
