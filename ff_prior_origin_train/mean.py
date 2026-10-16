"""Strong TRAIN-only mean using the same balanced empirical distribution."""
from pathlib import Path
import copy
import numpy as np
from ff_prior.storage import digest,read_json
from ff_prior_coverage_train.storage import atomic_json
from ff_prior_latent.learning import Codec,feature


def fit(directory,arrays,codec,coverage_ids):
    z,bases,conditions,ids,splits,_=arrays
    mask=np.array([s=='TRAIN' for s in splits]);slots=np.flatnonzero(mask)
    coverage=set(coverage_ids)
    if coverage-set(np.asarray(ids)[mask]):raise ValueError('coverage group not TRAIN')
    new=np.array([ids[i] in coverage for i in slots]);n=len(slots)
    if not any(new) or all(new):raise ValueError('both balanced TRAIN groups required')
    w=np.where(new,.5*n/sum(new),.5*n/sum(~new))
    c=np.stack([feature(conditions[i],bases[i]) for i in slots])
    cm=np.average(c,axis=0,weights=w)
    cs=np.sqrt(np.average((c-cm)**2,axis=0,weights=w));cs=np.where(cs<1e-8,1.,cs)
    y=np.stack([codec.encode(z[i],bases[i]) for i in slots])
    d=np.c_[np.ones(n),(c-cm)/cs]
    penalty=np.eye(d.shape[1])*10.;penalty[0,0]=0.
    ridge=np.linalg.solve(d.T@(w[:,None]*d)+penalty,d.T@(w[:,None]*y))
    state=copy.deepcopy(codec.state)
    fit_ids=[ids[i] for i in slots]
    state.update(feature_mean=cm,feature_scale=cs,ridge=ridge,fit_ids=fit_ids,
        fit_hash=digest({'z':z[mask],'bases':bases[mask],'conditions':[conditions[i] for i in slots],'ids':fit_ids}),
        mean_contract={'TRAIN_only':True,'balanced_group_weight':[.5,.5],
                       'coverage_ids':sorted(coverage),'lambda':10.,'basis_hash':digest(codec.basis)},
        mean_weight_hash=digest(w))
    if not np.isfinite(ridge).all():raise ValueError('nonfinite weighted mean')
    path=Path(directory)/'weighted_mean.json'
    if path.exists():
        if digest(read_json(path))!=digest(state):raise ValueError('weighted mean resume changed')
    else:atomic_json(path,state)
    return Codec(state)
