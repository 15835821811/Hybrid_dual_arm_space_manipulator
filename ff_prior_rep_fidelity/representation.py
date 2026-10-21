"""Fixed bases and explicit preprocessing; no fitting or trajectory search."""
import numpy as np
from ff_prior_goal.projection import smooth
from .protocol import P

def fixed_basis(original,rank):
    original=np.asarray(original,dtype=np.float64)
    assert original.shape==(19,6) and rank in P['ranks']
    assert np.allclose(original.T@original,np.eye(6),atol=1e-12,rtol=0)
    if rank==19:return np.eye(19)
    vectors=[v.copy() for v in original.T]
    candidates=[np.eye(19)[-1]]+[np.cos(np.pi*(np.arange(19)+.5)*k/19) for k in range(19)]
    for c in candidates:
        if len(vectors)>=rank:break
        u=c.copy()
        for _ in range(2):
            for v in vectors:u-=v*(v@u)
        norm=np.linalg.norm(u)
        if norm>1e-10:vectors.append(u/norm)
    b=np.stack(vectors,axis=1)
    assert b.shape==(19,rank) and np.allclose(b.T@b,np.eye(rank),atol=1e-12,rtol=0)
    return b

def project(z,base,basis):
    z=np.asarray(z,dtype=np.float64)
    if basis.shape[1]==19:return z.copy()  # exact identity, no fabricated compression benefit
    return np.asarray(base)+basis@(basis.T@(z-base))

def prepare(raw,base,basis,q0,lower,upper,quick,mode):
    """raw is the representation output, as it would be at sampler deployment."""
    if mode=='raw':return raw.copy(),{'input':raw.copy(),'policy':'no_prepare'}
    details={'input':raw.copy(),'policy':mode}
    if mode=='preserve_prepare' and quick(raw):
        return raw.copy(),dict(details,unchanged=True,shrink=1.,reason='already_quick_feasible')
    if mode=='legacy_prepare':
        smoothed=smooth(raw,q0,P['legacy_smooth_weight'])
        clipped=np.clip(smoothed,lower,upper)
        candidate=project(clipped,base,basis)
        details.update(smoothed=smoothed,clipped=clipped,projected=candidate)
    else:candidate=raw.copy();details.update(projected=candidate)
    residual=candidate-base
    for scale in P['shrink_scales']:
        # Canonical scale=1 avoids an avoidable roundoff-only change to a valid candidate.
        trial=candidate.copy() if scale==1 else project(base+scale*residual,base,basis)
        if quick(trial):return trial,dict(details,unchanged=bool(np.array_equal(trial,raw)),shrink=scale)
    return candidate,dict(details,unchanged=False,shrink=None,quick_feasible=False)
