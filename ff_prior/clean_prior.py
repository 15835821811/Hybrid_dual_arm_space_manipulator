"""TRAIN-only temporal clean-estimate regularizer; not a physical validator.

The current generator occupies a smooth, low-rank temporal subspace. Project
denoised estimates there, never noisy states. Goals and terminal configurations
are not inputs to fitting. All resulting candidates still require full gates.
"""
from pathlib import Path
import numpy as np
from .storage import digest,atomic_json,read_json


class TemporalPrior:
    def __init__(self,projector,q0,metadata):
        self.projector=np.asarray(projector,dtype=np.float64)
        self.q0=np.asarray(q0,dtype=np.float64)
        self.metadata=metadata
        if self.projector.shape!=(19,19) or self.q0.shape!=(17,) or not np.isfinite(self.projector).all():
            raise ValueError('invalid clean temporal prior')
        if not np.allclose(self.projector@self.projector,self.projector,atol=1e-12,rtol=0):
            raise ValueError('temporal projector must be idempotent')

    @classmethod
    def fit(cls,z,q0,sample_ids,splits,robot_hash,initial_state_id):
        z=np.asarray(z,dtype=np.float64)
        if not splits or len(splits)!=len(z) or any(s!='TRAIN' for s in splits) or len(set(sample_ids))!=len(z):
            raise ValueError('temporal prior is unique TRAIN-only')
        if z.shape!=(len(z),19,17) or not np.isfinite(z).all():
            raise ValueError('invalid temporal fitting array')
        matrix=(z-np.asarray(q0)).transpose(1,0,2).reshape(19,-1)
        u,s,_=np.linalg.svd(matrix,full_matrices=False)
        energy=np.cumsum(s*s)
        rank=int(np.searchsorted(energy/energy[-1],1-1e-7)+1) if energy[-1]>0 else 0
        projector=u[:,:rank]@u[:,:rank].T
        return cls(projector,q0,{'kind':'TRAIN_TEMPORAL_SVD','retained_energy':1-1e-7,'rank':rank,
            'fit_split':'TRAIN','sample_ids':list(sample_ids),'training_array_hash':digest(z),
            'robot_hash':robot_hash,'initial_state_id':initial_state_id,
            'maximum_training_reconstruction_error_rad':float(abs(projector@matrix-matrix).max())})

    def project(self,z):
        z=np.asarray(z,dtype=np.float64)
        if z.shape!=(19,17) or not np.isfinite(z).all():
            raise ValueError('nonfinite or mismatched clean estimate')
        return self.q0+self.projector@(z-self.q0)

    def save(self,path):
        v={'projector':self.projector,'q0':self.q0,'metadata':self.metadata}
        atomic_json(path,dict(v,hash=digest(v)))

    @classmethod
    def load(cls,path,expected_hash,robot_hash,initial_state_id):
        value=read_json(path); stored=value.pop('hash')
        if digest(value)!=stored or stored!=expected_hash or value['metadata']['robot_hash']!=robot_hash or value['metadata']['initial_state_id']!=initial_state_id or value['metadata']['fit_split']!='TRAIN':
            raise ValueError('clean prior identity/hash mismatch')
        return cls(**value)
