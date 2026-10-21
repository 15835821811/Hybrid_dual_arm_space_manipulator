import numpy as np
from ff_prior_latent.learning import Codec,feature
from ff_prior.storage import digest

class FullCodec(Codec):
    @classmethod
    def fit(cls,z,bases,conditions,ids,splits):
        if not splits or set(splits)!={'TRAIN'}:raise ValueError('TRAIN-only fit required')
        z,bases=np.asarray(z),np.asarray(bases)
        if z.shape!=bases.shape or z.shape!=(len(ids),19,17) or len(set(ids))!=len(ids):raise ValueError('fit identity')
        y=(z-bases).reshape(len(z),323)
        c=np.stack([feature(v,b) for v,b in zip(conditions,bases)])
        cm,cs=c.mean(0),c.std(0);cs=np.where(cs<1e-8,1.,cs)
        design=np.c_[np.ones(len(c)),(c-cm)/cs]
        penalty=np.eye(design.shape[1])*10.;penalty[0,0]=0
        ridge=np.linalg.solve(design.T@design+penalty,design.T@y)
        rest=y-design@ridge
        state={'basis':np.eye(19),'feature_mean':cm,'feature_scale':cs,'ridge':ridge,
          'latent_mean':rest.mean(0),'latent_scale':np.maximum(rest.std(0),1e-4),
          'fit_ids':ids,'fit_split':'TRAIN','fit_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids}),
          'rank':19,'dim':323,'basis_fitted':False,'basis_kind':'identity_no_temporal_compression'}
        return cls(state)

    def encode(self,z,base):return (np.asarray(z)-base).reshape(323)
    def decode(self,y,base):return np.asarray(base)+np.asarray(y).reshape(19,17)
