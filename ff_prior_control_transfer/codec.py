import copy
import numpy as np
from ff_prior_latent.learning import Codec
from ff_prior.storage import read_json,digest
from .protocol import SOURCE,P

class FullCodec(Codec):
    @classmethod
    def fit(cls,z,bases,conditions,ids,splits):
        if not splits or set(splits)!={'TRAIN'}:raise ValueError('TRAIN-only fit required')
        old=read_json(SOURCE/'learning/aligned/codec.json');low=Codec(old)
        assert low.rank==6 and low.dim==102 and set(low.state['fit_ids'])<=set(ids)
        _,_,vt=np.linalg.svd(low.basis.T,full_matrices=True)
        basis=np.c_[low.basis,vt[6:].T]
        residual=np.asarray(z)-np.asarray(bases)
        detail=np.einsum('tk,ntj->nkj',basis[:,6:],residual).reshape(len(z),221)
        # All detail coordinates exist. Their physical amplitudes are TRAIN-only;
        # floor controls small numerical modes, while max support includes every teacher.
        scale=np.maximum(detail.std(0),1e-4)
        limit=np.maximum(P['detail_bound_sigma'],abs(detail/scale).max(0)+1.)
        state=copy.deepcopy(old);state.update(basis=basis,rank=19,dim=323,
            ridge=np.c_[low.ridge,np.zeros((66,221))],latent_mean=np.r_[low.latent_mean,np.zeros(221)],
            latent_scale=np.r_[low.latent_scale,scale],detail_limit=limit,fit_ids=ids,
            fit_hash=digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids}),
            basis_kind='old_six_columns_plus_complete_orthogonal_complement',
            source_codec_hash=digest(old),low_state=old,detail_fit_split='TRAIN')
        return cls(state)
