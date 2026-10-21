import numpy as np
import torch
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_origin_learn_bounded.distribution import population
from ff_prior.diffusion import schedule
from .protocol import ROOT,DATA
from .codec import FullCodec

def load():
    audit=read_json(DATA/'training_input_audit.json');index=audit['records']
    with np.load(DATA/'loaded_training_arrays.npz') as a:z,bases=a['Z'].copy(),a['bases'].copy()
    rows=[]
    for e in index:
        assert file_digest(ROOT/e['record'])==e['record_sha256']
        r=read_json(ROOT/e['record']);assert r['sample_id']==e['sample_id'] and r['split']==e['split']
        assert e['validation_level'] in ('MODEL_VALID','EXECUTION_VERIFIED') and r['training_eligible']==(e['split']=='TRAIN')
        rows.append(r)
    conditions=[r['condition'] for r in rows];ids=[e['sample_id'] for e in index];splits=[e['split'] for e in index]
    states=[e['initial_state_id'] for e in audit['runtime_proofs']]
    assert digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states})==read_json(DATA/'learning/goal_loss/training.json')['metadata']['data_hash']
    tr=np.flatnonzero(np.asarray(splits)=='TRAIN');va=np.flatnonzero(np.asarray(splits)=='VAL')
    assert len(tr)==14222 and len(va)==48
    codec=FullCodec.fit(z[tr],bases[tr],[conditions[i] for i in tr],[ids[i] for i in tr],['TRAIN']*len(tr))
    centers=np.stack([codec.center(c,b) for c,b in zip(conditions,bases)])
    x=codec.normalized(np.stack([codec.encode(v,b) for v,b in zip(z,bases)]),centers)
    c=np.stack([codec.condition(v,b) for v,b in zip(conditions,bases)])
    reconstructed=np.stack([codec.decode(codec.coefficients(v,m),b) for v,m,b in zip(x,centers,bases)])
    error=float(abs(reconstructed-z).max());assert error<=1e-12
    _,keys,_,pop=population(conditions,ids,splits)
    data={'x':torch.tensor(x,dtype=torch.float32),'c':torch.tensor(c,dtype=torch.float32),
        'centers':centers,'tr':tr,'va':va,'keys':keys,'population':pop,'alpha':schedule(),
        'ids':{sid:i for i,sid in enumerate(ids)},'reconstruction_max_rad':error}
    return codec,(z,bases,conditions,ids,splits,states),rows,data
