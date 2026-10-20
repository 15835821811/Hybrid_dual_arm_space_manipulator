import numpy as np
import torch
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.diffusion import schedule
from ff_prior_origin_learn_bounded.distribution import population
from .protocol import ROOT,CONTROL

def load(codec):
    audit=read_json(CONTROL/'training_input_audit.json');index=audit['records'];n=len(index)
    with np.load(CONTROL/'loaded_training_arrays.npz') as a:z=a['Z'].copy();bases=a['bases'].copy()
    rows=[read_json(ROOT/e['record']) for e in index]
    if any(file_digest(ROOT/e['record'])!=e['record_sha256'] for e in index):raise ValueError('training row changed')
    conditions=[r['condition'] for r in rows];ids=[e['sample_id'] for e in index];splits=[e['split'] for e in index]
    states=[e['initial_state_id'] for e in audit['runtime_proofs']]
    tr=[i for i,s in enumerate(splits) if s=='TRAIN'];va=[i for i,s in enumerate(splits) if s=='VAL']
    t=read_json(CONTROL/'learning/goal_loss/training.json')
    if len(tr)!=14222 or len(va)!=48 or digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states})!=t['metadata']['data_hash']:raise ValueError('TRAIN content hash mismatch')
    centers=np.stack([codec.center(c,b) for c,b in zip(conditions,bases)])
    x=torch.tensor(codec.normalized(np.stack([codec.encode(v,b) for v,b in zip(z,bases)]),centers),dtype=torch.float32)
    c=torch.tensor(np.stack([codec.condition(q,b) for q,b in zip(conditions,bases)]),dtype=torch.float32)
    summary=read_json(CONTROL/'sensitivity_summary.json');mat={}
    for k,b in summary['bindings'].items():
        if file_digest(ROOT/b['array'])!=b['array_sha256']:raise ValueError('G bytes changed')
        with np.load(ROOT/b['array']) as a:mat[k]=a['G'].copy()
    g=np.stack([mat[rows[i]['task']['task_hash']] for i in tr])
    if digest(g)!=t['metadata']['sensitivity_matrix_hash']:raise ValueError('G content hash mismatch')
    slots,keys,_,pop=population(conditions,ids,splits)
    data={'x':x,'c':c,'centers':centers,'alpha':schedule(),'tr':np.array(tr),'va':va,'keys':keys,'population':pop,
      'lookup':{i:j for j,i in enumerate(tr)},'ids':{sid:i for i,sid in enumerate(ids)}}
    return (z,bases,conditions,ids,splits,states),rows,torch.tensor(g*codec.latent_scale[None,None,:],dtype=torch.float32),data
