import numpy as np
import torch
from ff_prior.diffusion import prediction_to_clean_noise,schedule
from ff_prior_origin_learn_bounded.distribution import population
from .protocol import PROTOCOL as P

def prepare(arrays,codec,rows,targets,new_ids):
    z,bases,conditions,ids,splits,states=arrays
    centers=np.stack([codec.center(c,b) for c,b in zip(conditions,bases)])
    encoded=np.stack([codec.encode(v,b) for v,b in zip(z,bases)])
    x=torch.tensor(codec.normalized(encoded,centers),dtype=torch.float32)
    c=torch.tensor(np.stack([codec.condition(q,b) for q,b in zip(conditions,bases)]),dtype=torch.float32)
    slots,keys,weights,pop=population(conditions,ids,splits)
    groups=[[i for i,sid in sorted(enumerate(ids),key=lambda p:p[1])
             if sid in new_ids and rows[i]['parent_sample_id']==t['parent_sample_id']] for t in targets]
    if any(len(g)!=4 for g in groups):raise ValueError('16 groups of four fixed corrections required')
    return {'x':x,'c':c,'centers':centers,'slots':slots,'keys':keys,'population':pop,'groups':groups,'alpha':schedule()}

def random_batch(step,batch,dimension):
    rng=torch.Generator().manual_seed(P['training_seed']+step)
    return torch.randint(0,1000,(batch,),generator=rng),torch.randn((batch,dimension),generator=rng)

def terms(model,clean,cond,t,noise,alpha,G):
    a=alpha[t,None];noisy=a.sqrt()*clean+(1-a).sqrt()*noise
    target=a.sqrt()*noise-(1-a).sqrt()*clean
    v=model(noisy,t,cond);pred,_=prediction_to_clean_noise(noisy,v,a,'v')
    return (v-target).square().mean(),torch.einsum('bij,bj->bi',G,pred-clean).square().mean(),pred

def predict(model,data,i,t,seed):
    clean=data['x'][i:i+1];cond=data['c'][i:i+1];a=data['alpha'][t]
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed))
    noisy=a.sqrt()*clean+(1-a).sqrt()*noise
    pred,_=prediction_to_clean_noise(noisy,model(noisy,torch.tensor([t]),cond),a,'v')
    return pred[0]

def vector_gradient(value,model):
    return torch.cat([v.reshape(-1) for v in torch.autograd.grad(value,tuple(model.parameters()),retain_graph=True)])

def gradient_statistics(vectors):
    norms={k:float(torch.linalg.vector_norm(v)) for k,v in vectors.items()}
    cos={a+'__'+b:float(torch.dot(vectors[a],vectors[b])/max(norms[a]*norms[b],1e-30))
         for a in vectors for b in vectors if a<b}
    return {'norms':norms,'cosines':cos}

def set_vector(model,vector):
    torch.nn.utils.vector_to_parameters(vector.detach().clone(),model.parameters())

def get_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
