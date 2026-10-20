"""Read-only AdamW algebra; no optimizer.step or physical propagation."""
import copy
import torch
from ff_prior_paired_objectives.objectives import get_vector,set_vector,predict,vector_gradient
from .optimization import optimizer_hash

def reconstruct_adamw(model,gradient,state):
    params=list(model.parameters());groups=state['param_groups'];assert len(groups)==1
    group=groups[0];assert not group['amsgrad'] and not group.get('maximize',False)
    gs=[];offset=0
    for p in params:gs.append(gradient[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gs]))
    factor=torch.clamp(1./(norm+1e-6),max=1.);result=[];after=copy.deepcopy(state);b1,b2=group['betas']
    for pid,p,g in zip(group['params'],params,gs):
        old=state['state'].get(pid,{})
        step=old.get('step',torch.tensor(0.)).clone()+1
        m=old.get('exp_avg',torch.zeros_like(p)).clone();v=old.get('exp_avg_sq',torch.zeros_like(p)).clone()
        g=g*factor;m.lerp_(g,1-b1);v.mul_(b2).addcmul_(g,g,value=1-b2)
        w=p.detach().clone().mul_(1-group['lr']*group['weight_decay'])
        denom=(v.sqrt()/(1-b2**float(step))**.5).add_(group['eps'])
        w.addcdiv_(m,denom,value=-group['lr']/(1-b1**float(step)))
        result.append(w.reshape(-1));after['state'][pid]={'step':step,'exp_avg':m,'exp_avg_sq':v}
    return torch.cat(result),after

def state_close(a,b):
    assert a['param_groups']==b['param_groups'] and set(a['state'])==set(b['state'])
    for k,st in a['state'].items():
        assert set(st)==set(b['state'][k])
        for name,v in st.items():
            other=b['state'][k][name]
            if isinstance(v,torch.Tensor):
                if name=='step':assert torch.equal(v,other)
                else:assert torch.allclose(v,other,rtol=2e-6,atol=1e-9),(k,name)
            else:assert v==other
    return True
