"""Existing evidence only: neural autograd and algebra; NO optimizer.step/physics."""
import sys, json, hashlib, zipfile, copy
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from ff_prior.storage import digest, file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_delivery_train.learning import load as warm_load
from ff_prior_endpoint_precision.data import load as data_load
from ff_prior_endpoint_precision.protocol import WARM, CONTROL
from ff_prior_paired_objectives.objectives import predict, terms
OUT=ROOT/'artifacts/v7_1/endpoint_precision_bounded_001'
DEST=Path(__file__).parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def vec(model): return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
def grad(loss,model):return torch.cat([p.reshape(-1) for p in torch.autograd.grad(loss,tuple(model.parameters()))]).detach()
def setvec(model,v):
    j=0
    with torch.no_grad():
        for p in model.parameters():p.copy_(v[j:j+p.numel()].view_as(p));j+=p.numel()
def loss_pose(ee,goals):
    ee=np.asarray(ee);goals=np.asarray(goals)
    pos=np.linalg.norm(ee[:,:3,3]-goals[:,:3,3],axis=1)
    rot=Rotation.from_matrix(goals[:,:3,:3].transpose(0,2,1)@ee[:,:3,:3]).magnitude()
    q=np.r_[pos/.005,rot/np.deg2rad(1)]
    return float(np.max(q*q)+.05*np.mean(q*q)),q
def manual_adamw(weights,gradient,params,group):
    gs=[];j=0
    for p in params:gs.append(gradient[j:j+p.numel()].view_as(p));j+=p.numel()
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gs]))
    factor=torch.clamp(1./(norm+1e-6),max=1.)
    result=[];j=0
    b1,b2=group['betas']
    for p,g in zip(params,gs):
        w=weights[j:j+p.numel()].view_as(p).clone();j+=p.numel()
        g=g*factor
        m=torch.zeros_like(g).lerp_(g,1-b1)
        v=torch.zeros_like(g).mul_(b2).addcmul_(g,g,value=1-b2)
        w.mul_(1-group['lr']*group['weight_decay'])
        denom=(v.sqrt()/(1-b2)**.5).add_(group['eps'])
        w.addcdiv_(m,denom,value=-group['lr']/(1-b1))
        result.append(w.reshape(-1))
    return torch.cat(result)-weights,float(norm)
torch.set_num_threads(2)
run=read(OUT/'run.json');ident=run['identity'];checked=[]
with zipfile.ZipFile(OUT/'executed_source.zip') as z:
    for n,h in ident['source_hashes'].items():
        assert hashlib.sha256(z.read(n)).hexdigest()==h
        assert file_digest(ROOT/n)==h
for n,h in ident['binding']['files'].items():assert file_digest(ROOT/n)==h
assert read(OUT/'physical_contract.json')['acceptance']==read(CONTROL/'physical_contract.json')['acceptance']
model,codec=warm_load(WARM/'learning/goal_loss');warm_hash=weight_hash(model)
arrays,rows,G,data=data_load(codec)
stats={'source_files_checked':len(ident['source_hashes']),'bound_files_checked':len(ident['binding']['files']),
 'producer_head':run['producer_head'],'physical_acceptance_unchanged':True,'cases':[],
 'new_physical_calls_by_review':0,'new_optimizer_steps_by_review':0}
fail_reasons=Counter();case_reasons=Counter();accepted_fractions=Counter();by_noise=Counter();maxerr=Counter();all_states_empty=True
for path in sorted((OUT/'diagnostic').glob('*/result.json')):
    r=read(path);b=torch.load(path.parent/'before.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(b['weights']);model.eval();initial=vec(model)
    assert weight_hash(model)==warm_hash==r['input_weight_hash']==r['output_weight_hash']
    assert b['optimizer']['state']=={};assert len(b['optimizer']['param_groups'])==1
    i,t,seed=b['specs'][0]; assert arrays[4][i]=='TRAIN'
    pr=read(path.parent/'gradient_0.json');a=np.load(path.parent/'gradient_0.npz');p=np.load(path.parent/'parameter_proposals.npz')
    pred=predict(model,data,i,t,seed)
    assert np.array_equal(pred.detach().numpy().astype(float),a['x'])
    assert file_digest(path.parent/'gradient_0.npz')==pr['arrays_sha256']
    assert digest(a['x'])==pr['x_hash'] and pr['input_weight_hash']==warm_hash
    assert pr['task_hash']==rows[i]['task']['task_hash']
    goals=rows[i]['task']['condition']['goals']
    values=np.array([[loss_pose(e,goals)[0] for e in pair] for pair in a['perturbation_ee']])
    assert np.allclose(values,a['perturbation_losses'],rtol=1e-12,atol=1e-12)
    fd=(values[:,1]-values[:,0])/(2*pr['epsilon'])
    assert np.allclose(fd,a['gradient'],rtol=1e-12,atol=1e-12)
    assert abs(loss_pose(pr['central']['ee'],goals)[0]-pr['true_loss'])<1e-10
    assert a['gradient'].shape==(102,) and pr['calls']['coupled']==212
    wg=grad((pred*torch.tensor(a['gradient'],dtype=pred.dtype)).sum(),model)
    noise=torch.randn((1,102),generator=torch.Generator().manual_seed(seed))
    vl,ll,_=terms(model,data['x'][[i]],data['c'][[i]],torch.tensor([t]),noise,data['alpha'],G[[data['lookup'][i]]])
    bg=grad(vl+.05*ll,model)
    for name,v in [('base_gradient',bg),('world_gradient',wg)]:
        e=float(np.max(abs(v.numpy()-p[name])));maxerr[name]=max(maxerr[name],e)
        assert np.array_equal(v.numpy(),p[name]),(path,name,e)
    scale=.1*float(bg.norm())/float(wg.norm());assert scale==r['world_coefficient']
    for name,g in [('base',bg),('combined',bg+scale*wg)]:
        delta,norm=manual_adamw(initial,g,list(model.parameters()),b['optimizer']['param_groups'][0])
        e=float(np.max(abs(delta.numpy()-p[name+'_delta'])));maxerr[name+'_delta']=max(maxerr[name+'_delta'],e)
        assert np.allclose(delta.numpy(),p[name+'_delta'],rtol=0,atol=6e-8),(path,name,e)
    def record_check(record,spec):
        ri,rt,rs=spec;x=predict(model,data,ri,rt,rs).detach().numpy().astype(float)
        assert digest(x)==record['x_hash']
        z=codec.decode(codec.coefficients(x,data['centers'][ri]),arrays[1][ri]);assert digest(z)==record['Z_hash']
        l,q=loss_pose(record['ee'],rows[ri]['task']['condition']['goals'])
        assert abs(l-record['loss'])<1e-9
        assert np.allclose(q,np.r_[np.asarray(record['position_m'])/.005,np.asarray(record['rotation_rad'])/np.deg2rad(1)],rtol=1e-12,atol=1e-12)
        return q
    checkspec=b['specs']+b['controls'];before=[record_check(rec,s) for rec,s in zip(r['before'],checkspec)]
    trials=[];reasons=set()
    for trial in r['trials']:
        setvec(model,initial+trial['fraction']*torch.from_numpy(p['combined_delta']))
        after=[record_check(rec,s) for rec,s in zip(trial['after'],checkspec)]
        why=[]
        for k,(v,w,rec) in enumerate(zip(before,after,trial['after'])):
            label='event' if k==0 else 'control'
            if not np.isfinite(w).all():why.append(label+'_nonfinite')
            if not max(w)<max(v)-1e-9:why.append(label+'_Q_not_decreased')
            if not np.all(w[v<=1]<=1):why.append(label+'_passed_group_crossed')
            if not rec['domain']:why.append(label+'_domain')
        assert trial['accepted']==(not why)
        fail_reasons.update(why);reasons.update(why)
        trials.append({'fraction':trial['fraction'],'accepted':trial['accepted'],'reasons':why,
                       'Q_before':[float(max(v)) for v in before],'Q_after':[float(max(v)) for v in after]})
    assert r['accepted']==any(t['accepted'] for t in trials)
    assert r['actual_coupled_calls']==212+2+2*len(trials)
    assert not r['updates_retained'] and r['ordinary_updates_committed']==0
    if r['accepted']:accepted_fractions[str(r['fraction'])]+=1;by_noise[str(t)]+=1
    else:case_reasons.update(reasons)
    stats['cases'].append({'id':path.parent.name,'t':t,'sample_id':arrays[3][i],'accepted':r['accepted'],'trials':trials,
       'base_delta_vs_combined_l2':float(np.linalg.norm(p['base_delta']-p['combined_delta'])),
       'base_linear_world_direction':float(np.dot(p['world_gradient'],p['base_delta'])),
       'combined_linear_world_direction':float(np.dot(p['world_gradient'],p['combined_delta']))})
    print(path.parent.name,'checked',flush=True)
stats.update(accepted=sum(c['accepted'] for c in stats['cases']),accepted_per_noise=dict(by_noise),accepted_fractions=dict(accepted_fractions),
 trial_failure_reasons=dict(fail_reasons),rejected_case_reasons=dict(case_reasons),max_reconstruction_abs_error=dict(maxerr),
 all_saved_optimizer_states_empty=all_states_empty)
stats['reconstructed_coupled_calls']=sum(212+2+2*len(c['trials']) for c in stats['cases'])
assert stats['reconstructed_coupled_calls']==read(OUT/'ledger.json')['calls']['diagnostic.coupled']==read(OUT/'query_budget.json')['reserved_coupled_upper']
stats['absent_dependent_directories']={n:not (OUT/n).exists() for n in ['learning','world_events','unguided','candidates','executions']}
assert all(stats['absent_dependent_directories'].values())
stats['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read(ROOT/'paper/PAPER_CLAIM_AUDIT.json')
stats['status']='PASS_READONLY_RECONSTRUCTION'
(DEST/'independent_checks.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in stats.items() if k!='cases'},ensure_ascii=False,indent=2))
