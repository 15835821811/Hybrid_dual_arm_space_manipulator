import time
import numpy as np
import torch
from ff_prior.storage import read_json,digest
from ff_prior.diffusion import schedule,prediction_to_clean_noise
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.guidance import public_goal_initializer
from ff_prior_delivery_guidance.sampling import candidate,LatentProjector,ddim_update
from ff_prior_feasible.validation import validate
from ff_prior_generalization.experiment import execute
from ff_prior_expand256_train.learning import load as control_load
from .protocol import CONTROL,PROTOCOL as P

def unguided_sample(bundle,base,task,seed):
    model,codec=bundle;before=model.calls;center=codec.center(task['condition'],base)
    c=torch.tensor(codec.condition(task['condition'],base),dtype=torch.float32)[None]
    x=torch.randn((1,codec.dim),generator=torch.Generator().manual_seed(seed)).double()
    alpha=schedule().double();times=np.linspace(999,0,16).round().astype(int);steps=[]
    for j,t in enumerate(times):
        with torch.no_grad():clean,_=prediction_to_clean_noise(x,model(x.float(),torch.tensor([int(t)]),c).double(),alpha[t],'v')
        next_a=alpha[times[j+1]] if j<15 else torch.tensor(1.,dtype=torch.float64)
        x,res=ddim_update(x,clean,alpha[t],next_a);steps.append({'step':j,'t':int(t),'residual':res})
    if model.calls-before!=16:raise ValueError('unguided network budget')
    return codec.decode(codec.coefficients(x[0].numpy(),center),base),steps

def unguided(out,bundle,arrays,rows,data,cache,profile,ledger,budget,progress):
    groups={'A':control_load(CONTROL/'learning/goal_loss'),'B':bundle}
    for sj,seed in enumerate(P['val_seeds']):
        for j,i in enumerate(data['va']):
            row=rows[i];task=row['task'];rr,ii,cc,_=cache.get(row);base=arrays[1][i]
            for name,b in groups.items():
                key=f'unguided_{sj}_{j}_{name}';ledger.begin(key,'unguided',21600.,192)
                progress('M3_unguided_VAL',seed=sj+1,target=j+1,method=name)
                before=rr.calls.copy();budget.reserve(3)
                raw,steps=unguided_sample(b,base,task,seed+j*997)
                projector=LatentProjector(task,rr,ii,cc,profile,budget.deadline,codec=b[1],base=base)
                z,prep=projector.prepare(raw)
                result=validate(z,task,rr,ii,cc,profile,budget.deadline)
                result.update(method=name,sample_id=arrays[3][i],seed=seed+j*997,seed_group=sj,steps=steps,
                    raw_Z=raw,prepared_Z=z,preparation=prep,physics_guidance_calls=0,task_hash=task['task_hash'],endpoint_scope='prepared un-guided full16 DDIM; raw path stored, no claimed raw endpoint',
                    calls={k:rr.calls[k]-before[k] for k in before})
                atomic_json(out/'unguided'/str(sj)/name/f'{j:02d}.json',result);ledger.finish({'status':'DONE','eligible':result['eligible']},result['calls'])

def dev(out,bundle,cache,profile,ledger,budget,progress):
    entries=read_json(CONTROL/'frozen_tasks.json')['entries']
    atomic_json(out/'frozen_tasks.json',{'entries':entries,'reuse_from':str(CONTROL),'heldout_TEST':False})
    for sj,seed in enumerate(P['sampling_seeds']):
        for ti,e in enumerate(entries):
            task=e['task'];sid=e['sample_id'];rr,ii,cc,_=cache.get(e);base=public_goal_initializer(task,rr,ii);records=[]
            for k in range(5):
                ledger.begin(f'candidate_{sj}_{sid}_{k}','candidate',21600.,180);budget.reserve(113)
                progress('M3_frozen_DEV_B_candidates',seed=sj+1,task=ti+1,candidate=k+1)
                result=candidate('late4_diffusion',k,task,base,seed+ti*101+k*997,bundle,rr,ii,cc,profile,budget.deadline)
                result.update(method='world_B',sample_id=sid,seed_group=sj,split=e['split']);records.append(result)
                atomic_json(out/'candidates'/str(sj)/sid/f'{k:02d}.json',result);ledger.finish({'status':result['status']},result['model_calls'])
            chosen=min((v for v in records if v['eligible']),key=lambda v:(v['score'],v['candidate_id']),default=None)
            directory=out/'executions'/str(sj)/sid
            ledger.begin(f'execution_{sj}_{sid}','execution',21600.,36)
            atomic_json(directory/'selection.json',{'selected_id':chosen['candidate_id'] if chosen else None,'selection_before_execution':True,'task_hash':task['task_hash']})
            before=rr.calls.copy();progress('M3_frozen_DEV_B_native_replay',seed=sj+1,task=ti+1)
            if chosen:r=execute(directory,task,chosen['Z'],rr,ii,cc,profile,budget.deadline)
            else:r={'status':'NO_CANDIDATE'};atomic_json(directory/'outcome.json',r)
            ledger.finish(r,{k:rr.calls[k]-before[k] for k in before})
