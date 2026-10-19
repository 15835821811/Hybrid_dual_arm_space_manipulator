import numpy as np
import torch
from ff_prior.storage import digest,read_json,file_digest
from ff_prior.diffusion import weight_hash
from ff_prior_coverage_train.storage import atomic_json,atomic_npz
from ff_prior_world_endpoint import endpoint
from .objectives import predict,terms,vector_gradient,gradient_statistics,get_vector,set_vector
from .protocol import ARMS,PROTOCOL as P

def run(out,model,codec,data,arrays,rows,G,cache,ledger,deadline,progress):
    path=out/'single_step_summary.json'
    if path.exists():return read_json(path)
    groups=data['groups'];chosen=[]
    for obstacle in (False,True):
        pool=[g[0] for g in groups if bool(rows[g[0]]['scene']['objects'])==obstacle]
        chosen+=sorted(pool,key=lambda i:digest({'paired_single_step':arrays[3][i]}))[:2]
    lookup={int(v):j for j,v in enumerate(data['slots'])};original=get_vector(model)
    original_hash=weight_hash(model);cases=[]
    for j,i in enumerate(chosen):
        for k,t in enumerate(P['world_noise_times']):
            case=j*2+k;key=f'single_step_{case:02d}';dest=out/'single_step'/f'{case:02d}.json'
            if key in ledger.data['completed']:
                cases.append(read_json(dest));continue
            ledger.begin(key,'single_step',P['wall_budget_s'],8)
            progress('TRAIN_single_step_diagnostic',case=case+1,total=8,noise_t=t)
            set_vector(model,original);seed=P['world_noise_seed']+10000+case
            rr,ii,_,_=cache.get(rows[i]);before=rr.calls.copy()
            pred=predict(model,data,i,t,seed);xx=pred.detach().numpy().astype(float)
            value,g,proof,arr=endpoint.gradient(xx,codec,data['centers'][i],arrays[1][i],rows[i]['task'],rr,ii,deadline)
            proof.update(sample_id=arrays[3][i],input_weight_hash=original_hash,noise_t=t,noise_seed=seed)
            gradient_path=dest.with_name(dest.stem+'_gradient.json')
            atomic_npz(gradient_path.with_suffix('.npz'),**arr)
            atomic_json(gradient_path,dict(proof,arrays_sha256=file_digest(gradient_path.with_suffix('.npz'))))
            noise=torch.randn((1,102),generator=torch.Generator().manual_seed(seed))
            def losses():
                v,l,p=terms(model,data['x'][i:i+1],data['c'][i:i+1],torch.tensor([t]),noise,data['alpha'],G[lookup[i]:lookup[i]+1])
                return v,l,endpoint.attach_gradient(p[0],value,g)
            v,l,w=losses();vectors={name:vector_gradient(term,model) for name,term in [('v',v),('local',.05*l),('world',.1*w)]}
            truegrad=vector_gradient(w,model);norm=float(truegrad.norm());direction=truegrad/max(norm,1e-30)
            def evaluate_parameter(vector):
                set_vector(model,vector)
                with torch.no_grad():x=predict(model,data,i,t,seed).numpy().astype(float)
                f,r=endpoint.evaluate(x,codec,data['centers'][i],arrays[1][i],rows[i]['task'],rr,ii,deadline)
                return dict(r,loss=f,weight_hash=weight_hash(model),x_hash=digest(x))
            central=[]
            for eps in (P['parameter_difference'],P['parameter_difference']/2):
                lo=evaluate_parameter(original-eps*direction);hi=evaluate_parameter(original+eps*direction)
                slope=(hi['loss']-lo['loss'])/(2*eps);error=abs(slope-norm)/max(abs(slope),norm,1e-8)
                central.append({'epsilon':eps,'minus':lo,'plus':hi,'slope':slope,'expected':norm,'relative_error':error})
            descent=[dict(evaluate_parameter(original-eps*direction),step=eps) for eps in P['parameter_descent_steps']]
            optimizer_steps={}
            for arm,(lw,ww) in ARMS.items():
                set_vector(model,original);optimizer=torch.optim.AdamW(model.parameters(),lr=P['lr'])
                v,l,w=losses();loss=v+lw*l+ww*w;optimizer.zero_grad();loss.backward()
                gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();after=get_vector(model)
                optimizer_steps[arm]=dict(evaluate_parameter(after),gradient_norm=float(gradnorm),
                    parameter_displacement=float((after-original).norm()),total_training_loss=float(loss.detach()))
            set_vector(model,original)
            passed=bool(proof['status']=='PASS' and norm>0 and np.isfinite(norm)
                and all(c['relative_error']<=P['parameter_direction_tolerance'] for c in central)
                and any(c['loss']<value-max(1e-7,abs(value)*1e-7) for c in descent)
                and all(np.isfinite(c['loss']) for c in optimizer_steps.values()) and weight_hash(model)==original_hash)
            result={'status':'PASS' if passed else 'BLOCKED','case':case,'sample_id':arrays[3][i],
                'task_hash':rows[i]['task']['task_hash'],'t':t,'seed':seed,'before_loss':value,
                'initial_weight_hash':original_hash,'component_gradients':gradient_statistics(vectors),
                'parameter_central_checks':central,'negative_gradient_steps':descent,'optimizer_steps':optimizer_steps,
                'TRAIN_only':True,'updates_retained':False,'calls':{k:rr.calls[k]-before[k] for k in before}}
            atomic_json(dest,result);ledger.finish({'status':result['status']},result['calls']);cases.append(result)
    result={'status':'PASS' if len(cases)==8 and all(r['status']=='PASS' for r in cases) else 'BLOCKED',
            'cases':cases,'initial_weight_hash':original_hash,'TRAIN_only':True,'weight_unchanged':weight_hash(model)==original_hash}
    atomic_json(path,result);return result
