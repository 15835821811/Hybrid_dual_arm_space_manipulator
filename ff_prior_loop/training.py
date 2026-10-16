"""New TRAIN data are consumed; population identity is distinct from an initial state."""
from pathlib import Path
import time
import numpy as np
import torch
from ff_prior.diffusion import NoiseTransformer,schedule,save_torch,weight_hash,prediction_to_clean_noise
from ff_prior.dataset import Scaler
from ff_prior.clean_prior import TemporalPrior
from ff_prior.storage import atomic_json,read_json,digest,file_digest

CONTRACT={'version':'multi_origin_residual_v1','representation':'Z_minus_public_goal_initializer',
    'prediction':'v','updates':600,'batch':16,'lr':2e-4,'seed':20261061,
    'architecture':{'hidden':128,'layers':4,'heads':4},'fit':'TRAIN_only',
    'terminal_loss_weight':4.,'selection':'fixed_VAL_physical_clean_MSE_at_0_500_999_every_20_updates',
    'initialization':'fresh_weights; previous frozen model is a separate baseline',
    'population_token_is_initial_state':False,'initial_velocity':'zero only'}


def load(directory,scope):
    directory=Path(directory); state=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    m=state['metadata']
    if m['contract']!=CONTRACT or m['scope']!=scope: raise ValueError('training scope mismatch')
    if file_digest(directory/'scaler.json')!=m['scaler_file_hash']: raise ValueError('scaler bytes changed')
    scaler=Scaler.load(directory/'scaler.json',scope['canonical_robot_hash'],scope['fit_population_token'])
    prior=TemporalPrior.load(directory/'clean_prior.json',m['prior_hash'],scope['canonical_robot_hash'],scope['fit_population_token'])
    model=NoiseTransformer(m['condition_dim']); model.load_state_dict(state['weights']); model.eval(); model.prediction_type='v'
    return model,scaler,prior


def train(directory,arrays,robot_hash,progress,deadline):
    z,bases,conditions,ids,splits,states=arrays; directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    if len(set(ids))!=len(ids) or z.shape!=bases.shape or z.shape!=(len(ids),19,17): raise ValueError('invalid dataset identity')
    if set(splits)!={'TRAIN','VAL'} or not np.isfinite(z-bases).all(): raise ValueError('invalid data or partition')
    tr=np.array([s=='TRAIN' for s in splits]); va=~tr; residual=z-bases
    train_ids=[i for i,s in zip(ids,splits) if s=='TRAIN']; val_ids=[i for i,s in zip(ids,splits) if s=='VAL']
    scope={'canonical_robot_hash':robot_hash,'fit_population_token':'multi_origin_population:'+digest(sorted(set(np.array(states)[tr]))),
        'fit_initial_state_ids':sorted(set(np.array(states)[tr])),'schema':'multi_origin_residual_v1',
        'inference_transfer':'unchanged physical robot; explicit full initial state and scene audit required'}
    tc=[c for c,s in zip(conditions,splits) if s=='TRAIN']; vc=[c for c,s in zip(conditions,splits) if s=='VAL']
    scaler=Scaler.fit(residual[tr],tc,['TRAIN']*sum(tr),robot_hash,scope['fit_population_token'])
    prior=TemporalPrior.fit(residual[tr],np.zeros(17),train_ids,['TRAIN']*sum(tr),robot_hash,scope['fit_population_token'])
    for name,obj in (('scaler.json',scaler),('clean_prior.json',prior)):
        if not (directory/name).exists(): obj.save(directory/name)
    metadata={'contract':CONTRACT,'scope':scope,'condition_dim':len(scaler.condition(tc[0])),
        'data_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states}),
        'training_ids':train_ids,'validation_ids':val_ids,'scaler_file_hash':file_digest(directory/'scaler.json'),
        'prior_hash':read_json(directory/'clean_prior.json')['hash']}
    if (directory/'training.json').exists():
        report=read_json(directory/'training.json')
        if report['metadata']!=metadata or file_digest(directory/'model.pt')!=report['checkpoint_sha256']: raise ValueError('training identity changed')
        load(directory,scope); return report
    torch.set_num_threads(2); torch.manual_seed(CONTRACT['seed'])
    model=NoiseTransformer(metadata['condition_dim']); original=weight_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4); alpha=schedule()
    x=torch.tensor(scaler.z(residual[tr]),dtype=torch.float32); xv=torch.tensor(scaler.z(residual[va]),dtype=torch.float32)
    c=torch.tensor(np.stack([scaler.condition(v) for v in tc]),dtype=torch.float32)
    cv=torch.tensor(np.stack([scaler.condition(v) for v in vc]),dtype=torch.float32)
    vg=torch.Generator().manual_seed(CONTRACT['seed']+1); vn=torch.randn(xv.shape,generator=vg)
    wt=torch.ones(1,19,1); wt[:,-1]=4.
    scale=torch.tensor(scaler.z_scale,dtype=torch.float32)
    start=0; best=float('inf'); history=[]; counts=np.zeros(len(x),dtype=int); order=[]; wall_before=0.
    if (directory/'progress.pt').exists():
        saved=torch.load(directory/'progress.pt',map_location='cpu',weights_only=False)
        if saved['metadata']!=metadata: raise ValueError('training resume mismatch')
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['rng'])
        start,best,history,counts,order,original,wall_before=(saved[k] for k in ('updates','best','history','counts','order','original','wall_s'))
    def diagnostics(shuffle=False):
        result=[]
        model.eval()
        for t in (0,500,999):
            tt=torch.full((len(xv),),t); a=alpha[tt,None,None]; noisy=a.sqrt()*xv+(1-a).sqrt()*vn
            with torch.no_grad():
                clean,_=prediction_to_clean_noise(noisy,model(noisy,tt,cv.roll(1,0) if shuffle else cv),a,'v')
                loss=((clean-xv)*scale).square()
            result.append({'t':t,'rmse_rad':float(loss.mean().sqrt()),'terminal_rmse_rad':float(loss[:,-1].mean().sqrt()),
                'selection_mse':float((loss*wt).mean())})
        return result
    wall=time.perf_counter()
    for step in range(start,600):
        if time.time()>=deadline: raise RuntimeError('TRAINING_WALL_BUDGET_EXHAUSTED')
        if len(order)<16: order.extend(torch.randperm(len(x)).tolist())
        batch=order[:16]; order=order[16:]; np.add.at(counts,batch,1)
        model.train(); t=torch.randint(0,1000,(16,)); noise=torch.randn_like(x[batch]); a=alpha[t,None,None]
        prediction=model(a.sqrt()*x[batch]+(1-a).sqrt()*noise,t,c[batch]); target=a.sqrt()*noise-(1-a).sqrt()*x[batch]
        loss=((prediction-target).square()*wt).mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('nonfinite training gradient')
        optimizer.step()
        if (step+1)%20==0:
            bands=diagnostics(); val=sum(b['selection_mse'] for b in bands)/3
            history.append({'update':step+1,'train_weighted_v_mse':float(loss.detach()),'val_clean_selection_mse':val,'noise_bands':bands})
            if val<best:
                best=val; save_torch(directory/'model.pt',{'metadata':metadata,'weights':model.state_dict(),'update':step+1})
            progress('train_actual_augmented_data',updates=step+1,total=600,val_clean_mse=val)
        save_torch(directory/'progress.pt',{'metadata':metadata,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':torch.get_rng_state(),'updates':step+1,'best':best,'history':history,'counts':counts,'order':order,
            'original':original,'wall_s':wall_before+time.perf_counter()-wall})
    model,_,_=load(directory,scope); clone,_,_=load(directory,scope)
    with torch.no_grad(): reload_error=float((model(xv,torch.zeros(len(xv),dtype=torch.long),cv)-clone(xv,torch.zeros(len(xv),dtype=torch.long),cv)).abs().max())
    report={'status':'PASS' if all(counts>0) and weight_hash(model)!=original and reload_error==0 else 'FAIL',
        'metadata':metadata,'updates':600,'selected_update':torch.load(directory/'model.pt',weights_only=False)['update'],
        'initial_weight_hash':original,'selected_weight_hash':weight_hash(model),'checkpoint_sha256':file_digest(directory/'model.pt'),
        'train_draw_counts':dict(zip(train_ids,counts.tolist())),'all_TRAIN_rows_seen':bool(all(counts>0)),
        'VAL_gradient_updates':0,'reload_error':reload_error,'history':history,'noise_bands':diagnostics(),
        'shuffled_condition_noise_bands':diagnostics(True),'zero_residual_val_rmse_rad':float(np.sqrt(np.mean(residual[va]**2))),
        'prior_rank':prior.metadata['rank'],'wall_s':wall_before+time.perf_counter()-wall,'benefit':'REQUIRES_PAIRED_EXECUTION_EVIDENCE'}
    atomic_json(directory/'training.json',report); return report
