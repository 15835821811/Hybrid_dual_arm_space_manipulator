"""TRAIN-only residual diffusion around a public-goal initializer."""
import time
from pathlib import Path
import numpy as np
import torch
from ff_prior.diffusion import NoiseTransformer, schedule, save_torch, weight_hash, ddim, prediction_to_clean_noise
from ff_prior.dataset import Scaler
from ff_prior.clean_prior import TemporalPrior
from ff_prior.storage import atomic_json, read_json, digest, file_digest

CONTRACT={'representation':'Z_minus_public_goal_initializer_v1','prediction':'v',
          'schedule':'linear_beta_1e-4_0.02_1000','updates':200,'seed':20261015,
          'hidden':128,'layers':4,'heads':4,'lr':2e-4,'batch':16,'fit_split':'TRAIN'}


class OffsetScaler:
    def __init__(self,scaler,base): self.scaler,self.base=scaler,np.asarray(base,dtype=np.float64)
    def condition(self,c): return self.scaler.condition(c)
    def z(self,z): return self.scaler.z(z-self.base)
    def inverse_z(self,x): return self.base+self.scaler.inverse_z(x)


class OffsetPrior:
    def __init__(self,prior,base): self.prior,self.base,self.metadata=prior,base,prior.metadata
    def project(self,z): return self.base+self.prior.project(z-self.base)


def load(directory,robot_hash,initial_id):
    directory=Path(directory)
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    m=saved['metadata']
    if m['contract']!=CONTRACT or m['robot_hash']!=robot_hash or m['initial_id']!=initial_id:
        raise ValueError('residual checkpoint contract mismatch')
    if file_digest(directory/'scaler.json')!=m['scaler_file_hash']:
        raise ValueError('residual scaler mismatch')
    scaler=Scaler.load(directory/'scaler.json',robot_hash,initial_id)
    prior=TemporalPrior.load(directory/'clean_prior.json',m['prior_hash'],robot_hash,initial_id)
    model=NoiseTransformer(m['condition_dim']); model.load_state_dict(saved['weights'])
    model.prediction_type='v'; model.eval()
    return model,scaler,prior


def sample(model,scaler,prior,base,task,seed,guidance=None):
    model.clean_prior=OffsetPrior(prior,base)
    return ddim(model,OffsetScaler(scaler,base),task,16,seed,guidance)


def train(directory,z,bases,conditions,ids,splits,robot_hash,initial_id,progress,deadline):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    z,bases=np.asarray(z),np.asarray(bases)
    if z.shape!=bases.shape or z.shape!=(len(ids),19,17) or len(set(ids))!=len(ids):
        raise ValueError('residual data shape/identity mismatch')
    if set(splits)!={'TRAIN','VAL'} or not np.isfinite(z).all() or not np.isfinite(bases).all():
        raise ValueError('residual training accepts finite TRAIN/VAL only')
    tr=np.array([s=='TRAIN' for s in splits]); va=~tr
    residual=z-bases
    tc=[c for c,s in zip(conditions,splits) if s=='TRAIN']
    vc=[c for c,s in zip(conditions,splits) if s=='VAL']
    scaler=Scaler.fit(residual[tr],tc,['TRAIN']*sum(tr),robot_hash,initial_id)
    prior=TemporalPrior.fit(residual[tr],np.zeros(17),[i for i,s in zip(ids,splits) if s=='TRAIN'],
                            ['TRAIN']*sum(tr),robot_hash,initial_id)
    for name,obj in (('scaler.json',scaler),('clean_prior.json',prior)):
        if not (directory/name).exists(): obj.save(directory/name)
    meta={'contract':CONTRACT,'robot_hash':robot_hash,'initial_id':initial_id,
          'data_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits}),
          'scaler_file_hash':file_digest(directory/'scaler.json'),
          'prior_hash':read_json(directory/'clean_prior.json')['hash'],
          'condition_dim':len(scaler.condition(tc[0])),'training_ids':[i for i,s in zip(ids,splits) if s=='TRAIN'],
          'data_quality_scope':'REAL_MODEL_PRIOR_TRANSFORMED_TO_RESIDUAL',
          'private_teacher_fields_used':False}
    if (directory/'training.json').exists():
        report=read_json(directory/'training.json')
        if report['metadata']!=meta or file_digest(directory/'model.pt')!=report['checkpoint_sha256']:
            raise ValueError('completed residual training identity changed')
        load(directory,robot_hash,initial_id)
        return report
    torch.set_num_threads(2); torch.manual_seed(CONTRACT['seed'])
    model=NoiseTransformer(meta['condition_dim']); original=weight_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONTRACT['lr'])
    x=torch.tensor(scaler.z(residual[tr]),dtype=torch.float32)
    xv=torch.tensor(scaler.z(residual[va]),dtype=torch.float32)
    c=torch.tensor(np.stack([scaler.condition(v) for v in tc]),dtype=torch.float32)
    cv=torch.tensor(np.stack([scaler.condition(v) for v in vc]),dtype=torch.float32)
    start,best,history,wall_before,cpu_before=0,float('inf'),[],0.,0.
    if (directory/'progress.pt').exists():
        saved=torch.load(directory/'progress.pt',weights_only=False,map_location='cpu')
        if saved['metadata']!=meta: raise ValueError('residual resume data/contract conflict')
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        start,best,history,original,wall_before,cpu_before=(saved[k] for k in ('updates','best','history','original_hash','wall_s','cpu_s'))
    alpha=schedule(); vg=torch.Generator().manual_seed(CONTRACT['seed']+10)
    vt=torch.randint(0,1000,(len(xv),),generator=vg); vn=torch.randn(xv.shape,generator=vg)
    av=alpha[vt,None,None]; noisy_val=av.sqrt()*xv+(1-av).sqrt()*vn
    target_val=av.sqrt()*vn-(1-av).sqrt()*xv
    wall,cpu=time.perf_counter(),time.process_time()
    for step in range(start,CONTRACT['updates']):
        if time.time()>=deadline: raise RuntimeError('TRAINING_WALL_BUDGET_EXHAUSTED')
        model.train(); batch=torch.randint(0,len(x),(min(16,len(x)),))
        t=torch.randint(0,1000,(len(batch),)); noise=torch.randn_like(x[batch]); a=alpha[t,None,None]
        loss=((model(a.sqrt()*x[batch]+(1-a).sqrt()*noise,t,c[batch])-(a.sqrt()*noise-(1-a).sqrt()*x[batch]))**2).mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite residual loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        model.eval()
        with torch.no_grad(): val=float(((model(noisy_val,vt,cv)-target_val)**2).mean())
        history.append({'update':step+1,'train_v_mse':float(loss.detach()),'val_v_mse':val})
        if val<best:
            best=val; save_torch(directory/'model.pt',{'metadata':meta,'weights':model.state_dict(),'update':step+1})
        save_torch(directory/'progress.pt',{'metadata':meta,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':torch.get_rng_state(),'updates':step+1,'best':best,'history':history,'original_hash':original,
            'wall_s':wall_before+time.perf_counter()-wall,'cpu_s':cpu_before+time.process_time()-cpu})
        if (step+1)%20==0: progress('training',updates=step+1,total_updates=200)
    model,scaler,prior=load(directory,robot_hash,initial_id)
    clone,_,_=load(directory,robot_hash,initial_id)
    with torch.no_grad(): reload_error=float(abs(model(noisy_val,vt,cv)-clone(noisy_val,vt,cv)).max())
    bands=[]
    for timestep in (0,500,999):
        t=torch.full((len(xv),),timestep); a=alpha[t,None,None]; noisy=a.sqrt()*xv+(1-a).sqrt()*vn
        with torch.no_grad():
            clean,_=prediction_to_clean_noise(noisy,model(noisy,t,cv),a,'v')
            shuffled,_=prediction_to_clean_noise(noisy,model(noisy,t,cv.roll(1,0)),a,'v')
        target=residual[va]
        bands.append({'t':timestep,'val_rmse_rad':float(np.sqrt(np.mean((scaler.inverse_z(clean.numpy())-target)**2))),
            'shuffled_condition_rmse_rad':float(np.sqrt(np.mean((scaler.inverse_z(shuffled.numpy())-target)**2)))})
    report={'status':'PASS' if weight_hash(model)!=original and reload_error==0 else 'FAIL','metadata':meta,
        'updates':200,'training_val_forward_calls':400,'reload_forward_calls':2,'diagnostic_forward_calls':6,
        'selected_weight_hash':weight_hash(model),'checkpoint_sha256':file_digest(directory/'model.pt'),
        'reload_error':reload_error,'checkpoint_selection':'VAL_V_MSE_ONLY','history':history,'noise_bands':bands,
        'zero_residual_val_rmse_rad':float(np.sqrt(np.mean(residual[va]**2))),
        'train_residual_rms_rad':float(np.sqrt(np.mean(residual[tr]**2))),
        'train_absolute_rms_from_initial_rad':float(np.sqrt(np.mean((z[tr]-np.array(tc[0]['initial_features']))**2))),
        'prior_rank':prior.metadata['rank'],'wall_s':wall_before+time.perf_counter()-wall,
        'cpu_s':cpu_before+time.process_time()-cpu,'benefit':'NOT_ESTABLISHED'}
    atomic_json(directory/'training.json',report)
    return report
