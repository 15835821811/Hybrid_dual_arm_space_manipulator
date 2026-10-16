"""Matched-data and matched-RNG v-loss / local coupled-pose-loss ablation."""
from pathlib import Path
import time
import numpy as np
import torch
from ff_prior_latent.learning import Codec,LatentDenoiser,sample
from ff_prior_pilot_train import learning as frozen_learning
from ff_prior.diffusion import schedule,prediction_to_clean_noise,save_torch as base_save_torch,weight_hash
from ff_prior.storage import read_json,digest,file_digest
from .storage import atomic_json,retry_write


def save_torch(path,value):
    return retry_write(base_save_torch,path,value)

CONTRACT={'version':'qualified_coverage_frozen_codec_finetune_1','updates':6000,'batch':64,'lr':2e-4,'seed':20261501,
    'architecture':'same centered temporal latent MLP as previous round','goal_loss_weight':.05,
    'goal_loss':'mean squared local coupled pose change of denoised error in 34-direction surrogate',
    'selection':'fixed final update; no VAL gradients/checkpoint selection','TRAIN_only_fit':True,
    'warm_start':'verified 10k model; fixed codec fitted to original 10000 TRAIN only','obstacle_conditioned':False}


ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'artifacts/v7_1/qualified_pilot_train_002/learning/goal_loss'


def fit_codec(out,arrays):
    z,bases,conditions,ids,splits,_=arrays
    _,codec=frozen_learning.load(FROZEN)
    fit_ids=codec.state['fit_ids'];slots={sid:i for i,sid in enumerate(ids)}
    if any(sid not in slots or splits[slots[sid]]!='TRAIN' for sid in fit_ids):raise ValueError('frozen codec fit rows are not inherited TRAIN')
    chosen=[slots[sid] for sid in fit_ids]
    if codec.state['fit_hash']!=digest({'z':z[chosen],'bases':bases[chosen],
        'conditions':[conditions[i] for i in chosen],'ids':fit_ids}):raise ValueError('frozen codec fit arrays changed')
    path=Path(out)/'codec.json'
    if path.exists():
        if digest(read_json(path))!=digest(codec.state):raise ValueError('frozen codec identity changed')
    else:atomic_json(path,codec.state)
    return codec


def objective(prediction,target,noisy,alpha,clean,G,weight):
    denoised,_=prediction_to_clean_noise(noisy,prediction,alpha,'v')
    vmse=(prediction-target).square().mean()
    pose=torch.einsum('bij,bj->bi',G,denoised-clean).square().mean() if G is not None else vmse*0.
    return vmse+weight*pose,vmse,pose


def load(directory):
    directory=Path(directory);r=read_json(directory/'training.json')
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False);m=saved['metadata']
    if m!=r['metadata'] or m['contract']!=CONTRACT or file_digest(directory/'model.pt')!=r['checkpoint_sha256']:
        raise ValueError('new checkpoint identity mismatch')
    if file_digest(directory/'codec.json')!=m['codec_sha256']:raise ValueError('codec changed')
    codec=Codec(read_json(directory/'codec.json'));model=LatentDenoiser(codec.dim,len(codec.feature_mean))
    model.load_state_dict(saved['weights']);model.eval();return model,codec


def train(directory,arrays,codec,matrices,mode,sensitivity_report,progress,deadline):
    if mode not in ('v_only','goal_loss'):raise ValueError('unknown loss mode')
    if mode=='goal_loss' and matrices is None:
        result={'status':'BLOCKED','reason':'TRAIN_SENSITIVITY_INCOMPLETE'};atomic_json(Path(directory)/'blocked.json',result);return result
    z,bases,conditions,ids,splits,states=arrays;directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if set(splits)!={'TRAIN','VAL'}:raise ValueError('TRAIN/VAL required')
    cp=directory/'codec.json'
    if cp.exists():
        if digest(read_json(cp))!=digest(codec.state):raise ValueError('codec changed')
    else:atomic_json(cp,codec.state)
    tr=np.array([s=='TRAIN' for s in splits]);va=~tr;tids=np.asarray(ids)[tr].tolist()
    encoded=np.stack([codec.encode(v,b) for v,b in zip(z,bases)])
    reconstruction=bases+np.einsum('tk,nkj->ntj',codec.basis,encoded.reshape(len(z),codec.rank,17))
    errors=np.max(abs(reconstruction-z),axis=(1,2));old_fit=set(codec.state['fit_ids'])
    new=np.array([s=='TRAIN' and sid not in old_fit for sid,s in zip(ids,splits)])
    representation={'frozen_codec_fit_rows':len(old_fit),'TRAIN_max_abs_projection_error_rad':float(max(errors[tr])),
        'new_TRAIN_rows':int(sum(new)),'new_TRAIN_max_abs_projection_error_rad':float(max(errors[new])) if any(new) else None,
        'scope':'representation diagnostic; not physical validation or checkpoint selection'}
    meta={'contract':CONTRACT,'loss_mode':mode,'codec_sha256':file_digest(cp),'training_ids':tids,
        'validation_ids':np.asarray(ids)[va].tolist(),'data_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states}),
        'warm_start_checkpoint_sha256':file_digest(FROZEN/'model.pt'),'representation':representation,
        'sensitivity_hash':digest(sensitivity_report),'sensitivity_matrix_hash':digest(matrices) if matrices is not None else None}
    if (directory/'training.json').exists():
        report=read_json(directory/'training.json')
        if report['metadata']!=meta:raise ValueError('completed training identity changed')
        load(directory);return report
    c=np.stack([codec.condition(v,b) for v,b in zip(conditions,bases)])
    centers=np.stack([codec.center(v,b) for v,b in zip(conditions,bases)])
    y=encoded
    xall=codec.normalized(y,centers)
    torch.set_num_threads(2);torch.manual_seed(CONTRACT['seed'])
    model,old_codec=frozen_learning.load(FROZEN)
    if digest(old_codec.state)!=digest(codec.state):raise ValueError('warm start codec differs')
    original=weight_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4);alpha=schedule()
    x=torch.tensor(xall[tr],dtype=torch.float32);ct=torch.tensor(c[tr],dtype=torch.float32)
    xv=torch.tensor(xall[va],dtype=torch.float32);cv=torch.tensor(c[va],dtype=torch.float32)
    G=torch.tensor(matrices*codec.latent_scale[None,None,:],dtype=torch.float32) if matrices is not None else None
    vg=torch.Generator().manual_seed(CONTRACT['seed']+1);vn=torch.randn(xv.shape,generator=vg)
    counts=np.zeros(len(x),dtype=int);order=[];history=[];rng_proofs=[];start=0;wall_before=0.
    if (directory/'progress.pt').exists():
        saved=torch.load(directory/'progress.pt',map_location='cpu',weights_only=False)
        if saved['metadata']!=meta:raise ValueError('training resume identity changed')
        model.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        start,counts,order,history,rng_proofs,original,wall_before=(saved[k] for k in ('updates','counts','order','history','rng_proofs','original','wall_s'))
    def diagnostics():
        result=[];model.eval()
        for t in (0,500,999):
            tt=torch.full((len(xv),),t);a=alpha[t];noisy=a.sqrt()*xv+(1-a).sqrt()*vn
            with torch.no_grad():clean,_=prediction_to_clean_noise(noisy,model(noisy,tt,cv),a,'v')
            coefficients=codec.coefficients(clean.numpy(),centers[va])
            actual=bases[va]+np.einsum('tk,nkj->ntj',codec.basis,coefficients.reshape(sum(va),codec.rank,17))
            result.append({'t':t,'full_trajectory_rmse_rad':float(np.sqrt(np.mean((actual-z[va])**2)))})
        return result
    wall=time.perf_counter();weight=CONTRACT['goal_loss_weight'] if mode=='goal_loss' else 0.
    for step in range(start,CONTRACT['updates']):
        if time.time()>=deadline:raise RuntimeError('TRAINING_WALL_BUDGET_EXHAUSTED')
        if len(order)<CONTRACT['batch']:order.extend(torch.randperm(len(x)).tolist())
        batch=order[:CONTRACT['batch']];order=order[CONTRACT['batch']:];np.add.at(counts,batch,1)
        model.train();t=torch.randint(0,1000,(CONTRACT['batch'],));noise=torch.randn_like(x[batch]);a=alpha[t,None]
        noisy=a.sqrt()*x[batch]+(1-a).sqrt()*noise;target=a.sqrt()*noise-(1-a).sqrt()*x[batch]
        loss,vmse,pose=objective(model(noisy,t,ct[batch]),target,noisy,a,x[batch],G[batch] if G is not None else None,weight)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite training gradient')
        optimizer.step()
        if (step+1)%100==0:
            rng_proofs.append(digest({'step':step+1,'batch':batch,'t':t.numpy(),'noise':noise.numpy()}))
            history.append({'update':step+1,'v_mse':float(vmse.detach()),'local_pose_mse':float(pose.detach()),'loss':float(loss.detach()),'VAL_noise_bands':diagnostics()})
            progress('train_augmented_diffusion',mode=mode,updates=step+1,total=CONTRACT['updates'])
        if (step+1)%20==0:
            save_torch(directory/'progress.pt',{'metadata':meta,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                'updates':step+1,'counts':counts,'order':order,'history':history,'rng_proofs':rng_proofs,'original':original,'wall_s':wall_before+time.perf_counter()-wall})
    save_torch(directory/'model.pt',{'metadata':meta,'weights':model.state_dict()})
    report={'status':'PASS' if all(counts>0) and weight_hash(model)!=original else 'FAIL','metadata':meta,
        'updates':CONTRACT['updates'],'checkpoint_sha256':file_digest(directory/'model.pt'),'initial_weight_hash':original,'trained_weight_hash':weight_hash(model),
        'train_draw_counts':dict(zip(tids,counts.tolist())),'all_TRAIN_rows_seen':bool(all(counts>0)),
        'VAL_gradient_updates':0,'VAL_checkpoint_selection':False,'noise_bands':diagnostics(),'noise_band_scope':'full trajectory including discarded temporal component; corruption same across this ablation',
        'history':history,'rng_proofs':rng_proofs,'latent_dimensions':codec.dim,'wall_s':wall_before+time.perf_counter()-wall,
        'benefit':'REQUIRES_FROZEN_PHYSICAL_COMPARISON'}
    atomic_json(directory/'training.json',report);load(directory);return report
