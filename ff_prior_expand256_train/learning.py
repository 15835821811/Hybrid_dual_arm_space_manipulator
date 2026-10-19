"""One exact warm start; v + existing local G loss; fixed final checkpoint."""
from pathlib import Path
import time
import numpy as np
import torch
from ff_prior.storage import read_json,file_digest,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.diffusion import schedule,prediction_to_clean_noise,weight_hash
from ff_prior_latent.learning import Codec,LatentDenoiser
from ff_prior_delivery_train.learning import load as parent_load
from ff_prior_origin_learn_bounded.learning import objective,save_torch
from ff_prior_origin_learn_bounded.distribution import population,CoverageSampler
from .protocol import OLD,PROTOCOL

CONTRACT={'version':'expand256_exact13966_warm_v_local_G_1','updates':6000,'batch':64,'lr':2e-4,
          'seed':20263001,'local_G_weight':.05,'codec':'exact warm model codec',
          'loss':'v MSE plus 0.05 teacher-local 34-direction pose surrogate; not full world-gradient training',
          'checkpoint':'fixed final update','VAL_gradients':False,'VAL_checkpoint_selection':False}


def resume_permission(journal,checkpoint_updates):
    if journal is None:
        if checkpoint_updates:raise ValueError('checkpoint without training journal')
        return 'FRESH'
    if journal['status']=='PENDING':
        if checkpoint_updates==journal['end_update']:
            return 'RECOVER_COMMITTED_CHECKPOINT'
        raise RuntimeError('UNCOMMITTED_TRAINING_SEGMENT_UNKNOWN_WORK; do not replay')
    if journal['committed_updates']!=checkpoint_updates:
        raise ValueError('training journal/checkpoint mismatch')
    return 'CONTINUE_COMMITTED'


def load(directory):
    directory=Path(directory);r=read_json(directory/'training.json')
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    if saved['metadata']!=r['metadata'] or r['metadata']['contract']!=CONTRACT or file_digest(directory/'model.pt')!=r['checkpoint_sha256']:
        raise ValueError('E256T checkpoint binding changed')
    codec=Codec(read_json(directory/'codec.json'))
    if digest(codec.state)!=r['metadata']['codec_hash']:raise ValueError('E256T codec changed')
    model=LatentDenoiser(codec.dim,len(codec.feature_mean));model.load_state_dict(saved['weights']);model.eval()
    if weight_hash(model)!=r['trained_weight_hash']:raise ValueError('reloaded weights differ')
    return model,codec


def train(directory,arrays,codec,matrices,sreport,new_ids,progress,deadline):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    z,bases,conditions,ids,splits,states=arrays
    tr=np.asarray(splits)=='TRAIN';va=np.asarray(splits)=='VAL'
    if sum(tr)!=13966+len(new_ids) or not 128<=len(new_ids)<=256 or sum(va)!=48 or not np.all(tr|va):raise ValueError('unexpected TRAIN/VAL population')
    model,old_codec=parent_load(OLD/'learning/goal_loss')
    if digest(codec.state)!=digest(old_codec.state):raise ValueError('warm-start codec must be unchanged')
    old_report=read_json(OLD/'learning/goal_loss/training.json')
    original=weight_hash(model)
    if original!=old_report['trained_weight_hash']:raise ValueError('warm weights mismatch')
    atomic_json(directory/'codec.json',codec.state)
    tids=np.asarray(ids)[tr].tolist();vids=np.asarray(ids)[va].tolist()
    encoded=np.stack([codec.encode(v,b) for v,b in zip(z,bases)])
    centers=np.stack([codec.center(c,b) for c,b in zip(conditions,bases)])
    conditions_encoded=np.stack([codec.condition(c,b) for c,b in zip(conditions,bases)])
    values=codec.normalized(encoded,centers)
    reconstruction=bases+np.einsum('tk,nkj->ntj',codec.basis,encoded.reshape(len(z),codec.rank,17))
    errors=np.max(abs(reconstruction-z),axis=(1,2))
    meta={'contract':CONTRACT,'training_ids':tids,'validation_ids':vids,'new_E256_ids':sorted(new_ids),
          'warm_checkpoint_sha256':file_digest(OLD/'learning/goal_loss/model.pt'),'codec_hash':digest(codec.state),
          'data_hash':digest({'z':z,'bases':bases,'conditions':conditions,'ids':ids,'splits':splits,'states':states}),
          'sensitivity_summary_hash':digest(sreport),'sensitivity_matrix_hash':digest(matrices),
          'representation_error_rad_max':float(max(errors[tr]))}
    if (directory/'training.json').exists():
        report=read_json(directory/'training.json')
        if report['metadata']!=meta:raise ValueError('completed training identity changed')
        load(directory);return report
    torch.set_num_threads(2);torch.manual_seed(CONTRACT['seed'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONTRACT['lr']);alpha=schedule()
    x=torch.tensor(values[tr],dtype=torch.float32);c=torch.tensor(conditions_encoded[tr],dtype=torch.float32)
    G=torch.tensor(matrices*codec.latent_scale[None,None,:],dtype=torch.float32)
    xv=torch.tensor(values[va],dtype=torch.float32);cv=torch.tensor(conditions_encoded[va],dtype=torch.float32)
    vn=torch.randn(xv.shape,generator=torch.Generator().manual_seed(CONTRACT['seed']+1))
    _,keys,_,population_proof=population(conditions,ids,splits)
    sampler=CoverageSampler(keys,CONTRACT['seed']+2)
    counts=np.zeros(len(tids),dtype=np.int64);history=[];segments=[];start=0;wall_before=0.
    saved=None
    if (directory/'progress.pt').exists():
        saved=torch.load(directory/'progress.pt',map_location='cpu',weights_only=False)
        if saved['metadata']!=meta:raise ValueError('checkpoint metadata mismatch')
        start=int(saved['updates'])
    journal_path=directory/'training_journal.json'
    permission=resume_permission(read_json(journal_path) if journal_path.exists() else None,start)
    if saved:
        model.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        sampler=CoverageSampler(keys,CONTRACT['seed']+2,saved['sampler'])
        counts=saved['counts'];history=saved['history'];segments=saved['segments'];wall_before=saved['wall_s']
        if permission=='RECOVER_COMMITTED_CHECKPOINT':
            atomic_json(directory/'training_recovery.json',{'status':permission,'updates':start,'replayed_updates':0})
    if start%20 or not 0<=start<=6000:raise ValueError('invalid committed update boundary')
    atomic_json(journal_path,{'status':'COMMITTED','committed_updates':start})
    def diagnostics():
        model.eval();answer=[]
        for t in (0,500,999):
            a=alpha[t];noisy=a.sqrt()*xv+(1-a).sqrt()*vn
            with torch.no_grad():clean,_=prediction_to_clean_noise(noisy,model(noisy,torch.full((len(xv),),t),cv),a,'v')
            coeff=codec.coefficients(clean.numpy(),centers[va])
            actual=bases[va]+np.einsum('tk,nkj->ntj',codec.basis,coeff.reshape(sum(va),codec.rank,17))
            answer.append({'t':t,'trajectory_rmse_rad':float(np.sqrt(np.mean((actual-z[va])**2)))})
        return answer
    wall=time.perf_counter()
    for begin in range(start,6000,20):
        end=begin+20
        atomic_json(journal_path,{'status':'PENDING','start_update':begin+1,'end_update':end,'last_committed_updates':begin})
        proofs=[]
        for step in range(begin,end):
            if time.time()>=deadline:raise RuntimeError('TRAINING_TIME_BUDGET_EXHAUSTED')
            batch=sampler.take(64);np.add.at(counts,batch,1)
            model.train();t=torch.randint(0,1000,(64,));noise=torch.randn_like(x[batch]);a=alpha[t,None]
            noisy=a.sqrt()*x[batch]+(1-a).sqrt()*noise;target=a.sqrt()*noise-(1-a).sqrt()*x[batch]
            loss,vmse,pose=objective(model(noisy,t,c[batch]),target,noisy,a,x[batch],G[batch],.05)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite training gradient')
            optimizer.step()
            proofs.append(digest({'update':step+1,'batch':batch,'t':t.numpy(),'noise':noise.numpy()}))
            if (step+1)%100==0:
                history.append({'update':step+1,'v_mse':float(vmse.detach()),'local_pose_mse':float(pose.detach()),
                                'loss':float(loss.detach()),'VAL_noise_diagnostic':diagnostics()})
                progress('actual_TRAIN_updates',updates=step+1,total=6000,TRAIN_rows=int(sum(tr)))
        segments.append({'start_update':begin+1,'end_update':end,'draws':1280,'batch_rng_digest':digest(proofs)})
        save_torch(directory/'progress.pt',{'metadata':meta,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':torch.get_rng_state(),'sampler':sampler.state(),'updates':end,'counts':counts,'history':history,
            'segments':segments,'wall_s':wall_before+time.perf_counter()-wall})
        atomic_json(journal_path,{'status':'COMMITTED','committed_updates':end})
    save_torch(directory/'model.pt',{'metadata':meta,'weights':model.state_dict()})
    report={'status':'PASS' if all(counts>0) and weight_hash(model)!=original else 'FAIL',
            'metadata':meta,'updates':6000,'segments':segments,'actual_TRAIN_draws':int(sum(counts)),
            'train_draw_counts':dict(zip(tids,counts.tolist())),'all_TRAIN_rows_seen':bool(all(counts>0)),
            'initial_weight_hash':original,'trained_weight_hash':weight_hash(model),
            'checkpoint_sha256':file_digest(directory/'model.pt'),'VAL_gradient_updates':0,'VAL_checkpoint_selection':False,
            'history':history,'VAL_noise_diagnostic':diagnostics(),'sampling_population':population_proof,
            **sampler.report(tids),'wall_s':wall_before+time.perf_counter()-wall,
            'world_endpoint_supervision_claim':'teacher-local surrogate only; not nonlinear exact endpoint loss',
            'benefit':'REQUIRES_FROZEN_DEVELOPMENT_COMPARISON'}
    atomic_json(directory/'training.json',report)
    load(directory)
    return report
