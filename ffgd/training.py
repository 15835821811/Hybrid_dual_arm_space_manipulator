"""Real v-prediction training with TRAIN-only scalers and atomic resumable state."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .model import ConditionalDenoiser
from .orchestration import atomic_json,check_budget,source_hash,ORCH,read_state


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path,payload):
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp);os.replace(tmp,path)
    atomic_json(path.with_suffix('.identity.json'),dict(sha256=file_hash(path),update=payload['update'],dataset_hash=payload['dataset_hash']))


def load_model(path,device='cpu'):
    path=Path(path);identity=json.loads(path.with_suffix('.identity.json').read_text())
    if file_hash(path)!=identity['sha256']:raise ValueError('CHECKPOINT_HASH_MISMATCH')
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=ConditionalDenoiser(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['ema']);model.eval()
    return model,checkpoint,identity


def normalize_condition(tokens,padding,checkpoint,device):
    x=(np.asarray(tokens)-checkpoint['condition_mean'])/checkpoint['condition_scale']
    x=np.where(np.asarray(padding)[...,None],0.,x)
    return dict(tokens=torch.tensor(x,device=device,dtype=torch.float32),padding_mask=torch.tensor(padding,device=device,dtype=torch.bool))


def train(path,config,resume=None):
    torch.set_num_threads(1)
    if not torch.cuda.is_available() and config['device']=='cuda':raise RuntimeError('CUDA_REQUIRED_BY_TRAINING_CONFIG')
    device=torch.device(config['device']);seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_per_process_memory_fraction(.75)
    directory=Path(config['dataset']);manifest=json.loads((directory/'dataset_manifest.json').read_text())
    if manifest['data_hash']!=file_hash(directory/'prior.npz'):raise ValueError('DATASET_HASH_MISMATCH')
    if manifest['accepted']<config['minimum_dataset_size']:raise ValueError('INSUFFICIENT_DECLARED_DATASET')
    with np.load(directory/'prior.npz') as data:arrays={k:data[k] for k in data.files}
    trainids=np.flatnonzero(arrays['is_train']);valids=np.flatnonzero(~arrays['is_train'])
    if config.get('debug_train_subset'):trainids=trainids[:config['debug_train_subset']]
    if len(trainids)==0 or len(valids)==0:raise ValueError('MISSING_TRAIN_OR_VAL')
    if set(arrays['group'][trainids]) & set(arrays['group'][valids]):raise ValueError('FAMILY_SPLIT_LEAK')
    z=arrays['z'];mean=z[trainids].mean(0);scale=np.maximum(z[trainids].std(0),1e-8)
    train_tokens=arrays['tokens'][trainids][~arrays['padding'][trainids]]
    cm=train_tokens.mean(0);cs=np.maximum(train_tokens.std(0),.1)
    provenance=dict(mean=mean,scale=scale,condition_mean=cm,condition_scale=cs,
                    dataset_hash=manifest['data_hash'],dataset_manifest=manifest,normalizer_split='TRAIN',
                    model_config=config['model'],training_config=config,source_hash=source_hash())
    model=ConditionalDenoiser(**config['model']).to(device);ema=copy.deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    start_update=0;best=float('inf')
    if resume:
        _,saved,identity=load_model(resume,device)
        if saved['dataset_hash']!=manifest['data_hash'] or saved['training_config']!=config:raise ValueError('RESUME_IDENTITY_MISMATCH')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer'])
        start_update=saved['update'];best=saved['best_val'];torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        for key in ('mean','scale','condition_mean','condition_scale'):np.testing.assert_array_equal(provenance[key],saved[key])
    x=torch.tensor((z-mean)/scale,device=device,dtype=torch.float32)
    condition=normalize_condition(arrays['tokens'],arrays['padding'],provenance,device)
    alpha=torch.cumprod(1-torch.linspace(.0001,.02,1000,device=device),0)
    train_index=torch.tensor(trainids,device=device)
    valgen=torch.Generator(device=device).manual_seed(seed+9000)
    valnoise=torch.randn((len(valids),x.shape[1]),generator=valgen,device=device)
    valt=torch.randint(0,1000,(len(valids),),generator=valgen,device=device)
    def objective(net,ids,t,noise):
        clean=x[ids];a=alpha[t,None].sqrt();s=(1-alpha[t,None]).sqrt()
        noisy=a*clean+s*noise;target=a*noise-s*clean
        prediction=net(noisy,t,dict(tokens=condition['tokens'][ids],padding_mask=condition['padding_mask'][ids]))
        return (prediction-target).square().mean()
    def validation():
        ema.eval();loss=0.
        with torch.no_grad():
            for i in range(0,len(valids),config['batch_size']):
                ids=torch.tensor(valids[i:i+config['batch_size']],device=device)
                loss+=float(objective(ema,ids,valt[i:i+len(ids)],valnoise[i:i+len(ids)]))*len(ids)
        return loss/len(valids)
    initial_val=validation();start=time.perf_counter();last_loss=None;updates=start_update
    atomic_json(path/'training_config.json',config)
    atomic_json(path/'coverage.json',dict(train=len(trainids),val=len(valids),free_dim=x.shape[1],minimum_coordinate_std=float(scale.min()),
                covariance_rank=int(np.linalg.matrix_rank(z[trainids]-mean)),normalizer_split='TRAIN',
                family_disjoint=True,debug_only=bool(config.get('debug_train_subset')),initial_val_v_mse=initial_val))
    def checkpoint(update):
        return dict(**provenance,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),update=update,
                    best_val=best,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    numpy_rng=np.random.get_state(),python_rng=random.getstate(),iterator='random_with_replacement_rng_state',
                    robot_model_hash=manifest['model_hash'])
    try:
        for update in range(start_update+1,config['updates']+1):
            check_budget();model.train()
            ids=train_index[torch.randint(len(train_index),(config['batch_size'],),device=device)]
            t=torch.randint(0,1000,(len(ids),),device=device);noise=torch.randn_like(x[ids])
            optimizer.zero_grad(set_to_none=True);loss=objective(model,ids,t,noise)
            if not torch.isfinite(loss):raise RuntimeError('NONFINITE_TRAINING_LOSS')
            loss.backward();gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not np.isfinite(gradient_norm):raise RuntimeError('NONFINITE_TRAINING_GRADIENT')
            optimizer.step();updates=update;last_loss=float(loss.detach())
            with torch.no_grad():
                for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,1-config['ema_decay'])
            if update%config['log_every']==0 or update==1:
                record=dict(update=update,train_v_mse=last_loss,gradient_norm=gradient_norm,wall_seconds=time.perf_counter()-start)
                with (path/'loss.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            if update%config['validate_every']==0 or update==config['updates']:
                value=validation();improved=value<best
                if improved:best=value
                record=dict(update=update,val_v_mse=value,best_val_v_mse=best,selection='minimum fixed VAL v-MSE, EMA weights')
                with (path/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                payload=checkpoint(update);save_checkpoint(path/'last.pt',payload)
                if improved:save_checkpoint(path/'best.pt',payload)
                s=read_state();s['latest_verified_checkpoint']=dict(path=str(path/'last.pt'),sha256=file_hash(path/'last.pt'),update=update)
                atomic_json(ORCH/'state.json',s)
                print(json.dumps(record),flush=True)
    except Exception:
        save_checkpoint(path/'interrupted.pt',checkpoint(updates));raise
    return dict(passed=updates==config['updates'],updates=updates,initial_val_v_mse=initial_val,best_val_v_mse=best,
                last_train_v_mse=last_loss,train=len(trainids),val=len(valids),free_dim=x.shape[1],
                checkpoint=str(path/'best.pt'),checkpoint_hash=file_hash(path/'best.pt'),dataset_hash=manifest['data_hash'],
                seconds=time.perf_counter()-start,architecture_status='NOT_RUN',learning_benefit_status='NOT_RUN')
