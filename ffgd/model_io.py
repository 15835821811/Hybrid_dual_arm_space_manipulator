"""Checkpoint/scaler-only inference IO. Never reads training trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .model import ConditionalDenoiser


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
