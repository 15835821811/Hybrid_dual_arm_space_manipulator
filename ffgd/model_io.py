"""Checkpoint/scaler-only inference IO. Never reads training trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .model import ConditionalDenoiser


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_free_coordinates(checkpoint,duration):
    """Convert learned free-coordinate units to physical spline coordinates.

    duration_squared_v1 uses z_physical = T_seconds**2 * z_label. This is
    invertible for T>0 and changes neither the codec nullspace nor task time.
    Checkpoint arrays are never mutated; repeated binding at the same T is safe.
    """
    if not np.isfinite(duration) or duration<=0:raise ValueError('INVALID_FREE_COORDINATE_DURATION')
    schema=checkpoint['dataset_manifest'].get('free_coordinate_schema','physical_position_v1')
    if schema=='physical_position_v1':return checkpoint
    if schema!='duration_squared_v1':raise ValueError('UNSUPPORTED_FREE_COORDINATE_SCHEMA')
    if '_bound_free_duration' in checkpoint:
        if checkpoint['_bound_free_duration']!=duration:raise ValueError('FREE_COORDINATES_ALREADY_BOUND_TO_OTHER_DURATION')
        return checkpoint
    factor=float(duration)*float(duration)
    if not np.isfinite(factor) or factor<=0:raise ValueError('INVALID_FREE_COORDINATE_FACTOR')
    return dict(checkpoint,mean=np.asarray(checkpoint['mean'])*factor,
                scale=np.asarray(checkpoint['scale'])*factor,_bound_free_duration=float(duration))


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
