"""Bounded same-noise comparison of a known train/inference schema mismatch."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .problem_io import load_problem
from .model_io import load_model,normalize_condition
from .conditioning import encode_condition,CONDITION_SCHEMA
from .api import boundary_codec
from .physics import DifferentiableTree
from .propagation import Propagator
from .geometry import Geometry
from .costs import MultiSpaceEnergy
from .diffusion import sample,SamplingBudget
from .orchestration import atomic_json,check_budget


def run(output,config):
    torch.set_num_threads(1);model,checkpoint,identity=load_model(config['checkpoint'],config.get('device','cpu'))
    dataset=Path(checkpoint['dataset_manifest']['dataset_path'])/'prior.npz'
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=checkpoint['dataset_manifest']['data_hash']:raise ValueError('CONDITION_AUDIT_DATA_HASH_MISMATCH')
    with np.load(dataset) as data:
        training=data['tokens'][data['is_train']];padding=data['padding'][data['is_train']]
        scene_tokens=int(np.count_nonzero(training[...,4][~padding]))
    r,snapshot,task,problem=load_problem(config['problem']);codec=boundary_codec(r,snapshot,task.duration)
    prop=Propagator(DifferentiableTree(r),snapshot);energy=MultiSpaceEnergy(prop,codec,Geometry(r),task)
    rows=[];samples={}
    for include_scene in (True,False):
        check_budget();tokens=encode_condition(r,snapshot,task,prop.H.numpy(),include_scene=include_scene)
        condition=normalize_condition(tokens[None],np.zeros((1,len(tokens)),bool),checkpoint,config.get('device','cpu'))
        result=sample(model,condition,codec.free_dim,config['seed'],SamplingBudget(steps=60,guide_steps=0),mean=checkpoint['mean'],scale=checkpoint['scale'])
        z=result['free'];name='untrained_scene_token' if include_scene else CONDITION_SCHEMA;samples[name]=z
        times=np.linspace(0,task.duration,1025)
        rows.append(dict(condition=name,token_count=len(tokens),network_calls=result['network_calls'],initial_noise_hash=result['initial_noise_hash'],
                         normalized_free_rms=float(np.sqrt(np.mean(((z-checkpoint['mean'])/checkpoint['scale'])**2))),
                         native_energy=energy.native_evaluate(z),
                         maximum_acceleration_ratio=float(np.max(abs(codec.evaluate(z,times,2))/r.spec.planner_acceleration_limits)),
                         scene_geometry_count=len(energy.geometry.ids),all_pairs=len(energy.geometry.pairs)))
        atomic_json(output/'comparison.json',rows)
    np.savez_compressed(output/'diagnostic_samples.npz',**samples)
    return dict(passed=scene_tokens==0 and rows[0]['initial_noise_hash']==rows[1]['initial_noise_hash'],
                scope='condition contract diagnostic, no selected/executable plan and no extra main candidate',
                checkpoint=identity,training_scene_token_count=scene_tokens,comparisons=rows,
                finding='Scene tokens were never trained; physical scene is identical in both native evaluations. No goal or obstacle was changed.')
