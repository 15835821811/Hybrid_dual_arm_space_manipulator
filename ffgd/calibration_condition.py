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
    if config.get('mode')=='train_mean_factorial':return train_mean_factorial(output,config)
    torch.set_num_threads(1);model,checkpoint,identity=load_model(config['checkpoint'],config.get('device','cpu'))
    dataset=Path(checkpoint['dataset_manifest']['dataset_path'])/'prior.npz'
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=checkpoint['dataset_manifest']['data_hash']:raise ValueError('CONDITION_AUDIT_DATA_HASH_MISMATCH')
    with np.load(dataset) as data:
        training=data['tokens'][data['is_train']];padding=data['padding'][data['is_train']]
        scene_tokens=int(np.count_nonzero(training[...,4][~padding]))
    r,snapshot,task,problem=load_problem(config['problem']);codec=boundary_codec(r,snapshot,task.duration,origin=checkpoint['dataset_manifest'].get('boundary_origin','pivot_zero'))
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


def train_mean_factorial(output,config):
    """Isolated conditioning intervention, never an executable planning entry.

    Only network tokens are ablated; all native measurements use the original
    public initial state and task. TRAIN token statistics supply no trajectory.
    This diagnoses sensitivity, not a permissible substitute for conditioning.
    """
    torch.set_num_threads(1)
    device=config.get('device','cpu')
    model,checkpoint,identity=load_model(config['checkpoint'],device)
    dataset=Path(checkpoint['dataset_manifest']['dataset_path'])/'prior.npz'
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=checkpoint['dataset_manifest']['data_hash']:
        raise ValueError('CONDITION_AUDIT_DATA_HASH_MISMATCH')
    with np.load(dataset) as data:
        train=data['tokens'][data['is_train']];padding=data['padding'][data['is_train']]
        observed=train[~padding]
        q_mean=observed[observed[:,0]==1,5:22].mean(0)
        pose_mean=observed[observed[:,3]==1,9:18].mean(0)
    r,snapshot,task,problem=load_problem(config['problem'])
    original_state_hash=hashlib.sha256(snapshot.state.tobytes()).hexdigest()
    problem_hash=hashlib.sha256(Path(config['problem']).read_bytes()).hexdigest()
    codec=boundary_codec(r,snapshot,task.duration,origin=checkpoint['dataset_manifest'].get('boundary_origin','pivot_zero'))
    prop=Propagator(DifferentiableTree(r),snapshot)
    energy=MultiSpaceEnergy(prop,codec,Geometry(r),task)
    original=encode_condition(r,snapshot,task,prop.H.numpy())
    rows=[];samples={};times=np.linspace(0,task.duration,1025)
    for neutral_q,neutral_pose in ((False,False),(True,False),(False,True),(True,True)):
        check_budget();tokens=original.copy()
        if neutral_q:tokens[0,5:22]=q_mean
        if neutral_pose:tokens[tokens[:,3]==1,9:18]=pose_mean
        condition=normalize_condition(tokens[None],np.zeros((1,len(tokens)),bool),checkpoint,device)
        result=sample(model,condition,codec.free_dim,config['seed'],SamplingBudget(steps=60,guide_steps=0),mean=checkpoint['mean'],scale=checkpoint['scale'])
        z=result['free'];name=f'q_mean_{int(neutral_q)}_pose_mean_{int(neutral_pose)}';samples[name]=z
        rows.append(dict(intervention=name,network_calls=result['network_calls'],initial_noise_hash=result['initial_noise_hash'],
            condition_hash=hashlib.sha256(tokens.tobytes()).hexdigest(),
            normalized_free_rms=float(np.sqrt(np.mean(((z-checkpoint['mean'])/checkpoint['scale'])**2))),
            native_energy_original_problem=energy.native_evaluate(z),
            maximum_velocity_ratio=float(np.max(abs(codec.evaluate(z,times,1))/r.spec.planner_velocity_limits)),
            maximum_acceleration_ratio=float(np.max(abs(codec.evaluate(z,times,2))/r.spec.planner_acceleration_limits))))
        atomic_json(output/'comparison.json',rows)
    np.savez_compressed(output/'non_executable_diagnostic_samples.npz',**samples)
    unchanged=(original_state_hash==hashlib.sha256(snapshot.state.tobytes()).hexdigest()
               and problem_hash==hashlib.sha256(Path(config['problem']).read_bytes()).hexdigest())
    return dict(passed=unchanged and len({x['initial_noise_hash'] for x in rows})==1,
        scope='Diagnostic integrity only; ablated conditions are NOT allowed in main planning, no selection or execution',
        original_problem_unchanged=unchanged,checkpoint=identity,training_samples=len(train),
        interventions=['initial q token TRAIN mean','pose position/rotation token TRAIN mean'],
        comparisons=rows,main_method_changed=False)
