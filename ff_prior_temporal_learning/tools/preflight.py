import json
import time
import numpy as np
import torch
from ff_prior.storage import digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.diffusion import weight_hash
from ff_prior_temporal_learning.protocol import ROOT,PACKAGE,P
from ff_prior_temporal_learning.inputs import identity
from ff_prior_temporal_learning.data import load_data,source_environment,development_environment
from ff_prior_temporal_learning.features import public_features,decode_reference,encode_reference
from ff_prior_temporal_learning.representation import prepare
from ff_prior_temporal_learning.models import SequenceNetwork,sample

def main():
    torch.set_num_threads(1);checks=[];ident=identity()
    def check(name,value):assert value,name;checks.append(name)
    codec,arrays,rows,features=load_data(lambda *a,**k:None)
    check('eight_TRAIN_four_VAL',len(arrays['train']['x'])==8 and len(arrays['validation']['x'])==4)
    check('codec_TRAIN_only',codec['fit_split']=='TRAIN' and codec['fit_count']==8)
    for row,f in zip(rows,features):
        env,scene=source_environment(row);r=env.robot
        with np.load(ROOT/row['reference'],allow_pickle=False) as a:c=a['coefficients'].copy()
        reconstructed=decode_reference(encode_reference(c,f,codec),f,codec)
        check(row['unit']+'_codec_roundtrip',np.max(abs(reconstructed-c))<1e-12)
        prepared,report=prepare(c,f['q0'],f['v0'],r.lower,r.upper,r.spec.planner_velocity_limits,r.spec.planner_acceleration_limits)
        check(row['unit']+'_qualified_prepare_identity',not report['changed'] and np.array_equal(prepared.c,c) and all(report['gates'].values()))
        del env
    for scene_name in ['stationary','drifting']:
        env,scene=development_environment(scene_name);before=env.initial.values.copy();f=public_features(env,scene)
        check(scene_name+'_full_condition_schema_matches',f['schema']==codec['schema'])
        check(scene_name+'_zero_physics_initial_preserved',env.data.time==0 and np.array_equal(before,env.initial.values))
        check(scene_name+'_temporal_public_condition',f['local_condition'].shape==(271,36) and not np.array_equal(f['local_condition'][0],f['local_condition'][-1]))
        del env
    torch.manual_seed(20261022);model=SequenceNetwork(arrays['train']['global_condition'].shape[1])
    # Synthetic only: test differentiability of all 16 DDIM steps, no actual TRAIN update.
    g=torch.zeros((2,model.condition_dim));local=torch.zeros((2,271,36));initial=weight_hash(model)
    start=time.monotonic();x,steps=sample(model,g,local,[123,456],'temporal_diffusion')
    check('complete_sixteen_DDIM',len(steps)==16 and x.shape==(2,271,17))
    loss=(x-1).square().mean();loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    check('sampler_VJP_finite_nonzero',all(torch.isfinite(v).all() for v in grads) and sum(float(v.abs().sum()) for v in grads)>0)
    check('zero_optimizer_steps_in_preflight',initial==weight_hash(model))
    with torch.no_grad():again,_=sample(model,g,local,[123,456],'temporal_diffusion')
    check('frozen_noise_reproducible',torch.equal(x.detach(),again))
    check('full_input_gt_legacy65',model.condition_dim>1000)
    check('formal_budget_within_P3',P['wall_s']+377.453<=21600 and P['max_bytes']+1152681165<=20*1024**3)
    check('all_new_step_budget_fixed',P['max_native_replay_steps']==16*27000)
    check('identity_preserved_by_tests',identity()==ident)
    result=dict(passed=True,count=len(checks),checks=checks,source_hashes=ident['source_hashes'],
        condition_dimension=model.condition_dim,local_condition_dimension=36,free_output_shape=[271,17],
        parameters=sum(p.numel() for p in model.parameters()),synthetic_forward_backward_seconds=time.monotonic()-start,
        actual_native_steps=0,optimizer_steps=0,not_formal_TRAIN_diagnostic=True)
    atomic_json(PACKAGE/'evidence/software_tests.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},ensure_ascii=False))

if __name__=='__main__':main()
