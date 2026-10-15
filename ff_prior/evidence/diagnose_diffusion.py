"""Read-only TRAIN/VAL and saved DEV failure diagnosis, never reads TEST arrays."""
from pathlib import Path
import argparse
import time
from collections import Counter
import numpy as np
import torch
from ff_prior.diffusion import load_checkpoint,schedule,prediction_to_clean_noise
from ff_prior.storage import read_json,atomic_json,file_digest
from ff_prior.pipeline import audit,config_from
from ff_prior.spline import Reference


def main(run_dir,output,checkpoint_dir=None,plans_dir=None):
    wall=time.perf_counter()
    run=Path(run_dir)
    manifest=read_json(run/'run.json')
    dataset=read_json(run/'dataset.json')
    checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else run/'learning'
    plans_dir=Path(plans_dir) if plans_dir else run
    model,scaler=load_checkpoint(checkpoint_dir,manifest['identity']['robot_hash'],manifest['identity']['initial_state_id'])
    torch.set_num_threads(2)
    _,robot,_,_=audit(config_from('configs/ff_prior/v7_1.proposed.yaml'))
    results={'source_run':str(run),'checkpoint_sha256':file_digest(checkpoint_dir/'model.pt'),
        'prediction_type':model.prediction_type,
        'data_scope':['TRAIN','VAL','saved DEV plans'],'TEST_read':False,'bands':{},'candidates':{}}
    for split in ('TRAIN','VAL'):
        ids=dataset['eligible'][split]
        rows=[read_json(run/'records'/f'{i}.json') for i in ids]
        z=np.stack([np.load(run/r['arrays'])['Z'] for r in rows])
        x=torch.tensor(scaler.z(z),dtype=torch.float32)
        c=torch.tensor(np.stack([scaler.condition(r['condition']) for r in rows]),dtype=torch.float32)
        noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(9182))
        bands=[]
        with torch.no_grad():
            for t in (0,66,250,500,750,999):
                a=schedule()[t]
                xt=a.sqrt()*x+(1-a).sqrt()*noise
                pred=model(xt,torch.full((len(x),),t),c)
                estimate,epsilon=prediction_to_clean_noise(xt,pred,a,model.prediction_type)
                shuffled=model(xt,torch.full((len(x),),t),c.flip(0))
                shuffled_clean,shuffled_epsilon=prediction_to_clean_noise(xt,shuffled,a,model.prediction_type)
                bands.append({'t':t,'alpha_bar':float(a),'epsilon_error_amplification':float(((1-a)/a).sqrt()),
                    'epsilon_mse':float(((epsilon-noise)**2).mean()),'x0_mse_normalized':float(((estimate-x)**2).mean()),
                    'x0_rmse_rad':float(np.sqrt(np.mean(((estimate-x).numpy()*scaler.z_scale)**2))),
                    'shuffled_condition_epsilon_mse':float(((shuffled_epsilon-noise)**2).mean()),
                    'shuffled_condition_x0_rmse_rad':float(np.sqrt(np.mean(((shuffled_clean-x).numpy()*scaler.z_scale)**2)))})
        results['bands'][split]=bands
        if split=='TRAIN':
            centered=z-z.mean(axis=0)
            sv=np.linalg.svd(centered.reshape(len(z),-1),compute_uv=False)
            energy=np.cumsum(sv**2)/np.sum(sv**2)
            results['training_distribution']={'samples':len(z),'families':len({r['family_id'] for r in rows}),
                'q17_amplitude_max_rad':float(abs(z-robot.spec.planner_zero).max()),
                'latent_dimension':323,'pca_ranks_95_99_999':[int(np.searchsorted(energy,e)+1) for e in (.95,.99,.999)],
                'durations_s':sorted({r['condition']['duration'] for r in rows}),
                'normalized_abs_max':float(abs(x).max())}
    for method in ('structured_prior_same_guidance','diffusion_in_denoising_guidance','diffusion_no_guidance'):
        failures=Counter(); values=[]
        for p in (plans_dir/'benchmark').glob(f'*_{method}/planning.json'):
            result=read_json(p)
            for c in result['candidates']:
                z=np.asarray(c['Z'])
                ref=Reference(z,robot.spec.planner_zero,10)
                velocity=max(abs(v/robot.spec.planner_velocity_limits).max() for v in ref.bounds(1))
                acceleration=max(abs(v/robot.spec.planner_acceleration_limits).max() for v in ref.bounds(2))
                excess=max(float((z-robot.upper).max()),float((robot.lower-z).max()),0.)
                failures.update(k for k,v in c['validation']['gates'].items() if not v)
                values.append({'task':result['task_hash'],'candidate_id':c['candidate_id'],
                    'domain_excess_rad':excess,'maximum_amplitude_rad':float(abs(z-robot.spec.planner_zero).max()),
                    'velocity_limit_ratio':float(velocity),'acceleration_limit_ratio':float(acceleration)})
        results['candidates'][method]={'failed_gates':dict(failures),'values':values}
    results['wall_s']=time.perf_counter()-wall
    atomic_json(output,results)
    print({k:v for k,v in results.items() if k!='candidates'})
    print({m:{'failures':v['failed_gates'],'domain_excess_range':[min(x['domain_excess_rad'] for x in v['values']),max(x['domain_excess_rad'] for x in v['values'])],
        'acceleration_ratio_range':[min(x['acceleration_limit_ratio'] for x in v['values']),max(x['acceleration_limit_ratio'] for x in v['values'])]} for m,v in results['candidates'].items() if v['values']})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--output',required=True)
    p.add_argument('--checkpoint-dir'); p.add_argument('--plans-dir')
    a=p.parse_args(); main(a.run_dir,a.output,a.checkpoint_dir,a.plans_dir)
