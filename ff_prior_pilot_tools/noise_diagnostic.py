"""Reproduce saved VAL corruption and compare learned v with zero v/conditional mean."""
from pathlib import Path
import argparse
import importlib
import numpy as np
import torch
from ff_prior.storage import read_json,atomic_json,file_digest,digest
from ff_prior.diffusion import schedule,prediction_to_clean_noise

ROOT=Path(__file__).resolve().parents[1]
STAGES={'one_k':('qualified_expansion_002','ff_prior_expand.learning'),
        'ten_k':('qualified_pilot_train_002','ff_prior_pilot_train.learning'),
        'coverage':('qualified_coverage_train_001','ff_prior_coverage_train.learning')}


def analyze(stage,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('existing diagnostic retained; choose a new output before repeating inference')
    run,module=STAGES[stage];source=ROOT/'artifacts/v7_1'/run;directory=source/'learning/goal_loss'
    training=read_json(directory/'training.json');index=read_json(source/'training_index.json')['records']
    mask=np.array([e['split']=='VAL' for e in index]);entries=[e for e in index if e['split']=='VAL']
    assert [e['sample_id'] for e in entries]==training['metadata']['validation_ids']
    with np.load(source/'loaded_training_arrays.npz') as a:z,bases=a['Z'][mask],a['bases'][mask]
    for i,e in enumerate(entries):
        for key in ('record','arrays'):assert file_digest(ROOT/e[key])==e[key+'_sha256']
        with np.load(ROOT/e['arrays']) as a:assert np.array_equal(z[i],a['Z'])
    conditions=[read_json(ROOT/e['record'])['condition'] for e in entries]
    model,codec=importlib.import_module(module).load(directory);torch.set_num_threads(2)
    parent=None;parent_directory=None
    if stage=='coverage':
        learning=importlib.import_module(module);parent_directory=learning.FROZEN
        parent,parent_codec=learning.frozen_learning.load(parent_directory)
        assert digest(parent_codec.state)==digest(codec.state)
    centers=np.stack([codec.center(c,b) for c,b in zip(conditions,bases)])
    x=torch.tensor(codec.normalized(np.stack([codec.encode(v,b) for v,b in zip(z,bases)]),centers),dtype=torch.float32)
    c=torch.tensor(np.stack([codec.condition(v,b) for v,b in zip(conditions,bases)]),dtype=torch.float32)
    seed=training['metadata']['contract']['seed']+1
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(seed));alphas=schedule()
    def rmse(coefficients):
        actual=bases+np.einsum('tk,nkj->ntj',codec.basis,coefficients.reshape(len(z),codec.rank,17))
        return float(np.sqrt(np.mean((actual-z)**2)))
    ridge=rmse(centers);rows=[]
    for t in (0,500,999):
        a=alphas[t];noisy=a.sqrt()*x+(1-a).sqrt()*noise;target=a.sqrt()*noise-(1-a).sqrt()*x
        with torch.no_grad():
            prediction=model(noisy,torch.full((len(x),),t),c)
            clean,_=prediction_to_clean_noise(noisy,prediction,a,'v')
            zero,_=prediction_to_clean_noise(noisy,torch.zeros_like(prediction),a,'v')
        rows.append({'t':t,'network_trajectory_rmse_rad':rmse(codec.coefficients(clean.numpy(),centers)),
            'zero_v_trajectory_rmse_rad':rmse(codec.coefficients(zero.numpy(),centers)),
            'conditional_ridge_rmse_rad':ridge,'network_v_mse':float((prediction-target).square().mean()),'zero_v_mse':float(target.square().mean())})
        if parent is not None:
            with torch.no_grad():
                pp=parent(noisy,torch.full((len(x),),t),c)
                pc,_=prediction_to_clean_noise(noisy,pp,a,'v')
            rows[-1].update(parent_same_corruption_trajectory_rmse_rad=rmse(codec.coefficients(pc.numpy(),centers)),
                            parent_same_corruption_v_mse=float((pp-target).square().mean()))
    for actual,saved in zip(rows,training['noise_bands']):assert abs(actual['network_trajectory_rmse_rad']-saved['full_trajectory_rmse_rad'])<1e-10
    weight=training['metadata']['contract']['goal_loss_weight']
    ratios=[weight*r['local_pose_mse']/max(r['v_mse'],1e-30) for r in training['history']]
    result={'status':'PASS_REPRODUCED_SAVED_VAL_NOISE_DIAGNOSTICS','source':str(source),'checkpoint_sha256':file_digest(directory/'model.pt'),
        'VAL_rows':len(entries),'TEST_rows':0,'corruption_seed':seed,'new_network_forward_batch_calls':model.calls,
        'extra_physics_calls':0,'parameter_updates':0,'bands':rows,'latent_dimensions':codec.dim,
        'weighted_pose_to_v_loss_ratio_min_median_max':[min(ratios),float(np.median(ratios)),max(ratios)],
        'physical_goal_success_claim':False,'checkpoint_selection_or_protocol_change':False,
        'scope':'same saved VAL corruption as training diagnostics; trajectory reconstruction, not generated task success or physical superiority'}
    assert model.calls==3
    if parent is not None:
        assert parent.calls==3
        result.update(parent_checkpoint_sha256=file_digest(parent_directory/'model.pt'),
            parent_forward_batch_calls=parent.calls,total_diagnostic_forward_batch_calls=model.calls+parent.calls,
            parent_comparison_scope='same VAL rows, codec, input tensors, corruption seed and noise levels; no parameter updates or checkpoint selection')
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'diagnostic.json',result)
    lines=['# 已训练模型的高噪声诊断复现','',f'模型：{run}；VAL {len(entries)}条；仅3次批量网络前向，无训练更新或物理调用。','',
        '| 噪声t | 网络RMSE rad | 零v预测RMSE rad | 条件均值RMSE rad |','|---|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["t"]} | {r["network_trajectory_rmse_rad"]:.9f} | {r["zero_v_trajectory_rmse_rad"]:.9f} | {r["conditional_ridge_rmse_rad"]:.9f} |')
    if parent is not None:
        lines+=['','另用旧10k权重做3次批量前向，诊断合计6次；与新权重使用完全相同的VAL样本、编码器和腐蚀噪声。',
                '| 噪声t | 新权重RMSE rad | 旧10k权重同噪声RMSE rad |','|---|---:|---:|']
        for r in rows:lines.append(f'| {r["t"]} | {r["network_trajectory_rmse_rad"]:.9f} | {r["parent_same_corruption_trajectory_rmse_rad"]:.9f} |')
    lines+=['',f'训练日志中加权局部目标损失/v损失的最小值、中位数、最大值：{result["weighted_pose_to_v_loss_ratio_min_median_max"]}。',
        '目标损失的位姿分量已经按5mm/1°归一化，不能仅从0.05权重推断它不起作用。',
        '这些数值复现原保存噪声诊断，反映轨迹重建质量。它们不是完整DDIM采样后的世界目标成功率，也不是物理约束通过率；不据此选择检查点或修改已冻结训练协议。']
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=STAGES,default='one_k');parser.add_argument('--output',required=True)
    args=parser.parse_args();r=analyze(args.stage,ROOT/args.output);print({'status':r['status'],'bands':r['bands']})
