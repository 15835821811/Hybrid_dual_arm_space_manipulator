"""Publish the completed report from audited evidence without new experiments."""
from pathlib import Path
from datetime import datetime
import numpy as np
from ff_prior.storage import read_json,atomic_json,digest,file_digest


def main():
    out=Path('artifacts/v7_1/equal_budget_residual_001'); docs=Path('ff_prior_cost/docs')
    summary=read_json(out/'summary.json'); status=read_json(out/'status.json')
    audit=read_json(out/'independent_verification.json'); train=read_json(out/'learning/training.json')
    resume=read_json('ff_prior_cost/evidence/resume_verification.json')
    assert summary['status']=='FINISHED' and audit['status']=='PASS' and resume['status']=='PASS'
    methods=summary['methods']; m0='structured_final'; m1='frozen_diffusion_two'; m2='frozen_diffusion_final'; m3='trained_diffusion_final'
    comparisons={}
    identical=True
    for split,values in methods.items():
        a,b,c=values[m0],values[m1],values[m2]
        comparisons[split]={'query_saving_final_vs_two':1-c['coupled_queries']/b['coupled_queries'],
            'wall_saving_final_vs_two':1-c['candidate_wall_s']/b['candidate_wall_s'],
            'final_queries_over_structured':c['coupled_queries']/a['coupled_queries'],
            'final_wall_over_structured':c['candidate_wall_s']/a['candidate_wall_s']}
        for p in (out/'candidates'/split/m2).glob('*.json'):
            r=read_json(p); other=read_json(out/'candidates'/split/m3/p.name)
            identical=identical and r.get('reference_hash')==other.get('reference_hash')
    projections=[]; decomposition=[]
    for p in (out/'candidates').glob('*/*/*.json'):
        r=read_json(p)
        for step in r.get('sampling',{}).get('steps',[]): decomposition.append(step['decomposition_residual_after'])
        if r['status']=='DONE':
            active=[g for g in r['sampling']['guidance'] if g.get('active')]
            projections.append({'split':p.parents[1].name,'method':r['method'],
                'before_normalized_error':float(np.linalg.norm(active[-1]['initial_normalized_pose_error'])),
                'repair_norm_rad':sum(g['goal_correction_norm_rad'] for g in active),
                'iterations':sum(len(g['history']) for g in active)})
    derived={'comparisons':comparisons,'selected_and_frozen_final_references_identical':identical,
        'maximum_DDIM_decomposition_residual':max(decomposition),'projection_metrics':projections}
    atomic_json('ff_prior_cost/evidence/derived_analysis.json',derived)
    label=read_json('ff_prior_cost/evidence/teacher_representation_audit.json')
    best_new=min(train['history'],key=lambda r:np.mean(r['val_clean_mse']))
    new_loss=float(np.mean(best_new['val_clean_mse'])); old_loss=float(np.mean(train['initial_val_clean_mse']))
    report=f'''# 同预算修正与残差训练：最终分析

本轮后台运行 `artifacts/v7_1/equal_budget_residual_001` 已完成，耗时 **{status['elapsed_s']:.3f}秒**；执行源码提交 `3116e9e7`。72/72回归测试通过。独立只读证据审计通过，已完成resume核验，没有重复工作。

## 最重要的结论

**仅在最后一步修正，减少了Diffusion反复破坏再修复世界目标的计算；这次小规模微调没有带来可采用的新权重，Diffusion相对样条的额外收益仍未成立。**

VAL完成后、DEV开始前冻结的推荐方案为 `structured_final`。原四个DEV任务、每方法五候选与原种子保持；没有根据DEV更换方案或重新训练。

| 分组 | 方法 | MODEL_VALID | 世界目标合格 | 同时合格 | 预选执行与重放 | 候选总耦合查询 | 候选wall秒 |
|---|---|---:|---:|---:|---:|---:|---:|
'''
    names={m0:'样条+末步修正',m1:'原Diffusion+两步修正',m2:'原Diffusion+末步修正',m3:'训练选择分支（保留原权重）'}
    for split,values in methods.items():
        for m,r in values.items():
            n=r['candidate_denominator']; t=r['task_denominator']
            report+=f"| {split} | {names[m]} | {r['model_valid']}/{n} | {r['goal_valid']}/{n} | {r['eligible']}/{n} | {r['execution_verified']}/{t} | {r['coupled_queries']} | {r['candidate_wall_s']:.3f} |\n"
    report+='\n## 修正成本与失败机制\n\n'
    for split,r in comparisons.items():
        report+=f"- {split}：末步修正相对两步修正减少耦合查询 **{100*r['query_saving_final_vs_two']:.2f}%**，候选wall减少 **{100*r['wall_saving_final_vs_two']:.2f}%**；相对样条，查询仍为 **{r['final_queries_over_structured']:.3f}倍**、wall为 **{r['final_wall_over_structured']:.3f}倍**。\n"
    report+='\n逐步诊断仅使用6 TRAIN和6 VAL目标，各一个固定种子，额外前向探测单独计账：\n\n'
    for split in ('TRAIN','VAL'):
        rows=[r for r in summary['diagnosis'] if r['split']==split]
        d={k:float(np.median([r[k] for r in rows])) for k in ('initializer','first_network','repaired_14','network_15','prior_15','smooth_15','repaired_15')}
        report+=f"- {split}的归一化12维位姿误差中位数：初值 {d['initializer']:.4f}；高噪声首次clean估计 {d['first_network']:.4f}；第14步修正后 {d['repaired_14']:.4f}；第15步网络预测后 {d['network_15']:.4f}；时间先验投影后 {d['prior_15']:.4f}；平滑后 {d['smooth_15']:.4f}；最终修正后 {d['repaired_15']:.4f}。\n"
    report+=f'''
这些直接观测支持：前一步已经满足的世界目标没有被后续网络更新和时间投影硬性保留，导致再修正。平滑在这些案例中影响较小。共享修正器本身不变，把最终修正留到不会再被网络覆盖的位置，可以节约求解。

DDIM重构一致性最大残差为 {max(decomposition):.3g}；没有非有限候选或额外浮点放宽。当前观测不支持把失败直接归咎于微重力或以固定基座替代。自由漂浮改变了逆映射，但本轮同一真实耦合模型下样条成功，说明还有采样与学习问题。归一化误差范数不是毫米值；成功判据仍逐臂5mm/1度及所有物理门槛。

## 数据与训练结果

- 独立辅助集合：30/30 TRAIN、6/6 VAL修正标签合格。它们继承原TRAIN/VAL家族，冻结同一世界目标，来自公共初值+真实自由漂浮前向修正，没有使用私有终态。原166 TRAIN集合和旧checkpoint保留。
- TRAIN最大目标位置误差 {label['TRAIN']['max_goal_position_m']*1000:.6f}mm，VAL {label['VAL']['max_goal_position_m']*1000:.6f}mm。它们只有MODEL_VALID标签，没有执行认证；VAL标签只作验证，不进入梯度。
- 这些新标签在冻结rank6时间先验下重建RMSE为TRAIN {label['TRAIN']['frozen_prior_projection_rmse_rad']:.3g}rad、VAL {label['VAL']['frozen_prior_projection_rmse_rad']:.3g}rad。因此没有证据说这批合格数据无法使用，或其主要问题是当前时间基底无法表达。
- 8个TRAIN样本的80次固定高噪声拟合，loss比值为 {train['sanity']['ratio']:.4f}，通过预设<=0.3门槛。这只检验计算链和小样本拟合能力。
- 正式240次微调实际执行，batch16、lr1e-4、v与clean残差监督，终点权重4，scaler和时间先验冻结。训练耗时（含sanity与选择）{train['wall_s']:.3f}秒。
- 原权重固定三噪声段VAL平均clean MSE为 {old_loss:.6f}；所有新权重中最好的检查点为第{best_new['update']}步，MSE {new_loss:.6f}，相对原权重恶化 {(new_loss/old_loss-1)*100:.2f}%。故按事先规则选择第 **{train['selected_update']}步**。

**“训练选择分支”实际保留了旧权重，不能写成新Diffusion模型训练成功。** 它与原模型末步修正的全部参考轨迹哈希一致：{identical}。重复执行不构成新的独立模型证据。

训练集拟合与VAL退化符合过拟合或训练目标与规划目的不一致的现象，但这轮没有逐项隔离数据量、损失权重、学习率或条件表示的作用，不能宣布其中某一个是唯一原因。训练使用关节残差监督；真实世界目标在前向诊断和验收中检查，未实现或宣称可微全动力学损失。

## 真实物理审计

- 正式200候选、{audit['formal_network_calls']}次网络调用，最多113次耦合查询/候选（110修正+3模型验证），实际最大 {audit['maximum_candidate_queries']}。所有方法相同预算上限；实际未耗尽的额度不强行补算。
- {audit['real_torque_executions']}次预选67通道力矩执行及对应独立ctrl重放，共 {audit['physics_steps']}物理步。这是MuJoCo真实力矩积分，不是硬件实验。
- 最大hold位置误差 {audit['maximum_hold_position_m']*1000:.6f}mm，姿态 {np.rad2deg(audit['maximum_hold_rotation_rad']):.6f}度；最小全机净空 {audit['minimum_execution_clearance_m']*1000:.6f}mm。验收门槛保持仅不穿透>=0m，没有以实测净空替换门槛。
- 初态、模型、17→67映射、全机几何、2mm几何细化、自由基座、关节/力矩/动量/跟踪与1e-10重放容差保持。仅各自预选执行的样本获得EXECUTION_VERIFIED，其余模型样本不升级。
- 独立审计核对源快照、checkpoint身份、同任务种子、预算账本、原始ctrl哈希/native状态和hold误差；审计额外模型/网络/物理调用为0。旧结果与用户修改字节保持，main和旧分支保持。

## 本轮目标的完成情况

软件实现：完成并通过72项回归；固定目标标签：36条通过模型与目标验证；采样调度：完成同预算实测；选定物理执行与独立重放：见上表；新训练权重的收益：未成立且未采用；Diffusion相对样条必要性：{summary['diffusion_necessity']}。

主张范围仅本轮6 VAL+原4个开发用DEV的smoke；同一DEV已多轮使用，不是独立泛化证据。控制点差异不等于避障拓扑或strict_multimodal。没有启动pilot、study、strict_multimodal、TEST或生产规模，也没有上传大数据。

本次证据支持继续保留样条+末步耦合修正作为默认规划方案，Diffusion末步修正作为研究对照。若继续学习研究，需要另行冻结针对条件泛化/任务误差的诊断，不应据此自动扩大数据或计算规模。本轮不自动启动这些后续实验。

## 终端复查

```powershell
.\\ff_prior_cost\\job.ps1 status
.\\ff_prior_cost\\job.ps1 report
.\\ff_prior_cost\\job.ps1 resume
```

完成后的resume返回NO_WORK_ALREADY_COMPLETE，并核对全部运行文件未改变。详细目标见EXPERIMENT_PLAN.md，原始运行在artifacts/v7_1/equal_budget_residual_001，紧凑JSON证据提交在ff_prior_cost/evidence；权重与完整轨迹仅保留本地。
'''
    tracker='''# 实验跟踪

| Run ID | 阶段 | 实际结果 | 状态 |
|---|---|---|---|
| equal_budget_residual_001 | M0 | 冻结旧状态，72/72回归，源码提交3116e9e7 | COMPLETE |
| equal_budget_residual_001 | M1 | 12个逐步诊断，30 TRAIN+6 VAL教师合格 | COMPLETE |
| equal_budget_residual_001 | M2 | sanity通过，240次微调；VAL选回原权重，收益未成立 | COMPLETE_NEGATIVE_RESULT |
| equal_budget_residual_001 | M3 | 先VAL冻结选择后原DEV；200候选与预选执行 | COMPLETE |
| equal_budget_residual_001 | M4 | 原始证据审计、恢复不重复、报告 | COMPLETE |
'''
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    for name,content in (('ANALYSIS',report),('EXPERIMENT_TRACKER',tracker)):
        for filename in (name+'_'+stamp+'.md',name+'.md'):
            (docs/filename).write_text(content,encoding='utf-8')
            line=f'| {datetime.now():%Y-%m-%d %H:%M:%S} | experiment-plan | ff_prior_cost/docs/{filename} | experiment | audited equal-budget smoke results |\n'
            for p in (Path('MANIFEST.md'),docs/'MANIFEST.md'):
                with p.open('a',encoding='utf-8') as f:f.write(line)
    print({'report':str(docs/'ANALYSIS.md'),'derived':derived['comparisons']})


if __name__=='__main__':main()
